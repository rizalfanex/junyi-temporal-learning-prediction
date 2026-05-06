from __future__ import annotations

import torch
from torch import nn


class TemporalInputProjection(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        num_numeric_features: int,
        embedding_dim: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embedding_dim, padding_idx=0) for vocab_size in vocab_sizes]
        )
        self.num_numeric_features = num_numeric_features
        self.numeric_projection = nn.Linear(num_numeric_features, embedding_dim) if num_numeric_features else None
        input_dim = embedding_dim * len(vocab_sizes) + (embedding_dim if num_numeric_features else 0)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        parts = []
        for idx, embedding in enumerate(self.embeddings):
            parts.append(embedding(categorical[:, :, idx]))
        if self.numeric_projection is not None:
            parts.append(self.numeric_projection(numeric))
        return self.input_projection(torch.cat(parts, dim=-1))


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        num_numeric_features: int,
        sequence_length: int,
        embedding_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = TemporalInputProjection(
            vocab_sizes=vocab_sizes,
            num_numeric_features=num_numeric_features,
            embedding_dim=embedding_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(categorical, numeric)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        encoded = self.encoder(x, src_key_padding_mask=~mask.bool())
        lengths = mask.long().sum(dim=1).clamp(min=1)
        last_indices = lengths - 1
        pooled = encoded[torch.arange(encoded.size(0), device=encoded.device), last_indices]
        return self.head(pooled).squeeze(-1)


class TemporalGRU(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        num_numeric_features: int,
        embedding_dim: int,
        d_model: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = TemporalInputProjection(
            vocab_sizes=vocab_sizes,
            num_numeric_features=num_numeric_features,
            embedding_dim=embedding_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(categorical, numeric)
        lengths = mask.long().sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        pooled = hidden[-1]
        return self.head(pooled).squeeze(-1)


def enable_mc_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
