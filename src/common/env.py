from __future__ import annotations

import platform
from typing import Any

import psutil


def environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }
    try:
        import torch

        summary["torch"] = torch.__version__
        summary["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            summary["cuda_device_count"] = torch.cuda.device_count()
            summary["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        summary["torch_error"] = repr(exc)
    return summary
