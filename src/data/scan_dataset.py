from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import configured_path, ensure_project_dirs, load_config
from src.common.env import environment_summary
from src.common.logging_utils import setup_logger


ID_COLUMNS = {"uuid", "ucid", "upid", "level1_id", "level2_id", "level3_id", "level4_id"}
VALUE_COUNT_COLUMNS = {
    "is_correct",
    "is_hint_used",
    "is_downgrade",
    "is_upgrade",
    "level",
    "difficulty",
    "subject",
    "learning_stage",
    "content_kind",
    "gender",
    "user_grade",
    "user_city",
    "is_self_coach",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Junyi Academy CSV files without loading them fully.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Only scan a small sample from each CSV.")
    parser.add_argument("--full", action="store_true", help="Force a full streaming scan.")
    return parser.parse_args()


def parse_datetime(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def update_type_flags(flags: dict[str, bool], value: str) -> None:
    lowered = value.lower()
    if lowered not in {"true", "false", "0", "1"}:
        flags["bool"] = False
    try:
        int(value)
    except ValueError:
        flags["int"] = False
    try:
        float(value)
    except ValueError:
        flags["float"] = False


def inferred_type(flags: dict[str, bool], column: str) -> str:
    if "timestamp" in column.lower() or "date" in column.lower():
        return "datetime-like"
    if flags["bool"]:
        return "bool-like"
    if flags["int"]:
        return "int-like"
    if flags["float"]:
        return "float-like"
    return "string"


def scan_file(path: Path, sample_rows: int, full_scan: bool) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    start = time.time()
    rows_scanned = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing = Counter()
        type_flags = {c: {"bool": True, "int": True, "float": True} for c in columns}
        unique_sets = {c: set() for c in columns if c in ID_COLUMNS or c in VALUE_COUNT_COLUMNS}
        value_counts = {c: Counter() for c in columns if c in VALUE_COUNT_COLUMNS}
        sample_values = {c: [] for c in columns}
        numeric_min: dict[str, float | None] = {c: None for c in columns}
        numeric_max: dict[str, float | None] = {c: None for c in columns}
        numeric_valid = Counter()
        time_min: dict[str, datetime | None] = {c: None for c in columns}
        time_max: dict[str, datetime | None] = {c: None for c in columns}
        time_valid = Counter()

        limit = None if full_scan else sample_rows
        for row in reader:
            rows_scanned += 1
            if limit is not None and rows_scanned > limit:
                rows_scanned -= 1
                break
            for column in columns:
                raw_value = row.get(column, "")
                value = raw_value.strip() if raw_value is not None else ""
                if value == "":
                    missing[column] += 1
                    continue
                if len(sample_values[column]) < 5 and value not in sample_values[column]:
                    sample_values[column].append(value)
                update_type_flags(type_flags[column], value)
                if column in unique_sets:
                    unique_sets[column].add(value)
                if column in value_counts:
                    value_counts[column][value] += 1
                try:
                    numeric_value = float(value)
                    numeric_valid[column] += 1
                    numeric_min[column] = numeric_value if numeric_min[column] is None else min(numeric_min[column], numeric_value)
                    numeric_max[column] = numeric_value if numeric_max[column] is None else max(numeric_max[column], numeric_value)
                except ValueError:
                    pass
                if "timestamp" in column.lower() or "date" in column.lower():
                    dt = parse_datetime(value)
                    if dt is not None:
                        time_valid[column] += 1
                        time_min[column] = dt if time_min[column] is None else min(time_min[column], dt)
                        time_max[column] = dt if time_max[column] is None else max(time_max[column], dt)

    inventory = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / 1024**2, 3),
        "rows_scanned": rows_scanned,
        "row_count": rows_scanned if full_scan else "",
        "full_scan": full_scan,
        "column_count": len(columns),
        "columns": columns,
        "elapsed_sec": round(time.time() - start, 3),
    }
    schema_rows: list[dict[str, Any]] = []
    for column in columns:
        row = {
            "file": path.name,
            "column": column,
            "inferred_type": inferred_type(type_flags[column], column),
            "missing_count": missing[column],
            "missing_rate_scanned": round(missing[column] / rows_scanned, 6) if rows_scanned else 0.0,
            "unique_count": len(unique_sets[column]) if column in unique_sets else "",
            "sample_values": json.dumps(sample_values[column], ensure_ascii=False),
            "numeric_min": numeric_min[column] if numeric_valid[column] else "",
            "numeric_max": numeric_max[column] if numeric_valid[column] else "",
            "time_min": time_min[column].isoformat(sep=" ") if time_valid[column] else "",
            "time_max": time_max[column].isoformat(sep=" ") if time_valid[column] else "",
        }
        schema_rows.append(row)

    value_rows: list[dict[str, Any]] = []
    for column, counter in value_counts.items():
        for value, count in counter.most_common(50):
            value_rows.append({"file": path.name, "column": column, "value": value, "count": count})
    return inventory, schema_rows, value_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    path: Path,
    inventories: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    value_rows: list[dict[str, Any]],
    env: dict[str, Any],
) -> None:
    schema_by_file: dict[str, list[dict[str, Any]]] = {}
    for row in schema_rows:
        schema_by_file.setdefault(row["file"], []).append(row)
    value_by_file_col: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in value_rows:
        value_by_file_col.setdefault((row["file"], row["column"]), []).append(row)

    lines = [
        "# Junyi Dataset Inventory",
        "",
        "This report is generated from the raw CSV files using streaming reads.",
        "",
        "## Environment",
        "",
    ]
    for key, value in env.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Files", ""])
    lines.append("| File | Size MB | Rows Scanned | Full Scan | Columns |")
    lines.append("|---|---:|---:|---|---:|")
    for inv in inventories:
        lines.append(
            f"| {inv['file']} | {inv['size_mb']} | {inv['rows_scanned']} | {inv['full_scan']} | {inv['column_count']} |"
        )
    for inv in inventories:
        file_name = inv["file"]
        lines.extend(["", f"## {file_name}", ""])
        lines.append("Columns: `" + "`, `".join(inv["columns"]) + "`")
        lines.extend(["", "| Column | Type | Missing | Missing Rate | Unique Count | Range / Time Range |", "|---|---|---:|---:|---:|---|"])
        for row in schema_by_file[file_name]:
            range_text = ""
            if row["time_min"] or row["time_max"]:
                range_text = f"{row['time_min']} to {row['time_max']}"
            elif row["numeric_min"] != "" or row["numeric_max"] != "":
                range_text = f"{row['numeric_min']} to {row['numeric_max']}"
            lines.append(
                f"| {row['column']} | {row['inferred_type']} | {row['missing_count']} | "
                f"{row['missing_rate_scanned']} | {row['unique_count']} | {range_text} |"
            )
        for (value_file, column), rows in value_by_file_col.items():
            if value_file != file_name:
                continue
            pairs = ", ".join(f"{r['value']}={r['count']}" for r in rows[:10])
            lines.extend(["", f"Value counts for `{column}`: {pairs}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logger("scan_dataset")
    sample_rows = args.sample_rows or int(config["scan"]["sample_rows"])
    full_scan = bool(config["scan"]["full_scan"])
    if args.quick:
        full_scan = False
    if args.full:
        full_scan = True

    raw_dir = configured_path(config, "raw_data_dir")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    inventories: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    for path in files:
        logger.info("Scanning %s (full_scan=%s)", path.name, full_scan)
        inv, schema, values = scan_file(path, sample_rows=sample_rows, full_scan=full_scan)
        inventories.append(inv)
        schema_rows.extend(schema)
        value_rows.extend(values)
        logger.info("Finished %s: %s rows scanned in %.2fs", path.name, inv["rows_scanned"], inv["elapsed_sec"])

    metrics_dir = configured_path(config, "metrics_dir")
    reports_dir = configured_path(config, "reports_dir")
    write_csv(metrics_dir / "dataset_inventory.csv", inventories)
    write_csv(metrics_dir / "dataset_schema.csv", schema_rows)
    write_csv(metrics_dir / "dataset_value_counts.csv", value_rows)
    write_markdown_report(
        reports_dir / "dataset_inventory.md",
        inventories,
        schema_rows,
        value_rows,
        environment_summary(),
    )
    logger.info("Wrote reports/dataset_inventory.md and CSV summaries under outputs/metrics")


if __name__ == "__main__":
    main()
