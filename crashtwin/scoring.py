from __future__ import annotations

import csv
import json
from pathlib import Path

from .io import BenchmarkItem
from .metrics import METRIC_NAMES


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def _read_metrics(metrics_path: Path) -> tuple[dict[str, float | None], str]:
    if not metrics_path.is_file():
        return {metric: None for metric in METRIC_NAMES}, "missing_metrics"

    with metrics_path.open("r") as handle:
        payload = json.load(handle)
    metrics_payload = payload.get("metrics", payload)
    status = str(payload.get("status", "ok"))
    metrics = {metric: _as_float(metrics_payload.get(metric)) for metric in METRIC_NAMES}
    return metrics, status


def collect_scores(
    *,
    benchmark: tuple[BenchmarkItem, ...],
    per_video_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for item in benchmark:
        metrics_path = per_video_dir / item.video_id / "metrics.json"
        metrics, status = _read_metrics(metrics_path)
        row: dict[str, object] = {
            "video_id": item.video_id,
            "split": item.split,
            "status": status,
            **metrics,
        }
        rows.append(row)

    per_video_csv = output_dir / "per_video_metrics.csv"
    fieldnames = ["video_id", "split", *METRIC_NAMES, "status"]
    with per_video_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _format_float(row[key]) if key in METRIC_NAMES else row[key]
                    for key in fieldnames
                }
            )

    _write_summary(rows, output_dir / "summary_metrics.csv")
    _write_failures(rows, output_dir / "failed_videos.csv")


def _write_summary(rows: list[dict[str, object]], output_csv: Path) -> None:
    fieldnames = [*METRIC_NAMES, "num_videos", "num_failed"]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        summary: dict[str, object] = {}
        for metric in METRIC_NAMES:
            values = [_as_float(row.get(metric)) for row in rows]
            valid = [value for value in values if value is not None]
            summary[metric] = sum(valid) / len(valid) if valid else None
        summary["num_videos"] = len(rows)
        summary["num_failed"] = sum(1 for row in rows if row["status"] != "ok")
        writer.writerow(
            {
                key: _format_float(summary[key]) if key in METRIC_NAMES else summary[key]
                for key in fieldnames
            }
        )


def _write_failures(rows: list[dict[str, object]], output_csv: Path) -> None:
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "split", "status"])
        writer.writeheader()
        for row in rows:
            if row["status"] != "ok":
                writer.writerow(
                    {
                        "video_id": row["video_id"],
                        "split": row["split"],
                        "status": row["status"],
                    }
                )
