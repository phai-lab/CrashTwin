from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .io import BenchmarkItem


@dataclass(frozen=True)
class BenchmarkFileScan:
    benchmark: tuple[BenchmarkItem, ...]
    benchmark_root: Path
    missing_auto_json: tuple[str, ...]
    missing_vehicle_specs: tuple[str, ...]

    @property
    def has_errors(self) -> bool:
        return bool(self.missing_auto_json or self.missing_vehicle_specs)


def scan_benchmark_files(
    benchmark_root: Path | str, benchmark: tuple[BenchmarkItem, ...]
) -> BenchmarkFileScan:
    root = Path(benchmark_root)
    missing_auto_json: list[str] = []
    missing_vehicle_specs: list[str] = []

    for item in benchmark:
        auto_json = root / "benchmark" / "auto_json" / f"{item.video_id}_auto.json"
        vehicle_specs = (
            root / "benchmark" / "vehicle_specs" / f"{item.video_id}_vehicle_specs.json"
        )
        if not auto_json.is_file():
            missing_auto_json.append(item.video_id)
        if not vehicle_specs.is_file():
            missing_vehicle_specs.append(item.video_id)

    return BenchmarkFileScan(
        benchmark=benchmark,
        benchmark_root=root,
        missing_auto_json=tuple(missing_auto_json),
        missing_vehicle_specs=tuple(missing_vehicle_specs),
    )


def format_benchmark_file_errors(scan: BenchmarkFileScan) -> str:
    lines: list[str] = []
    if scan.missing_auto_json:
        lines.append(f"Missing auto_json files: {len(scan.missing_auto_json)}")
        lines.extend(f"  - {video_id}" for video_id in scan.missing_auto_json[:20])
        if len(scan.missing_auto_json) > 20:
            lines.append(f"  - ... {len(scan.missing_auto_json) - 20} more")
    if scan.missing_vehicle_specs:
        lines.append(f"Missing vehicle_specs files: {len(scan.missing_vehicle_specs)}")
        lines.extend(f"  - {video_id}" for video_id in scan.missing_vehicle_specs[:20])
        if len(scan.missing_vehicle_specs) > 20:
            lines.append(f"  - ... {len(scan.missing_vehicle_specs) - 20} more")
    return "\n".join(lines)


def write_benchmark_file_report(scan: BenchmarkFileScan, output_csv: Path | str) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_auto = set(scan.missing_auto_json)
    missing_specs = set(scan.missing_vehicle_specs)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_id", "split", "auto_json", "vehicle_specs", "status"],
        )
        writer.writeheader()
        for item in scan.benchmark:
            auto_ok = item.video_id not in missing_auto
            specs_ok = item.video_id not in missing_specs
            writer.writerow(
                {
                    "video_id": item.video_id,
                    "split": item.split,
                    "auto_json": "ok" if auto_ok else "missing",
                    "vehicle_specs": "ok" if specs_ok else "missing",
                    "status": "ok" if auto_ok and specs_ok else "missing",
                }
            )
