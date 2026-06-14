from __future__ import annotations

import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LONG_FORM_RE = re.compile(
    r"^(?P<left>.+)__(?P<index>\d{6})_(?P<split>real|syn|synthetic)_test_(?P<right>.+)__output$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BenchmarkItem:
    video_id: str
    split: str


@dataclass(frozen=True)
class PredictionScan:
    benchmark: tuple[BenchmarkItem, ...]
    matched: dict[str, Path]
    missing: tuple[str, ...]
    duplicates: dict[str, tuple[Path, ...]]
    unknown: tuple[Path, ...]

    @property
    def has_errors(self) -> bool:
        return bool(self.missing or self.duplicates or self.unknown)


def canonical_video_id(path: Path | str) -> str | None:
    """Return the benchmark video ID represented by an input video filename."""

    filename = Path(path)
    if filename.suffix.lower() != ".mp4":
        return None

    stem = filename.stem
    match = LONG_FORM_RE.match(stem)
    if match:
        left = match.group("left")
        right = match.group("right")
        return left if left == right else None
    return stem


def read_benchmark(path: Path | str) -> tuple[BenchmarkItem, ...]:
    benchmark_path = Path(path)
    with benchmark_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "split"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"{benchmark_path} must contain columns: {', '.join(sorted(required))}"
            )
        rows = tuple(
            BenchmarkItem(video_id=row["video_id"].strip(), split=row["split"].strip())
            for row in reader
            if row.get("video_id", "").strip()
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for item in rows:
        if item.video_id in seen:
            duplicates.append(item.video_id)
        seen.add(item.video_id)
    if duplicates:
        raise ValueError(f"Duplicate video IDs in benchmark: {', '.join(duplicates)}")
    return rows


def scan_prediction_folder(
    predictions: Path | str, benchmark: Iterable[BenchmarkItem]
) -> PredictionScan:
    prediction_dir = Path(predictions)
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction folder not found: {prediction_dir}")

    benchmark_items = tuple(benchmark)
    expected = {item.video_id for item in benchmark_items}
    matched: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    unknown: list[Path] = []

    for path in sorted(prediction_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        video_id = canonical_video_id(path)
        if video_id is None or video_id not in expected:
            unknown.append(path)
            continue
        if video_id in matched:
            duplicates.setdefault(video_id, [matched[video_id]]).append(path)
            continue
        matched[video_id] = path

    missing = tuple(item.video_id for item in benchmark_items if item.video_id not in matched)
    return PredictionScan(
        benchmark=benchmark_items,
        matched=matched,
        missing=missing,
        duplicates={key: tuple(value) for key, value in duplicates.items()},
        unknown=tuple(unknown),
    )


def format_scan_errors(scan: PredictionScan) -> str:
    lines: list[str] = []
    if scan.missing:
        lines.append(f"Missing videos: {len(scan.missing)}")
        lines.extend(f"  - {video_id}" for video_id in scan.missing[:20])
        if len(scan.missing) > 20:
            lines.append(f"  - ... {len(scan.missing) - 20} more")
    if scan.duplicates:
        lines.append(f"Duplicated benchmark IDs: {len(scan.duplicates)}")
        for video_id, paths in list(scan.duplicates.items())[:20]:
            names = ", ".join(path.name for path in paths)
            lines.append(f"  - {video_id}: {names}")
    if scan.unknown:
        lines.append(f"Unknown video files: {len(scan.unknown)}")
        lines.extend(f"  - {path.name}" for path in scan.unknown[:20])
        if len(scan.unknown) > 20:
            lines.append(f"  - ... {len(scan.unknown) - 20} more")
    return "\n".join(lines)


def write_validation_report(scan: PredictionScan, output_csv: Path | str) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duplicate_ids = set(scan.duplicates)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "split", "status", "path"])
        writer.writeheader()
        for item in scan.benchmark:
            if item.video_id in scan.missing:
                writer.writerow(
                    {"video_id": item.video_id, "split": item.split, "status": "missing", "path": ""}
                )
            elif item.video_id in duplicate_ids:
                paths = ";".join(str(path) for path in scan.duplicates[item.video_id])
                writer.writerow(
                    {
                        "video_id": item.video_id,
                        "split": item.split,
                        "status": "duplicate",
                        "path": paths,
                    }
                )
            else:
                writer.writerow(
                    {
                        "video_id": item.video_id,
                        "split": item.split,
                        "status": "ok",
                        "path": str(scan.matched[item.video_id]),
                    }
                )
        for path in scan.unknown:
            writer.writerow({"video_id": "", "split": "", "status": "unknown", "path": str(path)})


def stage_predictions(
    scan: PredictionScan,
    stage_dir: Path | str,
    copy_mode: str = "symlink",
) -> Path:
    """Create a normalized input folder with one '<video_id>.mp4' per benchmark item."""

    if scan.has_errors:
        raise ValueError("Cannot stage predictions with validation errors.")
    if copy_mode not in {"symlink", "hardlink", "copy"}:
        raise ValueError("copy_mode must be one of: symlink, hardlink, copy")

    stage_path = Path(stage_dir)
    stage_path.mkdir(parents=True, exist_ok=True)

    for item in scan.benchmark:
        src = scan.matched[item.video_id].resolve()
        dst = stage_path / f"{item.video_id}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy_mode == "symlink":
            dst.symlink_to(src)
        elif copy_mode == "hardlink":
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    return stage_path

