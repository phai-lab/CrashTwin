#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from crashtwin.io import (  # noqa: E402
    format_scan_errors,
    read_benchmark,
    scan_prediction_folder,
    stage_predictions,
    write_validation_report,
)
from crashtwin.metadata import format_metadata_errors, scan_metadata, write_metadata_report  # noqa: E402
from crashtwin.preprocess import run_preprocess  # noqa: E402
from crashtwin.reconstruct import run_reconstruction  # noqa: E402
from crashtwin.scoring import collect_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrashTwin evaluation for one method.")
    parser.add_argument("--method-name", required=True, help="Name used in output tables.")
    parser.add_argument("--predictions", required=True, type=Path, help="Folder of generated .mp4 files.")
    parser.add_argument(
        "--benchmark",
        default=REPO_ROOT / "benchmark" / "crashtwin_344.csv",
        type=Path,
        help="CrashTwin benchmark CSV.",
    )
    parser.add_argument(
        "--config",
        default=REPO_ROOT / "configs" / "default.yaml",
        type=Path,
        help="Official evaluation config.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output folder.")
    parser.add_argument(
        "--metadata-root",
        default=REPO_ROOT,
        type=Path,
        help="Toolkit root containing benchmark metadata files.",
    )
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs passed to Docker.")
    parser.add_argument(
        "--copy-mode",
        default="symlink",
        choices=("symlink", "hardlink", "copy"),
        help="How to stage input videos under the output folder.",
    )
    parser.add_argument("--skip-preprocess", action="store_true", help="Reuse existing preprocessed files.")
    parser.add_argument("--skip-reconstruction", action="store_true", help="Reuse existing reconstruction files.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running Docker.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = read_benchmark(args.benchmark)
    scan = scan_prediction_folder(args.predictions, benchmark)

    args.output.mkdir(parents=True, exist_ok=True)
    write_validation_report(scan, args.output / "input_validation.csv")
    if scan.has_errors:
        print(format_scan_errors(scan), file=sys.stderr)
        print(f"Validation report: {args.output / 'input_validation.csv'}", file=sys.stderr)
        return 2

    metadata_scan = scan_metadata(args.metadata_root, benchmark)
    write_metadata_report(metadata_scan, args.output / "metadata_validation.csv")
    if metadata_scan.has_errors:
        print(format_metadata_errors(metadata_scan), file=sys.stderr)
        print(f"Metadata report: {args.output / 'metadata_validation.csv'}", file=sys.stderr)
        return 2

    staged_inputs = stage_predictions(scan, args.output / "staged_inputs", copy_mode=args.copy_mode)
    per_video_dir = args.output / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preprocess:
        run_preprocess(
            repo_root=REPO_ROOT,
            inputs=staged_inputs,
            benchmark=args.benchmark,
            metadata_root=args.metadata_root,
            output=per_video_dir,
            config=args.config,
            gpus=args.gpus,
            dry_run=args.dry_run,
        )
    if not args.skip_reconstruction:
        run_reconstruction(
            repo_root=REPO_ROOT,
            benchmark=args.benchmark,
            metadata_root=args.metadata_root,
            per_video_dir=per_video_dir,
            config=args.config,
            gpus=args.gpus,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        collect_scores(benchmark=benchmark, per_video_dir=per_video_dir, output_dir=args.output)
        print(f"Summary: {args.output / 'summary_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
