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
from crashtwin.benchmark_files import (  # noqa: E402
    format_benchmark_file_errors,
    scan_benchmark_files,
    write_benchmark_file_report,
)
from crashtwin.preprocess import run_preprocess  # noqa: E402
from crashtwin.reconstruct import run_reconstruction  # noqa: E402
from crashtwin.scoring import collect_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrashTwin-Eval for one generated-video model.")
    parser.add_argument("--method-name", required=True, help="Model name used in output tables.")
    parser.add_argument("--predictions", required=True, type=Path, help="Folder of generated .mp4 files.")
    parser.add_argument(
        "--benchmark",
        default=REPO_ROOT / "benchmark" / "crashtwin_eval.csv",
        type=Path,
        help="CrashTwin-Eval manifest CSV.",
    )
    parser.add_argument(
        "--config",
        default=REPO_ROOT / "configs" / "default.yaml",
        type=Path,
        help="Official evaluation config.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output folder.")
    parser.add_argument(
        "--benchmark-root",
        default=REPO_ROOT,
        type=Path,
        help="Repository root containing CrashTwin-Eval benchmark files.",
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

    benchmark_file_scan = scan_benchmark_files(args.benchmark_root, benchmark)
    write_benchmark_file_report(
        benchmark_file_scan, args.output / "benchmark_file_validation.csv"
    )
    if benchmark_file_scan.has_errors:
        print(format_benchmark_file_errors(benchmark_file_scan), file=sys.stderr)
        print(
            f"Benchmark-file report: {args.output / 'benchmark_file_validation.csv'}",
            file=sys.stderr,
        )
        return 2

    staged_inputs = stage_predictions(scan, args.output / "staged_inputs", copy_mode=args.copy_mode)
    per_video_dir = args.output / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preprocess:
        run_preprocess(
            repo_root=REPO_ROOT,
            inputs=staged_inputs,
            benchmark=args.benchmark,
            benchmark_root=args.benchmark_root,
            output=per_video_dir,
            config=args.config,
            gpus=args.gpus,
            dry_run=args.dry_run,
        )
    if not args.skip_reconstruction:
        run_reconstruction(
            repo_root=REPO_ROOT,
            benchmark=args.benchmark,
            benchmark_root=args.benchmark_root,
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
