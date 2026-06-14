#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from crashtwin.io import format_scan_errors, read_benchmark, scan_prediction_folder, write_validation_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CrashTwin prediction video names.")
    parser.add_argument("--predictions", required=True, type=Path, help="Folder of generated .mp4 files.")
    parser.add_argument(
        "--benchmark",
        default=REPO_ROOT / "benchmark" / "crashtwin_344.csv",
        type=Path,
        help="CrashTwin benchmark CSV.",
    )
    parser.add_argument("--output-csv", type=Path, help="Optional validation report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = read_benchmark(args.benchmark)
    scan = scan_prediction_folder(args.predictions, benchmark)
    if args.output_csv:
        write_validation_report(scan, args.output_csv)

    print(f"Expected videos: {len(scan.benchmark)}")
    print(f"Matched videos: {len(scan.matched)}")
    print(f"Missing videos: {len(scan.missing)}")
    print(f"Duplicated IDs: {len(scan.duplicates)}")
    print(f"Unknown files: {len(scan.unknown)}")
    if scan.has_errors:
        print(format_scan_errors(scan), file=sys.stderr)
        return 2
    print("Input validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

