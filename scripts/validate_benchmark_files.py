#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from crashtwin.io import read_benchmark  # noqa: E402
from crashtwin.benchmark_files import (  # noqa: E402
    format_benchmark_file_errors,
    scan_benchmark_files,
    write_benchmark_file_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CrashTwin-Eval benchmark files.")
    parser.add_argument(
        "--benchmark-root",
        default=REPO_ROOT,
        type=Path,
        help="Repository root containing benchmark/auto_json and benchmark/vehicle_specs.",
    )
    parser.add_argument(
        "--benchmark",
        default=REPO_ROOT / "benchmark" / "crashtwin_eval.csv",
        type=Path,
        help="CrashTwin-Eval manifest CSV.",
    )
    parser.add_argument("--output-csv", type=Path, help="Optional validation report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = read_benchmark(args.benchmark)
    scan = scan_benchmark_files(args.benchmark_root, benchmark)
    if args.output_csv:
        write_benchmark_file_report(scan, args.output_csv)

    print(f"Expected videos: {len(scan.benchmark)}")
    print(f"Missing auto_json files: {len(scan.missing_auto_json)}")
    print(f"Missing vehicle_specs files: {len(scan.missing_vehicle_specs)}")
    if scan.has_errors:
        print(format_benchmark_file_errors(scan), file=sys.stderr)
        return 2
    print("Benchmark-file validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
