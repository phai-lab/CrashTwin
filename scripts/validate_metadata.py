#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from crashtwin.io import read_benchmark  # noqa: E402
from crashtwin.metadata import (  # noqa: E402
    format_metadata_errors,
    scan_metadata,
    write_metadata_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CrashTwin benchmark metadata files.")
    parser.add_argument(
        "--metadata-root",
        default=REPO_ROOT,
        type=Path,
        help="Toolkit root containing benchmark/auto_json and benchmark/vehicle_specs.",
    )
    parser.add_argument(
        "--benchmark",
        default=REPO_ROOT / "benchmark" / "crashtwin_344.csv",
        type=Path,
        help="CrashTwin benchmark CSV.",
    )
    parser.add_argument("--output-csv", type=Path, help="Optional metadata report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = read_benchmark(args.benchmark)
    scan = scan_metadata(args.metadata_root, benchmark)
    if args.output_csv:
        write_metadata_report(scan, args.output_csv)

    print(f"Expected videos: {len(scan.benchmark)}")
    print(f"Missing auto_json files: {len(scan.missing_auto_json)}")
    print(f"Missing vehicle_specs files: {len(scan.missing_vehicle_specs)}")
    if scan.has_errors:
        print(format_metadata_errors(scan), file=sys.stderr)
        return 2
    print("Metadata validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

