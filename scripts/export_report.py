#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print CrashTwin summary metrics as a Markdown table.")
    parser.add_argument("summary_csv", type=Path, help="Path to summary_metrics.csv.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.summary_csv.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("No rows found.")
        return 1

    fieldnames = rows[0].keys()
    print("| " + " | ".join(fieldnames) + " |")
    print("| " + " | ".join("---" for _ in fieldnames) + " |")
    for row in rows:
        print("| " + " | ".join(row.get(field, "") for field in fieldnames) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

