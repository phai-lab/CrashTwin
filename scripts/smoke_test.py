#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("[smoke]", " ".join(str(arg) for arg in args))
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrashTwin release smoke checks.")
    parser.add_argument("--predictions", required=True, type=Path, help="Folder of generated videos.")
    parser.add_argument("--output", required=True, type=Path, help="Smoke output folder.")
    parser.add_argument(
        "--metadata-root",
        default=REPO_ROOT,
        type=Path,
        help="Toolkit root containing benchmark metadata files.",
    )
    parser.add_argument("--gpus", default="0", help="GPU IDs for dry-run command rendering.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = REPO_ROOT / "benchmark" / "crashtwin_344.csv"

    run(
        [
            sys.executable,
            "-B",
            "scripts/validate_inputs.py",
            "--predictions",
            str(args.predictions),
            "--benchmark",
            str(benchmark),
            "--output-csv",
            str(args.output / "input_validation.csv"),
        ]
    )

    run(
        [
            sys.executable,
            "-B",
            "scripts/validate_metadata.py",
            "--metadata-root",
            str(args.metadata_root),
            "--benchmark",
            str(benchmark),
            "--output-csv",
            str(args.output / "metadata_validation.csv"),
        ]
    )

    run(
        [
            sys.executable,
            "-B",
            "scripts/evaluate.py",
            "--method-name",
            "smoke",
            "--predictions",
            str(args.predictions),
            "--benchmark",
            str(benchmark),
            "--output",
            str(args.output),
            "--metadata-root",
            str(args.metadata_root),
            "--gpus",
            args.gpus,
            "--dry-run",
        ]
    )

    print("Smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
