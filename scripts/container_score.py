#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path("/crashtwin")
sys.path.insert(0, str(REPO_ROOT))

from crashtwin.io import read_benchmark  # noqa: E402
from crashtwin.scoring import collect_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CrashTwin metrics inside Docker.")
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--per-video-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = read_benchmark(args.benchmark)
    collect_scores(
        benchmark=benchmark,
        per_video_dir=args.per_video_dir,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
