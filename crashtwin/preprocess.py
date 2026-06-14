from __future__ import annotations

from pathlib import Path

from .runner import container_path, run_compose_service


def run_preprocess(
    *,
    repo_root: Path,
    inputs: Path,
    benchmark: Path,
    benchmark_root: Path,
    output: Path,
    config: Path,
    gpus: str,
    dry_run: bool = False,
) -> None:
    command = [
        "python3",
        "/crashtwin/scripts/container_preprocess.py",
        "--inputs",
        container_path(repo_root, inputs),
        "--benchmark",
        container_path(repo_root, benchmark),
        "--benchmark-root",
        container_path(repo_root, benchmark_root),
        "--output",
        container_path(repo_root, output),
        "--config",
        container_path(repo_root, config),
        "--gpus",
        gpus,
    ]
    run_compose_service(repo_root, "preprocess", command, gpus=gpus, dry_run=dry_run)
