from __future__ import annotations

from pathlib import Path

from .runner import container_path, run_compose_service


def run_reconstruction(
    *,
    repo_root: Path,
    benchmark: Path,
    benchmark_root: Path,
    per_video_dir: Path,
    config: Path,
    gpus: str,
    dry_run: bool = False,
) -> None:
    command = [
        "python3",
        "/crashtwin/scripts/container_reconstruct.py",
        "--benchmark",
        container_path(repo_root, benchmark),
        "--benchmark-root",
        container_path(repo_root, benchmark_root),
        "--per-video-dir",
        container_path(repo_root, per_video_dir),
        "--config",
        container_path(repo_root, config),
        "--gpus",
        gpus,
    ]
    run_compose_service(repo_root, "reconstruct", command, gpus=gpus, dry_run=dry_run)
