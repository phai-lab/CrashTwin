from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


def repo_relative(repo_root: Path, path: Path | str) -> str:
    resolved_root = repo_root.resolve()
    resolved_path = Path(path).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path must be inside repository root: {resolved_path}") from exc


def container_path(repo_root: Path, path: Path | str) -> str:
    return f"/crashtwin/{repo_relative(repo_root, path)}"


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    printable = " ".join(shlex.quote(arg) for arg in args)
    print(f"[crashtwin] {printable}")
    if dry_run:
        return

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(args, cwd=cwd, env=merged_env, check=True)


def run_compose_service(
    repo_root: Path,
    service: str,
    command_args: list[str],
    *,
    gpus: str,
    dry_run: bool = False,
) -> None:
    compose_file = repo_root / "docker" / "docker-compose.yaml"
    env = {
        "CRASHTWIN_REPO_ROOT": str(repo_root.resolve()),
        "CRASHTWIN_GPUS": gpus,
        "CRASHTWIN_PREPROCESS_TAG": os.environ.get(
            "CRASHTWIN_PREPROCESS_TAG",
            "draft-20260614-env",
        ),
        "CRASHTWIN_RECONSTRUCT_TAG": os.environ.get(
            "CRASHTWIN_RECONSTRUCT_TAG",
            "draft-20260613",
        ),
    }
    args = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        service,
        *command_args,
    ]
    run_command(args, cwd=repo_root, env=env, dry_run=dry_run)
