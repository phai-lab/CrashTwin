#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path("/crashtwin")
THIRD_PARTY = REPO_ROOT / "third_party"
DEFAULT_CONDA_ENVS = Path("/workspace/conda_envs")
DEFAULT_DROID_ENV = Path("/root/miniconda3/envs/droid_metric")
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrashTwin preprocessing inside Docker.")
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--fps", default=20, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Optional video limit for smoke tests.")
    return parser.parse_args()


def read_video_ids(path: Path) -> list[str]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["video_id"].strip() for row in reader if row.get("video_id", "").strip()]


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_executable(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}. Expected executable at: {path}")


def run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[crashtwin-container]", " ".join(args), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(args, cwd=cwd, env=merged_env, check=True)


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["NVIDIA_VISIBLE_DEVICES"] = args.gpus

    conda_envs = Path(os.environ.get("CRASHTWIN_CONDA_ENVS", str(DEFAULT_CONDA_ENVS)))
    checkpoint_dir = Path(os.environ.get("CRASHTWIN_CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIR)))
    sam2_python = conda_envs / "sam2" / "bin" / "python"
    searaft_python = conda_envs / "searaft" / "bin" / "python"
    mapanything_python = conda_envs / "mapanything" / "bin" / "python"
    droid_python = Path(os.environ.get("CRASHTWIN_DROID_PYTHON", str(DEFAULT_DROID_ENV / "bin" / "python")))

    require_executable(sam2_python, "SAM2 Python environment")
    require_executable(searaft_python, "SEA-RAFT Python environment")
    require_executable(mapanything_python, "MapAnything Python environment")
    require_executable(droid_python, "DROID/Metric3D Python environment")
    require_file(checkpoint_dir / "metric_depth_vit_giant2_800k.pth", "Metric3D checkpoint")
    require_file(checkpoint_dir / "droid.pth", "DROID-SLAM checkpoint")

    video_ids = read_video_ids(args.benchmark)
    if args.limit is not None:
        video_ids = video_ids[: args.limit]

    for index, video_id in enumerate(video_ids, start=1):
        print(f"[crashtwin-container] preprocess {index}/{len(video_ids)} {video_id}", flush=True)
        save = args.output / video_id
        save.mkdir(parents=True, exist_ok=True)

        src_video = args.inputs / f"{video_id}.mp4"
        src_auto = args.benchmark_root / "benchmark" / "auto_json" / f"{video_id}_auto.json"
        src_specs = (
            args.benchmark_root
            / "benchmark"
            / "vehicle_specs"
            / f"{video_id}_vehicle_specs.json"
        )
        require_file(src_video, "input video")
        require_file(src_auto, "auto_json")
        require_file(src_specs, "vehicle_specs")

        work_video = save / f"{video_id}.mp4"
        normalized = save / f"{video_id}_1080p.mp4"
        shutil.copy2(src_video, work_video)
        shutil.copy2(src_auto, save / f"{video_id}_auto.json")
        shutil.copy2(src_specs, save / f"{video_id}_vehicle_specs.json")

        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(work_video),
                "-vf",
                f"scale=1920:1080:flags=lanczos,fps={args.fps},setsar=1:1,setdar=16/9",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "high",
                "-level",
                "4.0",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-c:a",
                "copy",
                str(normalized),
            ]
        )
        normalized.replace(work_video)

        run(
            [
                str(sam2_python),
                "mydemo_from_json.py",
                str(save / f"{video_id}_auto.json"),
                str(work_video),
                str(save / f"{video_id}_sam2_track.mp4"),
                str(save / f"{video_id}_sam2_masks.npz"),
            ],
            cwd=THIRD_PARTY / "sam2",
        )

        rgb_dir = save / f"{video_id}_RGB"
        depth_dir = save / f"{video_id}_DEPTH"
        poses_dir = save / f"{video_id}_OUT"
        rgb_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)
        poses_dir.mkdir(exist_ok=True)

        run(
            [str(droid_python), "scripts/sample.py", str(work_video), "--sample-fps", str(args.fps), "--out-dir", str(rgb_dir)],
            cwd=THIRD_PARTY / "droid_metric",
        )
        run(
            [
                str(mapanything_python),
                "calculate_intr.py",
                "--image_dir",
                str(rgb_dir),
                "--output_file",
                str(save / f"intrinsic_{video_id}.txt"),
            ],
            cwd=THIRD_PARTY / "map-anything",
        )
        run(
            [
                str(droid_python),
                "depth.py",
                "--images",
                str(rgb_dir),
                "--out",
                str(depth_dir),
                "--intr",
                str(save / f"intrinsic_{video_id}.txt"),
                "--checkpoint",
                str(checkpoint_dir / "metric_depth_vit_giant2_800k.pth"),
            ],
            cwd=THIRD_PARTY / "droid_metric",
        )
        run(
            [
                str(droid_python),
                "slam.py",
                "--images",
                str(rgb_dir),
                "--depth",
                str(depth_dir),
                "--intr",
                str(save / f"intrinsic_{video_id}.txt"),
                "--out-poses",
                str(poses_dir),
                "--checkpoint",
                str(checkpoint_dir / "droid.pth"),
            ],
            cwd=THIRD_PARTY / "droid_metric",
        )
        run(
            [
                str(droid_python),
                "traj_to_json.py",
                "--input_dir",
                str(poses_dir),
                "--output_file",
                str(save / f"{video_id}_trajectories.json"),
            ],
            cwd=THIRD_PARTY / "droid_metric",
        )
        run(
            [
                str(searaft_python),
                "st1_incompressibility.py",
                "--sam2_npz",
                str(save / f"{video_id}_sam2_masks.npz"),
                "--video",
                str(work_video),
                "--flow",
                "searaft",
                "--compute-temporal",
                "--cache-frames",
                "--output",
                str(save / f"{video_id}_st1.json"),
                "--temporal-output",
                str(save / f"{video_id}_temporal_warp.json"),
            ],
            cwd=THIRD_PARTY / "SEA-RAFT",
            env={"PYTHONPATH": str(THIRD_PARTY / "SEA-RAFT" / "core")},
        )
        run(
            [
                str(searaft_python),
                str(THIRD_PARTY / "compute_apd.py"),
                str(rgb_dir),
                str(save / f"{video_id}_sam2_masks.npz"),
                str(save / f"{video_id}_apd.json"),
            ],
            cwd=REPO_ROOT,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
