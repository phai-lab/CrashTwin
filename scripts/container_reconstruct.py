#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/crashtwin")
CENTERTRACK = REPO_ROOT / "third_party" / "centertrack"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
PYTHON = sys.executable
PHYSICS_PENALTY = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrashTwin 3D reconstruction inside Docker.")
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--per-video-dir", required=True, type=Path)
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


def run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[crashtwin-container]", " ".join(args), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(args, cwd=cwd, env=merged_env, check=True)


def read_focal_length(path: Path) -> str:
    with path.open("r") as handle:
        return handle.readline().strip()


def read_vehicle_masses(path: Path) -> tuple[str, str]:
    with path.open("r") as handle:
        data = json.load(handle)
    return str(data["left"]["mass_kg"]), str(data["opponent"]["mass_kg"])


def read_tracking_ids(path: Path) -> tuple[str, str]:
    with path.open("r") as handle:
        data = json.load(handle)
    pairs: dict[int, int] = {}
    for dets in data.values():
        for det in dets:
            if "sam2_car_id" in det and "tracking_id" in det:
                pairs[int(det["sam2_car_id"])] = int(det["tracking_id"])
    ordered = sorted(pairs.items())
    if len(ordered) < 2:
        raise RuntimeError(f"Could not infer two tracking IDs from {path}")
    return str(ordered[0][1]), str(ordered[-1][1])


def load_json(path: Path) -> dict[str, object]:
    with path.open("r") as handle:
        return json.load(handle)


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clipped_physics_metric(value: object) -> float | None:
    result = finite_float(value)
    return min(result, PHYSICS_PENALTY) if result is not None else None


def physics_metrics_with_penalty(save: Path) -> dict[str, float | None]:
    flag = load_json(save / "first_contact_flag.json")
    if flag.get("collided") is False:
        return {
            "J_p": PHYSICS_PENALTY,
            "J_H": PHYSICS_PENALTY,
            "J_E": PHYSICS_PENALTY,
        }

    jp = load_json(save / "momentum_j.json")
    jh = load_json(save / "momentum_jh.json")
    return {
        "J_p": clipped_physics_metric(jp.get("j_p")),
        "J_H": clipped_physics_metric(jh.get("J_H", jh.get("j_p"))),
        "J_E": clipped_physics_metric(jp.get("j_e")),
    }


def write_metrics(save: Path, video_id: str) -> None:
    st1 = load_json(save / f"{video_id}_st1.json")
    temporal = load_json(save / f"{video_id}_temporal_warp.json")
    apd = load_json(save / f"{video_id}_apd.json")
    physics = physics_metrics_with_penalty(save)
    dynamics_path = save / f"step0_instance_dynamics_{video_id}.json"
    dynamics = load_json(dynamics_path) if dynamics_path.is_file() else {}

    metrics = {
        "E_flow": st1.get("ST1"),
        "E_warp": temporal.get("E_warp"),
        **physics,
        "S_ID": dynamics.get("video_metric_weighted"),
        "D_ad": apd.get("apd"),
    }
    status = "ok" if all(value is not None for value in metrics.values()) else "missing_metric"
    with (save / "metrics.json").open("w") as handle:
        json.dump({"status": status, "metrics": metrics}, handle, indent=2)


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["NVIDIA_VISIBLE_DEVICES"] = args.gpus

    checkpoint_dir = Path(os.environ.get("CRASHTWIN_CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIR)))
    centertrack_model = checkpoint_dir / "nuScenes_3Dtracking.pth"
    require_file(centertrack_model, "CenterTrack checkpoint")

    video_ids = read_video_ids(args.benchmark)
    if args.limit is not None:
        video_ids = video_ids[: args.limit]

    for index, video_id in enumerate(video_ids, start=1):
        print(f"[crashtwin-container] reconstruct {index}/{len(video_ids)} {video_id}", flush=True)
        save = args.per_video_dir / video_id
        video = save / f"{video_id}.mp4"
        intrinsic = save / f"intrinsic_{video_id}.txt"
        trajectories = save / f"{video_id}_trajectories.json"
        depth_dir = save / f"{video_id}_DEPTH"
        vehicle_specs = save / f"{video_id}_vehicle_specs.json"
        detections = save / f"{video_id}_results.json"
        tracked_video = save / f"{video_id}_tracked.mp4"

        require_file(video, "normalized video")
        require_file(intrinsic, "intrinsics")
        require_file(trajectories, "camera trajectory JSON")
        require_file(vehicle_specs, "vehicle specs")
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Missing depth directory: {depth_dir}")

        focal_length = read_focal_length(intrinsic)
        run(
            [
                PYTHON,
                "demo.py",
                "tracking,ddd",
                "--load_model",
                str(centertrack_model),
                "--dataset",
                "nuscenes",
                "--pre_hm",
                "--track_thresh",
                "0.1",
                "--demo",
                str(video),
                "--video_h",
                "1080",
                "--video_w",
                "1920",
                "--test_focal_length",
                focal_length,
                "--save_video",
                "--save_results",
                "--results_out_path",
                str(detections),
            ],
            cwd=CENTERTRACK / "src",
        )
        run(
            [
                PYTHON,
                "visualize_tracking.py",
                str(detections),
                str(video),
                str(tracked_video),
                "cars",
                focal_length,
            ],
            cwd=CENTERTRACK / "src",
        )
        run(
            [
                PYTHON,
                "traj_recon_tr_rot.py",
                "--video_id",
                video_id,
                "--trajectories_file",
                str(trajectories),
                "--intrinsic_file",
                str(intrinsic),
                "--depth_images_path",
                str(depth_dir),
                "--detections_json",
                str(detections),
                "--savepath",
                str(save),
                "--override_dims_step0",
                "--vehicle_specs_json",
                str(vehicle_specs),
            ],
            cwd=CENTERTRACK,
        )

        kalman = save / f"kalman_smoothed_{video_id}_yaw_on.mp4_results.json"
        require_file(kalman, "Kalman-smoothed trajectory JSON")
        left_mass, opponent_mass = read_vehicle_masses(vehicle_specs)
        left_tid, opponent_tid = read_tracking_ids(kalman)

        common = [
            "--kalman-json",
            str(kalman),
            "--fps",
            str(args.fps),
            "--pre-frames",
            "5",
            "--post-frames",
            "5",
            "--tracking-ids",
            left_tid,
            opponent_tid,
            "--masses",
            left_mass,
            opponent_mass,
        ]
        run(
            [
                PYTHON,
                "check_first_contact_collision.py",
                "--kalman-json",
                str(kalman),
                "--fps",
                str(args.fps),
                "--tracking-ids",
                left_tid,
                opponent_tid,
                "--output",
                str(save / "first_contact_flag.json"),
            ],
            cwd=CENTERTRACK,
        )
        run(
            [
                PYTHON,
                "momentum_residual_windo_per_dir_col_l2.py",
                *common,
                "--j-output",
                str(save / "momentum_j.json"),
                "--velocity-plot",
                str(save / "velocity_profiles.png"),
            ],
            cwd=CENTERTRACK,
        )
        run(
            [
                PYTHON,
                "angular_momentum_residual_windo_per_dir_col.py",
                *common,
                "--j-output",
                str(save / "momentum_jh.json"),
                "--velocity-plot",
                str(save / "angular_velocity_profiles.png"),
            ],
            cwd=CENTERTRACK,
        )
        run(
            [
                PYTHON,
                "motion_bounds_check.py",
                "--kalman-json",
                str(kalman),
                "--fps",
                str(args.fps),
                "--pre-frames",
                "5",
                "--post-frames",
                "5",
                "--tracking-ids",
                left_tid,
                opponent_tid,
                "--ignore-first-frames",
                "20",
                "--decel-g-max",
                "1.0",
                "--accel-g-max",
                "0.4",
                "--yaw-rate-max",
                "0.7",
                "--smooth-vel",
                "0",
                "--output",
                str(save / "bounds_summary.json"),
                "--speed-plot",
                str(save / "speed_plot.png"),
                "--yaw-plot",
                str(save / "yaw_plot.png"),
            ],
            cwd=CENTERTRACK,
        )
        write_metrics(save, video_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
