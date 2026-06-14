#!/usr/bin/env python3
"""Check collision via 3D-box first contact and output a JSON flag.

This script reuses the 3D first-contact logic from
`momentum_residual_windo_per_dir_col_l2.py` without modifying it. It loads
Kalman-smoothed trajectories, selects the pair (either provided or auto),
evaluates first contact using the same thresholds, and writes a compact JSON
containing a boolean flag `collided` plus a few useful fields.

Example:

    python check_first_contact_collision.py \
        --kalman-json savepath/VV_209/kalman_smoothed_VV_209.mp4_results.json \
        --fps 30 \
        --output /workspace/Phy_metric/VV_209/first_contact_flag.json

Optionally specify the two tracking IDs:

    python check_first_contact_collision.py \
        --kalman-json ... \
        --tracking-ids 1478 1479 \
        --fps 30 \
        --output flag.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

# Import the existing logic and constants (kept as-is in that module)
from momentum_residual_windo_per_dir_col_l2 import (
    _first_contact_frame_3d,
    _select_collision_pair,
    BBOX3D_CONSEC,
    BBOX3D_DIMS,
    BBOX3D_SCALE,
    BBOX3D_SMOOTH,
)
from detect_collision_point import TrajectoryPoint, load_trajectories


def _max_run_length(flags: Sequence[bool]) -> int:
    run = 0
    best = 0
    for ok in flags:
        if ok:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _first_satisfied_frame(frames: List[int], satisfied: List[bool], consec: int) -> Optional[int]:
    if consec <= 1:
        for i, ok in enumerate(satisfied):
            if ok:
                return int(frames[i])
        return None
    run = 0
    for i, ok in enumerate(satisfied):
        run = run + 1 if ok else 0
        if run >= consec:
            return int(frames[i - consec + 1])
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect collision via 3D-box first contact and output a JSON flag",
    )
    parser.add_argument(
        "--kalman-json",
        required=True,
        help="Path to kalman_smoothed_*.mp4_results.json",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate (frames per second)",
    )
    parser.add_argument(
        "--tracking-ids",
        type=int,
        nargs=2,
        help="Tracking IDs of the two vehicles (default: auto-select closest pair)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the collision flag",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 加强鲁棒：加载轨迹时捕获异常，失败则按未碰撞输出
    try:
        trajectories: Dict[int, List[TrajectoryPoint]] = load_trajectories(args.kalman_json)
    except Exception as exc:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "kalman_json": os.path.abspath(args.kalman_json),
            "tracking_ids": [None, None],
            "fps": float(args.fps),
            "collided": False,
            "first_contact_frame": None,
            "first_contact_time": None,
            "fc_frame_any": None,
            "fc_time_any": None,
            "min_ratio": None,
            "max_consecutive_satisfied": 0,
            "frames_checked": 0,
            "reason": "load_error",
            "error": str(exc),
            "params": {
                "k_scale": float(BBOX3D_SCALE),
                "consec": int(BBOX3D_CONSEC),
                "smooth_window": int(BBOX3D_SMOOTH),
                "dims_mode": str(BBOX3D_DIMS),
            },
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved 3D first-contact collision flag to {args.output}")
        return

    if not trajectories:
        # 无轨迹时直接输出未碰撞
        tid_a = tid_b = None
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "kalman_json": os.path.abspath(args.kalman_json),
            "tracking_ids": [None, None],
            "fps": float(args.fps),
            "collided": False,
            "first_contact_frame": None,
            "first_contact_time": None,
            "fc_frame_any": None,
            "fc_time_any": None,
            "min_ratio": None,
            "max_consecutive_satisfied": 0,
            "frames_checked": 0,
            "reason": "no_trajectories",
            "params": {
                "k_scale": float(BBOX3D_SCALE),
                "consec": int(BBOX3D_CONSEC),
                "smooth_window": int(BBOX3D_SMOOTH),
                "dims_mode": str(BBOX3D_DIMS),
            },
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved 3D first-contact collision flag to {args.output}")
        return

    # 处理指定 ID 的健壮性：若提供的 ID 不存在或为 0，则视为未发生碰撞并直接输出
    provided_ids = args.tracking_ids
    if provided_ids is not None:
        a, b = int(provided_ids[0]), int(provided_ids[1])
        ids_valid = (
            a > 0 and b > 0 and a in trajectories and b in trajectories and
            len(trajectories.get(a, [])) > 0 and len(trajectories.get(b, [])) > 0
        )
        if not ids_valid:
            tid_a = a
            tid_b = b
            collided = False
            first_contact_frame = None
            fc_frame = None
            min_ratio = None
            max_consecutive = 0
            frames_checked = 0

            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            payload = {
                "kalman_json": os.path.abspath(args.kalman_json),
                "tracking_ids": [int(tid_a), int(tid_b)],
                "fps": float(args.fps),
                "collided": False,
                "first_contact_frame": None,
                "first_contact_time": None,
                "fc_frame_any": None,
                "fc_time_any": None,
                "min_ratio": None,
                "max_consecutive_satisfied": 0,
                "frames_checked": 0,
                "reason": "invalid_tracking_ids",
                "params": {
                    "k_scale": float(BBOX3D_SCALE),
                    "consec": int(BBOX3D_CONSEC),
                    "smooth_window": int(BBOX3D_SMOOTH),
                    "dims_mode": str(BBOX3D_DIMS),
                },
            }

            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"Saved 3D first-contact collision flag to {args.output}")
            return
        tid_a, tid_b = a, b
    else:
        # 未指定 ID：尝试自动选择；失败则按未碰撞处理
        try:
            tid_a, tid_b, _closest = _select_collision_pair(trajectories, None)
        except Exception as exc:
            tid_a = tid_b = None
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            payload = {
                "kalman_json": os.path.abspath(args.kalman_json),
                "tracking_ids": [None, None],
                "fps": float(args.fps),
                "collided": False,
                "first_contact_frame": None,
                "first_contact_time": None,
                "fc_frame_any": None,
                "fc_time_any": None,
                "min_ratio": None,
                "max_consecutive_satisfied": 0,
                "frames_checked": 0,
                "reason": "auto_select_failed",
                "error": str(exc),
                "params": {
                    "k_scale": float(BBOX3D_SCALE),
                    "consec": int(BBOX3D_CONSEC),
                    "smooth_window": int(BBOX3D_SMOOTH),
                    "dims_mode": str(BBOX3D_DIMS),
                },
            }

            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"Saved 3D first-contact collision flag to {args.output}")
            return

    # Reuse the same constants/behavior as the existing script
    try:
        fc_frame, series = _first_contact_frame_3d(
            trajectories,
            tid_a,
            tid_b,
            args.kalman_json,
            k=BBOX3D_SCALE,
            consec=BBOX3D_CONSEC,
            smooth_window=BBOX3D_SMOOTH,
            dims_mode=BBOX3D_DIMS,
        )
    except Exception as exc:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "kalman_json": os.path.abspath(args.kalman_json),
            "tracking_ids": [int(tid_a), int(tid_b)],
            "fps": float(args.fps),
            "collided": False,
            "first_contact_frame": None,
            "first_contact_time": None,
            "fc_frame_any": None,
            "fc_time_any": None,
            "min_ratio": None,
            "max_consecutive_satisfied": 0,
            "frames_checked": 0,
            "reason": "fc_compute_error",
            "error": str(exc),
            "params": {
                "k_scale": float(BBOX3D_SCALE),
                "consec": int(BBOX3D_CONSEC),
                "smooth_window": int(BBOX3D_SMOOTH),
                "dims_mode": str(BBOX3D_DIMS),
            },
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved 3D first-contact collision flag to {args.output}")
        return

    collided = False
    reason: Optional[str] = None
    first_contact_frame: Optional[int] = None
    min_ratio: Optional[float] = None
    max_consecutive: int = 0
    frames_checked: int = 0

    if series is not None:
        frames = [int(f) for f in series.get("frames", [])]
        dists = np.asarray(series.get("dist", []), dtype=float)
        thrs = np.asarray(series.get("thr", []), dtype=float)
        frames_checked = len(frames)

        if frames and dists.size == thrs.size == len(frames):
            satisfied = (dists <= thrs)
            max_consecutive = _max_run_length(satisfied.tolist())
            collided = bool(max_consecutive >= int(BBOX3D_CONSEC))
            first_contact_frame = _first_satisfied_frame(frames, satisfied.tolist(), int(BBOX3D_CONSEC))
            ratio = np.divide(dists, thrs, out=np.full_like(dists, np.inf), where=thrs > 0)
            min_ratio = float(np.min(ratio)) if ratio.size else None
            if not collided:
                reason = "no_consecutive_contact"
        else:
            reason = "series_empty_or_mismatch"
    else:
        reason = "series_missing"

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "kalman_json": os.path.abspath(args.kalman_json),
        "tracking_ids": [int(tid_a), int(tid_b)],
        "fps": float(args.fps),
        "collided": bool(collided),
        "first_contact_frame": int(first_contact_frame) if first_contact_frame is not None else None,
        "first_contact_time": (float(first_contact_frame) / float(args.fps)) if first_contact_frame is not None else None,
        "fc_frame_any": int(fc_frame) if fc_frame is not None else None,
        "fc_time_any": (float(fc_frame) / float(args.fps)) if fc_frame is not None else None,
        "min_ratio": float(min_ratio) if min_ratio is not None else None,
        "max_consecutive_satisfied": int(max_consecutive),
        "frames_checked": int(frames_checked),
        "reason": reason,
        "params": {
            "k_scale": float(BBOX3D_SCALE),
            "consec": int(BBOX3D_CONSEC),
            "smooth_window": int(BBOX3D_SMOOTH),
            "dims_mode": str(BBOX3D_DIMS),
        },
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Saved 3D first-contact collision flag to {args.output}")


if __name__ == "__main__":
    main()
