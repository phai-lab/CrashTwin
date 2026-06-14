#!/usr/bin/env python3
"""Physical consistency check: LONG (longitudinal accel) and YAW (yaw rate).

Features
- Loads kalman_smoothed_*.mp4_results.json
- Excludes burn-in frames and a collision window
- Computes per-vehicle LONG and YAW metrics, scores per policy
- Exports a compact JSON and two single-image plots (speed, yaw)

CLI (kept only the requested options)
  --kalman-json, --fps, --pre-frames, --post-frames,
  --collision-frame, --tracking-ids, --ignore-first-frames,
  --decel-g-max, --accel-g-max, --yaw-rate-max,
  --smooth-vel, --output, --speed-plot, --yaw-plot
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from detect_collision_point import (
    TrajectoryPoint,
    compute_closest_approach,
)

G = 9.80665  # m/s^2


@dataclass
class Det:
    frame: int
    pos: np.ndarray  # shape (3,)
    yaw: Optional[float]  # radians; 'rot_y' in JSON


def _load_pos_yaw(json_path: str) -> Dict[int, List[Det]]:
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: Dict[int, List[Det]] = {}
    for k, dets in data.items():
        try:
            frame = int(k)
        except Exception:
            continue
        for det in dets:
            tid = det.get("tracking_id")
            loc = det.get("loc")
            if tid is None or loc is None or not isinstance(loc, (list, tuple)) or len(loc) != 3:
                continue
            yaw = det.get("rot_y")
            out.setdefault(int(tid), []).append(
                Det(frame=frame, pos=np.asarray(loc, dtype=float), yaw=float(yaw) if yaw is not None else None)
            )
    for lst in out.values():
        lst.sort(key=lambda d: d.frame)
    return out


def _unwrap_angle_diff(a1: float, a2: float) -> float:
    d = a2 - a1
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _pairwise_velocity(frames: np.ndarray, pos: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    if frames.size < 2:
        return np.array([]), np.zeros((0, 3))
    dt_frames = np.diff(frames).astype(float)
    valid = dt_frames > 0
    if not np.any(valid):
        return np.array([]), np.zeros((0, 3))
    dt = dt_frames[valid] / fps
    dx = np.diff(pos, axis=0)[valid]
    v = dx / dt[:, None]
    t_mid = (frames[:-1] + frames[1:]) / (2.0 * fps)
    t_mid = t_mid[valid]
    return t_mid, v


def _pairwise_accel(t_mid: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if v.shape[0] < 2:
        return np.array([]), np.zeros((0, 3))
    dt = np.diff(t_mid)
    valid = dt > 1e-9
    if not np.any(valid):
        return np.array([]), np.zeros((0, 3))
    dv = np.diff(v, axis=0)[valid]
    a = dv / dt[valid][:, None]
    t_acc = (t_mid[:-1] + t_mid[1:]) / 2.0
    t_acc = t_acc[valid]
    return t_acc, a


def _pairwise_yaw_rate(frames: np.ndarray, yaw: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    if frames.size < 2:
        return np.array([]), np.array([])
    dt_frames = np.diff(frames).astype(float)
    valid = dt_frames > 0
    if not np.any(valid):
        return np.array([]), np.array([])
    dy = np.array([_unwrap_angle_diff(yaw[i], yaw[i + 1]) for i in range(len(yaw) - 1)], dtype=float)
    dy = dy[valid]
    dt = dt_frames[valid] / fps
    r = dy / dt
    t_mid = (frames[:-1] + frames[1:]) / (2.0 * fps)
    t_mid = t_mid[valid]
    return t_mid, r


def _apply_exclusions(frames_like: np.ndarray, ignore_first_frames: int, collision_frame: Optional[int], pre: int, post: int) -> np.ndarray:
    keep = np.ones_like(frames_like, dtype=bool)
    keep &= frames_like >= ignore_first_frames
    if collision_frame is not None:
        start = int(collision_frame - pre)
        end = int(collision_frame + post)
        keep &= ~((frames_like >= start) & (frames_like <= end))
    return keep


def _select_pair(traj: Dict[int, List[Det]], tracking_ids: Optional[Sequence[int]]) -> Tuple[int, int]:
    if tracking_ids and len(tracking_ids) == 2:
        return int(tracking_ids[0]), int(tracking_ids[1])
    # auto-pick closest pair by positions
    pos_traj = {tid: [TrajectoryPoint(frame=d.frame, position=d.pos) for d in dets] for tid, dets in traj.items()}
    tids = sorted(pos_traj.keys())
    best = None
    best_ids = None
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a = tids[i]
            b = tids[j]
            ca = compute_closest_approach(a, pos_traj[a], b, pos_traj[b])
            if ca is None:
                continue
            if best is None or ca.distance < best.distance:
                best = ca
                best_ids = (a, b)
    if best_ids is None:
        raise RuntimeError("No valid pair found for bounds check")
    return best_ids[0], best_ids[1]


def _infer_collision_frame_from_pair(traj: Dict[int, List[Det]], tid_a: int, tid_b: int) -> Optional[int]:
    pa = [TrajectoryPoint(frame=d.frame, position=d.pos) for d in traj.get(tid_a, [])]
    pb = [TrajectoryPoint(frame=d.frame, position=d.pos) for d in traj.get(tid_b, [])]
    ca = compute_closest_approach(tid_a, pa, tid_b, pb)
    if ca is None:
        return None
    if ca.collision_frame is not None:
        return int(ca.collision_frame)
    return int(round(0.5 * (ca.frame_a + ca.frame_b)))


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return arr
    w = int(max(1, window))
    if arr.ndim == 1:
        kernel = np.ones(w, dtype=float) / float(w)
        return np.convolve(arr, kernel, mode="same")
    out = np.empty_like(arr, dtype=float)
    kernel = np.ones(w, dtype=float) / float(w)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(arr[:, j], kernel, mode="same")
    return out


def _build_spans(times: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    if times.size == 0 or mask.size == 0:
        return spans
    order = np.argsort(times)
    t = times[order]
    m = mask[order]
    dt = np.diff(t)
    half = 0.5 * (np.median(dt) if dt.size > 0 else 1.0 / 30.0)
    i = 0
    while i < len(t):
        if not m[i]:
            i += 1
            continue
        start = t[i] - half
        j = i
        while j + 1 < len(t) and m[j + 1]:
            j += 1
        end = t[j] + half
        spans.append((float(start), float(end)))
        i = j + 1
    return spans


def _merge_spans(spans: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _analyze_one(
    dets: List[Det],
    fps: float,
    ignore_first_frames: int,
    collision_frame: Optional[int],
    pre_frames: int,
    post_frames: int,
    decel_g_max: float,
    accel_g_max: float,
    yaw_rate_max: float,
    smooth_vel: int,
):
    frames = np.array([d.frame for d in dets], dtype=float)
    pos = np.stack([d.pos for d in dets]) if dets else np.zeros((0, 3))
    yaw_list = [d.yaw for d in dets]
    has_yaw = all(y is not None for y in yaw_list) and len(yaw_list) == len(dets)
    yaw = np.array(yaw_list, dtype=float) if has_yaw else None

    # velocity, accel
    t_v, v = _pairwise_velocity(frames, pos, fps)
    v = _moving_average(v, smooth_vel) if v.size else v
    t_a, a = _pairwise_accel(t_v, v)
    acc_frames = (t_a * fps).astype(int)
    keep_long = _apply_exclusions(acc_frames, ignore_first_frames, collision_frame, pre_frames, post_frames)

    # longitudinal component along velocity direction (use v at previous index)
    v_for_dir = v[:-1]
    v_mag = np.linalg.norm(v_for_dir, axis=1)
    dir_mask = v_mag > 1e-3
    a_long = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        if i < v_for_dir.shape[0] and dir_mask[i]:
            u = v_for_dir[i] / v_mag[i]
            a_long[i] = float(np.dot(a[i], u))
        else:
            a_long[i] = 0.0
    long_g = a_long / G
    decel_g = np.maximum(0.0, -long_g)
    accel_g = np.maximum(0.0, long_g)

    kept_decel = decel_g[keep_long]
    kept_accel = accel_g[keep_long]
    viol_long_mask = (kept_decel > decel_g_max) | (kept_accel > accel_g_max)
    p_long = float(np.sum(viol_long_mask)) / float(kept_decel.size) if kept_decel.size else 0.0
    r_long = max(
        (float(np.max(kept_decel)) / decel_g_max) if kept_decel.size and decel_g_max > 0 else 0.0,
        (float(np.max(kept_accel)) / accel_g_max) if kept_accel.size and accel_g_max > 0 else 0.0,
    )

    # yaw
    yaw_info = {"available": False, "p": 0.0, "r": 0.0}
    if has_yaw and yaw is not None:
        t_yaw, yaw_rate = _pairwise_yaw_rate(frames, yaw, fps)
        yaw_frames = (t_yaw * fps).astype(int)
        keep_yaw = _apply_exclusions(yaw_frames, ignore_first_frames, collision_frame, pre_frames, post_frames)
        kept_yaw = yaw_rate[keep_yaw]
        viol_yaw_mask = np.abs(kept_yaw) > yaw_rate_max
        p_yaw = float(np.sum(viol_yaw_mask)) / float(kept_yaw.size) if kept_yaw.size else 0.0
        r_yaw = (float(np.max(np.abs(kept_yaw))) / yaw_rate_max) if kept_yaw.size and yaw_rate_max > 0 else 0.0
        yaw_info = {
            "available": True,
            "t": t_yaw,
            "yaw_rate": yaw_rate,
            "keep_mask": keep_yaw,
            "viol_mask": viol_yaw_mask,
            "p": p_yaw,
            "r": r_yaw,
        }

    return {
        "acc": {
            "t_v": t_v,
            "speed": np.linalg.norm(v, axis=1) if v.size else np.array([]),
            "t_a": t_a,
            "keep_mask": keep_long,
            "viol_mask": viol_long_mask if keep_long.size else np.array([]),
            "p": p_long,
            "r": r_long,
        },
        "yaw": yaw_info,
    }


def _score(p: float, r: float) -> int:
    if p <= 0.20:
        return 0
    if r > 2.0:
        return -2
    if r > 1.5:
        return -1
    return 0


def _plot_speed_with_violations(lines, violation_spans, collision_time, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed; skipping speed plot generation")
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for t, y, label, color in lines:
        ax.plot(t, y, label=label, color=color, linewidth=2)
    for s, e in violation_spans:
        ax.axvspan(s, e, color="red", alpha=0.18)
    if collision_time is not None:
        ax.axvline(collision_time, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Longitudinal speed with violations")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|v| (m/s)")
    ax.grid(True, alpha=0.3)
    if len(lines) > 1:
        ax.legend(loc="upper right")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_yaw_with_violations(lines, violation_spans, collision_time, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed; skipping yaw plot generation")
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for t, y, label, color in lines:
        ax.plot(t, y, label=label, color=color, linewidth=2)
    for s, e in violation_spans:
        ax.axvspan(s, e, color="red", alpha=0.18)
    if collision_time is not None:
        ax.axvline(collision_time, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Yaw rate with violations")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("yaw rate (rad/s)")
    ax.grid(True, alpha=0.3)
    if len(lines) > 1:
        ax.legend(loc="upper right")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check longitudinal acceleration and yaw-rate bounds outside collision window")
    p.add_argument("--kalman-json", required=True, help="Path to kalman_smoothed_*.mp4_results.json")
    p.add_argument("--fps", type=float, default=30.0, help="Video frame rate (fps)")
    p.add_argument("--pre-frames", type=int, default=5, help="Collision window: frames before")
    p.add_argument("--post-frames", type=int, default=5, help="Collision window: frames after")
    p.add_argument("--collision-frame", type=int, help="Collision frame index (optional)")
    p.add_argument("--tracking-ids", type=int, nargs=2, help="Two tracking IDs (default: auto-select closest pair)")
    p.add_argument("--ignore-first-frames", type=int, default=20, help="Ignore first N frames as burn-in")
    p.add_argument("--decel-g-max", type=float, default=1.0, help="Max allowed braking deceleration in g")
    p.add_argument("--accel-g-max", type=float, default=0.4, help="Max allowed forward acceleration in g")
    p.add_argument("--yaw-rate-max", type=float, default=0.7, help="Max allowed absolute yaw-rate (rad/s)")
    p.add_argument("--smooth-vel", type=int, default=0, help="Moving-average window for velocity (frames); 0 disables")
    p.add_argument("--output", help="Path to save result JSON")
    p.add_argument("--speed-plot", help="Path to save speed plot image")
    p.add_argument("--yaw-plot", help="Path to save yaw-rate plot image")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.kalman_json):
        raise FileNotFoundError(args.kalman_json)

    traj = _load_pos_yaw(args.kalman_json)
    if len(traj) < 2:
        raise RuntimeError("Need at least two tracked trajectories for bounds check")

    tid_a, tid_b = _select_pair(traj, args.tracking_ids)
    collision_frame = args.collision_frame
    if collision_frame is None:
        collision_frame = _infer_collision_frame_from_pair(traj, tid_a, tid_b)
    collision_time = (float(collision_frame) / args.fps) if collision_frame is not None else None

    res_a = _analyze_one(
        traj.get(tid_a, []), args.fps, args.ignore_first_frames, collision_frame,
        args.pre_frames, args.post_frames, args.decel_g_max, args.accel_g_max,
        args.yaw_rate_max, args.smooth_vel,
    )
    res_b = _analyze_one(
        traj.get(tid_b, []), args.fps, args.ignore_first_frames, collision_frame,
        args.pre_frames, args.post_frames, args.decel_g_max, args.accel_g_max,
        args.yaw_rate_max, args.smooth_vel,
    )

    vehicles_out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tid, res in [(tid_a, res_a), (tid_b, res_b)]:
        long_p = float(res["acc"]["p"])
        long_r = float(res["acc"]["r"])
        long_pen = _score(long_p, long_r)

        yaw_p = float(res["yaw"].get("p", 0.0))
        yaw_r = float(res["yaw"].get("r", 0.0))
        yaw_pen = _score(yaw_p, yaw_r)

        vehicles_out[str(tid)] = {
            "long": {"p": long_p, "r": long_r, "penalty": long_pen},
            "yaw": {"p": yaw_p, "r": yaw_r, "penalty": yaw_pen},
        }

    long_metric = float(np.mean([v["long"]["penalty"] for v in vehicles_out.values()]))
    yaw_metric = float(np.mean([v["yaw"]["penalty"] for v in vehicles_out.values()]))

    summary = {
        "vehicles": vehicles_out,
        "metrics": {"long_metric": long_metric, "yaw_metric": yaw_metric},
        "policy": {"violation_ratio_threshold": 0.20, "levels": {"1pt": 1.5, "2pt": 2.0}},
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved summary to {args.output}")
    else:
        print(json.dumps(summary, indent=2))

    # Build plots
    speed_lines = []
    spans_long: List[Tuple[float, float]] = []
    for tid, res, color in [(tid_a, res_a, "tab:blue"), (tid_b, res_b, "tab:orange")]:
        tv = res["acc"]["t_v"]
        sp = res["acc"]["speed"]
        ta = res["acc"]["t_a"]
        keep = res["acc"]["keep_mask"]
        viol = res["acc"].get("viol_mask", np.zeros_like(keep, dtype=bool))
        speed_lines.append((tv, sp, f"vehicle {tid}", color))
        if ta.size and viol.size:
            spans_long += _build_spans(ta[keep], viol)
    spans_long = _merge_spans(spans_long)

    if args.speed_plot:
        try:
            _plot_speed_with_violations(speed_lines, spans_long, collision_time, args.speed_plot)
        except Exception as e:
            print(f"Warning: failed to generate speed plot: {e}")

    yaw_lines = []
    spans_yaw_all: List[Tuple[float, float]] = []
    for tid, res, color in [(tid_a, res_a, "tab:green"), (tid_b, res_b, "tab:red")]:
        y = res["yaw"]
        if y.get("available", False):
            t_y = y["t"]
            yr = y["yaw_rate"]
            yaw_lines.append((t_y, yr, f"vehicle {tid}", color))
            yaw_keep = y["keep_mask"]
            yaw_viol = y["viol_mask"]
            if t_y.size and yaw_viol.size:
                spans_yaw_all += _build_spans(t_y[yaw_keep], yaw_viol)
    spans_yaw_all = _merge_spans(spans_yaw_all)

    if args.yaw_plot and yaw_lines:
        try:
            _plot_yaw_with_violations(yaw_lines, spans_yaw_all, collision_time, args.yaw_plot)
        except Exception as e:
            print(f"Warning: failed to generate yaw plot: {e}")


if __name__ == "__main__":
    main()
