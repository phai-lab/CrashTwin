#!/usr/bin/env python3
"""Contact spin direction (yaw-only) via impulse.

We estimate spin direction at impact using vehicle mass, contact point, and
change in COM velocity. For vehicle i:
  J_i = m_i (v_i^+ - v_i^-)              # linear impulse (planar)
  r_c = pos_c - pos_i                    # lever arm (planar)
In vehicle body frame (x forward, y left), the angular impulse about z is
  τ_z = r_x J_y - r_y J_x
and the signed, dimensionless spin indicator is
  K_i = τ_z / (||J_i|| · ||r_c||) ∈ [-1, 1],  with K>0 meaning CCW (left).

Optionally, we compare sign(K_i) to the sign of post-pre yaw delta to produce a
binary score per vehicle (1 if signs match and non-zero, else 0); the summary
"j_p" is the mean of those two scores.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse helpers from existing scripts to keep this file minimal
from detect_collision_point import (
    ClosestApproach,
    TrajectoryPoint,
    compute_closest_approach,
    load_trajectories,
)
from angular_momentum_residual_windo_per_dir_col import (
    _first_contact_frame_3d,
    _load_rot_per_frame,
    _load_dims_per_frame,
    _value_at_frame_or_nearest,
    _position_at_frame_or_nearest,
    _contact_point_from_3d_boxes,
)
from momentum_residual_windo_per_dir_col_l2 import (
    _collect_window_points,
    _estimate_velocity,
)


@dataclass
class SpinInputs:
    tracking_ids: List[int]
    trajectories: Dict[int, List[TrajectoryPoint]]
    masses: Dict[int, float]
    collision_frame: int
    fps: float
    pre_frames: int
    post_frames: int
    include_y_axis: bool


@dataclass
class VehicleSpin:
    tracking_id: int
    mass: float
    v_before: np.ndarray
    v_after: np.ndarray
    yaw_at_collision: float
    r_body: np.ndarray
    J_body: np.ndarray
    K: Optional[float]
    yaw_rate_before: Optional[float]
    yaw_rate_after: Optional[float]
    delta_omega: Optional[float]
    score: Optional[float]


@dataclass
class SpinResult:
    collision_frame: int
    collision_time: float
    vehicles: List[VehicleSpin]
    j_p: float


# -------------------------- small helpers --------------------------

def _ensure_two_tracking_ids(tracking_ids: Sequence[int]) -> Tuple[int, int]:
    if len(tracking_ids) != 2:
        raise ValueError(f"Expected exactly two tracking IDs, got {tracking_ids}")
    return int(tracking_ids[0]), int(tracking_ids[1])


def _select_collision_pair(
    trajectories: Dict[int, List[TrajectoryPoint]],
    provided_ids: Optional[Sequence[int]] = None,
) -> Tuple[int, int, ClosestApproach]:
    if provided_ids and len(provided_ids) == 2:
        tid_a, tid_b = _ensure_two_tracking_ids(provided_ids)
        result = compute_closest_approach(tid_a, trajectories[tid_a], tid_b, trajectories[tid_b])
        if result is None:
            raise RuntimeError("Failed to compute closest approach for provided IDs")
        return tid_a, tid_b, result

    ids = sorted(trajectories.keys())
    if len(ids) < 2:
        raise ValueError("Need at least two trajectories")
    best: Optional[ClosestApproach] = None
    best_ids: Optional[Tuple[int, int]] = None
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            r = compute_closest_approach(a, trajectories[a], b, trajectories[b])
            if r is None:
                continue
            if best is None or r.distance < best.distance:
                best = r
                best_ids = (a, b)
    if best is None or best_ids is None:
        raise RuntimeError("Could not find valid pair for collision analysis")
    return best_ids[0], best_ids[1], best


def _infer_collision_frame(closest: ClosestApproach) -> int:
    if closest.collision_frame is not None:
        return closest.collision_frame
    return int(round(0.5 * (closest.frame_a + closest.frame_b)))


def _sign(x: Optional[float]) -> int:
    if x is None or not np.isfinite(x):
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _estimate_yaw_rate(frames: List[int], yaw_map: Dict[int, float], fps: float) -> Optional[float]:
    ys, ts = [], []
    for f in sorted(frames):
        if f in yaw_map:
            ys.append(float(yaw_map[f]))
            ts.append(f / fps)
    if len(ys) < 2:
        return None
    yarr = np.unwrap(np.array(ys, dtype=float))
    tarr = np.array(ts, dtype=float)
    t_centered = tarr - tarr.mean()
    denom = float(np.sum(t_centered ** 2))
    if denom <= 1e-9:
        dt = tarr[-1] - tarr[0]
        if dt <= 0:
            return None
        return float((yarr[-1] - yarr[0]) / dt)
    y_centered = yarr - yarr.mean()
    return float(np.sum(t_centered * y_centered) / denom)


def _delta_yaw_from_windows(
    yaw_map: Dict[int, float],
    collision_frame: int,
    pre_frames: int,
    post_frames: int,
    fps: float,
    *,
    min_count: int = 3,
) -> Tuple[Optional[float], Dict[str, object]]:
    """Compute robust Δyaw using short windows around t_c with unwrap.

    - Pre window: take last up to pre_frames frames strictly < t_c
    - Post window: take first up to post_frames frames strictly > t_c
    - Unwrap each window's yaw, take mean yaw_pre/yaw_post, then Δyaw = wrap(yaw_post - yaw_pre)
    - Returns (delta_yaw, debug_info)
    """
    dbg: Dict[str, object] = {}
    keys = sorted(int(k) for k in yaw_map.keys())
    pre = [f for f in keys if f < int(collision_frame)]
    post = [f for f in keys if f > int(collision_frame)]
    pre_sel = pre[-min(len(pre), max(1, min_count, int(pre_frames))) : ] if pre else []
    post_sel = post[: min(len(post), max(1, min_count, int(post_frames))) ] if post else []
    dbg["yaw_pre_frames"] = pre_sel
    dbg["yaw_post_frames"] = post_sel

    if not pre_sel or not post_sel:
        return None, dbg

    yaw_pre = np.array([float(yaw_map[f]) for f in pre_sel], dtype=float)
    yaw_post = np.array([float(yaw_map[f]) for f in post_sel], dtype=float)
    yaw_pre_u = np.unwrap(yaw_pre)
    yaw_post_u = np.unwrap(yaw_post)
    yaw_pre_mean = float(np.mean(yaw_pre_u))
    yaw_post_mean = float(np.mean(yaw_post_u))
    delta = yaw_post_mean - yaw_pre_mean
    # wrap to [-pi, pi]
    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    dbg.update({
        "yaw_pre": yaw_pre.tolist(),
        "yaw_post": yaw_post.tolist(),
        "yaw_pre_unwrap": yaw_pre_u.tolist(),
        "yaw_post_unwrap": yaw_post_u.tolist(),
        "yaw_pre_mean": yaw_pre_mean,
        "yaw_post_mean": yaw_post_mean,
        "delta_yaw": delta,
    })
    return delta, dbg


def _extrapolate_velocity_at(
    traj: List[TrajectoryPoint],
    collision_frame: int,
    k: int,
    side: str,
    fps: float,
) -> np.ndarray:
    """Estimate instantaneous velocity near t_c from one side without regression.

    Strategy:
    - Pick the nearest frame to t_c on the requested side (strictly < or > t_c).
    - Build up to k adjacent velocity samples using consecutive frame differences
      on that same side, picking those whose mid-frame is closest to t_c.
    - Return the weighted average of those velocities. Weight = 1/(|Δframe|+ε).
    - Output is 3D (vx, 0, vz), Y cleared unless user asks for Y elsewhere.
    """
    if not traj:
        return np.zeros(3, dtype=float)

    frames = sorted({int(pt.frame) for pt in traj})
    if side == "before":
        base_candidates = [f for f in frames if f < int(collision_frame)]
        if not base_candidates:
            return np.zeros(3, dtype=float)
        f0 = max(base_candidates)
    else:
        base_candidates = [f for f in frames if f > int(collision_frame)]
        if not base_candidates:
            return np.zeros(3, dtype=float)
        f0 = min(base_candidates)

    f_to_pos = {int(pt.frame): pt.position for pt in traj}
    # Build all consecutive velocity samples and filter by side
    triplets = []  # (dist_to_tc_in_frames, vx, vz)
    for i in range(1, len(frames)):
        f_prev, f_curr = frames[i - 1], frames[i]
        if f_prev not in f_to_pos or f_curr not in f_to_pos:
            continue
        mid = 0.5 * (f_prev + f_curr)
        if side == "before" and mid >= collision_frame:
            continue
        if side == "after" and mid <= collision_frame:
            continue
        dt = (f_curr - f_prev) / float(fps)
        if dt <= 0:
            continue
        dp = f_to_pos[f_curr] - f_to_pos[f_prev]
        vx, vz = float(dp[0] / dt), float(dp[2] / dt)
        dist = abs(mid - collision_frame)
        triplets.append((dist, vx, vz))

    if not triplets:
        return np.zeros(3, dtype=float)

    triplets.sort(key=lambda t: t[0])
    m = max(1, min(int(k), len(triplets)))
    sel = triplets[:m]
    # Weighted average, closer to t_c gets higher weight
    eps = 1e-6
    weights = np.array([1.0 / (d + eps) for d, _, _ in sel], dtype=float)
    weights /= float(weights.sum())
    vx = float(np.sum([w * v for w, (_, v, _) in zip(weights, sel)]))
    vz = float(np.sum([w * v for w, (_, _, v) in zip(weights, sel)]))
    return np.array([vx, 0.0, vz], dtype=float)


def _extrapolate_velocity_at_debug(
    traj: List[TrajectoryPoint],
    collision_frame: int,
    k: int,
    side: str,
    fps: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Like _extrapolate_velocity_at but returns debug details about segments/weights."""
    dbg: Dict[str, object] = {"side": side, "k": int(k)}
    if not traj:
        dbg.update({"frames": [], "pairs": [], "vel_xz": [], "weights": [], "t_c": float(collision_frame) / float(fps)})
        return np.zeros(3, dtype=float), dbg

    frames = sorted({int(pt.frame) for pt in traj})
    dbg["frames"] = frames
    if side == "before":
        cand = [f for f in frames if f < int(collision_frame)]
        sel = cand[-(k + 1):] if len(cand) >= 2 else cand
    else:
        cand = [f for f in frames if f > int(collision_frame)]
        sel = cand[: (k + 1)] if len(cand) >= 2 else cand
    dbg["sel_frames"] = sel

    f_to_pos = {int(pt.frame): pt.position for pt in traj}
    sel = [f for f in sel if f in f_to_pos]
    if len(sel) < 2:
        dbg.update({"pairs": [], "vel_xz": [], "weights": []})
        return np.zeros(3, dtype=float), dbg

    sel.sort()
    pos = np.stack([f_to_pos[f] for f in sel], axis=0)
    t = np.array(sel, dtype=float) / float(fps)

    dt = np.diff(t)
    valid = dt > 0
    mid_t = 0.5 * (t[:-1] + t[1:])
    dv = np.diff(pos[:, [0, 2]], axis=0)
    vel = np.zeros_like(dv)
    vel[valid] = dv[valid] / dt[valid, None]
    pairs = list(zip(sel[:-1], sel[1:]))
    pairs = [p for i, p in enumerate(pairs) if valid[i]]
    vel_valid = vel[valid]
    mid_t = mid_t[valid]
    dbg["pairs"] = pairs
    dbg["vel_xz"] = vel_valid.tolist()
    dbg["mid_t"] = mid_t.tolist()

    t_c = float(collision_frame) / float(fps)
    dbg["t_c"] = t_c

    # weights 1/(|mid - t_c| + eps)
    eps = 1e-6
    w = 1.0 / (np.abs(mid_t - t_c) + eps)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.ones_like(mid_t) / max(1, mid_t.size)
    dbg["weights"] = w.tolist()

    vx = float(np.sum(w * vel_valid[:, 0]))
    vz = float(np.sum(w * vel_valid[:, 1]))
    return np.array([vx, 0.0, vz], dtype=float), dbg


# ----------------------------- core flow ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate contact spin direction (yaw-only)")
    p.add_argument("--kalman-json", required=True, help="Path to kalman_smoothed_*.mp4_results.json")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--pre-frames", type=int, default=5)
    p.add_argument("--post-frames", type=int, default=5)
    p.add_argument("--collision-frame", type=int)
    p.add_argument("--tracking-ids", type=int, nargs=2)
    p.add_argument("--masses", type=float, nargs="+")
    p.add_argument("--default-mass", type=float, default=1500.0)
    p.add_argument("--include-y-axis", action="store_true")
    p.add_argument("--output")
    p.add_argument("--j-output")
    p.add_argument(
        "--no-brake-filter",
        action="store_true",
        help="Disable braking filter: use raw window velocities for impulse (v_after - v_before) instead of regression extrapolation at t_c",
    )
    p.add_argument(
        "--impulse-red",
        action="store_true",
        help="Draw impulse (J) arrows in red in visualization (default: use vehicle color)",
    )
    p.add_argument("--viz-style", choices=["minimal", "full"], default="minimal",
                   help="Visualization style: minimal (two impulses) or full (r, J, tau arcs)")
    p.add_argument(
        "--contact-j-sense",
        choices=["reaction", "action"],
        default="reaction",
        help="Sense for contact-point J arrow: reaction (on vehicle, J = m·(v+−v−)) or action (by vehicle, −J). Default: reaction",
    )
    p.add_argument(
        "--flip-arc",
        action="store_true",
        help="Flip visualization direction of rotation arc arrows (useful if your OpenCV arc looks reversed)",
    )
    p.add_argument(
        "--dashed-vector",
        choices=["momentum", "velocity", "impulse"],
        default="momentum",
        help="What to draw at dashed pre/post OBB centers: momentum (m*v), velocity (v), or impulse (J). Default: momentum",
    )
    p.add_argument(
        "--use-refined-contact",
        action="store_true",
        help="Use refined contact (min OBB distance in window) for contact-point J and torque visuals",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Verbose debug: print intermediate frames, segments, weights, and per-vehicle variables",
    )
    p.add_argument(
        "--softscore",
        action="store_true",
        help="Enable soft scoring: per-vehicle score = (match ? +1 : -1) * (|r×J| / sum_i |p_i^-|·|r_i|), aggregated as mean (baseline 0)",
    )
    # Visualization of contact point and OBBs (alias velocity-plot for compatibility)
    p.add_argument("--contact-viz", help="Path to save contact + OBB visualization (PNG)")
    p.add_argument("--velocity-plot", help="Alias for --contact-viz (compatibility)")
    return p.parse_args()


def _assemble_inputs(args: argparse.Namespace) -> SpinInputs:
    trajectories = load_trajectories(args.kalman_json)
    if not trajectories:
        raise RuntimeError("No trajectories found in Kalman results")
    tid_a, tid_b, closest = _select_collision_pair(trajectories, args.tracking_ids)

    collision_frame = args.collision_frame
    if collision_frame is None:
        fc, _ = _first_contact_frame_3d(
            trajectories, tid_a, tid_b, args.kalman_json,
            k=1.2, consec=3, smooth_window=0, dims_mode="median",
        )
        collision_frame = fc if fc is not None else _infer_collision_frame(closest)

    masses: Dict[int, float] = {}
    if args.masses:
        if len(args.masses) == 1:
            masses[tid_a] = args.masses[0]; masses[tid_b] = args.masses[0]
        elif len(args.masses) == 2:
            masses[tid_a] = args.masses[0]; masses[tid_b] = args.masses[1]
        else:
            raise ValueError("Provide one mass (shared) or two masses (per vehicle)")
    else:
        masses[tid_a] = args.default_mass; masses[tid_b] = args.default_mass

    return SpinInputs(
        tracking_ids=[tid_a, tid_b],
        trajectories=trajectories,
        masses=masses,
        collision_frame=collision_frame,
        fps=args.fps,
        pre_frames=args.pre_frames,
        post_frames=args.post_frames,
        include_y_axis=args.include_y_axis,
    )


def _compute_spin(args: argparse.Namespace, inputs: SpinInputs) -> SpinResult:
    tid_a, tid_b = inputs.tracking_ids
    traj_a = inputs.trajectories.get(tid_a, [])
    traj_b = inputs.trajectories.get(tid_b, [])
    if not traj_a or not traj_b:
        raise RuntimeError("Missing trajectory data for selected IDs")

    # Contact point from 3D boxes (fallback to midpoint)
    posA = _position_at_frame_or_nearest(traj_a, inputs.collision_frame)
    posB = _position_at_frame_or_nearest(traj_b, inputs.collision_frame)
    contact = None
    try:
        if posA is not None and posB is not None:
            contact = _contact_point_from_3d_boxes(
                args.kalman_json, tid_a, tid_b, inputs.collision_frame, posA, posB
            )
    except Exception:
        contact = None
    if contact is None:
        contact = 0.5 * (posA + posB) if (posA is not None and posB is not None) else (posA or posB or np.zeros(3))

    rot_by_id = _load_rot_per_frame(args.kalman_json)
    yaw_map_a = rot_by_id.get(tid_a, {})
    yaw_map_b = rot_by_id.get(tid_b, {})
    dims_by_id = _load_dims_per_frame(args.kalman_json)
    dims_a = _value_at_frame_or_nearest(dims_by_id.get(tid_a, {}), inputs.collision_frame)
    dims_b = _value_at_frame_or_nearest(dims_by_id.get(tid_b, {}), inputs.collision_frame)
    rot_a = yaw_map_a.get(inputs.collision_frame)
    rot_b = yaw_map_b.get(inputs.collision_frame)

    def _axes2d(yaw: float) -> Tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        # +X -> (c,-s), +Z -> (s,c) in (x,z)
        u = np.array([c, -s], dtype=float)
        v = np.array([s,  c], dtype=float)
        u /= (np.linalg.norm(u) + 1e-12)
        v /= (np.linalg.norm(v) + 1e-12)
        return u, v

    def _rect_corners(center_xz: np.ndarray, L: float, Wd: float, yaw: float) -> np.ndarray:
        u, v = _axes2d(yaw)
        a, b = 0.5 * float(L), 0.5 * float(Wd)
        return np.stack([
            center_xz - a * u - b * v,
            center_xz + a * u - b * v,
            center_xz + a * u + b * v,
            center_xz - a * u + b * v,
        ], axis=0)

    def _nearest_on_poly(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
        best_d, best_q = None, None
        for i in range(len(poly)):
            a = poly[i]; b = poly[(i + 1) % len(poly)]
            ab = b - a
            denom = float(ab @ ab)
            t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
            q = a + t * ab
            d = float(np.linalg.norm(q - p))
            if best_d is None or d < best_d:
                best_d, best_q = d, q
        return best_q if best_q is not None else poly[0]

    vehicles: List[VehicleSpin] = []
    _tmp_tau_abs: List[float] = []
    _tmp_match: List[bool] = []
    _tmp_pminus_abs: List[float] = []
    _tmp_r_abs: List[float] = []
    for is_a, (tid, traj, yaw_map, pos) in enumerate(((tid_a, traj_a, yaw_map_a, posA), (tid_b, traj_b, yaw_map_b, posB))):
        # Observed yaw change (Δyaw) using exactly one pre frame and one post frame
        try:
            before_points = _collect_window_points(traj, inputs.collision_frame, inputs.pre_frames, direction="before")
        except Exception:
            before_points = []
        try:
            after_points = _collect_window_points(traj, inputs.collision_frame, inputs.post_frames, direction="after")
        except Exception:
            after_points = []
        # Robust Δyaw from short windows with unwrap
        delta_yaw, yaw_dbg = _delta_yaw_from_windows(
            yaw_map, inputs.collision_frame, inputs.pre_frames, inputs.post_frames, inputs.fps, min_count=3
        )

        # Estimate COM velocities to form impulse J = m (v^+ - v^-)
        # Impulse computation: either regression-based (default) or raw window slope (no-brake-filter)
        if getattr(args, "no_brake_filter", False):
            # Strict finite-difference velocities at target frames with boundary clamp (no regression)
            frames_sorted = sorted({int(pt.frame) for pt in traj})
            f_to_pos = {int(pt.frame): pt.position for pt in traj}
            tc = int(inputs.collision_frame)
            target_pre = tc - int(inputs.pre_frames)
            target_post = tc + int(inputs.post_frames)

            pre_list = [f for f in frames_sorted if f < tc]
            post_list = [f for f in frames_sorted if f > tc]

            # Anchor frames (clamped to boundary)
            if pre_list:
                if target_pre < pre_list[0]:
                    f_pre_anchor = pre_list[0]
                else:
                    f_pre_anchor = max([f for f in pre_list if f <= target_pre], default=pre_list[-1])
            else:
                f_pre_anchor = None

            if post_list:
                if target_post > post_list[-1]:
                    f_post_anchor = post_list[-1]
                else:
                    # first >= target_post
                    candidates = [f for f in post_list if f >= target_post]
                    f_post_anchor = candidates[0] if candidates else post_list[-1]
            else:
                f_post_anchor = None

            # Finite diff around anchors
            def diff_backward(f1: int) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
                prevs = [f for f in frames_sorted if f < f1]
                if prevs:
                    f0 = prevs[-1]
                    if f0 in f_to_pos and f1 in f_to_pos and f1 > f0:
                        dt = (f1 - f0) / float(inputs.fps)
                        v = (f_to_pos[f1] - f_to_pos[f0]) / dt
                        return v, (f0, f1)
                return None

            def diff_forward(f0: int) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
                nexts = [f for f in frames_sorted if f > f0]
                if nexts:
                    f1 = nexts[0]
                    if f0 in f_to_pos and f1 in f_to_pos and f1 > f0:
                        dt = (f1 - f0) / float(inputs.fps)
                        v = (f_to_pos[f1] - f_to_pos[f0]) / dt
                        return v, (f0, f1)
                return None

            # v- prefer backward diff at pre-anchor; if not, forward diff at pre-anchor
            if f_pre_anchor is not None:
                res = diff_backward(f_pre_anchor)
                if res is None:
                    res = diff_forward(f_pre_anchor)
                if res is not None:
                    v_before, pair_pre = res
                else:
                    v_before, pair_pre = np.zeros(3), None
            else:
                v_before, pair_pre = np.zeros(3), None

            # v+ prefer forward diff at post-anchor; if not, backward diff at post-anchor
            if f_post_anchor is not None:
                res = diff_forward(f_post_anchor)
                if res is None:
                    res = diff_backward(f_post_anchor)
                if res is not None:
                    v_after, pair_post = res
                else:
                    v_after, pair_post = np.zeros(3), None
            else:
                v_after, pair_post = np.zeros(3), None

            if not inputs.include_y_axis:
                v_before = v_before.copy(); v_before[1] = 0.0
                v_after = v_after.copy(); v_after[1] = 0.0

            J_world = float(inputs.masses.get(tid, 0.0)) * (v_after - v_before)
            method_tag = "raw"
            v_minus = v_before
            v_plus = v_after
            # Anchor frames for debug: targets and used pairs
            f_minus_anchor = (pair_pre[1] if pair_pre else None)
            f_plus_anchor = (pair_post[1] if pair_post else None)
            pre_note = f"[target {target_pre}, anchor {f_pre_anchor}, used {pair_pre}]"
            post_note = f"[target {target_post}, anchor {f_post_anchor}, used {pair_post}]"
        else:
            # Smoothed instantaneous velocities near t_c from both sides (no regression)
            if getattr(args, "debug", False):
                v_minus, dbg_pre = _extrapolate_velocity_at_debug(traj, inputs.collision_frame, inputs.pre_frames, 'before', inputs.fps)
                v_plus, dbg_post = _extrapolate_velocity_at_debug(traj, inputs.collision_frame, inputs.post_frames, 'after',  inputs.fps)
            else:
                v_minus = _extrapolate_velocity_at(traj, inputs.collision_frame, inputs.pre_frames, 'before', inputs.fps)
                v_plus  = _extrapolate_velocity_at(traj, inputs.collision_frame, inputs.post_frames, 'after',  inputs.fps)
            J_world = float(inputs.masses.get(tid, 0.0)) * (v_plus - v_minus)
            method_tag = "smoothed"
            # Anchor frames for debug (nearest frames on each side of t_c)
            try:
                frames_sorted = sorted({int(pt.frame) for pt in traj})
                pre_candidates = [f for f in frames_sorted if f < int(inputs.collision_frame)]
                post_candidates = [f for f in frames_sorted if f > int(inputs.collision_frame)]
                f_minus_anchor = (max(pre_candidates) if pre_candidates else None)
                f_plus_anchor = (min(post_candidates) if post_candidates else None)
            except Exception:
                f_minus_anchor = None
                f_plus_anchor = None
            pre_note = ""; post_note = ""
        J_world_2d = J_world[[0, 2]]

        # Lever arm from center to contact (world XZ)
        r_world_2d = (contact - pos)[[0, 2]] if (contact is not None and pos is not None) else np.zeros(2)

        # Geometric torque direction: r x n where n is from contact to nearest OBB boundary
        yaw_use = yaw_map.get(inputs.collision_frame)
        if yaw_use is None and yaw_map:
            k = min(sorted(yaw_map.keys()), key=lambda k: abs(k - inputs.collision_frame))
            yaw_use = float(yaw_map[k])
        dims_use = dims_a if is_a == 0 else dims_b
        rot_use = rot_a if is_a == 0 else rot_b
        pos_use = pos
        # Rotate to body frame (x forward, y left)
        tau_geom = None
        K_val = 0.0
        if rot_use is not None:
            cbt, sbt = math.cos(float(rot_use)), math.sin(float(rot_use))
            Rwb = np.array([[cbt, sbt], [-sbt, cbt]], dtype=float)
            r_body = Rwb @ r_world_2d
            J_body = Rwb @ J_world_2d
            tau = float(r_body[0] * J_body[1] - r_body[1] * J_body[0])
            denom = float(np.linalg.norm(J_body) * np.linalg.norm(r_body))
            K_val = float(np.clip((tau / denom) if denom > 0 else 0.0, -1.0, 1.0))

        # Also compute geometric torque sign (for reference/visual if needed)
        if contact is not None and pos_use is not None and dims_use is not None and rot_use is not None:
            center_xz = np.array([float(pos_use[0]), float(pos_use[2])], dtype=float)
            L, Wd = float(dims_use[2]), float(dims_use[1])
            rect = _rect_corners(center_xz, L, Wd, float(rot_use))
            Cxz = np.array([float(contact[0]), float(contact[2])], dtype=float)
            near = _nearest_on_poly(Cxz, rect)
            r = Cxz - center_xz
            n = near - Cxz
            tau_geom = float(r[0] * n[1] - r[1] * n[0])

        # Scoring rule: compare predicted direction vs post-pre yaw delta sign
        s_pred = _sign(K_val)
        # delta_yaw already computed by window unwrap; if missing, leave None
        # Use BEV convention consistent with visualization: Δyaw>0 => CCW
        # Map Δyaw to on-screen CCW-positive convention.
        # Our BEV axis mapping makes positive yaw appear CW on screen;
        # use a fixed sign to align the scoring convention with visualization.
        yaw_score_sign = -1.0
        s_post = _sign((yaw_score_sign * delta_yaw) if (delta_yaw is not None) else None)
        # Binary or soft score (soft scoring finalized after loop)
        match = (s_pred != 0 and s_post != 0 and s_pred == s_post)
        if getattr(args, "softscore", False):
            score = 0.0  # placeholder; filled after weights computed
        else:
            score = 1.0 if match else 0.0
        _tmp_match.append(match)

        # Store |tau| = |r x J| (planar) for optional torque-weighted softscore
        try:
            tau_abs_i = abs(float(r_world_2d[0] * J_world_2d[1] - r_world_2d[1] * J_world_2d[0]))
        except Exception:
            tau_abs_i = 0.0
        _tmp_tau_abs.append(tau_abs_i)
        # Also store |p^-| and |r|
        try:
            pminus_abs = float(np.linalg.norm((float(inputs.masses.get(tid, 0.0)) * v_minus[[0, 2]])))
        except Exception:
            pminus_abs = 0.0
        _tmp_pminus_abs.append(pminus_abs)
        try:
            r_abs = float(np.linalg.norm(r_world_2d))
        except Exception:
            r_abs = 0.0
        _tmp_r_abs.append(r_abs)

        # ---- Debug prints (XZ only) -------------------------------------------------
        try:
            vm_xz = v_minus[[0, 2]] if v_minus is not None else np.zeros(2)
            vp_xz = v_plus[[0, 2]] if v_plus is not None else np.zeros(2)
            dv_xz = vp_xz - vm_xz
            mass_i = float(inputs.masses.get(tid, 0.0))
            J_xz = mass_i * dv_xz
            def fmt(vec):
                return f"[{vec[0]: .4f}, {vec[1]: .4f}]"
            print(f"Vehicle {tid} velocity/impulse debug ({method_tag}) [axes: world (x,z)]")
            anchor_m = (str(f_minus_anchor) if f_minus_anchor is not None else "N/A")
            anchor_p = (str(f_plus_anchor) if f_plus_anchor is not None else "N/A")
            extra_m = f" {pre_note}" if method_tag == "raw" else ""
            extra_p = f" {post_note}" if method_tag == "raw" else ""
            print(f"  v-   = {fmt(vm_xz)} m/s  (frame {anchor_m}){extra_m}")
            print(f"  v+   = {fmt(vp_xz)} m/s  (frame {anchor_p}){extra_p}")
            if getattr(args, "debug", False):
                # Print yaw frames used
                try:
                    yp = yaw_dbg.get("yaw_pre_frames") if 'yaw_dbg' in locals() else None
                    yq = yaw_dbg.get("yaw_post_frames") if 'yaw_dbg' in locals() else None
                    print(f"  yaw pre frames: {yp}")
                    print(f"  yaw post frames: {yq}")
                    print(f"  yaw pre unwrap: {[round(x,4) for x in yaw_dbg.get('yaw_pre_unwrap', [])]}")
                    print(f"  yaw post unwrap: {[round(x,4) for x in yaw_dbg.get('yaw_post_unwrap', [])]}")
                    print(f"  Δyaw(win) = {yaw_dbg.get('delta_yaw')}")
                except Exception:
                    pass
                # Print segments used for smoothed extrapolation
                if method_tag == "smoothed":
                    try:
                        def segdbg(d):
                            pairs = d.get("pairs", [])
                            w = d.get("weights", [])
                            vxz = d.get("vel_xz", [])
                            return pairs, w, vxz
                        p_pairs, p_w, p_v = segdbg(dbg_pre)
                        a_pairs, a_w, a_v = segdbg(dbg_post)
                        print(f"  pre segments: {p_pairs}")
                        print(f"  pre weights : {[round(x,4) for x in p_w]}")
                        print(f"  pre v_seg   : {[ [round(v[0],4), round(v[1],4)] for v in p_v ]}")
                        print(f"  post segments: {a_pairs}")
                        print(f"  post weights : {[round(x,4) for x in a_w]}")
                        print(f"  post v_seg   : {[ [round(v[0],4), round(v[1],4)] for v in a_v ]}")
                    except Exception:
                        pass
            print(f"  dv   = {fmt(dv_xz)} m/s")
            print(f"  mass = {mass_i:.1f} kg")
            print(f"  J    = {fmt(J_xz)} kg·m/s")
        except Exception:
            pass

        # Extra debug: r, tau, K, p- and torque terms
        if getattr(args, "debug", False):
            try:
                r_xz = r_world_2d
                tau_abs_i = abs(float(r_xz[0] * J_world_2d[1] - r_xz[1] * J_world_2d[0]))
                pminus_abs = float(np.linalg.norm((float(inputs.masses.get(tid, 0.0)) * (v_minus[[0, 2]]))))
                print(f"  r_xz = [{r_xz[0]: .4f}, {r_xz[1]: .4f}], |r|={np.linalg.norm(r_xz):.4f}")
                print(f"  tau_z = {float(r_xz[0] * J_world_2d[1] - r_xz[1] * J_world_2d[0]): .4f}, |tau|={tau_abs_i:.4f}")
                print(f"  K = {K_val}, |p^-|={pminus_abs:.4f}")
            except Exception:
                pass

        # Minimal fields retained; J_body/r_body left as zeros for compatibility
        yaw_final = float(yaw_use) if yaw_use is not None else 0.0
        vehicles.append(
            VehicleSpin(
                tracking_id=tid,
                mass=float(inputs.masses.get(tid, 0.0)),
                v_before=np.zeros(3),
                v_after=np.zeros(3),
                yaw_at_collision=yaw_final,
                r_body=np.zeros(2),
                J_body=np.zeros(2),
                K=K_val,
                yaw_rate_before=None,
                yaw_rate_after=None,
                delta_omega=delta_yaw,
                score=score,
            )
        )

    # Final scoring
    if getattr(args, "softscore", False) and len(vehicles) == len(_tmp_tau_abs) == len(_tmp_match):
        eps = 1e-12
        denom = sum((p * r) for p, r in zip(_tmp_pminus_abs, _tmp_r_abs))
        weights = [(t / denom) if denom > eps else 0.0 for t in _tmp_tau_abs]
        new_scores: List[float] = []
        for i, v in enumerate(vehicles):
            s = (1.0 if _tmp_match[i] else -1.0) * float(weights[i])
            v.score = s
            new_scores.append(s)
        j_p = float(np.mean(new_scores)) if new_scores else 0.0
    else:
        final_scores: List[float] = []
        for i, v in enumerate(vehicles):
            v.score = 1.0 if _tmp_match[i] else 0.0
            final_scores.append(v.score)
        j_p = float(np.mean(final_scores)) if final_scores else (0.0 if getattr(args, "softscore", False) else 0.5)
    result = SpinResult(
        collision_frame=inputs.collision_frame,
        collision_time=float(inputs.collision_frame / inputs.fps),
        vehicles=vehicles,
        j_p=j_p,
    )

    # Optional: draw contact + OBBs on XZ plane
    out_viz = args.contact_viz or args.velocity_plot
    if out_viz:
        try:
            _draw_contact_viz(
                output_path=out_viz,
                kalman_json=args.kalman_json,
                tid_a=tid_a,
                tid_b=tid_b,
                frame=inputs.collision_frame,
                pos_a=posA,
                pos_b=posB,
                contact=contact,
                masses=inputs.masses,
                fps=inputs.fps,
                pre_frames=inputs.pre_frames,
                post_frames=inputs.post_frames,
                include_y_axis=inputs.include_y_axis,
                viz_style=args.viz_style,
                use_raw_impulse=getattr(args, "no_brake_filter", False),
                impulse_red=getattr(args, "impulse_red", False),
                dashed_vector=getattr(args, "dashed_vector", "momentum"),
                use_refined_contact=getattr(args, "use_refined_contact", False),
                flip_arc=getattr(args, "flip_arc", False),
                contact_j_sense=getattr(args, "contact_j_sense", "reaction"),
            )
        except Exception as exc:
            print(f"Warning: failed to render contact visualization: {exc}")

    return result


def _draw_contact_viz(
    output_path: str,
    kalman_json: str,
    tid_a: int,
    tid_b: int,
    frame: int,
    pos_a: Optional[np.ndarray],
    pos_b: Optional[np.ndarray],
    contact: Optional[np.ndarray],
    *,
    img_size: Tuple[int, int] = (1920, 1080),
    scale: float = 12.0,
    masses: Optional[Dict[int, float]] = None,
    fps: float = 30.0,
    pre_frames: int = 5,
    post_frames: int = 5,
    include_y_axis: bool = False,
    viz_style: str = "minimal",
    use_raw_impulse: bool = False,
    impulse_red: bool = False,
    dashed_vector: str = "momentum",
    use_refined_contact: bool = False,
    flip_arc: bool = False,
    contact_j_sense: str = "reaction",
) -> None:
    try:
        import cv2
    except Exception:
        print("Warning: OpenCV not installed; skipping visualization.")
        return

    # Load dims and yaw nearest to the frame
    dims_by_id = _load_dims_per_frame(kalman_json)
    rot_by_id = _load_rot_per_frame(kalman_json)
    dims_a = _value_at_frame_or_nearest(dims_by_id.get(tid_a, {}), frame)
    dims_b = _value_at_frame_or_nearest(dims_by_id.get(tid_b, {}), frame)
    yaw_a = rot_by_id.get(tid_a, {}).get(frame)
    yaw_b = rot_by_id.get(tid_b, {}).get(frame)
    # Basic fallbacks
    if pos_a is None and pos_b is None:
        print("Warning: no positions to draw; skipping visualization.")
        return

    # Prepare canvas
    W, H = img_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    center = (W // 2, int(H * 0.9))

    # Grid
    grid_spacing = 50
    for x in range(0, W, grid_spacing):
        cv2.line(img, (x, 0), (x, H), (40, 40, 40), 1)
    for y in range(0, H, grid_spacing):
        cv2.line(img, (0, y), (W, y), (40, 40, 40), 1)

    # Axes (only in full mode to reduce arrow clutter)
    if viz_style == "full":
        cv2.arrowedLine(img, center, (center[0] + 120, center[1]), (0, 0, 255), 3)
        cv2.arrowedLine(img, center, (center[0], center[1] - 120), (0, 255, 0), 3)
        cv2.putText(img, "X (world)", (center[0] + 130, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, "Z (world)", (center[0] + 5, center[1] - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def to_px(p: np.ndarray) -> Tuple[int, int]:
        return int(p[0] * scale + center[0]), int(-p[2] * scale + center[1])

    # Helper: draw a world 2D vector as arrow in pixels (available to inner helpers)
    def _draw_vec(start_world: np.ndarray, vec_world: np.ndarray, color: Tuple[int, int, int], k: float = 1.0, thickness: int = 2):
        sx, sy = to_px(np.array([start_world[0], 0.0, start_world[1]], dtype=float))
        ex, ey = to_px(np.array([start_world[0] + k * vec_world[0], 0.0, start_world[1] + k * vec_world[1]], dtype=float))
        cv2.arrowedLine(img, (sx, sy), (ex, ey), color, thickness, tipLength=0.2)

    def _axes2d(yaw: float) -> Tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        # Match CenterTrack R=[[c,0,s],[0,1,0],[-s,0,c]] mapping:
        #   +X (length) -> (c, -s) in XZ, +Z (width) -> (s, c) in XZ
        u = np.array([c, -s], dtype=float)   # length axis projected to XZ
        v = np.array([s,  c], dtype=float)   # width  axis projected to XZ
        # normalize
        u = u / (np.linalg.norm(u) + 1e-12)
        v = v / (np.linalg.norm(v) + 1e-12)
        return u, v

    def _rect_corners(center_xz: np.ndarray, L: float, Wd: float, yaw: float) -> np.ndarray:
        u, v = _axes2d(yaw)
        a, b = 0.5 * float(L), 0.5 * float(Wd)
        return np.stack([
            center_xz - a * u - b * v,
            center_xz + a * u - b * v,
            center_xz + a * u + b * v,
            center_xz - a * u + b * v,
        ], axis=0)

    # Draw vehicle A OBB (if data available)
    color_a = (0, 200, 255)  # yellow-ish
    color_b = (255, 0, 255)  # magenta
    rectA = None
    rectB = None
    if pos_a is not None and dims_a is not None and yaw_a is not None:
        cA = np.array([float(pos_a[0]), float(pos_a[2])], dtype=float)
        L, Wd = float(dims_a[2]), float(dims_a[1])
        rectA = _rect_corners(cA, L, Wd, float(yaw_a))
        ptsA = np.array([[int(p[0] * scale + center[0]), int(-p[1] * scale + center[1]) ] for p in rectA], dtype=int)
        cv2.polylines(img, [ptsA.reshape(-1, 1, 2)], True, color_a, 3)
        # center point
        px, py = to_px(np.array([pos_a[0], 0.0, pos_a[2]], dtype=float))
        cv2.circle(img, (px, py), 5, color_a, -1)
        cv2.putText(img, f"ID {tid_a}", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_a, 2)

    if pos_b is not None and dims_b is not None and yaw_b is not None:
        cB = np.array([float(pos_b[0]), float(pos_b[2])], dtype=float)
        L, Wd = float(dims_b[2]), float(dims_b[1])
        rectB = _rect_corners(cB, L, Wd, float(yaw_b))
        ptsB = np.array([[int(p[0] * scale + center[0]), int(-p[1] * scale + center[1]) ] for p in rectB], dtype=int)
        cv2.polylines(img, [ptsB.reshape(-1, 1, 2)], True, color_b, 3)
        px, py = to_px(np.array([pos_b[0], 0.0, pos_b[2]], dtype=float))
        cv2.circle(img, (px, py), 5, color_b, -1)
        cv2.putText(img, f"ID {tid_b}", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_b, 2)

    # Helper: dashed polygon drawing (approximate by short line segments)
    def _draw_dashed_poly(points_px: np.ndarray, color: Tuple[int, int, int], thickness: int = 2,
                          dash_px: int = 10, gap_px: int = 6) -> None:
        pts = points_px.reshape(-1, 2).astype(int)
        edges = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        for p0, p1 in edges:
            x0, y0 = p0
            x1, y1 = p1
            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-3:
                continue
            ux = dx / seg_len
            uy = dy / seg_len
            s = 0.0
            draw = True
            while s < seg_len:
                run = min(dash_px if draw else gap_px, seg_len - s)
                if draw:
                    sx = int(x0 + ux * s)
                    sy = int(y0 + uy * s)
                    ex = int(x0 + ux * (s + run))
                    ey = int(y0 + uy * (s + run))
                    cv2.line(img, (sx, sy), (ex, ey), color, thickness, lineType=cv2.LINE_AA)
                s += run
                draw = not draw

    # Prepare trajectories to compute impulses for torque drawing
    trajs = load_trajectories(kalman_json)
    def _impulse_and_arm(tid: int, pos: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
        traj = trajs.get(tid, [])
        if pos is None or not traj:
            return np.zeros(2), np.zeros(2), 0.0
        if use_raw_impulse:
            # Raw velocities from window slopes
            try:
                bpts = _collect_window_points(traj, frame, pre_frames, direction="before")
                v_b = _estimate_velocity(bpts, fps)
            except Exception:
                v_b = np.zeros(3)
            try:
                apts = _collect_window_points(traj, frame, post_frames, direction="after")
                v_a = _estimate_velocity(apts, fps)
            except Exception:
                v_a = np.zeros(3)
            v_b = v_b.copy(); v_b[1] = 0.0
            v_a = v_a.copy(); v_a[1] = 0.0
            Jw = (masses.get(tid, 0.0) if masses else 0.0) * (v_a - v_b)[[0, 2]]
        else:
            # Extrapolated velocities at t_c from both sides
            v_minus = _extrapolate_velocity_at(traj, frame, pre_frames, 'before', fps)
            v_plus  = _extrapolate_velocity_at(traj, frame, post_frames, 'after',  fps)
            Jw = (masses.get(tid, 0.0) if masses else 0.0) * (v_plus - v_minus)[[0, 2]]
        rw = (contact - pos)[[0, 2]] if contact is not None else np.zeros(2)
        tau = float(rw[0] * Jw[1] - rw[1] * Jw[0])
        return Jw, rw, tau

    J_a, r_a, _tau_imp_a = _impulse_and_arm(tid_a, pos_a)
    J_b, r_b, _tau_imp_b = _impulse_and_arm(tid_b, pos_b)

    # Overlay dashed OBBs for pre/post windows, shifted horizontally for readability
    grid_spacing = 50
    dash_shift_cells = 8  # shift by N grid cells to avoid overlap
    shift_dx_world_pre = -(dash_shift_cells * grid_spacing) / float(scale)   # left for pre
    shift_dx_world_post = (dash_shift_cells * grid_spacing) / float(scale)   # right for post

    def _window_frames_for(traj: List[TrajectoryPoint], f0: int, window: int, direction: str) -> List[int]:
        frames = sorted({int(pt.frame) for pt in traj})
        if direction == "before":
            # strictly before contact frame
            cand = [f for f in frames if f < int(f0)]
            return cand[-window:] if cand else []
        else:
            # strictly after contact frame
            cand = [f for f in frames if f > int(f0)]
            return cand[:window] if cand else []

    def _window_pose(tid: int, traj: List[TrajectoryPoint], f0: int, window: int, direction: str):
        frs = _window_frames_for(traj, f0, window, direction)
        if not frs:
            return None
        # center from trajectory points at those frames (if present)
        pos_map = {int(pt.frame): pt.position for pt in traj}
        centers = []
        for f in frs:
            if f in pos_map:
                centers.append(np.array([float(pos_map[f][0]), float(pos_map[f][2])], dtype=float))
        if not centers:
            return None
        center_xz = np.mean(np.stack(centers, axis=0), axis=0)
        # yaw: unwrap then mean
        rot_map = rot_by_id.get(tid, {}) if 'rot_by_id' in locals() else {}
        yaws = [float(rot_map[f]) for f in frs if f in rot_map]
        if len(yaws) >= 2:
            yaws = np.unwrap(np.array(yaws, dtype=float))
            yaw_val = float(np.mean(yaws))
        elif len(yaws) == 1:
            yaw_val = float(yaws[0])
        else:
            yaw_val = None
        # dims: median across frames
        dims_map = dims_by_id.get(tid, {}) if 'dims_by_id' in locals() else {}
        dims_list = [np.asarray(dims_map[f], dtype=float) for f in frs if f in dims_map]
        if dims_list:
            dims_arr = np.stack(dims_list, axis=0)
            dims_med = np.median(dims_arr, axis=0)
        else:
            dims_med = None
        if yaw_val is None or dims_med is None:
            return None
        L, Wd = float(dims_med[2]), float(dims_med[1])
        return center_xz, L, Wd, yaw_val

    # Helper to compute window velocity (XZ) for a given label
    def _window_velocity_xz(tid: int, label: str) -> np.ndarray:
        traj = trajs.get(tid, [])
        if not traj:
            return np.zeros(2, dtype=float)
        direction = "before" if label == "pre" else "after"
        win = pre_frames if label == "pre" else post_frames
        if use_raw_impulse:
            # Window slope (raw)
            try:
                pts = _collect_window_points(traj, frame, win, direction=direction)
                v = _estimate_velocity(pts, fps)
            except Exception:
                v = np.zeros(3, dtype=float)
        else:
            # Extrapolated instantaneous velocity at t_c from corresponding side
            try:
                v = _extrapolate_velocity_at(traj, frame, win, 'before' if direction == 'before' else 'after', fps)
            except Exception:
                v = np.zeros(3, dtype=float)
        if not include_y_axis:
            v = v.copy(); v[1] = 0.0
        return v[[0, 2]]

    # Prepare dashed vectors per vehicle and label
    dash_vecs: Dict[Tuple[int, str], np.ndarray] = {}
    labels = ["pre", "post"]
    for tid in (tid_a, tid_b):
        for lab in labels:
            if dashed_vector == "impulse":
                # Use the same global impulse J for both pre/post
                Jw = J_a if tid == tid_a else J_b
                dash_vecs[(tid, lab)] = Jw
            else:
                v_xz = _window_velocity_xz(tid, lab)
                if dashed_vector == "velocity":
                    dash_vecs[(tid, lab)] = v_xz
                else:  # momentum
                    m = (masses.get(tid, 0.0) if masses else 0.0)
                    dash_vecs[(tid, lab)] = float(m) * v_xz

    # Determine unified scaling if vectors share the same units as contact J
    # - For momentum mode: unify scaling across {p^-, p^+, J} so lengths are comparable (triangle check).
    # - For impulse mode: unify across impulses as well.
    # - For velocity mode: dashed vectors use their own scale; J keeps its own (different units).
    unify_with_J = dashed_vector in ("momentum", "impulse")
    if dashed_vector == "momentum":
        vec_pool = [J_a, J_b] + list(dash_vecs.values())
        max_mag = max(1e-6, max(float(np.linalg.norm(v)) for v in vec_pool))
        k_shared = (120.0 / scale) / max_mag
        k_dashed = k_shared
        Jk = k_shared
    elif dashed_vector == "impulse":
        vec_pool = [J_a, J_b] + list(dash_vecs.values())
        max_mag = max(1e-6, max(float(np.linalg.norm(v)) for v in vec_pool))
        k_shared = (120.0 / scale) / max_mag
        k_dashed = k_shared
        Jk = k_shared
    else:
        # velocity mode: scale dashed independently; J keeps its own scale
        max_mag = 0.0
        for vec in dash_vecs.values():
            max_mag = max(max_mag, float(np.linalg.norm(vec)))
        k_dashed = (120.0 / scale) / max(1e-6, max_mag)

    # Impulse arrow color selection (reuse switch)
    imp_color_a = (0, 0, 255) if impulse_red else color_a
    imp_color_b = (0, 0, 255) if impulse_red else color_b

    def _draw_dashed_window(
        tid: int,
        traj: List[TrajectoryPoint],
        label: str,
        color: Tuple[int, int, int],
        imp_color: Optional[Tuple[int, int, int]] = None,
    ):
        cfg = _window_pose(tid, traj, frame, pre_frames if label == 'pre' else post_frames, 'before' if label=='pre' else 'after')
        if cfg is None:
            return
        center_xz, L, Wd, yaw_val = cfg
        # apply offset: pre to left, post to right (world units)
        dx = shift_dx_world_pre if label == 'pre' else shift_dx_world_post
        center_xz_shifted = center_xz + np.array([dx, 0.0], dtype=float)
        rect = _rect_corners(center_xz_shifted, L, Wd, float(yaw_val))
        pts = np.array([[int(p[0] * scale + center[0]), int(-p[1] * scale + center[1])] for p in rect], dtype=int)
        _draw_dashed_poly(pts, color, thickness=2, dash_px=12, gap_px=8)
        # small text tag
        tag_pos = pts.mean(axis=0).astype(int)
        cv2.putText(img, label, (int(tag_pos[0])+5, int(tag_pos[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # draw dashed-window vector at dashed center (momentum/velocity/impulse)
        vec = dash_vecs.get((tid, label))
        if vec is not None and np.linalg.norm(vec) > 0:
            _draw_vec(center_xz_shifted, vec, imp_color if imp_color is not None else color, k=k_dashed, thickness=3)
            # Console debug: dashed vector length (world and px)
            try:
                vnorm = float(np.linalg.norm(vec))
                length_px = float(k_dashed * scale * vnorm)
                kind = "P" if dashed_vector == "momentum" else ("V" if dashed_vector == "velocity" else "J")
                unit = "kg·m/s" if dashed_vector == "momentum" else ("m/s" if dashed_vector == "velocity" else "kg·m/s")
                print(f"Vehicle {tid} {label} dashed {kind}: |{kind}|={vnorm:.4f} {unit}, len≈{length_px:.1f}px")
            except Exception:
                pass

    # Draw pre/post for both vehicles
    _draw_dashed_window(tid_a, trajs.get(tid_a, []), 'pre', color_a, imp_color=imp_color_a)
    _draw_dashed_window(tid_a, trajs.get(tid_a, []), 'post', color_a, imp_color=imp_color_a)
    _draw_dashed_window(tid_b, trajs.get(tid_b, []), 'pre', color_b, imp_color=imp_color_b)
    _draw_dashed_window(tid_b, trajs.get(tid_b, []), 'post', color_b, imp_color=imp_color_b)

    # Draw original contact point unless using refined-only mode
    if not use_refined_contact:
        if contact is not None and np.all(np.isfinite(contact)):
            px, py = to_px(contact)
            cv2.circle(img, (px, py), 7, (0, 255, 255), -1)
            cv2.putText(img, "contact", (px + 10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # (draw_vec defined earlier; avoid redefining to prevent closure binding issues)

    # Normalize J magnitude for arrow scaling (may be unified above)
    if not unify_with_J:
        Jmax = max(1e-6, float(max(np.linalg.norm(J_a), np.linalg.norm(J_b))))
        Jk = (120.0 / scale) / Jmax

    # Draw r (from vehicle center to contact) and J at contact point (skip if using refined-only)
    if pos_a is not None and not use_refined_contact:
        cA_world = np.array([float(pos_a[0]), float(pos_a[2])], dtype=float)
        # remove r vector (center -> contact) from visualization
        if viz_style == "full" and np.linalg.norm(J_a) > 0:
            if contact is not None and np.all(np.isfinite(contact)):
                start = np.array([float(contact[0]), float(contact[2])], dtype=float)
            else:
                start = cA_world
            # Impulse arrow color (red if switch on, else vehicle color)
            imp_color_a = (0, 0, 255) if impulse_red else color_a
            # Sense: reaction (on vehicle) uses +J; action (by vehicle) uses -J
            J_draw = J_a if contact_j_sense == "reaction" else (-J_a)
            _draw_vec(start, J_draw, imp_color_a, k=Jk, thickness=3)

    if pos_b is not None and not use_refined_contact:
        cB_world = np.array([float(pos_b[0]), float(pos_b[2])], dtype=float)
        # remove r vector (center -> contact) from visualization
        if viz_style == "full" and np.linalg.norm(J_b) > 0:
            if contact is not None and np.all(np.isfinite(contact)):
                start = np.array([float(contact[0]), float(contact[2])], dtype=float)
            else:
                start = cB_world
            imp_color_b = (0, 0, 255) if impulse_red else color_b
            J_draw = J_b if contact_j_sense == "reaction" else (-J_b)
            _draw_vec(start, J_draw, imp_color_b, k=Jk, thickness=3)

    # Geometry contact rays: from contact to nearest OBB boundary points
    def _nearest_on_poly(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
        # p: (2,), poly: (N,2) CCW rectangle
        best_d = None
        best_q = None
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            ab = b - a
            t = 0.0
            denom = ab @ ab
            if denom > 1e-12:
                t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
            q = a + t * ab
            d = float(np.linalg.norm(q - p))
            if best_d is None or d < best_d:
                best_d = d
                best_q = q
        return best_q if best_q is not None else poly[0]

    tau_geom_a = 0.0
    tau_geom_b = 0.0
    if contact is not None and np.all(np.isfinite(contact)):
        cC2 = np.array([float(contact[0]), float(contact[2])], dtype=float)
        if rectA is not None and pos_a is not None:
            nearA = _nearest_on_poly(cC2, rectA)
            rA = cC2 - np.array([float(pos_a[0]), float(pos_a[2])], dtype=float)
            nA = nearA - cC2
            tau_geom_a = float(rA[0] * nA[1] - rA[1] * nA[0])
        if rectB is not None and pos_b is not None:
            nearB = _nearest_on_poly(cC2, rectB)
            rB = cC2 - np.array([float(pos_b[0]), float(pos_b[2])], dtype=float)
            nB = nearB - cC2
            tau_geom_b = float(rB[0] * nB[1] - rB[1] * nB[0])

    # Draw arc arrows around contact to indicate torque direction (per-vehicle)
    def _arc_arrow(center_xy: Tuple[int, int], radius: int, sign: float, color: Tuple[int, int, int]):
        if sign == 0:
            return
        # CCW if sign>0, CW if sign<0
        start, end = (-60, 240) if sign > 0 else (240, -60)
        cv2.ellipse(img, center_xy, (radius, radius), 0, start, end, color, 2)
        # arrow head at the end point, oriented tangentially
        ang = math.radians(end)
        ex = float(center_xy[0] + radius * math.cos(ang))
        ey = float(center_xy[1] + radius * math.sin(ang))
        # radial vector from center to end
        rx = ex - float(center_xy[0])
        ry = ey - float(center_xy[1])
        # screen-space tangent: CCW -> rotate radial by +90° CCW; CW -> +90° CW
        if sign > 0:  # CCW
            tx, ty = -ry, rx
        else:         # CW
            tx, ty = ry, -rx
        norm = math.hypot(tx, ty) or 1.0
        tx /= norm; ty /= norm
        # build two head legs rotated ±alpha from tangent
        alpha = math.radians(25.0)
        ca, sa = math.cos(alpha), math.sin(alpha)
        # rotate (tx,ty) by ±alpha (CCW rotation in screen coords)
        lx, ly = tx * ca - ty * sa, tx * sa + ty * ca
        rx2, ry2 = tx * ca + ty * sa, -tx * sa + ty * ca
        head_len = 9
        hx1 = int(ex - head_len * lx)
        hy1 = int(ey - head_len * ly)
        hx2 = int(ex - head_len * rx2)
        hy2 = int(ey - head_len * ry2)
        cv2.line(img, (int(ex), int(ey)), (hx1, hy1), color, 2)
        cv2.line(img, (int(ex), int(ey)), (hx2, hy2), color, 2)

    if contact is not None and np.all(np.isfinite(contact)) and not use_refined_contact:
        cx, cy = to_px(contact)
        # Torque sign based on displayed J_draw: s = sign((r x J_draw)_z)
        def _sign(v: float) -> int:
            return (1 if v > 0 else (-1 if v < 0 else 0))
        # r vectors from earlier computations (center->contact)
        J_draw_a = J_a if contact_j_sense == "reaction" else (-J_a)
        J_draw_b = J_b if contact_j_sense == "reaction" else (-J_b)
        # Default: invert sign for on-screen CCW visual (OpenCV y-down)
        sA = -_sign(float(r_a[0] * J_draw_a[1] - r_a[1] * J_draw_a[0]))
        sB = -_sign(float(r_b[0] * J_draw_b[1] - r_b[1] * J_draw_b[0]))
        if flip_arc:
            sA = -sA; sB = -sB
        _arc_arrow((cx, cy), 28, sA, color_a)
        _arc_arrow((cx, cy), 42, sB, color_b)

    # Refine contact point within the window by minimal OBB distance and draw it 8 grid cells upward
    try:
        trajs = trajs  # reuse
        pos_map_a = {int(pt.frame): pt.position for pt in trajs.get(tid_a, [])}
        pos_map_b = {int(pt.frame): pt.position for pt in trajs.get(tid_b, [])}

        fmin = int(frame - pre_frames)
        fmax = int(frame + post_frames)
        cand_frames = []
        for f in range(fmin, fmax + 1):
            if f in pos_map_a and f in pos_map_b \
               and (f in dims_by_id.get(tid_a, {}) and f in dims_by_id.get(tid_b, {})) \
               and (f in rot_by_id.get(tid_a, {}) and f in rot_by_id.get(tid_b, {})):
                cand_frames.append(f)

        def rect_at(tid: int, f: int) -> Optional[np.ndarray]:
            pos = (pos_map_a if tid == tid_a else pos_map_b).get(f)
            dims = dims_by_id.get(tid, {}).get(f)
            yawv = rot_by_id.get(tid, {}).get(f)
            if pos is None or dims is None or yawv is None:
                return None
            center_xz = np.array([float(pos[0]), float(pos[2])], dtype=float)
            L, Wd = float(dims[2]), float(dims[1])
            return _rect_corners(center_xz, L, Wd, float(yawv))

        def seg_closest_points(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
            da = a1 - a0
            db = b1 - b0
            r = a0 - b0
            A = float(da @ da)
            E = float(db @ db)
            F = float(db @ r)
            if A <= 1e-12 and E <= 1e-12:
                return a0, b0, float(np.linalg.norm(a0 - b0))
            if A <= 1e-12:
                s = 0.0
                t = np.clip(F / E, 0.0, 1.0)
            else:
                C = float(da @ r)
                if E <= 1e-12:
                    t = 0.0
                    s = np.clip(-C / A, 0.0, 1.0)
                else:
                    B = float(da @ db)
                    denom = A * E - B * B
                    if abs(denom) > 1e-12:
                        s = np.clip((B * F - C * E) / denom, 0.0, 1.0)
                    else:
                        s = 0.0
                    t = (B * s + F) / E
                    if t < 0.0:
                        t = 0.0
                        s = np.clip(-C / A, 0.0, 1.0)
                    elif t > 1.0:
                        t = 1.0
                        s = np.clip((B - C) / A, 0.0, 1.0)
            Pa = a0 + s * da
            Pb = b0 + t * db
            return Pa, Pb, float(np.linalg.norm(Pa - Pb))

        def poly_closest_points(polyA: np.ndarray, polyB: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
            best = None
            nA, nB = polyA.shape[0], polyB.shape[0]
            for i in range(nA):
                a0, a1 = polyA[i], polyA[(i + 1) % nA]
                for j in range(nB):
                    b0, b1 = polyB[j], polyB[(j + 1) % nB]
                    pa, pb, d = seg_closest_points(a0, a1, b0, b1)
                    if best is None or d < best[0]:
                        best = (d, pa, pb)
            return best[1], best[2], best[0] if best is not None else (polyA[0], polyB[0], 0.0)

        best_tuple = None  # (dmin, f, pa, pb)
        for f in cand_frames:
            rA = rect_at(tid_a, f)
            rB = rect_at(tid_b, f)
            if rA is None or rB is None:
                continue
            pa, pb, d = poly_closest_points(rA, rB)
            if best_tuple is None or d < best_tuple[0]:
                best_tuple = (d, f, pa, pb)
        if best_tuple is not None:
            dmin, fsel, pa, pb = best_tuple
            mid = 0.5 * (pa + pb)  # world XZ
            # draw the refined contact marker shifted upward by 8 grid cells (visual separation)
            rx = int(mid[0] * scale + center[0])
            ry = int(-mid[1] * scale + center[1] - 8 * grid_spacing)
            cv2.circle(img, (rx, ry), 6, (0, 255, 255), 2)
            cv2.putText(img, "refined", (rx + 8, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Helpers to draw with upward shift of 8 grid cells
            def to_px_ref(p_xz: np.ndarray) -> Tuple[int, int]:
                px = int(p_xz[0] * scale + center[0])
                py = int(-p_xz[1] * scale + center[1] - 8 * grid_spacing)
                return px, py
            def _draw_vec_shifted(start_world_xz: np.ndarray, vec_world_xz: np.ndarray, color: Tuple[int, int, int], k: float = 1.0, thickness: int = 2):
                sx, sy = to_px_ref(start_world_xz)
                ex, ey = to_px_ref(start_world_xz + k * vec_world_xz)
                cv2.arrowedLine(img, (sx, sy), (ex, ey), color, thickness, tipLength=0.2)

            # Draw refined-frame OBBs for both vehicles (shifted up 8 grids)
            try:
                # Vehicle A
                dims_af = dims_by_id.get(tid_a, {}).get(fsel)
                yaw_af  = rot_by_id.get(tid_a, {}).get(fsel)
                pos_af  = pos_map_a.get(fsel)
                if dims_af is not None and yaw_af is not None and pos_af is not None:
                    cA_xz = np.array([float(pos_af[0]), float(pos_af[2])], dtype=float)
                    L_a, W_a = float(dims_af[2]), float(dims_af[1])
                    rectA_ref = _rect_corners(cA_xz, L_a, W_a, float(yaw_af))
                    ptsA_ref = np.array([to_px_ref(p) for p in rectA_ref], dtype=int)
                    cv2.polylines(img, [ptsA_ref.reshape(-1, 1, 2)], True, color_a, 2)
                # Vehicle B
                dims_bf = dims_by_id.get(tid_b, {}).get(fsel)
                yaw_bf  = rot_by_id.get(tid_b, {}).get(fsel)
                pos_bf  = pos_map_b.get(fsel)
                if dims_bf is not None and yaw_bf is not None and pos_bf is not None:
                    cB_xz = np.array([float(pos_bf[0]), float(pos_bf[2])], dtype=float)
                    L_b, W_b = float(dims_bf[2]), float(dims_bf[1])
                    rectB_ref = _rect_corners(cB_xz, L_b, W_b, float(yaw_bf))
                    ptsB_ref = np.array([to_px_ref(p) for p in rectB_ref], dtype=int)
                    cv2.polylines(img, [ptsB_ref.reshape(-1, 1, 2)], True, color_b, 2)
            except Exception:
                pass

            # Draw vehicle center -> refined contact vectors (r_ref) at fsel (shifted)
            try:
                pos_af = pos_map_a.get(fsel)
                pos_bf = pos_map_b.get(fsel)
                if pos_af is not None:
                    cA_world_xz = np.array([float(pos_af[0]), float(pos_af[2])], dtype=float)
                    rA_ref = mid - cA_world_xz
                    _draw_vec_shifted(cA_world_xz, rA_ref, color_a, k=1.0, thickness=2)
                if pos_bf is not None:
                    cB_world_xz = np.array([float(pos_bf[0]), float(pos_bf[2])], dtype=float)
                    rB_ref = mid - cB_world_xz
                    _draw_vec_shifted(cB_world_xz, rB_ref, color_b, k=1.0, thickness=2)
            except Exception:
                pass

            # Draw J arrows at refined contact (shifted), with same scaling Jk
            try:
                start_ref = np.array([float(mid[0]), float(mid[1])], dtype=float)
                if np.linalg.norm(J_a) > 0:
                    imp_color_a2 = (0, 0, 255) if impulse_red else color_a
                    J_draw_a = J_a if contact_j_sense == "reaction" else (-J_a)
                    _draw_vec_shifted(start_ref, J_draw_a, imp_color_a2, k=Jk, thickness=3)
                if np.linalg.norm(J_b) > 0:
                    imp_color_b2 = (0, 0, 255) if impulse_red else color_b
                    J_draw_b = J_b if contact_j_sense == "reaction" else (-J_b)
                    _draw_vec_shifted(start_ref, J_draw_b, imp_color_b2, k=Jk, thickness=3)
            except Exception:
                pass

            # Draw torque-direction arc arrows at refined contact using r_ref x displayed J (shifted)
            try:
                def tau_sign_for(pos_center: Optional[np.ndarray], J_vec: np.ndarray) -> float:
                    if pos_center is None:
                        return 0.0
                    c = np.array([float(pos_center[0]), float(pos_center[2])], dtype=float)
                    r = mid - c
                    s = float(np.sign(r[0] * J_vec[1] - r[1] * J_vec[0]))
                    # Default invert for on-screen CCW; flip_arc toggles back
                    s = -s
                    return -s if flip_arc else s

                sA = tau_sign_for(pos_map_a.get(fsel), J_draw_a)
                sB = tau_sign_for(pos_map_b.get(fsel), J_draw_b)
                rcx, rcy = to_px_ref(mid)
                _arc_arrow((rcx, rcy), 24, sA, color_a)
                _arc_arrow((rcx, rcy), 36, sB, color_b)
            except Exception:
                pass

            print(f"Refined contact within window: frame={fsel}, d_min={dmin:.4f} m")
    except Exception as _err:
        print(f"Warning: refine-contact failed: {_err}")

    # Title
    cv2.putText(img, f"Collision frame {int(frame)}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Save
    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    import cv2  # ensure present for imwrite
    cv2.imwrite(output_path, img)
    print(f"Saved contact visualization to {output_path}")


def _print_summary(args: argparse.Namespace, inputs: SpinInputs, result: SpinResult) -> None:
    print("Contact spin direction (yaw-only)")
    print("=" * 60)
    print(f"Kalman results : {os.path.abspath(args.kalman_json)}")
    print(f"Tracking IDs    : {inputs.tracking_ids[0]}, {inputs.tracking_ids[1]}")
    print(f"Collision frame : {result.collision_frame} (t = {result.collision_time:.3f}s)")
    print(f"Frame window    : -{inputs.pre_frames} / +{inputs.post_frames} frames")
    print(f"FPS             : {inputs.fps:.3f}")
    print("-" * 60)
    for v in result.vehicles:
        print(f"Vehicle {v.tracking_id} (mass = {v.mass:.1f} kg, yaw={v.yaw_at_collision:.3f} rad)")
        print(f"  K = {v.K}")
        print(f"  Δω = {v.delta_omega}")
        print(f"  score = {v.score}")
        print("-" * 60)
    print(f"Aggregated j_p: {result.j_p}")


def _save_j_value(output_path: Optional[str], inputs: SpinInputs, result: SpinResult) -> None:
    if not output_path:
        return
    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    payload = {
        "tracking_ids": inputs.tracking_ids,
        "collision_frame": result.collision_frame,
        "collision_time": result.collision_time,
        "j_p": result.j_p,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Stored j_p to {output_path}")


def _serialize_result(args: argparse.Namespace, inputs: SpinInputs, result: SpinResult) -> Dict[str, object]:
    return {
        "kalman_json": os.path.abspath(args.kalman_json),
        "tracking_ids": inputs.tracking_ids,
        "collision_frame": result.collision_frame,
        "collision_time": result.collision_time,
        "fps": inputs.fps,
        "pre_frames": inputs.pre_frames,
        "post_frames": inputs.post_frames,
        "include_y_axis": inputs.include_y_axis,
        "j_p": result.j_p,
        "vehicles": [
            {
                "tracking_id": v.tracking_id,
                "mass": v.mass,
                "K": v.K,
                "delta_omega": v.delta_omega,
                "score": v.score,
            }
            for v in result.vehicles
        ],
    }


if __name__ == "__main__":
    args = parse_args()
    # Provide path for helpers that optionally read from environment
    try:
        os.environ["KALMAN_JSON_PATH"] = args.kalman_json
    except Exception:
        pass
    inputs = _assemble_inputs(args)
    result = _compute_spin(args, inputs)
    _print_summary(args, inputs, result)
    _save_j_value(args.j_output, inputs, result)
    if args.output:
        payload = _serialize_result(args, inputs, result)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved contact spin summary to {args.output}")
