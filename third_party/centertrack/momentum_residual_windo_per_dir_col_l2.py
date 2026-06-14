#!/usr/bin/env python3
"""Linear momentum residual analysis around a detected collision frame.

This script loads the Kalman-smoothed trajectory output from ``traj_recon.py``
(``kalman_smoothed_*.mp4_results.json``), isolates the two vehicles involved in
a collision, and evaluates the linear momentum residual defined by

    Δp = Σ m_i v_i^+ − Σ m_i v_i^−
    J_p = ||Δp|| / Σ m_i ||v_i^−||

where ``v_i^-`` and ``v_i^+`` denote the average pre- and post-impact velocity
vectors over configurable frame windows. Small values of ``J_p`` indicate that
the observed motion is consistent with an impulsive, nearly closed system.

Typical usage (auto-detect the closest pair and collision frame):

    python momentum_residual_window.py \
        --kalman-json savepath/VV_209/kalman_smoothed_VV_209.mp4_results.json \
        --fps 30 --pre-frames 5 --post-frames 5

You can override the collision frame or the vehicle IDs manually:

    python momentum_residual_window.py \
        --kalman-json ... \
        --tracking-ids 1478 1479 \
        --collision-frame 119 \
        --fps 30 --pre-frames 8 --post-frames 12

Set ``--output`` to store the computed metrics in JSON form for later use.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from detect_collision_point import (
    ClosestApproach,
    TrajectoryPoint,
    compute_closest_approach,
    load_trajectories,
)


@dataclass
class MomentumInputs:
    tracking_ids: List[int]
    trajectories: Dict[int, List[TrajectoryPoint]]
    masses: Dict[int, float]
    collision_frame: int
    fps: float
    pre_frames: int
    post_frames: int
    include_y_axis: bool
    total_frames: int
    # Optional debug series for 3D box first-contact (frames, dist, thr, ratio)
    bbox3d_series: Optional[Dict[str, List[float]]] = None


@dataclass
class VehicleMomentum:
    tracking_id: int
    mass: float
    velocity_before: np.ndarray
    velocity_after: np.ndarray

    @property
    def momentum_before(self) -> np.ndarray:
        return self.mass * self.velocity_before

    @property
    def momentum_after(self) -> np.ndarray:
        return self.mass * self.velocity_after


@dataclass
class MomentumResidual:
    collision_frame: int
    collision_time: float
    vehicles: List[VehicleMomentum]
    delta_p: np.ndarray
    delta_p_abs: np.ndarray
    j_p: float
    j_axis: np.ndarray
    denominator: float
    axis_denominator: np.ndarray
    # Kinetic energy terms and residual (JE)
    ek_before: float
    ek_after: float
    j_e: float


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# 3D bbox first-contact defaults (CLI stays unchanged; tweak here if needed)
USE_3D_BOX_FIRST_CONTACT: bool = True
BBOX3D_SCALE: float = 1.2  # e.g., set to 1.10 or 1.20 as you wish
BBOX3D_CONSEC: int = 3      # consecutive frames to declare contact
BBOX3D_SMOOTH: int = 0      # moving-average window for distance (0 disables)
BBOX3D_DIMS: str = "median" # or "per_frame"

def _ensure_two_tracking_ids(tracking_ids: Sequence[int]) -> Tuple[int, int]:
    if len(tracking_ids) != 2:
        raise ValueError(
            f"Momentum analysis expects exactly two tracking IDs, got {tracking_ids}",
        )
    return int(tracking_ids[0]), int(tracking_ids[1])


def _select_collision_pair(
    trajectories: Dict[int, List[TrajectoryPoint]],
    provided_ids: Optional[Sequence[int]] = None,
) -> Tuple[int, int, ClosestApproach]:
    if provided_ids and len(provided_ids) == 2:
        tid_a, tid_b = _ensure_two_tracking_ids(provided_ids)
        result = compute_closest_approach(tid_a, trajectories[tid_a], tid_b, trajectories[tid_b])
        if result is None:
            raise RuntimeError(
                f"Failed to compute closest-approach distance for IDs {tid_a}, {tid_b}",
            )
        return tid_a, tid_b, result

    if provided_ids:
        raise ValueError("Provide either zero or two tracking IDs for analysis")

    tracking_ids = sorted(trajectories.keys())
    if len(tracking_ids) < 2:
        raise ValueError("Need at least two trajectories to analyze a collision")

    best_pair: Optional[ClosestApproach] = None
    best_ids: Optional[Tuple[int, int]] = None

    for idx_a in range(len(tracking_ids)):
        for idx_b in range(idx_a + 1, len(tracking_ids)):
            tid_a = tracking_ids[idx_a]
            tid_b = tracking_ids[idx_b]
            result = compute_closest_approach(tid_a, trajectories[tid_a], tid_b, trajectories[tid_b])
            if result is None:
                continue
            if best_pair is None or result.distance < best_pair.distance:
                best_pair = result
                best_ids = (tid_a, tid_b)

    if best_pair is None or best_ids is None:
        raise RuntimeError("Could not find any valid trajectory pair for collision analysis")

    return best_ids[0], best_ids[1], best_pair


def _infer_collision_frame(closest: ClosestApproach) -> int:
    if closest.collision_frame is not None:
        return closest.collision_frame

    # Fall back to midpoint between closest frames (rounded to nearest int)
    return int(round(0.5 * (closest.frame_a + closest.frame_b)))


def _find_turning_point_frame(
    traj_a: List[TrajectoryPoint],
    traj_b: List[TrajectoryPoint],
    smoothing_window: int = 3,
    slope_tolerance_ratio: float = 0.02,
    min_flat_frames: int = 2,
) -> Optional[int]:
    """Detect the first frame where the distance stops shrinking appreciably.

    A light moving-average smooth is applied to reduce jitter before we examine
    the gradient. We then look for the earliest index where the smoothed series
    transitions from a clearly negative slope to a flat or positive slope for a
    couple of frames. If we cannot locate such a turning point we fall back to
    returning the frame of the smoothed minimum distance.
    """

    frame_to_pos_a = {pt.frame: pt.position for pt in traj_a}
    frame_to_pos_b = {pt.frame: pt.position for pt in traj_b}
    common_frames = sorted(set(frame_to_pos_a) & set(frame_to_pos_b))

    if len(common_frames) < 3:
        return None

    distances = np.array(
        [
            float(np.linalg.norm(frame_to_pos_a[frame] - frame_to_pos_b[frame]))
            for frame in common_frames
        ],
        dtype=float,
    )

    if np.allclose(distances, distances[0]):
        # No meaningful change in distance -> treat the earliest frame as the turning point.
        return common_frames[0]

    window = max(1, min(smoothing_window, len(distances)))
    if window > 1:
        kernel = np.ones(window, dtype=float) / window
        smoothed = np.convolve(distances, kernel, mode="same")
    else:
        smoothed = distances

    gradient = np.gradient(smoothed, common_frames)
    span = float(np.max(smoothed) - np.min(smoothed))
    tolerance = max(1e-4, slope_tolerance_ratio * span)

    # Short-circuit if we never observe a shrinking phase.
    if not np.any(gradient < -tolerance):
        return common_frames[int(np.argmin(smoothed))]

    def _has_recent_descent(idx: int) -> bool:
        start = max(0, idx - 3)
        return np.any(gradient[start:idx] < -tolerance)

    for idx in range(1, len(common_frames)):
        if gradient[idx] >= -tolerance and _has_recent_descent(idx):
            end = min(len(gradient), idx + min_flat_frames)
            if np.all(gradient[idx:end] >= -tolerance):
                return common_frames[idx]

    return common_frames[int(np.argmin(smoothed))]


# ---------------------------------------------------------------------------
# 3D bounding-box based first-contact frame (distance vs. scaled box radii)
# ---------------------------------------------------------------------------

def _load_dims_per_frame(json_path: str) -> Dict[int, Dict[int, np.ndarray]]:
    """Load per-frame 3D box dimensions (h,w,l) for each tracking ID.

    Returns: dict[tracking_id][frame] -> np.ndarray shape (3,)
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    dims_by_id: Dict[int, Dict[int, np.ndarray]] = {}
    for frame_key, dets in data.items():
        try:
            frame = int(frame_key)
        except Exception:
            continue
        for det in dets:
            tid = det.get("tracking_id")
            dim = det.get("dim")
            loc = det.get("loc")
            # Require both dim and loc to ensure the frame is usable
            if tid is None or dim is None or loc is None:
                continue
            if not isinstance(dim, (list, tuple)) or len(dim) != 3:
                continue
            dims_by_id.setdefault(int(tid), {})[frame] = np.asarray(dim, dtype=float)

    return dims_by_id


def _median_dims(dims_frames: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    if not dims_frames:
        return None
    arr = np.stack(list(dims_frames.values()))
    return np.median(arr, axis=0)


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr.size == 0:
        return arr
    window = int(max(1, window))
    kernel = np.ones(window, dtype=float) / float(window)
    # mode='same' keeps alignment with original indices
    return np.convolve(arr, kernel, mode="same")


def _first_contact_frame_3d(
    trajectories: Dict[int, List[TrajectoryPoint]],
    tid_a: int,
    tid_b: int,
    kalman_json: str,
    *,
    k: float = 1.15,
    consec: int = 3,
    smooth_window: int = 0,
    dims_mode: str = "median",  # or "per_frame"
) -> Tuple[Optional[int], Optional[Dict[str, List[float]]]]:
    """Compute earliest frame where d(loc) <= k*(rA+rB) holds for 'consec' frames.

    r = diag(dim)/2; diag = sqrt(h^2 + w^2 + l^2). Units assumed consistent (meters).
    If no consecutive satisfaction, fall back to frame with minimal ratio d/thr.

    Returns (frame, series_dict) where series_dict has keys: frames, dist, thr, ratio.
    """
    traj_a = trajectories.get(tid_a, [])
    traj_b = trajectories.get(tid_b, [])
    if not traj_a or not traj_b:
        return None, None

    frame_to_pos_a = {pt.frame: pt.position for pt in traj_a}
    frame_to_pos_b = {pt.frame: pt.position for pt in traj_b}
    frames_common = sorted(set(frame_to_pos_a) & set(frame_to_pos_b))
    if not frames_common:
        return None, None

    dims_by_id = _load_dims_per_frame(kalman_json)
    dims_a_map = dims_by_id.get(tid_a, {})
    dims_b_map = dims_by_id.get(tid_b, {})

    # Derive dims per strategy
    if dims_mode == "median":
        dims_a_med = _median_dims(dims_a_map)
        dims_b_med = _median_dims(dims_b_map)
        if dims_a_med is None or dims_b_med is None:
            # Fall back to per-frame if median not available
            dims_mode = "per_frame"
    
    dists: List[float] = []
    thrs: List[float] = []
    valid_frames: List[int] = []

    for f in frames_common:
        pos_a = frame_to_pos_a.get(f)
        pos_b = frame_to_pos_b.get(f)
        if pos_a is None or pos_b is None:
            continue
        d = float(np.linalg.norm(pos_a - pos_b))

        if dims_mode == "median":
            dim_a = dims_a_med
            dim_b = dims_b_med
        else:
            dim_a = dims_a_map.get(f)
            dim_b = dims_b_map.get(f)

        if dim_a is None or dim_b is None:
            # Cannot compute threshold for this frame
            continue

        # Compute radii from diag/2; ensure non-negative
        diag_a = float(np.linalg.norm(np.maximum(dim_a, 0.0)))
        diag_b = float(np.linalg.norm(np.maximum(dim_b, 0.0)))
        r_sum = 0.5 * (diag_a + diag_b)
        thr = float(k * r_sum)
        if not np.isfinite(thr) or thr <= 0.0:
            continue

        valid_frames.append(f)
        dists.append(d)
        thrs.append(thr)

    if not valid_frames:
        return None, None

    dists_arr = np.asarray(dists, dtype=float)
    thrs_arr = np.asarray(thrs, dtype=float)
    if smooth_window and smooth_window > 1:
        dists_arr = _moving_average(dists_arr, int(smooth_window))

    ratio = np.divide(dists_arr, thrs_arr, out=np.full_like(dists_arr, np.inf), where=thrs_arr > 0)
    satisfied = dists_arr <= thrs_arr

    # Find earliest index where 'consec' consecutive are True
    if consec <= 1:
        idx = int(np.argmax(satisfied)) if np.any(satisfied) else None
        fc_frame = valid_frames[idx] if idx is not None and satisfied[idx] else None
    else:
        run = 0
        fc_frame = None
        for i, ok in enumerate(satisfied):
            run = run + 1 if ok else 0
            if run >= consec:
                fc_frame = valid_frames[i - consec + 1]
                break

    if fc_frame is None:
        # Fallback to minimal ratio frame
        best_idx = int(np.argmin(ratio))
        fc_frame = valid_frames[best_idx]

    series = {
        "frames": [int(f) for f in valid_frames],
        "dist": dists_arr.tolist(),
        "thr": thrs_arr.tolist(),
        "ratio": ratio.tolist(),
    }
    return int(fc_frame), series


def _extract_velocity_series(
    trajectory: List[TrajectoryPoint],
    fps: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(trajectory) < 2:
        return None

    frames = np.array([pt.frame for pt in trajectory], dtype=float)
    positions = np.stack([pt.position for pt in trajectory])

    frame_deltas = np.diff(frames)
    if np.all(frame_deltas == 0):
        return None

    dt = frame_deltas / fps
    valid = dt > 0
    if not np.any(valid):
        return None

    diffs = np.diff(positions, axis=0)
    dt = dt[valid]
    diffs = diffs[valid]
    velocities = diffs / dt[:, None]
    times = (frames[:-1] + frames[1:]) / (2.0 * fps)
    times = times[valid]

    return times, velocities


def _collect_window_points(
    trajectory: List[TrajectoryPoint],
    collision_frame: int,
    window: int,
    *,
    direction: str,
) -> List[TrajectoryPoint]:
    if direction == "before":
        subset = [pt for pt in trajectory if pt.frame <= collision_frame]
        subset = subset[-(window + 1):]
    elif direction == "after":
        subset = [pt for pt in trajectory if pt.frame >= collision_frame]
        subset = subset[: (window + 1)]
    else:
        raise ValueError("direction must be 'before' or 'after'")

    subset.sort(key=lambda pt: pt.frame)

    unique_frames = {pt.frame for pt in subset}
    if len(unique_frames) < 2:
        raise ValueError(
            "Not enough unique frames in the selected window to estimate velocity",
        )

    return subset


def _estimate_velocity(points: List[TrajectoryPoint], fps: float) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("Need at least two trajectory points to estimate velocity")

    frames = np.array([pt.frame for pt in points], dtype=float)
    times = frames / fps
    positions = np.stack([pt.position for pt in points])

    times_centered = times - times.mean()
    denom = float(np.sum(times_centered ** 2))
    if denom <= 1e-9:
        dt = times[-1] - times[0]
        if dt <= 0:
            raise ValueError("Degenerate timestamps for velocity estimation")
        return (positions[-1] - positions[0]) / dt

    positions_centered = positions - positions.mean(axis=0)
    slope = (times_centered[:, None] * positions_centered).sum(axis=0) / denom
    return slope


def _mask_axis(vec: np.ndarray, include_y: bool) -> np.ndarray:
    if include_y:
        return vec
    masked = vec.copy()
    masked[1] = 0.0
    return masked


def _compute_vehicle_momentum(
    tracking_id: int,
    trajectory: List[TrajectoryPoint],
    mass: float,
    fps: float,
    collision_frame: int,
    pre_window: int,
    post_window: int,
    include_y_axis: bool,
) -> VehicleMomentum:
    try:
        before_points = _collect_window_points(
            trajectory,
            collision_frame,
            pre_window,
            direction="before",
        )
        velocity_before = _mask_axis(_estimate_velocity(before_points, fps), include_y_axis)
    except ValueError as exc:
        print(
            f"Warning: vehicle {tracking_id} lacks sufficient pre-collision frames; "
            f"using zero velocity before impact ({exc}).",
        )
        velocity_before = np.zeros(3)

    try:
        after_points = _collect_window_points(
            trajectory,
            collision_frame,
            post_window,
            direction="after",
        )
        velocity_after = _mask_axis(_estimate_velocity(after_points, fps), include_y_axis)
    except ValueError as exc:
        print(
            f"Warning: vehicle {tracking_id} lacks sufficient post-collision frames; "
            f"using zero velocity after impact ({exc}).",
        )
        velocity_after = np.zeros(3)

    return VehicleMomentum(
        tracking_id=tracking_id,
        mass=mass,
        velocity_before=velocity_before,
        velocity_after=velocity_after,
    )


def _assemble_inputs(args: argparse.Namespace) -> MomentumInputs:
    trajectories = load_trajectories(args.kalman_json)
    if not trajectories:
        raise RuntimeError("No trajectories found in Kalman results")

    tid_a, tid_b, closest = _select_collision_pair(trajectories, args.tracking_ids)
    traj_a = trajectories.get(tid_a, [])
    traj_b = trajectories.get(tid_b, [])

    all_frames = sorted({pt.frame for traj in trajectories.values() for pt in traj})
    total_frames = len(all_frames)

    collision_frame = args.collision_frame
    bbox3d_series: Optional[Dict[str, List[float]]] = None
    if collision_frame is None:
        # Prefer 3D box first-contact by default (CLI unchanged; tune constants above)
        if USE_3D_BOX_FIRST_CONTACT:
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
            if fc_frame is not None:
                collision_frame = fc_frame
                bbox3d_series = series

        # Fallbacks if still not determined
        if collision_frame is None:
            turning_point = _find_turning_point_frame(traj_a, traj_b)
            if turning_point is not None:
                collision_frame = turning_point
            else:
                collision_frame = _infer_collision_frame(closest)

    masses: Dict[int, float] = {}
    default_mass = args.default_mass

    if args.masses:
        if len(args.masses) == 1:
            masses[tid_a] = args.masses[0]
            masses[tid_b] = args.masses[0]
        elif len(args.masses) == 2:
            masses[tid_a] = args.masses[0]
            masses[tid_b] = args.masses[1]
        else:
            raise ValueError("Provide either one mass (shared) or two masses (per vehicle)")
    else:
        masses[tid_a] = default_mass
        masses[tid_b] = default_mass

    return MomentumInputs(
        tracking_ids=[tid_a, tid_b],
        trajectories=trajectories,
        masses=masses,
        collision_frame=collision_frame,
        fps=args.fps,
        pre_frames=args.pre_frames,
        post_frames=args.post_frames,
        include_y_axis=args.include_y_axis,
        total_frames=total_frames,
        bbox3d_series=bbox3d_series,
    )


def _compute_residual(inputs: MomentumInputs) -> MomentumResidual:
    tid_a, tid_b = inputs.tracking_ids
    traj_a = inputs.trajectories.get(tid_a, [])
    traj_b = inputs.trajectories.get(tid_b, [])

    if not traj_a or not traj_b:
        raise RuntimeError("Missing trajectory data for one of the selected IDs")

    vehicle_momentums: List[VehicleMomentum] = []
    vehicle_momentums.append(
        _compute_vehicle_momentum(
            tid_a,
            traj_a,
            inputs.masses[tid_a],
            inputs.fps,
            inputs.collision_frame,
            inputs.pre_frames,
            inputs.post_frames,
            inputs.include_y_axis,
        )
    )
    vehicle_momentums.append(
        _compute_vehicle_momentum(
            tid_b,
            traj_b,
            inputs.masses[tid_b],
            inputs.fps,
            inputs.collision_frame,
            inputs.pre_frames,
            inputs.post_frames,
            inputs.include_y_axis,
        )
    )

    momentum_before = sum((vm.momentum_before for vm in vehicle_momentums), np.zeros(3))
    momentum_after = sum((vm.momentum_after for vm in vehicle_momentums), np.zeros(3))
    delta_p = momentum_after - momentum_before
    delta_p_abs = np.abs(delta_p)

    # Per-axis diagnostic denominator (kept for j_axis reporting)
    axis_denominator = sum((np.abs(vm.momentum_before) for vm in vehicle_momentums), np.zeros(3))

    # L2-based denominator: sum_i m_i * ||v_i^-||_2, consistent with spec
    denom_l2_terms = [float(vm.mass * np.linalg.norm(vm.velocity_before)) for vm in vehicle_momentums]
    denominator = float(np.sum(denom_l2_terms))

    with np.errstate(divide="ignore", invalid="ignore"):
        j_axis = np.divide(
            delta_p_abs,
            axis_denominator,
            out=np.full(3, math.inf),
            where=axis_denominator > 0,
        )

    # L2 residual: ||Δp||_2 / (Σ m_i ||v_i^-||_2)
    numerator = float(np.linalg.norm(delta_p))
    j_p = (numerator / denominator) if denominator > 0 else math.inf

    collision_time = inputs.collision_frame / inputs.fps

    # Kinetic energy terms
    ek_before_terms = [0.5 * float(vm.mass) * float(np.linalg.norm(vm.velocity_before)) ** 2 for vm in vehicle_momentums]
    ek_after_terms = [0.5 * float(vm.mass) * float(np.linalg.norm(vm.velocity_after)) ** 2 for vm in vehicle_momentums]
    ek_before = float(np.sum(ek_before_terms))
    ek_after = float(np.sum(ek_after_terms))
    # Numerical stability and bounded JE: add epsilon to denom and clip to [0, 1]
    eps = 1e-6
    j_e_raw = (ek_after - ek_before) / max(ek_before, eps)
    j_e = float(max(0.0, min(1.0, j_e_raw)))

    return MomentumResidual(
        collision_frame=inputs.collision_frame,
        collision_time=collision_time,
        vehicles=vehicle_momentums,
        delta_p=delta_p,
        delta_p_abs=delta_p_abs,
        j_p=j_p,
        j_axis=j_axis,
        denominator=denominator,
        axis_denominator=axis_denominator,
        ek_before=ek_before,
        ek_after=ek_after,
        j_e=j_e,
    )


def _format_vector(vec: np.ndarray) -> str:
    return f"[{vec[0]:6.3f}, {vec[1]:6.3f}, {vec[2]:6.3f}]"


def _print_summary(args: argparse.Namespace, inputs: MomentumInputs, residual: MomentumResidual) -> None:
    tid_a, tid_b = inputs.tracking_ids
    print("Collision momentum residual analysis")
    print("=" * 60)
    print(f"Kalman results : {os.path.abspath(args.kalman_json)}")
    print(f"Tracking IDs    : {tid_a}, {tid_b}")
    print(f"Collision frame : {inputs.collision_frame} (t = {residual.collision_time:.3f}s)")
    if inputs.total_frames:
        print(f"Total frames    : {inputs.total_frames}")
    print(f"Frame window    : -{inputs.pre_frames} / +{inputs.post_frames} frames")
    axes_label = "X,Y,Z" if inputs.include_y_axis else "X,Z (Y suppressed)"
    print(f"FPS             : {inputs.fps:.3f}")
    print(f"Axes used       : {axes_label}")
    print("-" * 60)

    for vm in residual.vehicles:
        print(f"Vehicle {vm.tracking_id} (mass = {vm.mass:.1f} kg)")
        print(f"  v^- = {_format_vector(vm.velocity_before)} m/s")
        print(f"  v^+ = {_format_vector(vm.velocity_after)} m/s")
        print(f"  p^- = {_format_vector(vm.momentum_before)} kg·m/s")
        print(f"  p^+ = {_format_vector(vm.momentum_after)} kg·m/s")
        print("  Δp  = " + _format_vector(vm.momentum_after - vm.momentum_before))
        print("-" * 60)

    print("System momentum")
    print(f"  Σ p^- = {_format_vector(sum((vm.momentum_before for vm in residual.vehicles), np.zeros(3)))} kg·m/s")
    print(f"  Σ p^+ = {_format_vector(sum((vm.momentum_after for vm in residual.vehicles), np.zeros(3)))} kg·m/s")
    print(f"  Δp    = {_format_vector(residual.delta_p)} kg·m/s")
    print(f"  |Δp|  = {_format_vector(residual.delta_p_abs)} kg·m/s")
    print(f"  Σ|p^-|= {_format_vector(residual.axis_denominator)} kg·m/s")
    print(f"  J_axis= {_format_vector(residual.j_axis)}")
    print(f"  J_p   = {residual.j_p:.6f}")
    print(f"  Denom = {residual.denominator:.6f} (Σ m_i ||v_i^-||)")
    print("Kinetic energy")
    print(f"  E_k^- = {residual.ek_before:.6f}")
    print(f"  E_k^+ = {residual.ek_after:.6f}")
    print(f"  J_E   = {residual.j_e:.6f}  (max(0, (E_k^+ - E_k^-) / E_k^-), clipped [0,1], eps=1e-6)")
    print("=" * 60)


def _save_j_value(output_path: Optional[str], inputs: MomentumInputs, residual: MomentumResidual) -> None:
    if not output_path:
        return

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        "tracking_ids": inputs.tracking_ids,
        "collision_frame": residual.collision_frame,
        "collision_time": residual.collision_time,
        "j_p": residual.j_p,
        "j_e": residual.j_e,
        "j_axis": residual.j_axis.tolist(),
        "delta_p_abs": residual.delta_p_abs.tolist(),
        "denominator_axis": residual.axis_denominator.tolist(),
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Stored J_p metric at {output_path}")


def _plot_velocity_profiles(
    inputs: MomentumInputs,
    residual: MomentumResidual,
    output_path: Optional[str],
) -> None:
    if not output_path:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping velocity plot generation.")
        return

    profiles: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for tracking_id in inputs.tracking_ids:
        trajectory = inputs.trajectories.get(tracking_id, [])
        series = _extract_velocity_series(trajectory, inputs.fps)
        if series is not None:
            profiles[tracking_id] = series

    if not profiles:
        print("Warning: not enough trajectory data to plot velocities.")
        return

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    collision_time = residual.collision_time
    highlight_start = max(0.0, collision_time - 0.5)
    highlight_end = collision_time + 0.5
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, (tracking_id, (times, velocities)) in enumerate(sorted(profiles.items())):
        color = colors[idx % len(colors)]
        speed = np.linalg.norm(velocities, axis=1)
        axes[0].plot(times, speed, label=f"Vehicle {tracking_id}", color=color, linewidth=2)
        axes[1].plot(times, velocities[:, 0], label=f"Vehicle {tracking_id}", color=color, linewidth=2)
        axes[2].plot(times, velocities[:, 2], label=f"Vehicle {tracking_id}", color=color, linewidth=2)

    axes[0].set_ylabel("|v| (m/s)")
    axes[0].set_title("Velocity magnitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("v_x (m/s)")
    axes[1].set_title("Velocity X component")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("v_z (m/s)")
    axes[2].set_title("Velocity Z component")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.axvspan(highlight_start, highlight_end, color="red", alpha=0.15)
        ax.axvline(collision_time, color="red", linestyle="--", linewidth=1.5)

    fig.suptitle("Vehicle velocity profiles", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved velocity profiles plot to {output_path}")


def _serialize_result(args: argparse.Namespace, inputs: MomentumInputs, residual: MomentumResidual) -> Dict[str, object]:
    return {
        "kalman_json": os.path.abspath(args.kalman_json),
        "tracking_ids": inputs.tracking_ids,
        "collision_frame": inputs.collision_frame,
        "collision_time": residual.collision_time,
        "fps": inputs.fps,
        "pre_frames": inputs.pre_frames,
        "post_frames": inputs.post_frames,
        "include_y_axis": inputs.include_y_axis,
        "vehicles": [
            {
                "tracking_id": vm.tracking_id,
                "mass": vm.mass,
                "velocity_before": vm.velocity_before.tolist(),
                "velocity_after": vm.velocity_after.tolist(),
                "momentum_before": vm.momentum_before.tolist(),
                "momentum_after": vm.momentum_after.tolist(),
            }
            for vm in residual.vehicles
        ],
        "delta_p": residual.delta_p.tolist(),
        "delta_p_abs": residual.delta_p_abs.tolist(),
        "j_p": residual.j_p,
        "j_e": residual.j_e,
        "j_axis": residual.j_axis.tolist(),
        "denominator": residual.denominator,
        "denominator_axis": residual.axis_denominator.tolist(),
        "ek_before": residual.ek_before,
        "ek_after": residual.ek_after,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute linear momentum residual around a collision frame",
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
        "--pre-frames",
        type=int,
        default=5,
        help="Number of frames before the collision to include in v^- window",
    )
    parser.add_argument(
        "--post-frames",
        type=int,
        default=5,
        help="Number of frames after the collision to include in v^+ window",
    )
    parser.add_argument(
        "--collision-frame",
        type=int,
        help="Collision frame index (if omitted, auto-detected from closest approach)",
    )
    parser.add_argument(
        "--tracking-ids",
        type=int,
        nargs=2,
        help="Tracking IDs of the two vehicles (default: auto-select closest pair)",
    )
    parser.add_argument(
        "--masses",
        type=float,
        nargs="+",
        help="Vehicle masses in kg (one value for shared mass, or two for per-vehicle)",
    )
    parser.add_argument(
        "--default-mass",
        type=float,
        default=1500.0,
        help="Fallback mass in kg if --masses not provided",
    )
    parser.add_argument(
        "--include-y-axis",
        action="store_true",
        help="Include Y-axis when computing velocities and momentum (default: ignore Y)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the momentum residual result as JSON",
    )
    parser.add_argument(
        "--j-output",
        help="Optional JSON file to store only the J_p value and collision metadata",
    )
    parser.add_argument(
        "--velocity-plot",
        help="Optional path to save velocity profiles plot around the collision",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inputs = _assemble_inputs(args)
    residual = _compute_residual(inputs)
    _print_summary(args, inputs, residual)
    _save_j_value(args.j_output, inputs, residual)
    _plot_velocity_profiles(inputs, residual, args.velocity_plot)

    if args.output:
        payload = _serialize_result(args, inputs, residual)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved momentum residual summary to {args.output}")
