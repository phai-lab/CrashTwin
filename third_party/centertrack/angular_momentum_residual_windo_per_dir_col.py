#!/usr/bin/env python3
"""Angular momentum residual analysis around a detected collision frame (yaw-only).

We compute angular momentum about a contact point c:
    H_y(c) = sum_i [ m_i * ((r_i - c) × v_i)_y + I_{z,i} * ω_i ]
and define the normalized jump:
    J_H = | H_y^+ - H_y^- | / ( sum_i | H_{y,i}^- | + ε )

Smaller J_H indicates rotationally plausible behavior near impact.

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
from typing import Dict, List, Optional, Sequence, Tuple

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
    r_at_collision: np.ndarray
    velocity_before: np.ndarray
    velocity_after: np.ndarray
    inertia_iz: float
    yaw_rate_before: float
    yaw_rate_after: float

    @property
    def momentum_before(self) -> np.ndarray:
        # Angular momentum (yaw-only): L = m (r × v^-)_y + I_z * ω^- (as 3D vector with only Y)
        r = self.r_at_collision.copy()
        v = self.velocity_before.copy()
        r[1] = 0.0
        v[1] = 0.0
        cross_y = r[2] * v[0] - r[0] * v[2]
        orbital = np.array([0.0, self.mass * cross_y, 0.0])
        spin = np.array([0.0, self.inertia_iz * self.yaw_rate_before, 0.0])
        return orbital + spin

    @property
    def momentum_after(self) -> np.ndarray:
        # Angular momentum (yaw-only): L = m (r × v^+)_y + I_z * ω^+ (as 3D vector with only Y)
        r = self.r_at_collision.copy()
        v = self.velocity_after.copy()
        r[1] = 0.0
        v[1] = 0.0
        cross_y = r[2] * v[0] - r[0] * v[2]
        orbital = np.array([0.0, self.mass * cross_y, 0.0])
        spin = np.array([0.0, self.inertia_iz * self.yaw_rate_after, 0.0])
        return orbital + spin


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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# 3D bbox first-contact defaults (CLI stays unchanged; tweak here if needed)
USE_3D_BOX_FIRST_CONTACT: bool = True
BBOX3D_SCALE: float = 1.2  # e.g., set to 1.10 or 1.20 as you wish
BBOX3D_CONSEC: int = 3      # consecutive frames to declare contact
BBOX3D_SMOOTH: int = 0      # moving-average window for distance (0 disables)
BBOX3D_DIMS: str = "median" # or "per_frame"
ALPHA_IZ: float = 0.25      # spin inertia coefficient for I_z ≈ α m L^2
EPS_J: float = 1e-9         # small epsilon to stabilize denominators

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


def _value_at_frame_or_nearest(map_frames: Dict[int, np.ndarray], frame: int) -> Optional[np.ndarray]:
    if frame in map_frames:
        return map_frames[frame]
    if not map_frames:
        return None
    keys = sorted(map_frames.keys())
    best_k = min(keys, key=lambda k: abs(k - frame))
    return map_frames.get(best_k)


def _load_rot_per_frame(json_path: str) -> Dict[int, Dict[int, float]]:
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    rot_by_id: Dict[int, Dict[int, float]] = {}
    for frame_key, dets in data.items():
        try:
            frame = int(frame_key)
        except Exception:
            continue
        for det in dets:
            tid = det.get("tracking_id")
            rot = det.get("rot_y")
            loc = det.get("loc")
            if tid is None or rot is None or loc is None:
                continue
            rot_by_id.setdefault(int(tid), {})[frame] = float(rot)
    return rot_by_id


def _yaw_axes_from_rot(rot_y: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = math.cos(rot_y)
    s = math.sin(rot_y)
    ux = np.array([c, 0.0, s], dtype=float)
    uy = np.array([0.0, 1.0, 0.0], dtype=float)
    uz = np.array([-s, 0.0, c], dtype=float)
    return ux, uy, uz


def _obb_support_point(center: np.ndarray, dims: np.ndarray, rot_y: float, direction: np.ndarray) -> np.ndarray:
    h, w, l = float(dims[0]), float(dims[1]), float(dims[2])
    ex, ey, ez = w * 0.5, h * 0.5, l * 0.5
    ux, uy, uz = _yaw_axes_from_rot(rot_y)
    d = direction / (np.linalg.norm(direction) + 1e-9)
    sx = math.copysign(ex, float(np.dot(d, ux))) * ux
    sy = math.copysign(ey, float(np.dot(d, uy))) * uy
    sz = math.copysign(ez, float(np.dot(d, uz))) * uz
    return center + sx + sy + sz


def _contact_point_from_3d_boxes(
    kalman_json: str,
    tid_a: int,
    tid_b: int,
    frame: int,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute a contact point on the XZ plane using 2D OBB SAT + clipping.

    Collision (overlap):
      - Use SAT to detect overlap and get minimal penetration axis.
      - Compute intersection polygon by clipping one OBB with the other.
      - Return centroid of intersection polygon on XZ plane.

    Separation (no overlap):
      - Compute closest segment pair between the two OBB rectangles on XZ plane.
      - Return midpoint of the closest pair.

    Fallbacks:
      - If dims/rot missing or any numerical issue, return None (caller will fallback).
    """
    # Load per-frame dims (h,w,l) and yaw
    dims_by_id = _load_dims_per_frame(kalman_json)
    rot_by_id = _load_rot_per_frame(kalman_json)
    dims_a_map = dims_by_id.get(tid_a, {})
    dims_b_map = dims_by_id.get(tid_b, {})
    rot_a_map = rot_by_id.get(tid_a, {})
    rot_b_map = rot_by_id.get(tid_b, {})
    dims_a = _value_at_frame_or_nearest(dims_a_map, frame)
    dims_b = _value_at_frame_or_nearest(dims_b_map, frame)
    rot_a = rot_a_map.get(frame)
    rot_b = rot_b_map.get(frame)
    if dims_a is None or dims_b is None or rot_a is None or rot_b is None:
        return None

    # Build 2D OBBs on XZ plane (Y ignored)
    def _axes2d(yaw: float) -> Tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        # Match CenterTrack's rotation R=[[c,0,s],[0,1,0],[-s,0,c]]
        # Object +X (length) -> (c, -s) in XZ, Object +Z (width) -> (s, c) in XZ
        u = np.array([c, -s], dtype=float)    # length axis in (x,z)
        v = np.array([s,  c], dtype=float)    # width  axis in (x,z)
        return u / (np.linalg.norm(u) + 1e-12), v / (np.linalg.norm(v) + 1e-12)

    def _corners2d(center_xz: np.ndarray, L: float, W: float, yaw: float) -> np.ndarray:
        u, v = _axes2d(yaw)
        a, b = 0.5 * float(L), 0.5 * float(W)
        # CCW order: (-u - v), (u - v), (u + v), (-u + v)
        return np.stack([
            center_xz - a * u - b * v,
            center_xz + a * u - b * v,
            center_xz + a * u + b * v,
            center_xz - a * u + b * v,
        ], axis=0)

    def _project_axis(pts: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
        vals = pts @ (axis / (np.linalg.norm(axis) + 1e-12))
        return float(np.min(vals)), float(np.max(vals))

    def _sat_overlap_and_mtv(cA: np.ndarray, LA: float, WA: float, yawA: float,
                             cB: np.ndarray, LB: float, WB: float, yawB: float) -> Tuple[bool, Optional[np.ndarray], float]:
        # Prepare axes
        uA, vA = _axes2d(yawA)
        uB, vB = _axes2d(yawB)
        axes = [uA, vA, uB, vB]
        ptsA = _corners2d(cA, LA, WA, yawA)
        ptsB = _corners2d(cB, LB, WB, yawB)
        # Center delta for axis sign disambiguation
        dC = cB - cA
        min_overlap = float('inf')
        mtv_axis = None
        for ax in axes:
            a_min, a_max = _project_axis(ptsA, ax)
            b_min, b_max = _project_axis(ptsB, ax)
            # Interval overlap
            overlap = min(a_max, b_max) - max(a_min, b_min)
            if overlap <= 0.0:
                return False, None, 0.0
            if overlap < min_overlap:
                min_overlap = overlap
                # Ensure axis points from A -> B
                axn = ax / (np.linalg.norm(ax) + 1e-12)
                if (dC @ axn) < 0:
                    axn = -axn
                mtv_axis = axn
        return True, mtv_axis, float(min_overlap)

    def _clip_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
        # Sutherland–Hodgman polygon clipping (convex-clipper). Both CCW.
        def inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
            # Left-of-edge test for CCW inward half-plane
            return ((b - a)[0] * (p[1] - a[1]) - (b - a)[1] * (p[0] - a[0])) >= -1e-12

        def compute_intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            # Line p1->p2 with edge a->b intersection
            # Solve for t where cross(b-a, (a + t*(p2-p1)) - a) = 0
            d = p2 - p1
            e = b - a
            denom = e[0] * d[1] - e[1] * d[0]
            if abs(denom) < 1e-12:
                return p2  # Parallel; return endpoint to avoid NaN
            t = (e[0] * (p1[1] - a[1]) - e[1] * (p1[0] - a[0])) / denom
            return p1 + t * d

        output = subject.copy()
        for i in range(len(clipper)):
            input_list = output
            output = []
            A = clipper[i]
            B = clipper[(i + 1) % len(clipper)]
            if len(input_list) == 0:
                break
            S = input_list[-1]
            for E in input_list:
                if inside(E, A, B):
                    if not inside(S, A, B):
                        output.append(compute_intersection(S, E, A, B))
                    output.append(E)
                elif inside(S, A, B):
                    output.append(compute_intersection(S, E, A, B))
                S = E
            output = np.asarray(output, dtype=float) if len(output) else np.zeros((0, 2), dtype=float)
        return output

    def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
        n = len(poly)
        if n == 0:
            return np.zeros(2, dtype=float)
        if n == 1:
            return poly[0]
        if n == 2:
            return 0.5 * (poly[0] + poly[1])
        # Polygon centroid (shoelace)
        x = poly[:, 0]
        y = poly[:, 1]
        a = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if abs(a) < 1e-12:
            # Degenerate polygon -> average
            return np.mean(poly, axis=0)
        cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (3.0 * a)
        cz = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (3.0 * a)
        return np.array([cx, cz], dtype=float)

    def _edges(pts: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]

    def _segment_segment_closest(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        # Compute closest points on 2D segments p1-q1 and p2-q2
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        a = d1 @ d1
        e = d2 @ d2
        f = d2 @ r
        if a <= 1e-12 and e <= 1e-12:
            # Both degenerate
            return p1, p2, float(np.linalg.norm(p1 - p2))
        if a <= 1e-12:
            # First degenerate -> project p1 onto segment 2
            t = np.clip(f / e, 0.0, 1.0) if e > 1e-12 else 0.0
            c2 = p2 + t * d2
            return p1, c2, float(np.linalg.norm(p1 - c2))
        if e <= 1e-12:
            # Second degenerate -> project p2 onto segment 1
            s = np.clip(-(d1 @ r) / a, 0.0, 1.0)
            c1 = p1 + s * d1
            return c1, p2, float(np.linalg.norm(c1 - p2))
        b = d1 @ d2
        c = d1 @ r
        denom = a * e - b * b
        if denom != 0.0:
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
        else:
            s = 0.0
        t = (b * s + f) / e
        if t < 0.0:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        elif t > 1.0:
            t = 1.0
            s = np.clip((b - c) / a, 0.0, 1.0)
        c1 = p1 + s * d1
        c2 = p2 + t * d2
        return c1, c2, float(np.linalg.norm(c1 - c2))

    # Prepare 2D inputs
    cA = np.array([float(pos_a[0]), float(pos_a[2])], dtype=float)
    cB = np.array([float(pos_b[0]), float(pos_b[2])], dtype=float)
    LA, WA = float(dims_a[2]), float(dims_a[1])  # length, width
    LB, WB = float(dims_b[2]), float(dims_b[1])
    yawA, yawB = float(rot_a), float(rot_b)

    # SAT test and MTV
    overlap, mtv_axis, _ = _sat_overlap_and_mtv(cA, LA, WA, yawA, cB, LB, WB, yawB)

    if overlap:
        # Compute intersection polygon by clipping A by B (both CCW rectangles)
        ptsA = _corners2d(cA, LA, WA, yawA)
        ptsB = _corners2d(cB, LB, WB, yawB)
        interAB = _clip_polygon(ptsA, ptsB)
        if interAB.shape[0] == 0:
            # Robustness: also try B clipped by A and merge
            interBA = _clip_polygon(ptsB, ptsA)
            if interBA.shape[0] == 0:
                return None
            poly = interBA
        else:
            poly = interAB
        centroid = _polygon_centroid(poly)
        return np.array([centroid[0], 0.0, centroid[1]], dtype=float)

    # Not overlapping: compute closest segment pair on rectangles and return midpoint
    ptsA = _corners2d(cA, LA, WA, yawA)
    ptsB = _corners2d(cB, LB, WB, yawB)
    edgesA = _edges(ptsA)
    edgesB = _edges(ptsB)
    best = None
    best_pair = (None, None)
    for (p1, q1) in edgesA:
        for (p2, q2) in edgesB:
            c1, c2, dist = _segment_segment_closest(p1, q1, p2, q2)
            if (best is None) or (dist < best):
                best = dist
                best_pair = (c1, c2)
    if best_pair[0] is None or best_pair[1] is None:
        return None
    mid = 0.5 * (best_pair[0] + best_pair[1])
    return np.array([mid[0], 0.0, mid[1]], dtype=float)


def _estimate_yaw_rate(frames: List[int], yaw_map: Dict[int, float], fps: float) -> Optional[float]:
    ys = []
    ts = []
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
    slope = float(np.sum(t_centered * y_centered) / denom)
    return slope


def _position_at_frame_or_nearest(traj: List[TrajectoryPoint], frame: int) -> Optional[np.ndarray]:
    if not traj:
        return None
    # Try exact match first
    for pt in traj:
        if pt.frame == frame:
            return pt.position
    # nearest by frame index
    best_gap = None
    best_pos = None
    for pt in traj:
        gap = abs(int(pt.frame) - int(frame))
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_pos = pt.position
    return best_pos


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
    r_at_collision: np.ndarray,
    yaw_map: Dict[int, float],
    inertia_iz: float,
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

    # Estimate yaw rates on same windows
    before_frames = [pt.frame for pt in before_points] if 'before_points' in locals() else []
    after_frames = [pt.frame for pt in after_points] if 'after_points' in locals() else []
    yaw_before = _estimate_yaw_rate(before_frames, yaw_map, fps) or 0.0
    yaw_after = _estimate_yaw_rate(after_frames, yaw_map, fps) or 0.0

    return VehicleMomentum(
        tracking_id=tracking_id,
        mass=mass,
        r_at_collision=r_at_collision,
        velocity_before=velocity_before,
        velocity_after=velocity_after,
        inertia_iz=float(inertia_iz),
        yaw_rate_before=float(yaw_before),
        yaw_rate_after=float(yaw_after),
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

    # Compute reference contact point from 3D boxes at collision frame; fallback to midpoint
    posA = _position_at_frame_or_nearest(traj_a, inputs.collision_frame)
    posB = _position_at_frame_or_nearest(traj_b, inputs.collision_frame)
    contact = None
    try:
        if posA is not None and posB is not None:
            # args is only available in __main__; to avoid refactor, read from env variable if not present
            kalman_json = os.environ.get("KALMAN_JSON_PATH", None)
            # Best effort: use the path passed originally if available in global 'args'
            if 'args' in globals() and getattr(args, 'kalman_json', None):
                kalman_json = args.kalman_json
            if kalman_json:
                contact = _contact_point_from_3d_boxes(kalman_json, tid_a, tid_b, inputs.collision_frame, posA, posB)
    except Exception:
        contact = None
    if contact is None:
        if posA is None and posB is None:
            contact = np.zeros(3)
        elif posA is None:
            contact = posB
        elif posB is None:
            contact = posA
        else:
            contact = 0.5 * (posA + posB)

    rA = (posA - contact) if posA is not None else np.zeros(3)
    rB = (posB - contact) if posB is not None else np.zeros(3)

    # Prepare yaw maps and inertias I_z ≈ α m L^2 using length from dims at collision frame
    kalman_json = os.environ.get("KALMAN_JSON_PATH", None)
    if 'args' in globals() and getattr(args, 'kalman_json', None):
        kalman_json = args.kalman_json
    rot_by_id = _load_rot_per_frame(kalman_json) if kalman_json else {}
    yaw_map_a = rot_by_id.get(tid_a, {})
    yaw_map_b = rot_by_id.get(tid_b, {})
    dims_by_id = _load_dims_per_frame(kalman_json) if kalman_json else {}
    dims_a_map = dims_by_id.get(tid_a, {})
    dims_b_map = dims_by_id.get(tid_b, {})
    dims_a = _value_at_frame_or_nearest(dims_a_map, inputs.collision_frame)
    dims_b = _value_at_frame_or_nearest(dims_b_map, inputs.collision_frame)
    def _len_from_dims(d):
        if d is None:
            return 0.0
        return max(0.0, float(d[2]))  # length l from [h,w,l]
    L_a = _len_from_dims(dims_a)
    L_b = _len_from_dims(dims_b)
    Iza = ALPHA_IZ * inputs.masses[tid_a] * (L_a ** 2)
    Izb = ALPHA_IZ * inputs.masses[tid_b] * (L_b ** 2)

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
            rA,
            yaw_map_a,
            Iza,
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
            rB,
            yaw_map_b,
            Izb,
        )
    )

    momentum_before = sum((vm.momentum_before for vm in vehicle_momentums), np.zeros(3))
    momentum_after = sum((vm.momentum_after for vm in vehicle_momentums), np.zeros(3))
    # Yaw-only: only Y component is meaningful
    delta_y = momentum_after[1] - momentum_before[1]
    delta_p = np.array([0.0, float(delta_y), 0.0])
    delta_p_abs = np.abs(delta_p)

    # Denominator uses sum of absolute pre-impact angular momenta to avoid
    # cancellation: Σ_i |H_{y,i}^-|
    sum_abs_Hy_before = sum(abs(vm.momentum_before[1]) for vm in vehicle_momentums)
    axis_denominator = np.array([0.0, float(sum_abs_Hy_before), 0.0])
    denominator = float(sum_abs_Hy_before)

    with np.errstate(divide="ignore", invalid="ignore"):
        j_axis = np.array([0.0, float(delta_p_abs[1] / (axis_denominator[1] + EPS_J)), 0.0])

    j_p = float(delta_p_abs[1] / (denominator + EPS_J))

    collision_time = inputs.collision_frame / inputs.fps

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
    )


def _format_vector(vec: np.ndarray) -> str:
    return f"[{vec[0]:6.3f}, {vec[1]:6.3f}, {vec[2]:6.3f}]"


def _print_summary(args: argparse.Namespace, inputs: MomentumInputs, residual: MomentumResidual) -> None:
    tid_a, tid_b = inputs.tracking_ids
    print("Angular momentum residual analysis")
    # print("=" * 60)
    # print(f"Kalman results : {os.path.abspath(args.kalman_json)}")
    # print(f"Tracking IDs    : {tid_a}, {tid_b}")
    # print(f"Collision frame : {inputs.collision_frame} (t = {residual.collision_time:.3f}s)")
    # if inputs.total_frames:
    #     print(f"Total frames    : {inputs.total_frames}")
    # print(f"Frame window    : -{inputs.pre_frames} / +{inputs.post_frames} frames")
    # print(f"FPS             : {inputs.fps:.3f}")
    # print("-" * 60)

    # for vm in residual.vehicles:
    #     print(f"Vehicle {vm.tracking_id} (mass = {vm.mass:.1f} kg)")
    #     print(f"  ω^- = {vm.yaw_rate_before:6.3f} rad/s, ω^+ = {vm.yaw_rate_after:6.3f} rad/s")
    #     print(f"  L_y^- = {vm.momentum_before[1]:8.3f} kg·m^2/s")
    #     print(f"  L_y^+ = {vm.momentum_after[1]:8.3f} kg·m^2/s")
    #     print(f"  ΔL_y  = {(vm.momentum_after[1]-vm.momentum_before[1]):8.3f} kg·m^2/s")
    #     print("-" * 60)

    print("System angular momentum (yaw-only) about contact point")
    print(f"  Σ H_y^- = {sum(vm.momentum_before[1] for vm in residual.vehicles):8.3f} kg·m^2/s")
    print(f"  Σ H_y^+ = {sum(vm.momentum_after[1] for vm in residual.vehicles):8.3f} kg·m^2/s")
    print(f"  ΔH_y    = {residual.delta_p[1]:8.3f} kg·m^2/s")
    print(f"  |ΔH_y|  = {residual.delta_p_abs[1]:8.3f} kg·m^2/s")
    print(f"  Σ|H_y^-|= {residual.axis_denominator[1]:8.3f} kg·m^2/s")
    print(f"  J_H     = {residual.j_p:.6f}")
    print(f"  Denom   = {residual.denominator:.6f} kg·m^2/s (Σ|H_y^-|)")
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
        "J_H": residual.j_p,
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
        print("Warning: matplotlib not installed, skipping yaw-rate plot generation.")
        return

    # Build yaw-rate time series for involved vehicles
    yaw_by_id = _load_rot_per_frame(os.environ.get("KALMAN_JSON_PATH", "")) if os.environ.get("KALMAN_JSON_PATH") else {}
    if 'args' in globals() and getattr(args, 'kalman_json', None):
        yaw_by_id = _load_rot_per_frame(args.kalman_json)

    series_map: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for tracking_id in inputs.tracking_ids:
        yaw_map = yaw_by_id.get(tracking_id, {})
        if len(yaw_map) < 2:
            continue
        frames = np.array(sorted(yaw_map.keys()), dtype=float)
        times = frames / inputs.fps
        yaws = np.unwrap(np.array([yaw_map[int(f)] for f in frames], dtype=float))
        if len(times) >= 2:
            omega = np.gradient(yaws, times)
            series_map[int(tracking_id)] = (times, omega)

    if not series_map:
        print("Warning: not enough yaw data to plot.")
        return

    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), sharex=False)
    collision_time = residual.collision_time
    highlight_start = max(0.0, collision_time - 0.5)
    highlight_end = collision_time + 0.5
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, (tid, (times, omega)) in enumerate(sorted(series_map.items())):
        color = colors[idx % len(colors)]
        ax.plot(times, omega, label=f"ω (ID {tid})", color=color, linewidth=2)

    ax.set_ylabel("Yaw rate ω (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvspan(highlight_start, highlight_end, color="red", alpha=0.15)
    ax.axvline(collision_time, color="red", linestyle="--", linewidth=1.5)
    fig.suptitle("Yaw rate around collision", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved yaw-rate plot to {output_path}")


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
        # Clearer aliases for angular momentum residual (backward compatible):
        "delta_H_y": residual.delta_p[1],
        "J_H": residual.j_p,
        "j_axis": residual.j_axis.tolist(),
        "denominator": residual.denominator,
        "denominator_axis": residual.axis_denominator.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute angular momentum (yaw-only) residual around a collision frame",
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
    # Stabilize downstream helpers that optionally read from environment
    try:
        os.environ["KALMAN_JSON_PATH"] = args.kalman_json
    except Exception:
        pass
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
