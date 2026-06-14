#!/usr/bin/env python3
"""
Eight-step trajectory reconstruction pipeline with optional SAM2 alignment and depth correction support

Step 1: Read JSON and get transformation matrices
Step 2: Apply transformations to objects at origin
Step 3: Visualize transformed objects
Step 4: Read and visualize detections (with depth correction option)
Step 5: Transform detections to world frame
Step 5.5: Save world frame results in detection format
Step 6: Apply Kalman filtering and RTS smoothing
Step 7: Save Kalman results in detection format

===============================================================================
DEPTH CORRECTION FEATURE:
===============================================================================

Two trajectory reading functions are now available:

1. extract_detection_bboxes() - Normal method (original)
   - Uses detection depths as-is from CenterTrack output
   - Fast and straightforward

2. extract_detection_bboxes_depth_corrected() - Depth-corrected method (new)
   - Uses metric depth images to correct detection depths
   - Projects 3D bboxes to 2D, extracts depth from corresponding image regions
   - Replaces CenterTrack depth estimates with metric depth measurements
   - Requires depth images and camera matrix

To switch between methods:
- In main() function, comment/uncomment the desired OPTION in STEP 4
- OPTION 1: Normal trajectory reading (currently active)
- OPTION 2: Depth-corrected trajectory reading (currently commented)

Example usage of depth correction:
- Ensure depth images are available in {video_id}_DEPTH/ directory
- Uncomment OPTION 2 and comment OPTION 1 in main()
- Run the pipeline to see depth-corrected results

The depth correction process:
1. Creates initial 3D bbox using original detection values
2. Projects bbox to 2D image coordinates
3. Extracts depth statistics from corresponding depth image region
4. Uses mean depth to replace original depth estimate
5. Recreates 3D bbox with corrected depth value

Optional Step 0: SAM2-aligned trajectory filtering
- Reads SAM2 mask tracks and matches them to 2D detections
- Filters detections/trajectories to only include SAM2-consistent vehicles
- Rewrites tracking IDs so that discontinuities are merged per SAM2 track
- Supplies precise SAM2 masks for depth correction (replaces bbox-based sampling)
===============================================================================
"""

import argparse
import json
import numpy as np
import cv2
from typing import List, Dict, Tuple, Set, Optional
from tqdm import tqdm
import sys
import os
from copy import deepcopy
import math
from collections import defaultdict, Counter


DEFAULT_MIN_SEGMENT_FRAMES = 5

# Import CenterTrack's 3D utils
sys.path.append('/root/CenterTrack/src/lib/utils')
try:
    from ddd_utils import compute_box_3d, project_to_image, draw_box_3d
    HAS_DDD_UTILS = True
except ImportError:
    print("Warning: Could not import ddd_utils, using fallback implementation")
HAS_DDD_UTILS = False


# ============================================================================
# STEP 0: Align Tracking Trajectories with SAM2 Masks
# ============================================================================


class SAM2MaskLookup:
    """Utility for retrieving SAM2 masks aligned with detection frames."""

    def __init__(self, sam2_path: str, detection_min_frame: int):
        self.sam2_path = sam2_path
        self.detection_min_frame = detection_min_frame

        with np.load(sam2_path) as sam2_data:
            self.height = int(sam2_data['height'][0]) if 'height' in sam2_data else None
            self.width = int(sam2_data['width'][0]) if 'width' in sam2_data else None
            raw_car_ids = [str(cid) for cid in sam2_data['car_ids']]

            self.frame_arrays = {}
            self.mask_arrays = {}
            self.car_ids: List[str] = []

            for car_id in raw_car_ids:
                frames_key = f'frames_{car_id}'
                masks_key = f'masks_{car_id}'
                if frames_key not in sam2_data or masks_key not in sam2_data:
                    continue
                frames = sam2_data[frames_key]
                masks = sam2_data[masks_key]
                if frames.size == 0 or masks.size == 0:
                    continue
                self.car_ids.append(car_id)
                self.frame_arrays[car_id] = frames
                self.mask_arrays[car_id] = masks

        if not self.car_ids:
            self.sam_min_frame = detection_min_frame
            self.frame_offset = 0
            self.frame_index_lookup = {}
            return

        self.sam_min_frame = min(int(frames[0]) for frames in self.frame_arrays.values())
        self.frame_offset = detection_min_frame - self.sam_min_frame
        self.frame_index_lookup: Dict[str, Dict[int, int]] = {}

        for car_id, frames in self.frame_arrays.items():
            detection_frames = frames + self.frame_offset
            self.frame_index_lookup[car_id] = {int(det_frame): idx for idx, det_frame in enumerate(detection_frames)}

    def iter_frame_mask_pairs(self, car_id: str):
        """Yield tuples of (detection_frame, sam_frame, mask) for a given SAM2 car ID."""
        frames = self.frame_arrays.get(car_id)
        masks = self.mask_arrays.get(car_id)
        if frames is None or masks is None:
            return

        for sam_frame, mask in zip(frames, masks):
            detection_frame = int(sam_frame + self.frame_offset)
            yield detection_frame, int(sam_frame), mask

    def get_mask(self, car_id: str, detection_frame: int) -> Optional[np.ndarray]:
        lookup = self.frame_index_lookup.get(str(car_id))
        if not lookup:
            return None

        idx = lookup.get(int(detection_frame))
        if idx is None:
            return None

        return self.mask_arrays[str(car_id)][idx]

def _compute_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute bounding box [x1, y1, x2, y2] from a boolean mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    ys = coords[:, 0]
    xs = coords[:, 1]

    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())

    return x_min, y_min, x_max, y_max


def _bbox_iou(box_a: Tuple[float, float, float, float],
              box_b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))

    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0

    return inter_area / denom


def _bbox_overlap_ratio(box_a: Tuple[float, float, float, float],
                        box_b: Tuple[float, float, float, float]) -> float:
    """Compute overlap ratio using the smaller box area as denominator."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))

    min_area = min(area_a, area_b)
    if min_area <= 0.0:
        return 0.0

    return inter_area / min_area


def match_tracking_with_sam2(video_id: str,
                             trajectory_json_path: str,
                             detections_json_path: str,
                             savepath: str,
                             min_iou: float = 0.2,
                             relaxed_iou: float = 0.1,
                             min_segment_length: int = DEFAULT_MIN_SEGMENT_FRAMES,
                             override_dims_step0: bool = False,
                             vehicle_specs_json: Optional[str] = None) -> Tuple[str, str, Dict[int, List[int]], Optional[SAM2MaskLookup]]:
    """
    Filter tracking trajectories using SAM2 masks and return updated file paths.

    Args:
        video_id: Video identifier
        trajectory_json_path: Path to the original trajectory JSON (Step 1 input)
        detections_json_path: Path to detection results JSON
        savepath: Directory used for intermediate outputs
        min_iou: Minimum overlap score (IoU or containment ratio) for keeping a frame
        relaxed_iou: Relaxed overlap score for considering candidate frames during greedy search
        min_segment_length: Shortest allowed matched segment length (frames); shorter segments are discarded

    Returns:
        Tuple of (trajectory_json_path, detections_json_path, merge_mapping)
        where the first two elements point to possibly filtered files and the
        mapping tracks which original tracking IDs contributed to each SAM2 ID.
    """
    if savepath is None:
        savepath = '.'

    sam2_path = os.path.join(savepath, f"{video_id}_sam2_masks.npz")
    if not os.path.exists(sam2_path):
        print(f"  SAM2 mask file not found at {sam2_path}, skipping Step 0 alignment")
        return trajectory_json_path, detections_json_path, {}, None

    if not os.path.exists(detections_json_path):
        print(f"  Detection JSON not found at {detections_json_path}, skipping Step 0 alignment")
        return trajectory_json_path, detections_json_path, {}, None

    try:
        with open(detections_json_path, 'r') as f:
            detection_data = json.load(f)
    except Exception as exc:
        print(f"  Failed to load detections JSON ({detections_json_path}): {exc}")
        return trajectory_json_path, detections_json_path, {}, None

    detection_frame_numbers = sorted(int(k) for k in detection_data.keys())
    if not detection_frame_numbers:
        print("  Detection JSON is empty, skipping Step 0 alignment")
        return trajectory_json_path, detections_json_path, {}, None

    det_min_frame = detection_frame_numbers[0]
    try:
        sam2_lookup = SAM2MaskLookup(sam2_path, det_min_frame)
    except Exception as exc:
        print(f"  Failed to load SAM2 masks ({sam2_path}): {exc}")
        return trajectory_json_path, detections_json_path, {}, None

    if not sam2_lookup.car_ids:
        print("  SAM2 mask file contains no car tracks, skipping Step 0 alignment")
        return trajectory_json_path, detections_json_path, {}, sam2_lookup

    filtered_detections: Dict[str, List[Dict]] = {}
    matched_sam_frames: Set[int] = set()
    merge_mapping: Dict[int, List[int]] = {}
    # Map SAM2 car_id (str) -> new tracking_id (int)
    sam2_to_newid: Dict[str, int] = {}

    print(f"  SAM2 alignment: detected {len(sam2_lookup.car_ids)} mask tracks")

    def normalize_tracking_id(raw_id):
        if isinstance(raw_id, (int, np.integer)):
            return int(raw_id)
        if isinstance(raw_id, float) and float(raw_id).is_integer():
            return int(raw_id)
        if isinstance(raw_id, str):
            raw_id = raw_id.strip()
            if raw_id.isdigit():
                return int(raw_id)
        return None

    for car_idx, car_id_str in enumerate(sam2_lookup.car_ids):
        sam_track_frames = []
        overlap_candidates: Dict[str, Dict[str, object]] = {}

        for detection_frame, sam_frame, mask in sam2_lookup.iter_frame_mask_pairs(car_id_str):
            bbox = _compute_mask_bbox(mask)
            if bbox is None:
                continue

            sam_track_frames.append({
                'det_frame': int(detection_frame),
                'sam_frame': int(sam_frame),
                'bbox': bbox
            })

            detections_in_frame = detection_data.get(str(detection_frame), [])
            if not detections_in_frame:
                continue

            for det in detections_in_frame:
                if det.get('class') != 1:
                    continue
                det_bbox = det.get('bbox')
                if not det_bbox or len(det_bbox) != 4:
                    continue

                iou = _bbox_iou(bbox, det_bbox)
                contain_ratio = _bbox_overlap_ratio(bbox, det_bbox)
                overlap_score = max(iou, contain_ratio)

                if overlap_score < relaxed_iou:
                    continue

                tracking_id_raw = det.get('tracking_id')
                if tracking_id_raw is None:
                    continue

                track_key = str(tracking_id_raw)
                candidate = overlap_candidates.setdefault(track_key, {
                    'norm': normalize_tracking_id(tracking_id_raw),
                    'raw': tracking_id_raw,
                    'records': []
                })
                candidate['records'].append({
                    'frame': int(detection_frame),
                    'sam_frame': int(sam_frame),
                    'iou': float(iou),
                    'contain': float(contain_ratio),
                    'overlap': float(overlap_score),
                    'detection': det
                })

        if not sam_track_frames:
            print(f"    SAM2 car {car_id_str}: no mask frames with valid bounding boxes")
            continue

        if not overlap_candidates:
            print(f"    SAM2 car {car_id_str}: no overlapping detections found")
            continue

        unassigned_frames: Set[int] = {entry['det_frame'] for entry in sam_track_frames}
        assigned_records: Dict[int, Dict[str, object]] = {}
        used_tracking_ids: Set[str] = set()
        used_tracking_norms: Set[int] = set()

        def build_segments() -> List[Dict[str, object]]:
            segments: List[Dict[str, object]] = []

            for candidate in overlap_candidates.values():
                records = [
                    rec for rec in candidate['records']
                    if rec['frame'] in unassigned_frames and rec['overlap'] >= relaxed_iou
                ]
                if not records:
                    continue

                records.sort(key=lambda r: r['frame'])
                current_run: List[Dict[str, object]] = []

                def finalize_run(run: List[Dict[str, object]]) -> None:
                    if not run:
                        return
                    if any(rec['overlap'] < min_iou for rec in run):
                        return
                    segments.append({
                        'candidate': candidate,
                        'records': list(run),
                        'length': len(run),
                        'avg_overlap': float(sum(rec['overlap'] for rec in run) / len(run)),
                        'avg_iou': float(sum(rec['iou'] for rec in run) / len(run))
                    })

                for rec in records:
                    if not current_run:
                        current_run = [rec]
                        continue
                    prev_frame = current_run[-1]['frame']
                    if rec['frame'] == prev_frame + 1:
                        current_run.append(rec)
                    else:
                        finalize_run(current_run)
                        current_run = [rec]

                finalize_run(current_run)

            return segments

        greedy_segments = 0
        while True:
            segments = [seg for seg in build_segments() if seg['length'] >= min_segment_length]
            if not segments:
                break

            segments.sort(key=lambda seg: (seg['length'], seg['avg_overlap'], seg['avg_iou']), reverse=True)
            best = segments[0]
            greedy_segments += 1
            candidate = best['candidate']
            source_label = str(candidate['raw'])
            source_norm = candidate['norm']
            if source_norm is not None:
                used_tracking_norms.add(source_norm)
            used_tracking_ids.add(source_label)

            print(
                f"    SAM2 car {car_id_str}: segment {greedy_segments} from track {source_label} "
                f"({best['length']} frames, avg overlap {best['avg_overlap']:.2f})"
            )

            for rec in best['records']:
                frame_idx = rec['frame']
                if frame_idx not in unassigned_frames:
                    continue
                assigned_records[frame_idx] = {
                    'detection': deepcopy(rec['detection']),
                    'sam_frame': rec['sam_frame'],
                    'overlap': rec['overlap'],
                    'iou': rec['iou'],
                    'contain': rec['contain'],
                    'source_norm': source_norm,
                    'source_raw': candidate['raw']
                }
                unassigned_frames.discard(frame_idx)
                matched_sam_frames.add(rec['sam_frame'])

        if not assigned_records:
            print(f"    SAM2 car {car_id_str}: no segments met minimum length of {min_segment_length} frames")
            continue

        try:
            new_tracking_id = int(car_id_str)
        except ValueError:
            new_tracking_id = 1000 + car_idx

        # Record mapping for later logging and optional dim override
        sam2_to_newid[car_id_str] = new_tracking_id

        assigned_frames_sorted = sorted(assigned_records.keys())
        total_frames = len(assigned_frames_sorted)
        total_available = len(sam2_lookup.frame_arrays.get(car_id_str, []))
        gap_frames = len(unassigned_frames)

        if used_tracking_norms:
            merge_mapping[new_tracking_id] = [new_tracking_id] + sorted(used_tracking_norms)
        else:
            merge_mapping[new_tracking_id] = [new_tracking_id]

        used_list = sorted(used_tracking_ids)
        print(
            f"    SAM2 car {car_id_str}: matched {total_frames}/{total_available} frames, "
            f"using tracks {used_list}"
        )
        # Explicitly log the remapping for transparency
        print(f"      Mapping: SAM2 car_id {car_id_str} -> tracking_id {new_tracking_id}")
        if used_list:
            print(f"      Sources {used_list} -> tracking_id {new_tracking_id}")
        if gap_frames:
            print(f"      Remaining unmatched frames for SAM2 car {car_id_str}: {gap_frames}")

        for age_idx, frame_idx in enumerate(assigned_frames_sorted, start=1):
            record = assigned_records[frame_idx]
            det_copy = record['detection']
            det_copy['tracking_id'] = new_tracking_id
            det_copy['age'] = age_idx
            det_copy['active'] = total_frames
            det_copy['sam2_iou'] = float(record['iou'])
            det_copy['sam2_overlap'] = float(record['overlap'])
            det_copy['sam2_contain'] = float(record['contain'])
            det_copy['sam2_car_id'] = car_id_str
            det_copy['sam2_frame'] = int(record['sam_frame'])
            det_copy['original_tracking_id'] = record['source_raw']

            frame_key = str(frame_idx)
            filtered_detections.setdefault(frame_key, []).append(det_copy)

    if not filtered_detections:
        print("  No SAM2-aligned detections generated, Step 0 will be skipped")
        return trajectory_json_path, detections_json_path, merge_mapping, sam2_lookup

    for frame_key in filtered_detections:
        filtered_detections[frame_key].sort(key=lambda det: det.get('tracking_id', 0))

    # Optional: Override bbox dimensions using scenario specs (Step 0)
    if override_dims_step0:
        specs_path = vehicle_specs_json or os.path.join(savepath, f"{video_id}_vehicle_specs.json")
        try:
            with open(specs_path, 'r', encoding='utf-8') as fh:
                specs = json.load(fh)
            left_box = (specs.get('left') or {}).get('bounding_box')
            opp_box = (specs.get('opponent') or {}).get('bounding_box')
            if not left_box or not opp_box:
                print(f"  Warning: Missing bounding_box in specs JSON; skip Step0 dim override ({specs_path})")
            else:
                def to_hwl(box):
                    L = float(box['length_m'])
                    W = float(box['width_m'])
                    H = float(box['height_m'])
                    return [H, W, L]
                left_hwl = to_hwl(left_box)
                opp_hwl = to_hwl(opp_box)

                # Determine roles from SAM2 IDs: smaller -> left, larger -> opponent
                numeric_ids = []
                for cid in sam2_to_newid.keys():
                    try:
                        numeric_ids.append((int(cid), cid))
                    except Exception:
                        pass
                if len(numeric_ids) >= 2:
                    numeric_ids.sort(key=lambda x: x[0])
                    left_cid = numeric_ids[0][1]
                    opp_cid = numeric_ids[-1][1]
                    print(f"  SAM2 role mapping: left={left_cid}, opponent={opp_cid}")
                    # Apply override
                    changed = 0
                    for frame_key, dets in filtered_detections.items():
                        for det in dets:
                            cid = str(det.get('sam2_car_id'))
                            if cid == str(left_cid):
                                det['dim'] = left_hwl
                                changed += 1
                            elif cid == str(opp_cid):
                                det['dim'] = opp_hwl
                                changed += 1
                    print(f"  Step0 dim override applied to {changed} detections (specs: {specs_path})")
                    # Report tracking IDs per role
                    print(f"  Role->tracking: left->ID{sam2_to_newid.get(str(left_cid))}, opponent->ID{sam2_to_newid.get(str(opp_cid))}")
                else:
                    print("  Warning: Fewer than two SAM2 car_ids; cannot assign roles; skip dim override")
        except FileNotFoundError:
            print(f"  Warning: Specs file not found: {specs_path}; skip Step0 dim override")
        except Exception as exc:
            print(f"  Warning: Failed applying Step0 dim override: {exc}")

    filtered_detections = dict(sorted(filtered_detections.items(), key=lambda item: int(item[0])))

    filtered_detection_path = os.path.join(savepath, f"step0_matched_{video_id}.mp4_results.json")
    with open(filtered_detection_path, 'w') as f:
        json.dump(filtered_detections, f, indent=2)

    # (Entropy-based metric removed per request)

    # ----------------------------------------------------------------------
    # Instance dynamics metric: HHI or Entropy (toggle), per car and video-level
    # ----------------------------------------------------------------------
    def stability_metric_hhi_runs(id_seq: List[Optional[str]],
                       count_none: bool = True,
                       exclude_none_runs: bool = True) -> Dict[str, float]:
        n = len(id_seq)
        # HHI / Simpson only. If count_none=False, exclude None from categories and denominator
        base = id_seq if count_none else [x for x in id_seq if x is not None]
        m = len(base)
        if n <= 0 or m == 0:
            return {"n": n, "hhi": 1.0, "metric": 1.0}
        cnt = Counter(base)
        hhi = sum((c / m) ** 2 for c in cnt.values())
        metric = max(0.0, min(1.0, hhi))
        return {"n": n, "hhi": float(hhi), "metric": float(metric)}

    def stability_metric_entropy(id_seq: List[Optional[str]], count_none: bool = True) -> Dict[str, float]:
        n = len(id_seq)
        base = id_seq if count_none else [x for x in id_seq if x is not None]
        m = len(base)
        if n <= 0 or m == 0:
            return {"n": n, "K": 0, "H": 0.0, "metric": 0.0}
        cnt = Counter(base)
        K = len(cnt)
        if K <= 1:
            return {"n": n, "K": K, "H": 0.0, "metric": 0.0}
        H = 0.0
        for c in cnt.values():
            p = c / m
            if p > 0:
                H -= p * math.log(p)
        # Use raw entropy as the metric (no normalization)
        metric = float(H)
        return {"n": n, "K": K, "H": metric, "metric": metric}

    # Toggles (edit in code):
    ALLOW_ONE_ID_MERGE = False       # Merge two longest non-None runs (allow one ID change)
    EXCLUDE_NONE_FROM_METRIC = False # If True, None excluded from categories and denominator
    USE_ENTROPY_METRIC = True       # If True, use entropy-based metric; else use HHI
    


    def _build_runs(id_seq: List[Optional[str]]):
        runs = []
        if not id_seq:
            return runs
        start = 0
        current = id_seq[0]
        for idx in range(1, len(id_seq)):
            if id_seq[idx] != current:
                runs.append({"start": start, "end": idx, "len": idx - start, "label": current})
                start = idx
                current = id_seq[idx]
        runs.append({"start": start, "end": len(id_seq), "len": len(id_seq) - start, "label": current})
        return runs

    def _merge_two_longest_runs(id_seq: List[Optional[str]]):
        runs = _build_runs(id_seq)
        # consider only non-None runs
        candidates = [r for r in runs if r["label"] is not None]
        if len(candidates) < 2:
            return id_seq, None
        # sort by length desc
        candidates.sort(key=lambda r: r["len"], reverse=True)
        first = None
        second = None
        # pick two runs with different labels if possible
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if str(candidates[i]["label"]) != str(candidates[j]["label"]):
                    first, second = candidates[i], candidates[j]
                    break
            if first is not None:
                break
        # fallback: if all labels identical, nothing to merge that changes metric
        if first is None or second is None:
            return id_seq, None
        new_seq = list(id_seq)
        # merge second run into first run's label
        target_label = str(first["label"]) if first["label"] is not None else first["label"]
        for k in range(second["start"], second["end"]):
            new_seq[k] = target_label
        info = {
            "merged": True,
            "first_run": {k: first[k] for k in ("start", "end", "len")},
            "second_run": {k: second[k] for k in ("start", "end", "len")},
            "first_label": str(first["label"]) if first["label"] is not None else None,
            "second_label": str(second["label"]) if second["label"] is not None else None,
            "target_label": target_label,
        }
        return new_seq, info

    dynamics_cars = []
    total_n = 0
    sum_metric_weighted = 0.0
    # Aggregation for bbox size stability (weighted by available samples)
    sum_size_stability_weighted = 0.0
    total_size_weight = 0

    if sam2_lookup is not None and getattr(sam2_lookup, 'car_ids', None):
        for cid in sam2_lookup.car_ids:
            # Build full-frame sequence for this SAM2 car across its mask frames
            frames = [int(d) for d in sam2_lookup.frame_arrays.get(cid, [])]
            # Convert to detection frame index using the same offset as lookup used
            # We can reuse iterator which yields detection_frame in sorted order
            det_frames = []
            for det_frame, sam_frame, _ in sam2_lookup.iter_frame_mask_pairs(cid):
                det_frames.append(int(det_frame))
            det_frames.sort()

            seq = []
            for df in det_frames:
                dets = filtered_detections.get(str(df), [])
                id_val = None
                for det in dets:
                    if str(det.get('sam2_car_id')) == str(cid):
                        id_val = det.get('original_tracking_id')
                        if id_val is not None:
                            id_val = str(id_val)
                        break
                seq.append(id_val)

            # Skip cars with no observed IDs at all
            if not any(x is not None for x in seq):
                continue

            # raw metrics (before optional merge)
            dyn_entropy_raw = stability_metric_entropy(seq, count_none=(not EXCLUDE_NONE_FROM_METRIC))
            dyn_hhi_raw = stability_metric_hhi_runs(
                seq, count_none=(not EXCLUDE_NONE_FROM_METRIC), exclude_none_runs=True)
            seq_used = seq
            merge_info = None
            if ALLOW_ONE_ID_MERGE:
                seq_merged, merge_info = _merge_two_longest_runs(seq)
                seq_used = seq_merged
            # metrics after optional merge
            dyn_entropy = stability_metric_entropy(seq_used, count_none=(not EXCLUDE_NONE_FROM_METRIC))
            dyn_hhi = stability_metric_hhi_runs(
                seq_used, count_none=(not EXCLUDE_NONE_FROM_METRIC), exclude_none_runs=True)
            # selected metric for aggregation
            dyn = dyn_entropy if USE_ENTROPY_METRIC else dyn_hhi
            dyn_raw = dyn_entropy_raw if USE_ENTROPY_METRIC else dyn_hhi_raw

            # Derive switch events (ignore None->None)
            switch_events = []
            switch_count = 0
            def same(a,b):
                return (a is None and b is None) or (a == b)
            for idx in range(len(seq) - 1):
                a, b = seq[idx], seq[idx + 1]
                if not same(a, b):
                    switch_count += 1
                    switch_events.append({
                        "pos": idx + 1,  # between idx and idx+1
                        "from": a,
                        "to": b
                    })

            # -----------------------------
            # BBox size stability metrics
            # -----------------------------
            dims_list = []  # list of [H,W,L]
            widths = []
            heights = []
            areas = []
            for df in det_frames:
                dets_in = filtered_detections.get(str(df), [])
                det_sel = None
                for det in dets_in:
                    if str(det.get('sam2_car_id')) == str(cid):
                        det_sel = det
                        break
                if det_sel is None:
                    continue
                # 3D dims if present
                d = det_sel.get('dim')
                if isinstance(d, (list, tuple)) and len(d) >= 3:
                    try:
                        H, W, L = float(d[0]), float(d[1]), float(d[2])
                        if H > 0 and W > 0 and L > 0:
                            dims_list.append([H, W, L])
                    except Exception:
                        pass
                # 2D bbox width/height/area
                bb = det_sel.get('bbox')
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    try:
                        x1, y1, x2, y2 = map(float, bb)
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        if w > 0 and h > 0:
                            widths.append(w)
                            heights.append(h)
                            areas.append(w * h)
                    except Exception:
                        pass

            def _cv(vals):
                if len(vals) < 2:
                    return None
                arr = np.asarray(vals, dtype=float)
                m = float(np.mean(arr))
                s = float(np.std(arr))
                if m <= 1e-9:
                    return 0.0 if s <= 1e-9 else 1.0
                return max(0.0, s / m)

            dims_stability = None
            dims_samples = len(dims_list)
            if dims_samples >= 2:
                arr = np.asarray(dims_list, dtype=float)
                cv_h = _cv(arr[:, 0])
                cv_w = _cv(arr[:, 1])
                cv_l = _cv(arr[:, 2])
                cvs = [c for c in (cv_h, cv_w, cv_l) if c is not None]
                if cvs:
                    cv_mean = float(np.mean(cvs))
                    dims_stability = 1.0 / (1.0 + cv_mean)

            wh_stability = None
            wh_samples = min(len(widths), len(heights))
            if wh_samples >= 2:
                cv_w2 = _cv(widths)
                cv_h2 = _cv(heights)
                cvs = [c for c in (cv_w2, cv_h2) if c is not None]
                if cvs:
                    cv_mean = float(np.mean(cvs))
                    wh_stability = 1.0 / (1.0 + cv_mean)

            area_stability = None
            area_samples = len(areas)
            if area_samples >= 2:
                cv_a = _cv(areas)
                if cv_a is not None:
                    area_stability = 1.0 / (1.0 + cv_a)

            # Preferred size stability selection order: dims > wh > area
            size_stability = None
            size_samples = 0
            if dims_stability is not None:
                size_stability = dims_stability
                size_samples = dims_samples
            elif wh_stability is not None:
                size_stability = wh_stability
                size_samples = wh_samples
            elif area_stability is not None:
                size_stability = area_stability
                size_samples = area_samples

            entry = {
                "sam2_car_id": str(cid),
                "n": int(dyn.get("n", len(seq_used))),
                "sequence": seq,
                "switch_count": switch_count,
                "switch_events": switch_events,
                "metric": float(dyn.get("metric", 0.0)),
                # Always include both entropy and HHI variants
                "entropy": float(dyn_entropy.get("H", 0.0)),
                "K": int(dyn_entropy.get("K", 0)),
                "hhi": float(dyn_hhi.get("hhi", 0.0)),
                "metric_entropy": float(dyn_entropy.get("metric", 0.0)),
                "metric_hhi": float(dyn_hhi.get("metric", 0.0)),
            }
            # attach size stability
            entry["size_stability"] = float(size_stability) if size_stability is not None else None
            entry["dims_stability"] = float(dims_stability) if dims_stability is not None else None
            entry["wh_stability"] = float(wh_stability) if wh_stability is not None else None
            entry["area_stability"] = float(area_stability) if area_stability is not None else None
            entry["size_samples"] = int(size_samples)
            entry["dims_samples"] = int(dims_samples)
            entry["wh_samples"] = int(wh_samples)
            entry["area_samples"] = int(area_samples)
            # record optional merge info and raw metric for transparency
            entry["metric_raw"] = float(dyn_raw["metric"]) if isinstance(dyn_raw.get("metric"), (int, float)) else dyn_raw.get("metric")
            if merge_info is not None:
                entry["merge_applied"] = True
                entry["merged"] = merge_info
            else:
                entry["merge_applied"] = False
            dynamics_cars.append(entry)
            total_n += entry["n"]
            sum_metric_weighted += entry["n"] * entry["metric"]
            if size_stability is not None and size_samples > 0:
                sum_size_stability_weighted += size_stability * size_samples
                total_size_weight += size_samples

    if dynamics_cars:
        video_metric_weighted = float(sum_metric_weighted / total_n) if total_n > 0 else 1.0
        video_metric_mean = float(np.mean([c["metric"] for c in dynamics_cars]))
        video_metric_min = float(min([c["metric"] for c in dynamics_cars]))

        dynamics_summary = {
            "video_id": video_id,
            "num_cars": len(dynamics_cars),
            "defaults": {
                "count_none": (not EXCLUDE_NONE_FROM_METRIC),
                "use_entropy": USE_ENTROPY_METRIC,
                "allow_one_merge": ALLOW_ONE_ID_MERGE
            },
            "video_metric_weighted": video_metric_weighted,
            "video_metric_mean": video_metric_mean,
            "video_metric_min": video_metric_min,
            "video_size_stability_weighted": (
                float(sum_size_stability_weighted / total_size_weight)
                if total_size_weight > 0 else None
            ),
            "cars": dynamics_cars,
        }

        dynamics_json_path = os.path.join(savepath, f"step0_instance_dynamics_{video_id}.json")
        try:
            with open(dynamics_json_path, 'w') as f:
                json.dump(dynamics_summary, f, indent=2)
            label = "Entropy" if USE_ENTROPY_METRIC else "HHI"
            print(f"  Saved instance dynamics metric ({label}) to {dynamics_json_path}")
            print(f"  Video-level metric (weighted by n): {video_metric_weighted:.4f}")
            if total_size_weight > 0:
                print(f"  Video size stability (weighted): {sum_size_stability_weighted/total_size_weight:.4f}")
        except Exception as exc:
            print(f"  Warning: failed to write instance dynamics summary: {exc}")
        # raise NotImplementedError()
    try:
        with open(trajectory_json_path, 'r') as f:
            trajectory_data = json.load(f)
    except Exception as exc:
        print(f"  Failed to load trajectory JSON ({trajectory_json_path}): {exc}")
        print(f"  Using filtered detections only")
        return trajectory_json_path, filtered_detection_path, merge_mapping, sam2_lookup

    filtered_trajectory_data = {}
    for sam_frame in sorted(matched_sam_frames):
        key = f"{int(sam_frame):06d}"
        if key in trajectory_data:
            filtered_trajectory_data[key] = trajectory_data[key]

    if not filtered_trajectory_data:
        print("  No overlapping trajectory frames found, keeping original trajectory file")
        return trajectory_json_path, filtered_detection_path, merge_mapping, sam2_lookup

    filtered_trajectory_path = os.path.join(savepath, f"step0_matched_{video_id}_trajectories.json")
    with open(filtered_trajectory_path, 'w') as f:
        json.dump(filtered_trajectory_data, f, indent=2)

    print(f"  Saved SAM2-aligned detections to {filtered_detection_path}")
    print(f"  Saved SAM2-filtered trajectories to {filtered_trajectory_path}")


    return filtered_trajectory_path, filtered_detection_path, merge_mapping, sam2_lookup


# ============================================================================
# STEP 1: Read and Process JSON
# ============================================================================

def read_and_process_trajectory(json_path: str) -> List[np.ndarray]:
    """
    Read JSON trajectory file and convert to list of inverse transformation matrices

    Input: JSON file path
    Output: List of 4x4 inverse transformation matrices to transform from each frame to first frame

    Args:
        json_path: Path to trajectory JSON file

    Returns:
        List of 4x4 numpy arrays representing inverse transformations
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    # Sort frames by key to ensure chronological order
    frame_keys = sorted(raw_data.keys())

    # Convert to list of numpy matrices and compute inverses
    transform_matrices = []

    for frame_key in frame_keys:
        # Get transformation matrix from world to current frame
        T_current_to_world = np.array(raw_data[frame_key])

        # # Compute inverse to get transformation from current frame to world (first frame)
        # T_current_to_world = np.linalg.inv(T_world_to_current)

        transform_matrices.append(T_current_to_world)

    print(f"Processed {len(transform_matrices)} transformation matrices from {json_path}")
    return transform_matrices


# ============================================================================
# STEP 1.5: Project 3D Trajectories to Image Plane
# ============================================================================

def project_3d_to_image(points_3d: np.ndarray,
                       camera_matrix: np.ndarray) -> np.ndarray:
    """
    Project 3D points to image coordinates

    Args:
        points_3d: Nx3 array of 3D points [x, y, z] in camera frame
        camera_matrix: 3x3 camera intrinsic matrix or 3x4 projection matrix

    Returns:
        Nx2 array of image coordinates [u, v]
    """
    if HAS_DDD_UTILS and camera_matrix.shape[1] == 4:
        # Use CenterTrack's projection method with 3x4 matrix
        return project_to_image(points_3d, camera_matrix)
    else:
        # Use 3x3 intrinsic matrix method
        if camera_matrix.shape[1] == 4:
            # Convert 3x4 to 3x3 by dropping last column
            camera_matrix = camera_matrix[:, :3]

        # Project to image plane
        image_coords_homo = (camera_matrix @ points_3d.T).T

        # Convert from homogeneous to 2D image coordinates
        # Add small epsilon to avoid division by zero
        z_coords = image_coords_homo[:, 2:3]
        z_coords = np.where(np.abs(z_coords) < 1e-6, 1e-6, z_coords)

        image_coords = image_coords_homo[:, :2] / z_coords

        return image_coords


def draw_projected_bbox_on_image(img: np.ndarray,
                                bbox_3d: np.ndarray,
                                camera_matrix: np.ndarray,
                                color: tuple = (0, 255, 0),
                                thickness: int = 2,
                                use_centertrack_draw: bool = True) -> np.ndarray:
    """
    Draw projected 3D bounding box on image

    Args:
        img: Input image
        bbox_3d: 8x3 array of 3D bbox corners
        camera_matrix: 3x3 camera intrinsic matrix or 3x4 projection matrix
        color: BGR color for drawing
        thickness: Line thickness
        use_centertrack_draw: Whether to use CenterTrack's drawing method

    Returns:
        Image with projected bbox drawn
    """
    # Project 3D bbox corners to image
    bbox_2d = project_3d_to_image(bbox_3d, camera_matrix)
    bbox_2d = bbox_2d.astype(int)

    # Check if points are within image bounds
    h, w = img.shape[:2]
    valid_points = (bbox_2d[:, 0] >= 0) & (bbox_2d[:, 0] < w) & \
                   (bbox_2d[:, 1] >= 0) & (bbox_2d[:, 1] < h)

    if np.sum(valid_points) < 4:  # Need at least 4 visible points
        return img

    if HAS_DDD_UTILS and use_centertrack_draw:
        # Use CenterTrack's drawing method
        try:
            img = draw_box_3d(img, bbox_2d, c=color, same_color=True)
        except:
            # Fallback to simple drawing if CenterTrack method fails
            use_centertrack_draw = False

    if not use_centertrack_draw:
        # Simple line drawing method
        # Draw bottom face (first 4 corners)
        for i in range(4):
            j = (i + 1) % 4
            if valid_points[i] and valid_points[j]:
                cv2.line(img, tuple(bbox_2d[i]), tuple(bbox_2d[j]), color, thickness)

        # Draw top face (last 4 corners)
        for i in range(4, 8):
            j = 4 + ((i - 4 + 1) % 4)
            if valid_points[i] and valid_points[j]:
                cv2.line(img, tuple(bbox_2d[i]), tuple(bbox_2d[j]), color, thickness)

        # Draw vertical edges
        for i in range(4):
            if valid_points[i] and valid_points[i + 4]:
                cv2.line(img, tuple(bbox_2d[i]), tuple(bbox_2d[i + 4]), color, thickness)

    return img


def create_default_camera_matrix(img_width: int = 1920,
                                img_height: int = 1080,
                                fov_degrees: float = 90.0) -> np.ndarray:
    """
    Create a default camera intrinsic matrix

    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        fov_degrees: Field of view in degrees

    Returns:
        3x3 camera intrinsic matrix
    """
    # Calculate focal length from FOV
    fov_rad = np.deg2rad(fov_degrees)
    fx = img_width / (2.0 * np.tan(fov_rad / 2.0))
    fx = 960
    fy = 960
    fy = fx  # Assume square pixels

    # Principal point at image center
    cx = img_width / 2.0
    cy = img_height / 2.0

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    print(fx, fy, cx, cy)

    return camera_matrix


# ============================================================================
# Depth Image Processing Functions
# ============================================================================

def load_depth_image(depth_path: str, frame_idx: int) -> np.ndarray:
    """
    Load depth image for a specific frame

    Args:
        depth_path: Path to depth images directory
        frame_idx: Frame index (0-based)

    Returns:
        Depth image as numpy array
    """
    depth_file = os.path.join(depth_path, f"{frame_idx:06d}.npy")
    if not os.path.exists(depth_file):
        raise FileNotFoundError(f"Depth file not found: {depth_file}")

    depth_img = np.load(depth_file)
    return depth_img


def extract_depth_from_bbox(bbox_2d: np.ndarray, depth_img: np.ndarray) -> dict:
    """
    Extract depth statistics from projected 2D bounding box region

    Args:
        bbox_2d: Nx2 array of 2D bbox corners in image coordinates
        depth_img: Depth image (H x W)

    Returns:
        Dictionary with depth statistics
    """
    h, w = depth_img.shape

    # Get bounding rectangle of projected bbox
    valid_points = (bbox_2d[:, 0] >= 0) & (bbox_2d[:, 0] < w) & \
                   (bbox_2d[:, 1] >= 0) & (bbox_2d[:, 1] < h)

    if not np.any(valid_points):
        return {
            'mean_depth': 0.0,
            'min_depth': 0.0,
            'max_depth': 0.0,
            'std_depth': 0.0,
            'valid_pixels': 0
        }

    valid_bbox = bbox_2d[valid_points]
    x_min = max(0, int(valid_bbox[:, 0].min()))
    x_max = min(w, int(valid_bbox[:, 0].max()) + 1)
    y_min = max(0, int(valid_bbox[:, 1].min()))
    y_max = min(h, int(valid_bbox[:, 1].max()) + 1)

    # Extract depth region
    depth_region = depth_img[y_min:y_max, x_min:x_max]

    # Filter out invalid depth values (assuming 0 or very small values are invalid)
    valid_depth = depth_region[depth_region > 1.0]

    if len(valid_depth) == 0:
        return {
            'mean_depth': 0.0,
            'min_depth': 0.0,
            'max_depth': 0.0,
            'std_depth': 0.0,
            'valid_pixels': 0
        }

    return {
        'mean_depth': float(valid_depth.mean()),
        'min_depth': float(valid_depth.min()),
        'max_depth': float(valid_depth.max()),
        'std_depth': float(valid_depth.std()),
        'valid_pixels': len(valid_depth)
    }


def extract_depth_from_mask(depth_img: np.ndarray,
                            mask: np.ndarray,
                            lower_percentile: float = 40.0,
                            upper_percentile: float = 60.0,
                            min_valid_depth: float = 1.0) -> dict:
    """Extract depth statistics from a SAM2 mask region."""
    if mask is None:
        return {
            'mean_depth': 0.0,
            'min_depth': 0.0,
            'max_depth': 0.0,
            'std_depth': 0.0,
            'valid_pixels': 0
        }

    depth_values = depth_img[mask]
    if depth_values.size == 0:
        return {
            'mean_depth': 0.0,
            'min_depth': 0.0,
            'max_depth': 0.0,
            'std_depth': 0.0,
            'valid_pixels': 0
        }

    valid_depths = depth_values[np.isfinite(depth_values) & (depth_values > min_valid_depth)]
    if valid_depths.size == 0:
        return {
            'mean_depth': 0.0,
            'min_depth': 0.0,
            'max_depth': 0.0,
            'std_depth': 0.0,
            'valid_pixels': 0
        }

    low = np.percentile(valid_depths, lower_percentile)
    high = np.percentile(valid_depths, upper_percentile)

    central_band = valid_depths[(valid_depths >= low) & (valid_depths <= high)]
    if central_band.size == 0:
        central_band = valid_depths

    stats = {
        'mean_depth': float(np.mean(central_band)),
        'min_depth': float(np.min(central_band)),
        'max_depth': float(np.max(central_band)),
        'std_depth': float(np.std(central_band)),
        'valid_pixels': int(central_band.size)
    }

    return stats


def project_3d_bboxes_to_video(transform_matrices: List[np.ndarray],
                               original_video_path: str,
                               detections_json_path: str,
                               output_video_path: str = 'projected_3d_bboxes.mp4',
                               camera_matrix: np.ndarray = None,
                               depth_images_path: str = None,
                               video_id: str = None,
                               trajectory_file: str = None) -> None:
    """
    Project 3D bounding boxes from detections onto original video frames

    Args:
        transform_matrices: List of transformation matrices (current to world)
        original_video_path: Path to original video
        detections_json_path: Path to detection results JSON
        output_video_path: Path for output video
        camera_matrix: Camera intrinsic matrix (will create default if None)
        depth_images_path: Path to depth images directory (optional)
    """
    print(f"Projecting 3D bounding boxes onto video: {original_video_path}")

    # Load detection results
    from traj_recon import read_detection_results, extract_detection_bboxes
    detections = read_detection_results(detections_json_path)
    detection_bboxes = extract_detection_bboxes(detections)

    print(f"Loaded detections for {len(detection_bboxes)} frames")

    # Open original video
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {original_video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Create default camera matrix if not provided
    if camera_matrix is None:
        raise NotImplementedError("Camera matrix must be provided for depth correction")
        camera_matrix = create_default_camera_matrix(width, height)
        print("Using default camera matrix:")
        print(camera_matrix)

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get frame keys for mapping
    import json
    # if trajectory_file is None:
    #     trajectory_file = f'/root/CenterTrack/{video_id}_trajectories.json'
    with open(trajectory_file, 'r') as f:
        trajectory_data = json.load(f)
    frame_keys = sorted(trajectory_data.keys())

    # Process each frame
    frame_count = 0
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 200, 0), (200, 0, 255)]

    # Load depth image for current frame if depth path is provided
    depth_img = None
    if depth_images_path and os.path.exists(depth_images_path):
        print(f"Using depth images from: {depth_images_path}")

    print(f"Processing {min(total_frames, len(transform_matrices))} frames...")

    with tqdm(total=min(total_frames, len(transform_matrices)), desc="Projecting 3D bboxes") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= len(transform_matrices) or frame_count >= len(frame_keys):
                break

            frame_key = frame_keys[frame_count]

            # Load depth image for current frame
            if depth_images_path:
                try:
                    depth_img = load_depth_image(depth_images_path, frame_count)
                except FileNotFoundError:
                    depth_img = None

            # Draw detected objects' 3D bboxes in camera frame (original positions)
            if frame_key in detection_bboxes:
                for bbox_idx, bbox_3d in enumerate(detection_bboxes[frame_key]):
                    # Use different colors for different objects
                    color = colors[bbox_idx % len(colors)]

                    # Project 3D bbox to get 2D coordinates
                    bbox_2d = project_3d_to_image(bbox_3d, camera_matrix)

                    # Extract depth information if depth image is available
                    depth_stats = None
                    if depth_img is not None:
                        depth_stats = extract_depth_from_bbox(bbox_2d, depth_img)

                    # Draw 3D bbox
                    frame = draw_projected_bbox_on_image(frame, bbox_3d, camera_matrix,
                                                       color=color, thickness=2)

                    # Add object ID and depth info if available
                    if frame_key in detections and bbox_idx < len(detections[frame_key]):
                        det = detections[frame_key][bbox_idx]
                        tracking_id = det.get('tracking_id', bbox_idx)

                        # Get center of bbox for text placement
                        bbox_center_3d = np.mean(bbox_3d[:4], axis=0)  # Bottom face center
                        bbox_center_2d = project_3d_to_image(bbox_center_3d.reshape(1, -1), camera_matrix)[0]

                        # Check if center is in image bounds
                        if 0 <= bbox_center_2d[0] < width and 0 <= bbox_center_2d[1] < height:
                            text_x, text_y = int(bbox_center_2d[0]), int(bbox_center_2d[1])

                            # Draw tracking ID
                            cv2.putText(frame, f"ID{tracking_id}",
                                      (text_x, text_y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                            # Print depth information if available
                            if depth_stats and depth_stats['valid_pixels'] > 0:
                                print(f"Frame {frame_count:06d}, ID{tracking_id}: "
                                      f"Mean depth: {depth_stats['mean_depth']:.2f}m, "
                                      f"Range: {depth_stats['min_depth']:.2f}-{depth_stats['max_depth']:.2f}m, "
                                      f"Valid pixels: {depth_stats['valid_pixels']}")
                            elif depth_stats:
                                print(f"Frame {frame_count:06d}, ID{tracking_id}: No valid depth data")

            # Add frame information
            info_text = f"Frame: {frame_count:06d}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"Detected objects: {len(detection_bboxes.get(frame_key, []))}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "3D bounding boxes projected to image plane",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Write frame
            out.write(frame)

            frame_count += 1
            pbar.update(1)

    # Cleanup
    cap.release()
    out.release()

    print(f"✓ Projected 3D bboxes video saved to: {output_video_path}")


# ============================================================================
# STEP 2: Apply Transformations
# ============================================================================

def apply_transformations(transform_matrices: List[np.ndarray],
                         objects_at_origin: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply transformation matrices to objects at origin to get their positions in first frame reference

    Input:
        - List of transformation matrices (one per frame)
        - List of objects at origin (bbox corners for ego vehicle at origin)

    Output: List of transformed objects in first frame reference coordinate system

    Args:
        transform_matrices: List of 4x4 transformation matrices
        objects_at_origin: List of objects to transform (each frame has same object at origin)

    Returns:
        List of transformed objects (Nx3 arrays) in world coordinates
    """
    transformed_objects = []

    # If only one object provided, use it for all frames
    if len(objects_at_origin) == 1:
        object_at_origin = objects_at_origin[0]
        objects_at_origin = [object_at_origin] * len(transform_matrices)

    # Apply each transformation
    for i, (T, obj) in enumerate(zip(transform_matrices, objects_at_origin)):
        # Ensure object has homogeneous coordinates
        if obj.shape[1] == 3:
            # Add homogeneous coordinate
            obj_homo = np.hstack([obj, np.ones((obj.shape[0], 1))])
        else:
            obj_homo = obj

        # Apply transformation: T * points^T
        transformed_homo = (T @ obj_homo.T).T

        # Convert back to 3D coordinates
        transformed_3d = transformed_homo[:, :3] / transformed_homo[:, 3:4]

        transformed_objects.append(transformed_3d)

    print(f"Transformed {len(transformed_objects)} objects to world coordinates")
    return transformed_objects


# ============================================================================
# STEP 3: Visualize Transformed Objects
# ============================================================================

def visualize_transformed_objects(transformed_objects: List[np.ndarray],
                                 output_video: str = 'trajectory_visualization.mp4',
                                 fps: int = 10,
                                 img_size: Tuple[int, int] = (1920, 1080),
                                 scale: float = 10.0):
    """
    Visualize list of transformed objects (bboxes in world frame)

    Input: List of transformed objects from step 2
    Output: Video file showing the trajectory

    Args:
        transformed_objects: List of Nx3 arrays representing transformed bbox corners
        output_video: Output video file path
        fps: Frames per second
        img_size: Image size (width, height)
        scale: Pixels per meter for visualization
    """
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, img_size)

    # Visualization center - move ego vehicle down to see more ahead
    center = (img_size[0] // 2, int(img_size[1] * 0.9))  # 70% down from top

    # Track trajectory for trail
    trajectory_positions = []

    print(f"Creating visualization with {len(transformed_objects)} frames...")

    for frame_idx in tqdm(range(len(transformed_objects)), desc="Rendering"):
        # Create blank image
        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

        # Draw grid
        grid_spacing = 50
        grid_color = (40, 40, 40)
        for x in range(0, img_size[0], grid_spacing):
            cv2.line(img, (x, 0), (x, img_size[1]), grid_color, 1)
        for y in range(0, img_size[1], grid_spacing):
            cv2.line(img, (0, y), (img_size[0], y), grid_color, 1)

        # Draw coordinate axes
        cv2.arrowedLine(img, center, (center[0] + 80, center[1]), (0, 0, 255), 3)  # X-axis red
        cv2.arrowedLine(img, center, (center[0], center[1] - 80), (0, 255, 0), 3)  # Z-axis green
        cv2.putText(img, "X", (center[0] + 90, center[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(img, "Z", (center[0] + 5, center[1] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Get current bbox
        bbox = transformed_objects[frame_idx]

        # Calculate center position of bbox (mean of bottom 4 corners)
        bbox_center = np.mean(bbox[:4], axis=0)
        trajectory_positions.append(bbox_center)

        # Draw trajectory trail (show all history)
        if len(trajectory_positions) > 1:
            for i in range(1, len(trajectory_positions)):
                pt1 = trajectory_positions[i-1]
                pt2 = trajectory_positions[i]

                # Convert to pixel coordinates (bird's eye view: X-Z plane)
                px1 = int(pt1[0] * scale + center[0])
                py1 = int(-pt1[2] * scale + center[1])  # Negative Z for image coordinates
                px2 = int(pt2[0] * scale + center[0])
                py2 = int(-pt2[2] * scale + center[1])

                # Color gradient for trail - keep all history
                alpha = i / len(trajectory_positions)
                trail_color = (int(255 * (1 - alpha)), int(100), int(255 * alpha))
                cv2.line(img, (px1, py1), (px2, py2), trail_color, 3)

        # Convert bbox to pixel coordinates
        bbox_pixels = []
        for corner in bbox:
            px = int(corner[0] * scale + center[0])
            py = int(-corner[2] * scale + center[1])  # Negative Z for image coordinates
            bbox_pixels.append([px, py])
        bbox_pixels = np.array(bbox_pixels, np.int32)

        # Draw bounding box
        # Bottom face (first 4 corners)
        cv2.polylines(img, [bbox_pixels[:4].reshape((-1, 1, 2))], True, (0, 255, 0), 3)

        # Top face (last 4 corners)
        cv2.polylines(img, [bbox_pixels[4:].reshape((-1, 1, 2))], True, (0, 255, 0), 3)

        # Vertical edges
        for i in range(4):
            cv2.line(img, tuple(bbox_pixels[i]), tuple(bbox_pixels[i+4]), (0, 255, 0), 2)

        # Mark front of vehicle (midpoint between corners 1 and 2)
        front_center = ((bbox_pixels[1] + bbox_pixels[2]) // 2).astype(int)
        cv2.circle(img, tuple(front_center), 8, (0, 0, 255), -1)

        # Add text information
        current_pos = bbox_center
        info_texts = [
            f"Frame: {frame_idx:06d}",
            f"Position: X={current_pos[0]:.2f}m, Y={current_pos[1]:.2f}m, Z={current_pos[2]:.2f}m",
            f"Scale: 1 grid square = {grid_spacing/scale:.1f}m",
            "Red dot = Vehicle front"
        ]

        y_offset = 30
        for i, text in enumerate(info_texts):
            cv2.putText(img, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Write frame to video
        out.write(img)

    # Release video writer
    out.release()
    print(f"Video saved to {output_video}")

    # Save last frame as preview
    preview_path = output_video.replace('.mp4', '_preview.png')
    cv2.imwrite(preview_path, img)
    print(f"Preview image saved to {preview_path}")


# ============================================================================
# Helper function to create ego vehicle bbox at origin
# ============================================================================

def create_ego_bbox_at_origin(length: float = 4.5,
                              width: float = 2.0,
                              height: float = 1.5) -> np.ndarray:
    """
    Create ego vehicle bounding box at origin

    Args:
        length: Vehicle length in meters
        width: Vehicle width in meters
        height: Vehicle height in meters

    Returns:
        8x3 array of bbox corners
    """
    bbox = np.array([
        # Bottom 4 corners
        [-length/2, -width/2, 0],
        [length/2, -width/2, 0],
        [length/2, width/2, 0],
        [-length/2, width/2, 0],
        # Top 4 corners
        [-length/2, -width/2, height],
        [length/2, -width/2, height],
        [length/2, width/2, height],
        [-length/2, width/2, height]
    ])
    return bbox


# ============================================================================
# STEP 4: Read Detection Results
# ============================================================================

def read_detection_results(json_path: str) -> Dict:
    """
    Read detection results from JSON file

    Args:
        json_path: Path to detection results JSON

    Returns:
        Dictionary with frame_id -> list of detections
    """
    with open(json_path, 'r') as f:
        detections = json.load(f)

    print(f"Loaded detections for {len(detections)} frames")

    # Convert string keys to match trajectory format and filter class 1 only
    formatted_detections = {}
    for frame_id, dets in detections.items():
        # Convert frame number to 6-digit format to match trajectory
        formatted_key = f"{int(frame_id):06d}"

        # Filter only class 1 detections
        class1_dets = []
        for det in dets:
            if det.get('class') == 1:
                class1_dets.append(det)

        formatted_detections[formatted_key] = class1_dets

    print(f"Filtered to keep only class 1 detections")
    return formatted_detections


def extract_detection_bboxes(detections: Dict) -> Dict[str, List[np.ndarray]]:
    """
    Extract 3D bboxes from detection results using CenterTrack's method

    Args:
        detections: Detection dictionary

    Returns:
        Dictionary mapping frame_id to list of 3D bboxes
    """
    detection_bboxes = {}

    for frame_id, frame_dets in detections.items():
        frame_bboxes = []

        for det in frame_dets:
            if 'loc' in det and 'dim' in det and 'rot_y' in det and 'class' in det:
                # Filter: only keep class 1 detections
                if det['class'] != 1:
                    continue

                # Get 3D location and dimensions
                loc = det['loc']     # [x, y, z]
                dim = det['dim']     # [h, w, l]
                rot_y = det['rot_y']

                # ===== DEPTH CORRECTION POINT =====
                # TODO: This is where we can correct the depth using metric depth
                # loc[2] is the depth (z-coordinate) that can be adjusted
                # Example: loc[2] = corrected_depth_from_metric_depth_map
                # Note: Both compute_box_3d() and fallback use this loc value
                # ===================================

                if HAS_DDD_UTILS:
                    # Use CenterTrack's original method
                    bbox_3d = compute_box_3d(dim, loc, rot_y)
                else:
                    # Fallback to our implementation
                    h, w, l = dim
                    corners = np.array([
                        [-l/2, -w/2, 0], [l/2, -w/2, 0], [l/2, w/2, 0], [-l/2, w/2, 0],
                        [-l/2, -w/2, h], [l/2, -w/2, h], [l/2, w/2, h], [-l/2, w/2, h]
                    ])

                    R = np.array([
                        [np.cos(rot_y), 0, np.sin(rot_y)],
                        [0, 1, 0],
                        [-np.sin(rot_y), 0, np.cos(rot_y)]
                    ])

                    rotated_corners = (R @ corners.T).T
                    bbox_3d = rotated_corners + np.array(loc)

                frame_bboxes.append(bbox_3d)

        detection_bboxes[frame_id] = frame_bboxes

    return detection_bboxes


def extract_detection_bboxes_depth_corrected(detections: Dict,
                                            depth_images_path: str = None,
                                            camera_matrix: np.ndarray = None,
                                            sam2_lookup: Optional[SAM2MaskLookup] = None) -> Dict[str, List[np.ndarray]]:
    """
    Extract 3D bboxes from detection results with depth correction using metric depth

    Args:
        detections: Detection dictionary
        depth_images_path: Path to depth images directory (optional)
        camera_matrix: Camera intrinsic matrix for projection (optional)

    Returns:
        Dictionary mapping frame_id to list of depth-corrected 3D bboxes
    """
    detection_bboxes = {}

    for frame_id, frame_dets in detections.items():
        frame_bboxes = []

        # Load depth image for current frame if path is provided
        depth_img = None
        if depth_images_path:
            frame_num = int(frame_id)
            frame_idx = frame_num - 1 if frame_num > 0 else frame_num
            depth_img = load_depth_image(depth_images_path, frame_idx)

        for det in frame_dets:
            if 'loc' in det and 'dim' in det and 'rot_y' in det and 'class' in det:
                # Filter: only keep class 1 detections
                if det['class'] != 1:
                    continue

                # Get 3D location and dimensions
                loc = det['loc'].copy()  # [x, y, z] - make a copy to avoid modifying original
                dim = det['dim']         # [h, w, l]
                rot_y = det['rot_y']

                # ===== DEPTH CORRECTION USING METRIC DEPTH =====
                depth_stats = None
                mask_used = False

                detection_frame_num = int(frame_id)
                if depth_img is not None and sam2_lookup is not None:
                    candidate_ids = []
                    if 'sam2_car_id' in det:
                        candidate_ids.append(str(det['sam2_car_id']))
                    candidate_ids.append(str(det.get('tracking_id', '')))
                    if 'original_tracking_id' in det:
                        candidate_ids.append(str(det['original_tracking_id']))

                    seen = set()
                    for candidate in candidate_ids:
                        if not candidate or candidate in seen:
                            continue
                        seen.add(candidate)
                        mask = sam2_lookup.get_mask(candidate, detection_frame_num)
                        if mask is not None:
                            depth_stats = extract_depth_from_mask(depth_img, mask)
                            mask_used = True
                            if depth_stats['valid_pixels'] > 0 and depth_stats['mean_depth'] > 0:
                                break

                if mask_used and (depth_stats is None or depth_stats['valid_pixels'] == 0 or depth_stats['mean_depth'] <= 0):
                    mask_used = False
                    depth_stats = None

                if depth_img is not None and camera_matrix is not None and depth_stats is None:
                    # First, create initial bbox to get 2D projection
                    if HAS_DDD_UTILS:
                        initial_bbox_3d = compute_box_3d(dim, loc, rot_y)
                    else:
                        # Fallback implementation
                        h, w, l = dim
                        corners = np.array([
                            [-l/2, -w/2, 0], [l/2, -w/2, 0], [l/2, w/2, 0], [-l/2, w/2, 0],
                            [-l/2, -w/2, h], [l/2, -w/2, h], [l/2, w/2, h], [-l/2, w/2, h]
                        ])
                        R = np.array([
                            [np.cos(rot_y), 0, np.sin(rot_y)],
                            [0, 1, 0],
                            [-np.sin(rot_y), 0, np.cos(rot_y)]
                        ])
                        rotated_corners = (R @ corners.T).T
                        initial_bbox_3d = rotated_corners + np.array(loc)

                    # Project to 2D to get bbox region
                    bbox_2d = project_3d_to_image(initial_bbox_3d, camera_matrix)

                    # Extract depth statistics from projected region
                    depth_stats = extract_depth_from_bbox(bbox_2d, depth_img)

                if depth_stats and depth_stats['valid_pixels'] > 0 and depth_stats['mean_depth'] > 0:
                    corrected_depth = depth_stats['mean_depth']
                    source_text = " using SAM2 mask" if mask_used else ""
                    print(f"Frame {frame_id}: Correcting depth from {loc[2]:.2f}m to {corrected_depth:.2f}m{source_text}")
                    loc[2] = corrected_depth
                # ===================================================

                # Create final bbox with corrected depth
                if HAS_DDD_UTILS:
                    # Use CenterTrack's original method with corrected location
                    bbox_3d = compute_box_3d(dim, loc, rot_y)
                else:
                    # Fallback to our implementation with corrected location
                    h, w, l = dim
                    corners = np.array([
                        [-l/2, -w/2, 0], [l/2, -w/2, 0], [l/2, w/2, 0], [-l/2, w/2, 0],
                        [-l/2, -w/2, h], [l/2, -w/2, h], [l/2, w/2, h], [-l/2, w/2, h]
                    ])

                    R = np.array([
                        [np.cos(rot_y), 0, np.sin(rot_y)],
                        [0, 1, 0],
                        [-np.sin(rot_y), 0, np.cos(rot_y)]
                    ])

                    rotated_corners = (R @ corners.T).T
                    bbox_3d = rotated_corners + np.array(loc)

                frame_bboxes.append(bbox_3d)

        detection_bboxes[frame_id] = frame_bboxes

    return detection_bboxes


def visualize_detections_in_frame(detection_bboxes: Dict[str, List[np.ndarray]],
                                 detections_raw: Dict,
                                 output_video: str = 'detections_original.mp4',
                                 fps: int = 10,
                                 scale: float = 15.0):
    """
    Visualize detections in their original frame coordinates (camera view)

    Args:
        detection_bboxes: Dictionary of frame_id -> list of 3D bboxes
        detections_raw: Raw detection data with tracking IDs
        output_video: Output video path
        fps: Frames per second
        scale: Visualization scale (pixels per meter)
    """
    img_size = (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, img_size)

    center = (img_size[0] // 2, int(img_size[1] * 0.9))  # 70% down from top

    frame_ids = sorted(detection_bboxes.keys())
    print(f"Creating visualization for detections in {len(frame_ids)} frames...")

    # Store trails for each object (tracking_id -> list of positions)
    object_trails = {}
    ego_trail = []

    for frame_idx in tqdm(range(len(frame_ids)), desc="Rendering original"):
        frame_id = frame_ids[frame_idx]
        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

        # Draw grid
        grid_spacing = 50
        for x in range(0, img_size[0], grid_spacing):
            cv2.line(img, (x, 0), (x, img_size[1]), (40, 40, 40), 1)
        for y in range(0, img_size[1], grid_spacing):
            cv2.line(img, (0, y), (img_size[0], y), (40, 40, 40), 1)

        # Draw coordinate axes (camera frame)
        cv2.arrowedLine(img, center, (center[0] + 80, center[1]), (255, 0, 0), 3)  # X
        cv2.arrowedLine(img, center, (center[0], center[1] - 80), (0, 0, 255), 3)  # Z
        cv2.putText(img, "X (cam)", (center[0] + 90, center[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(img, "Z (cam)", (center[0] + 5, center[1] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw ego vehicle at origin
        ego_bbox = create_ego_bbox_at_origin()
        ego_pixels = []
        for corner in ego_bbox:
            px = int(corner[0] * scale + center[0])
            py = int(-corner[2] * scale + center[1])
            ego_pixels.append([px, py])
        ego_pixels = np.array(ego_pixels, np.int32)

        # Draw ego bbox in blue
        cv2.polylines(img, [ego_pixels[:4].reshape((-1, 1, 2))], True, (255, 100, 0), 2)
        cv2.polylines(img, [ego_pixels[4:].reshape((-1, 1, 2))], True, (255, 100, 0), 2)
        for i in range(4):
            cv2.line(img, tuple(ego_pixels[i]), tuple(ego_pixels[i+4]), (255, 100, 0), 2)

        # Add ego vehicle position to trail
        ego_pos = np.array([0, 0, 0])  # Ego is always at origin in camera frame
        ego_trail.append(ego_pos)

        # Draw ego trajectory trail (show all history)
        if len(ego_trail) > 1:
            for i in range(1, len(ego_trail)):  # Show all points
                pt1 = ego_trail[i-1]
                pt2 = ego_trail[i]
                px1 = int(pt1[0] * scale + center[0])
                py1 = int(-pt1[2] * scale + center[1])
                px2 = int(pt2[0] * scale + center[0])
                py2 = int(-pt2[2] * scale + center[1])

                alpha = i / len(ego_trail)
                trail_color = (int(255 * (1-alpha)), int(100), int(255 * alpha))
                cv2.line(img, (px1, py1), (px2, py2), trail_color, 2)

        # Update trails for current frame detections
        if frame_id in detection_bboxes and frame_id in detections_raw:
            for bbox_idx, bbox in enumerate(detection_bboxes[frame_id]):
                # Get corresponding detection data
                if bbox_idx < len(detections_raw[frame_id]):
                    det = detections_raw[frame_id][bbox_idx]
                    tracking_id = det.get('tracking_id', -1)

                    # Get object center position
                    obj_center = np.mean(bbox[:4], axis=0)

                    # Update trail for this object
                    if tracking_id not in object_trails:
                        object_trails[tracking_id] = []
                    object_trails[tracking_id].append(obj_center)

        # Draw ALL trails (including those of vehicles no longer visible)
        for tracking_id, trail in object_trails.items():
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    pt1 = trail[i-1]
                    pt2 = trail[i]
                    px1 = int(pt1[0] * scale + center[0])
                    py1 = int(-pt1[2] * scale + center[1])
                    px2 = int(pt2[0] * scale + center[0])
                    py2 = int(-pt2[2] * scale + center[1])

                    # Color based on tracking ID
                    color_idx = tracking_id % 6
                    trail_colors = [(100, 255, 100), (255, 255, 100), (255, 100, 255),
                                  (100, 255, 255), (255, 200, 100), (200, 100, 255)]
                    trail_color = trail_colors[color_idx]
                    cv2.line(img, (px1, py1), (px2, py2), trail_color, 2)

        # Draw current frame bboxes (only for currently visible objects)
        if frame_id in detection_bboxes and frame_id in detections_raw:
            for bbox_idx, bbox in enumerate(detection_bboxes[frame_id]):
                # Draw bbox
                bbox_pixels = []
                for corner in bbox:
                    px = int(corner[0] * scale + center[0])
                    py = int(-corner[2] * scale + center[1])
                    bbox_pixels.append([px, py])
                bbox_pixels = np.array(bbox_pixels, np.int32)

                # Get color based on tracking ID
                if bbox_idx < len(detections_raw[frame_id]):
                    det = detections_raw[frame_id][bbox_idx]
                    tracking_id = det.get('tracking_id', -1)
                    color_idx = tracking_id % 6
                    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255),
                             (0, 255, 255), (255, 200, 0), (200, 0, 255)]
                    detection_color = colors[color_idx]
                else:
                    detection_color = (0, 255, 0)

                # Draw detection bbox
                cv2.polylines(img, [bbox_pixels[:4].reshape((-1, 1, 2))], True, detection_color, 2)
                cv2.polylines(img, [bbox_pixels[4:].reshape((-1, 1, 2))], True, detection_color, 2)
                for i in range(4):
                    cv2.line(img, tuple(bbox_pixels[i]), tuple(bbox_pixels[i+4]), detection_color, 2)

                # Draw tracking ID
                if bbox_idx < len(detections_raw[frame_id]):
                    det = detections_raw[frame_id][bbox_idx]
                    tracking_id = det.get('tracking_id', -1)
                    bbox_center = np.mean(bbox_pixels[:4], axis=0).astype(int)
                    cv2.putText(img, f"ID{tracking_id}", tuple(bbox_center),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add info
        info_texts = [
            f"Frame: {frame_id}",
            f"Detections: {len(detection_bboxes.get(frame_id, []))}",
            "Blue: Ego vehicle, Green: Detected objects",
            "View: Camera frame (before transformation)"
        ]

        y_offset = 30
        for i, text in enumerate(info_texts):
            cv2.putText(img, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(img)

    out.release()
    print(f"Video saved to {output_video}")


# ============================================================================
# STEP 5: Transform Detections to World Frame
# ============================================================================

def transform_detections_to_world(detection_bboxes: Dict[str, List[np.ndarray]],
                                 transform_matrices: List[np.ndarray],
                                 frame_keys: List[str]) -> Dict[str, List[np.ndarray]]:
    """
    Transform detections from camera frame to world frame using transformation matrices

    Args:
        detection_bboxes: Dictionary of frame_id -> list of 3D bboxes in camera frame
        transform_matrices: List of transformation matrices (current to world)
        frame_keys: Sorted list of frame keys

    Returns:
        Dictionary of frame_id -> list of transformed 3D bboxes in world frame
    """
    transformed_detections = {}

    for idx, (frame_key, T) in enumerate(zip(frame_keys, transform_matrices)):
        if frame_key in detection_bboxes:
            frame_bboxes_world = []

            for bbox in detection_bboxes[frame_key]:
                # Convert to homogeneous coordinates
                bbox_homo = np.hstack([bbox, np.ones((bbox.shape[0], 1))])

                # Apply transformation
                bbox_world_homo = (T @ bbox_homo.T).T

                # Convert back to 3D
                bbox_world = bbox_world_homo[:, :3] / bbox_world_homo[:, 3:4]

                frame_bboxes_world.append(bbox_world)

            transformed_detections[frame_key] = frame_bboxes_world

    return transformed_detections


def visualize_detections_in_world(transformed_detections: Dict[str, List[np.ndarray]],
                                 transformed_ego: List[np.ndarray],
                                 detections_raw: Dict,
                                 frame_keys: List[str],
                                 output_video: str = 'detections_world.mp4',
                                 fps: int = 10,
                                 scale: float = 5.0):
    """
    Visualize detections and ego vehicle in world frame

    Args:
        transformed_detections: Dictionary of frame_id -> list of 3D bboxes in world frame
        transformed_ego: List of ego vehicle bboxes in world frame
        output_video: Output video path
        fps: Frames per second
    """
    img_size = (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, img_size)

    center = (img_size[0] // 2, int(img_size[1] * 0.9))  # 70% down from top

    frame_ids = sorted(transformed_detections.keys())

    # Store trails for objects in world frame
    all_ego_positions = []
    world_object_trails = {}

    print(f"Creating world frame visualization for {len(frame_ids)} frames...")

    for frame_idx in tqdm(range(len(frame_ids)), desc="Rendering world"):
        frame_id = frame_ids[frame_idx]
        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

        # Draw grid
        grid_spacing = 50
        for x in range(0, img_size[0], grid_spacing):
            cv2.line(img, (x, 0), (x, img_size[1]), (40, 40, 40), 1)
        for y in range(0, img_size[1], grid_spacing):
            cv2.line(img, (0, y), (img_size[0], y), (40, 40, 40), 1)

        # Draw world coordinate axes
        cv2.arrowedLine(img, center, (center[0] + 80, center[1]), (0, 0, 255), 3)  # X
        cv2.arrowedLine(img, center, (center[0], center[1] - 80), (0, 255, 0), 3)  # Z
        cv2.putText(img, "X (world)", (center[0] + 90, center[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, "Z (world)", (center[0] + 5, center[1] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Get ego bbox for current frame
        ego_bbox = transformed_ego[frame_idx]
        ego_center = np.mean(ego_bbox[:4], axis=0)
        all_ego_positions.append(ego_center)

        # Draw ego trajectory trail (show all history)
        if len(all_ego_positions) > 1:
            for i in range(1, len(all_ego_positions)):
                pt1 = all_ego_positions[i-1]
                pt2 = all_ego_positions[i]
                px1 = int(pt1[0] * scale + center[0])
                py1 = int(-pt1[2] * scale + center[1])
                px2 = int(pt2[0] * scale + center[0])
                py2 = int(-pt2[2] * scale + center[1])

                alpha = i / len(all_ego_positions)
                trail_color = (int(100 * (1 - alpha)), int(100), int(255 * alpha))
                cv2.line(img, (px1, py1), (px2, py2), trail_color, 2)

        # Draw ego vehicle
        ego_pixels = []
        for corner in ego_bbox:
            px = int(corner[0] * scale + center[0])
            py = int(-corner[2] * scale + center[1])
            ego_pixels.append([px, py])
        ego_pixels = np.array(ego_pixels, np.int32)

        # Draw ego in blue
        cv2.polylines(img, [ego_pixels[:4].reshape((-1, 1, 2))], True, (255, 100, 0), 3)
        cv2.polylines(img, [ego_pixels[4:].reshape((-1, 1, 2))], True, (255, 100, 0), 3)
        for i in range(4):
            cv2.line(img, tuple(ego_pixels[i]), tuple(ego_pixels[i+4]), (255, 100, 0), 2)

        # Update trails for current frame detections
        if frame_id in transformed_detections:
            for det_idx, bbox in enumerate(transformed_detections[frame_id]):
                # Get tracking ID from original detection data
                tracking_id = -1
                if frame_id in detections_raw and det_idx < len(detections_raw[frame_id]):
                    det = detections_raw[frame_id][det_idx]
                    tracking_id = det.get('tracking_id', -1)

                # Get object center position in world frame
                obj_center_world = np.mean(bbox[:4], axis=0)

                # Update trail for this object in world frame
                if tracking_id not in world_object_trails:
                    world_object_trails[tracking_id] = []
                world_object_trails[tracking_id].append(obj_center_world)

        # Draw ALL trails (including those of vehicles no longer visible)
        for tracking_id, trail in world_object_trails.items():
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    pt1 = trail[i-1]
                    pt2 = trail[i]
                    px1 = int(pt1[0] * scale + center[0])
                    py1 = int(-pt1[2] * scale + center[1])
                    px2 = int(pt2[0] * scale + center[0])
                    py2 = int(-pt2[2] * scale + center[1])

                    # Color based on tracking ID
                    color_idx = tracking_id % 6
                    trail_colors = [(100, 255, 100), (255, 255, 100), (255, 100, 255),
                                  (100, 255, 255), (255, 200, 100), (200, 100, 255)]
                    trail_color = trail_colors[color_idx]
                    cv2.line(img, (px1, py1), (px2, py2), trail_color, 2)

        # Draw current frame bboxes (only for currently visible objects)
        if frame_id in transformed_detections:
            for det_idx, bbox in enumerate(transformed_detections[frame_id]):
                # Get tracking ID from original detection data
                tracking_id = -1
                if frame_id in detections_raw and det_idx < len(detections_raw[frame_id]):
                    det = detections_raw[frame_id][det_idx]
                    tracking_id = det.get('tracking_id', -1)

                # Draw bbox
                bbox_pixels = []
                for corner in bbox:
                    px = int(corner[0] * scale + center[0])
                    py = int(-corner[2] * scale + center[1])
                    bbox_pixels.append([px, py])
                bbox_pixels = np.array(bbox_pixels, np.int32)

                # Color based on tracking ID
                color_idx = tracking_id % 6 if tracking_id >= 0 else det_idx % 6
                colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255),
                         (0, 255, 255), (255, 200, 0), (200, 0, 255)]
                detection_color = colors[color_idx]

                # Draw detection
                cv2.polylines(img, [bbox_pixels[:4].reshape((-1, 1, 2))], True, detection_color, 2)
                cv2.polylines(img, [bbox_pixels[4:].reshape((-1, 1, 2))], True, detection_color, 2)
                for i in range(4):
                    cv2.line(img, tuple(bbox_pixels[i]), tuple(bbox_pixels[i+4]), detection_color, 2)

                # Draw tracking ID
                bbox_center = np.mean(bbox_pixels[:4], axis=0).astype(int)
                cv2.putText(img, f"ID{tracking_id}", tuple(bbox_center),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add info
        info_texts = [
            f"Frame: {frame_id}",
            f"Detections: {len(transformed_detections.get(frame_id, []))}",
            f"Ego position: X={ego_center[0]:.1f}, Z={ego_center[2]:.1f}",
            "Blue: Ego, Green/Yellow/Purple: Other vehicles",
            "View: World frame (after transformation)"
        ]

        y_offset = 30
        for i, text in enumerate(info_texts):
            cv2.putText(img, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        out.write(img)

    out.release()
    print(f"Video saved to {output_video}")


# ============================================================================
# STEP 6: Kalman Filtering and RTS Smoothing (Position + Yaw)
# ============================================================================

class YawEKF:
    """
    Extended Kalman Filter for yaw angle with circular innovation handling
    State: [psi, omega] where psi=yaw angle, omega=yaw rate
    """

    def __init__(self,
                 initial_yaw: float = 0.0,
                 Q_psi: float = 1e-3,
                 Q_omega: float = 1e-2,
                 R_speed: float = 5e-3,
                 R_bbox: float = 5e-2,
                 speed_threshold: float = 0.2,
                 gate_threshold_deg: float = 120.0):
        """
        Initialize YawEKF

        Args:
            initial_yaw: Initial yaw angle in radians
            Q_psi: Process noise for yaw angle
            Q_omega: Process noise for yaw rate
            R_speed: Observation noise for motion-based heading
            R_bbox: Observation noise for bbox yaw
            speed_threshold: Minimum speed to trust motion heading (m/s)
            gate_threshold_deg: Maximum innovation before gating (degrees)
        """
        # State: [psi, omega]
        self.x = np.array([self.wrap_pi(initial_yaw), 0.0])

        # Covariance matrix
        self.P = np.eye(2) * 10.0  # Initial uncertainty

        # Process noise
        self.Q = np.diag([Q_psi, Q_omega])

        # Observation noise
        self.R_speed = R_speed
        self.R_bbox = R_bbox

        # Parameters
        self.speed_threshold = speed_threshold
        self.gate_threshold = np.deg2rad(gate_threshold_deg)

    @staticmethod
    def wrap_pi(angle: float) -> float:
        """Wrap angle to [-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def ang_diff(a: float, b: float) -> float:
        """Compute angular difference a - b in [-π, π]"""
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def predict(self, dt: float):
        """Prediction step"""
        # State transition matrix
        F = np.array([[1.0, dt],
                      [0.0, 1.0]])

        # Predict state
        self.x = F @ self.x
        self.x[0] = self.wrap_pi(self.x[0])  # Wrap predicted yaw

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, observation: float, use_speed_obs: bool = True, R_override: float = None):
        """
        Update step with circular innovation

        Args:
            observation: Observed yaw angle in radians
            use_speed_obs: Whether observation comes from speed (True) or bbox (False)
            R_override: Optional override for observation noise
        """
        # Choose observation noise
        if R_override is not None:
            R = R_override
        else:
            R = self.R_speed if use_speed_obs else self.R_bbox

        # Observation matrix
        H = np.array([1.0, 0.0]).reshape(1, -1)

        # Circular innovation
        innovation = self.ang_diff(observation, self.x[0])

        # Innovation gating
        if abs(innovation) > self.gate_threshold:
            print(f"Warning: Large yaw innovation {np.rad2deg(innovation):.1f}° gated")
            return  # Skip update

        # Standard Kalman update
        S = H @ self.P @ H.T + R
        K = self.P @ H.T / S

        # Update state
        self.x = self.x + K.flatten() * innovation
        self.x[0] = self.wrap_pi(self.x[0])  # Re-wrap after update

        # Update covariance
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P

    def get_yaw(self) -> float:
        """Get current filtered yaw"""
        return self.x[0]

    def get_yaw_rate(self) -> float:
        """Get current filtered yaw rate"""
        return self.x[1]


def compute_motion_heading(positions: List[np.ndarray], dt: float) -> List[Tuple[float, float]]:
    """
    Compute motion heading from position sequence

    Args:
        positions: List of 3D positions [x, y, z]
        dt: Time step

    Returns:
        List of (heading_angle, speed) tuples
    """
    motion_data = []

    for i in range(len(positions)):
        if i == 0:
            # No previous position available
            motion_data.append((0.0, 0.0))
        else:
            # Compute velocity
            p_prev, p_curr = positions[i-1], positions[i]

            # speed = np.sqrt(vx**2 + vy**2)
            # heading = np.arctan2(vy, vx) if speed > 1e-6 else 0.0
            vx = (p_curr[0] - p_prev[0]) / dt
            vz = (p_curr[2] - p_prev[2]) / dt

            speed = np.hypot(vx, vz)
            heading = np.arctan2(vz, vx) 


            motion_data.append((heading, speed))

    return motion_data


# ============================================================================
# STEP 6: Kalman Filtering and RTS Smoothing (Position + Yaw)
# ============================================================================

def kalman_filter_rts_smoother(trajectory_positions: List[np.ndarray],
                              dt: float = 1.0,
                              process_noise: float = 0.1,
                              measurement_noise: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply Kalman filtering followed by Rauch-Tung-Striebel (RTS) smoothing to trajectory data

    Args:
        trajectory_positions: List of 3D positions [x, y, z] for each frame
        dt: Time step between measurements
        process_noise: Process noise variance (how much we trust the motion model)
        measurement_noise: Measurement noise variance (how much we trust the observations)

    Returns:
        Tuple of (filtered_positions, smoothed_positions)
    """
    if len(trajectory_positions) < 2:
        return trajectory_positions, trajectory_positions

    n_frames = len(trajectory_positions)

    # State vector: [x, y, z, vx, vy, vz] - position and velocity
    state_dim = 6
    obs_dim = 3

    # Initialize state transition matrix (constant velocity model)
    F = np.eye(state_dim)
    F[0, 3] = dt  # x = x + vx*dt
    F[1, 4] = dt  # y = y + vy*dt
    F[2, 5] = dt  # z = z + vz*dt

    # Observation matrix (we only observe position, not velocity)
    H = np.zeros((obs_dim, state_dim))
    H[0, 0] = 1  # observe x
    H[1, 1] = 1  # observe y
    H[2, 2] = 1  # observe z

    # Process noise covariance matrix
    Q = np.eye(state_dim) * process_noise
    # Higher noise for velocity components
    Q[3:6, 3:6] *= 2.0

    # Measurement noise covariance matrix
    R = np.eye(obs_dim) * measurement_noise

    # Initialize state and covariance
    x_init = np.zeros(state_dim)
    x_init[:3] = trajectory_positions[0]  # Initial position
    # Initial velocity estimate (difference between first two positions)
    if len(trajectory_positions) > 1:
        x_init[3:6] = (trajectory_positions[1] - trajectory_positions[0]) / dt

    P_init = np.eye(state_dim) * 10.0  # Initial covariance

    # Storage for filtering
    x_filtered = []  # State estimates
    P_filtered = []  # Covariance matrices
    x_predicted = []  # Predicted states (for RTS)
    P_predicted = []  # Predicted covariances (for RTS)

    x = x_init.copy()
    P = P_init.copy()

    print(f"Applying Kalman filter to {n_frames} trajectory points...")

    # ========================================
    # FORWARD PASS: Kalman Filter
    # ========================================
    for i in range(n_frames):
        # Prediction step
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Store predictions for RTS smoother
        x_predicted.append(x_pred.copy())
        P_predicted.append(P_pred.copy())

        # Update step
        z = trajectory_positions[i]  # Current observation

        # Innovation
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R

        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)

        # State update
        x = x_pred + K @ y
        P = (np.eye(state_dim) - K @ H) @ P_pred

        # Store filtered results
        x_filtered.append(x.copy())
        P_filtered.append(P.copy())

    # Extract filtered positions
    filtered_positions = [x[:3] for x in x_filtered]

    print(f"✓ Kalman filtering complete")

    # ========================================
    # BACKWARD PASS: RTS Smoother
    # ========================================
    print(f"Applying RTS smoother...")

    # Initialize smoothed estimates with last filtered estimate
    x_smoothed = [None] * n_frames
    P_smoothed = [None] * n_frames

    x_smoothed[-1] = x_filtered[-1].copy()
    P_smoothed[-1] = P_filtered[-1].copy()

    # Backward pass
    for i in range(n_frames - 2, -1, -1):
        # Smoother gain
        A = P_filtered[i] @ F.T @ np.linalg.inv(P_predicted[i + 1])

        # Smoothed estimates
        x_smoothed[i] = x_filtered[i] + A @ (x_smoothed[i + 1] - x_predicted[i + 1])
        P_smoothed[i] = P_filtered[i] + A @ (P_smoothed[i + 1] - P_predicted[i + 1]) @ A.T

    # Extract smoothed positions
    smoothed_positions = [x[:3] for x in x_smoothed]

    print(f"✓ RTS smoothing complete")

    return filtered_positions, smoothed_positions


def _unwrap_angles_seq(angles: List[float]) -> np.ndarray:
    """Unwrap angle sequence by minimizing frame-to-frame circular differences."""
    if not angles:
        return np.array([], dtype=float)
    out = np.zeros(len(angles), dtype=float)
    out[0] = angles[0]
    for i in range(1, len(angles)):
        # Move current angle to nearest 2π branch of previous value
        out[i] = out[i - 1] + YawEKF.ang_diff(angles[i], out[i - 1])
    return out


def _stable_yaw_mask_by_median(yaw_angles: List[float],
                               window: int = 9,
                               threshold_deg: float = 12.0,
                               min_run: int = 6,
                               fill_gap: int = 1) -> np.ndarray:
    """
    Compute a boolean mask of stable yaw observations using a sliding-window median vote.

    - Unwrap yaw sequence to avoid ±π boundary issues
    - For each index, compute median within window and flag as stable
      if absolute residual is within threshold_deg
    - Remove stable runs shorter than min_run
    - Optionally fill small gaps (<= fill_gap) between stable runs
    """
    n = len(yaw_angles)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Unwrap angles to a continuous sequence
    yaw_unwrap = _unwrap_angles_seq(yaw_angles)

    half = max(0, window // 2)
    thr_rad = np.deg2rad(threshold_deg)
    stable = np.zeros(n, dtype=bool)

    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        median_val = float(np.median(yaw_unwrap[l:r]))
        resid = yaw_unwrap[i] - median_val
        stable[i] = abs(resid) <= thr_rad

    # Remove short stable runs
    if min_run > 1:
        i = 0
        while i < n:
            if stable[i]:
                j = i
                while j < n and stable[j]:
                    j += 1
                if (j - i) < min_run:
                    stable[i:j] = False
                i = j
            else:
                i += 1

    # Fill tiny gaps between stable runs (up to fill_gap frames)
    if fill_gap >= 1 and n >= 3:
        i = 0
        while i < n:
            if stable[i]:
                i += 1
                continue
            gap_start = i
            while i < n and not stable[i]:
                i += 1
            gap_len = i - gap_start
            if gap_len <= fill_gap and gap_start > 0 and i < n and stable[gap_start - 1] and stable[i]:
                stable[gap_start:i] = True

    return stable


def apply_yaw_filtering(yaw_angles: List[float],
                      positions: List[np.ndarray],
                      timestamps: List[float],
                      dt: float) -> Tuple[List[float], List[float]]:
    """
    Apply YawEKF filtering and RTS-like smoothing to yaw angle sequence

    Args:
        yaw_angles: List of raw yaw angles in radians
        positions: List of 3D positions for motion heading computation
        timestamps: List of timestamps
        dt: Time step

    Returns:
        Tuple of (filtered_yaw, smoothed_yaw)
    """
    if len(yaw_angles) < 2:
        return yaw_angles, yaw_angles

    # Initialize YawEKF with first yaw angle
    yaw_ekf = YawEKF(initial_yaw=yaw_angles[0])

    # Determine per-frame stability by median vote (no speed involved)
    STABLE_WINDOW = 9
    STABLE_THR_DEG = 12.0
    STABLE_MIN_RUN = 6
    STABLE_FILL_GAP = 1
    stable_mask = _stable_yaw_mask_by_median(
        yaw_angles,
        window=STABLE_WINDOW,
        threshold_deg=STABLE_THR_DEG,
        min_run=STABLE_MIN_RUN,
        fill_gap=STABLE_FILL_GAP,
    )

    # Forward pass: Kalman filtering
    filtered_yaw = []
    ekf_states = []  # Store states for RTS smoothing

    for i, (yaw_obs, timestamp) in enumerate(zip(yaw_angles, timestamps)):
        if i == 0:
            # First observation - just initialize
            filtered_yaw.append(yaw_ekf.get_yaw())
            ekf_states.append({
                'x': yaw_ekf.x.copy(),
                'P': yaw_ekf.P.copy()
            })
        else:
            # Predict step
            time_delta = dt if i < len(timestamps) else dt
            yaw_ekf.predict(time_delta)

            # Choose observation (yaw or yaw+pi) that best matches current state only
            state_yaw = yaw_ekf.get_yaw()
            candidates = [yaw_obs, YawEKF.wrap_pi(yaw_obs + np.pi)]
            # Align to nearest branch of current estimate and pick minimal innovation
            aligned = [YawEKF.wrap_pi(state_yaw + YawEKF.ang_diff(c, state_yaw)) for c in candidates]
            innovations = [abs(YawEKF.ang_diff(a, state_yaw)) for a in aligned]
            yaw_obs_adj = aligned[int(np.argmin(innovations))]

            # Update only if current frame is stable; otherwise skip bbox observation
            if stable_mask[i]:
                yaw_ekf.update(yaw_obs_adj, use_speed_obs=False)

            filtered_yaw.append(yaw_ekf.get_yaw())
            ekf_states.append({
                'x': yaw_ekf.x.copy(),
                'P': yaw_ekf.P.copy()
            })

    # Backward pass: RTS-like smoothing for yaw angles
    smoothed_yaw = filtered_yaw.copy()

    for i in range(len(smoothed_yaw) - 2, -1, -1):
        # Simple smoothing: blend filtered estimate with prediction from next state
        curr_yaw = smoothed_yaw[i]
        next_yaw = smoothed_yaw[i + 1]

        # Predict next yaw from current
        predicted_next = curr_yaw + ekf_states[i]['x'][1] * dt  # yaw + yaw_rate * dt
        predicted_next = YawEKF.wrap_pi(predicted_next)

        # Innovation between predicted and actual next
        innovation = YawEKF.ang_diff(next_yaw, predicted_next)

        # Smooth current estimate
        smoothing_gain = 0.3  # Smoothing strength
        correction = smoothing_gain * innovation
        smoothed_yaw[i] = YawEKF.wrap_pi(curr_yaw + correction)

    return filtered_yaw, smoothed_yaw


def apply_kalman_rts_to_all_trajectories(detection_bboxes: Dict[str, List[np.ndarray]],
                                       detections_raw: Dict,
                                       transform_matrices: List[np.ndarray],
                                       frame_keys: List[str],
                                       dt: float = 0.1,
                                       process_noise: float = 0.1,
                                       measurement_noise: float = 1.0,
                                       enable_yaw_filtering: bool = True,
                                       merge_mapping: Dict[int, List[int]] = None) -> Tuple[Dict[int, Dict[str, List[np.ndarray]]], Dict[int, List[int]]]:
    """
    Apply Kalman filtering and RTS smoothing to all vehicle trajectories

    Args:
        detection_bboxes: Detection data
        detections_raw: Raw detection data with tracking IDs
        transform_matrices: Transformation matrices
        frame_keys: Sorted frame keys
        dt: Time step
        process_noise: Process noise variance
        measurement_noise: Measurement noise variance
        enable_yaw_filtering: Whether to apply yaw angle filtering (default: True)

    Returns:
        Tuple of (smoothed_trajectories, merged_trajectory_mapping)
        - smoothed_trajectories: Dictionary with tracking_id -> {'raw': positions, 'filtered': positions, 'smoothed': positions, 'raw_yaw': yaw_angles, 'filtered_yaw': yaw_angles, 'smoothed_yaw': yaw_angles}
        - merged_trajectory_mapping: Dictionary mapping merged_id -> [original_id1, original_id2, ...]
    """
    print("\n" + "=" * 70)
    print("KALMAN FILTERING AND RTS SMOOTHING")
    print("=" * 70)
    print(f"Yaw angle filtering: {'ENABLED' if enable_yaw_filtering else 'DISABLED'}")

    # First, transform detections to world frame if not already done
    transformed_detections = transform_detections_to_world(detection_bboxes, transform_matrices, frame_keys)

    # Collect trajectories for each tracking ID
    trajectories = {}

    # Build complete trajectories for each tracking ID
    for frame_idx, frame_key in enumerate(frame_keys):
        if frame_key in transformed_detections and frame_key in detections_raw:
            for det_idx, bbox in enumerate(transformed_detections[frame_key]):
                if det_idx < len(detections_raw[frame_key]):
                    det = detections_raw[frame_key][det_idx]
                    tracking_id = det.get('tracking_id', -1)

                    if tracking_id >= 0:
                        if tracking_id not in trajectories:
                            trajectories[tracking_id] = {
                                'positions': [],
                                'yaw_angles': [],
                                'frame_indices': [],
                                'timestamps': []
                            }

                        # Use center of bottom face as position
                        position = np.mean(bbox[:4], axis=0)
                        trajectories[tracking_id]['positions'].append(position)

                        # Extract yaw angle from raw detection
                        yaw_angle = det.get('rot_y', 0.0)
                        trajectories[tracking_id]['yaw_angles'].append(yaw_angle)

                        trajectories[tracking_id]['frame_indices'].append(frame_idx)
                        trajectories[tracking_id]['timestamps'].append(frame_idx * dt)

    print(f"Found {len(trajectories)} distinct vehicle trajectories")

    # ========================================
    # TRAJECTORY MERGING: Track merged trajectory mappings
    # ========================================
    merged_trajectory_mapping = {}
    if merge_mapping:
        print("Applying SAM2 merge mapping to trajectories")
        for target_id, source_ids in merge_mapping.items():
            if target_id not in trajectories:
                continue
            merged_trajectory_mapping[target_id] = source_ids
            print(f"  Tracking ID {target_id} sourced from {source_ids}")

    # Apply Kalman filtering and RTS smoothing to each trajectory
    smoothed_trajectories = {}

    for tracking_id, traj_data in trajectories.items():
        positions = traj_data['positions']
        yaw_angles = traj_data['yaw_angles']
        timestamps = traj_data['timestamps']

        if len(positions) >= 3:  # Need at least 3 points for meaningful filtering
            print(f"Processing vehicle ID {tracking_id}: {len(positions)} points")

            # Apply position filtering
            filtered_pos, smoothed_pos = kalman_filter_rts_smoother(
                positions, dt, process_noise, measurement_noise
            )

            # Apply yaw angle filtering using YawEKF (if enabled)
            if enable_yaw_filtering:
                filtered_yaw, smoothed_yaw = apply_yaw_filtering(
                    yaw_angles, positions, timestamps, dt
                )
            else:
                # Keep original yaw angles without filtering
                filtered_yaw = yaw_angles.copy()
                smoothed_yaw = yaw_angles.copy()

            smoothed_trajectories[tracking_id] = {
                'raw': positions,
                'filtered': filtered_pos,
                'smoothed': smoothed_pos,
                'raw_yaw': yaw_angles,
                'filtered_yaw': filtered_yaw,
                'smoothed_yaw': smoothed_yaw,
                'frame_indices': traj_data['frame_indices'],
                'timestamps': traj_data['timestamps']
            }
        else:
            print(f"Skipping vehicle ID {tracking_id}: only {len(positions)} points (need ≥3)")
            # Keep original for short trajectories
            smoothed_trajectories[tracking_id] = {
                'raw': positions,
                'filtered': positions,
                'smoothed': positions,
                'raw_yaw': yaw_angles,
                'filtered_yaw': yaw_angles,
                'smoothed_yaw': yaw_angles,
                'frame_indices': traj_data['frame_indices'],
                'timestamps': traj_data['timestamps']
            }

    return smoothed_trajectories, merged_trajectory_mapping


def visualize_smoothed_trajectories(smoothed_trajectories: Dict[int, Dict[str, List[np.ndarray]]],
                                  output_video: str = 'smoothed_trajectories.mp4',
                                  fps: int = 10,
                                  scale: float = 12.0,
                                  banner_text: Optional[str] = None):
    """
    Visualize original vs filtered vs smoothed trajectories

    Args:
        smoothed_trajectories: Output from apply_kalman_rts_to_all_trajectories
        output_video: Output video file path
        fps: Frames per second
        scale: Visualization scale
    """
    img_size = (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, img_size)

    center = (img_size[0] // 2, int(img_size[1] * 0.9))

    # Find the maximum frame index to determine video length
    max_frame = 0
    for traj_data in smoothed_trajectories.values():
        if traj_data['frame_indices']:
            max_frame = max(max_frame, max(traj_data['frame_indices']))

    print(f"Creating smoothed trajectory visualization for {max_frame + 1} frames...")

    for frame_idx in tqdm(range(max_frame + 1), desc="Rendering smoothed"):
        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

        # Draw grid
        grid_spacing = 50
        for x in range(0, img_size[0], grid_spacing):
            cv2.line(img, (x, 0), (x, img_size[1]), (40, 40, 40), 1)
        for y in range(0, img_size[1], grid_spacing):
            cv2.line(img, (0, y), (img_size[0], y), (40, 40, 40), 1)

        # Draw coordinate axes
        cv2.arrowedLine(img, center, (center[0] + 80, center[1]), (0, 0, 255), 3)
        cv2.arrowedLine(img, center, (center[0], center[1] - 80), (0, 255, 0), 3)
        cv2.putText(img, "X (world)", (center[0] + 90, center[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, "Z (world)", (center[0] + 5, center[1] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw trajectories for each vehicle
        for tracking_id, traj_data in smoothed_trajectories.items():
            raw_positions = traj_data['raw']
            filtered_positions = traj_data['filtered']
            smoothed_positions = traj_data['smoothed']
            frame_indices = traj_data['frame_indices']

            color_idx = tracking_id % 6
            base_colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255),
                          (0, 255, 255), (255, 200, 0), (200, 0, 255)]
            base_color = base_colors[color_idx]

            # Find points up to current frame
            current_indices = [i for i, fi in enumerate(frame_indices) if fi <= frame_idx]

            if len(current_indices) > 1:
                # Draw raw trajectory (thin, darker)
                for i in range(1, len(current_indices)):
                    idx1, idx2 = current_indices[i-1], current_indices[i]
                    pt1, pt2 = raw_positions[idx1], raw_positions[idx2]

                    px1 = int(pt1[0] * scale + center[0])
                    py1 = int(-pt1[2] * scale + center[1])
                    px2 = int(pt2[0] * scale + center[0])
                    py2 = int(-pt2[2] * scale + center[1])

                    # Raw trajectory in dim color
                    dim_color = tuple(int(c * 0.3) for c in base_color)
                    cv2.line(img, (px1, py1), (px2, py2), dim_color, 1)

                # Draw filtered trajectory (medium thickness)
                for i in range(1, len(current_indices)):
                    idx1, idx2 = current_indices[i-1], current_indices[i]
                    pt1, pt2 = filtered_positions[idx1], filtered_positions[idx2]

                    px1 = int(pt1[0] * scale + center[0])
                    py1 = int(-pt1[2] * scale + center[1])
                    px2 = int(pt2[0] * scale + center[0])
                    py2 = int(-pt2[2] * scale + center[1])

                    # Filtered trajectory in medium color
                    med_color = tuple(int(c * 0.7) for c in base_color)
                    cv2.line(img, (px1, py1), (px2, py2), med_color, 2)

                # Draw smoothed trajectory (thick, bright)
                for i in range(1, len(current_indices)):
                    idx1, idx2 = current_indices[i-1], current_indices[i]
                    pt1, pt2 = smoothed_positions[idx1], smoothed_positions[idx2]

                    px1 = int(pt1[0] * scale + center[0])
                    py1 = int(-pt1[2] * scale + center[1])
                    px2 = int(pt2[0] * scale + center[0])
                    py2 = int(-pt2[2] * scale + center[1])

                    # Smoothed trajectory in full brightness
                    cv2.line(img, (px1, py1), (px2, py2), base_color, 3)

                # Draw current position
                if current_indices:
                    last_idx = current_indices[-1]
                    curr_pos = smoothed_positions[last_idx]
                    px = int(curr_pos[0] * scale + center[0])
                    py = int(-curr_pos[2] * scale + center[1])
                    cv2.circle(img, (px, py), 8, base_color, -1)
                    cv2.putText(img, f"ID{tracking_id}", (px + 15, py),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add legend and info
        info_texts = [
            f"Frame: {frame_idx}",
            "Thin line: Raw trajectory",
            "Medium line: Kalman filtered",
            "Thick line: RTS smoothed",
            f"Active vehicles: {len([t for t in smoothed_trajectories.values() if frame_idx in t['frame_indices']])}"
        ]

        # Optional banner (e.g., Yaw filtering ON/OFF)
        if banner_text:
            info_texts.insert(0, banner_text)

        y_offset = 30
        for i, text in enumerate(info_texts):
            cv2.putText(img, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(img)

    out.release()
    print(f"Smoothed trajectory video saved to {output_video}")


# ============================================================================
# Main function demonstrating the eight-step pipeline
# ============================================================================

def main(
    video_id=None,
    intrinsic_file=None,
    depth_images_path=None,
    trajectories_file=None,
    detections_json=None,
    savepath=None,
    override_dims_step0: bool = False,
    vehicle_specs_json: Optional[str] = None,
    skip_debug_videos: bool = False,
):
    print("=" * 70)
    print("EIGHT-STEP TRAJECTORY RECONSTRUCTION PIPELINE")
    print(f"Processing video: {video_id}")
    print("=" * 70)

    # Create output directory if savepath is provided
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        print(f"Output directory: {savepath}")
    else:
        savepath = '.'  # Use current directory as default
        print(f"Output directory: current directory")

    # ========================================
    # CONFIGURATION SWITCHES
    # ========================================
    # Switch to enable/disable yaw angle filtering
    #
    ENABLE_YAW_FILTERING = True   # -> Apply YawEKF filtering to yaw angles (recommended)
    # ENABLE_YAW_FILTERING = False  # -> Keep original raw yaw angles without filtering
    #
    # ENABLE_YAW_FILTERING = True  # Change this to False to disable yaw filtering

    print(f"Configuration:")
    print(f"  Yaw angle filtering: {'ENABLED' if ENABLE_YAW_FILTERING else 'DISABLED'}")
    if ENABLE_YAW_FILTERING:
        print(f"    -> YawEKF will filter noisy yaw angle observations")
        print(f"    -> Uses both bbox yaw and motion heading for robust estimation")
    else:
        print(f"    -> Raw yaw angles from detections will be preserved")
        print(f"    -> No circular innovation or yaw rate estimation")
    print()

    if trajectories_file is None:
        trajectory_json = f'/root/CenterTrack/{video_id}_trajectories.json'
    else:
        trajectory_json = trajectories_file

    if detections_json is None:
        detections_json = f'/root/CenterTrack/results/default_{video_id}.mp4_results.json'

    # ========================================
    # STEP 0: Optional SAM2 alignment
    # ========================================
    ENABLE_STEP0_SAM2 = True  # Toggle this flag to enable/disable Step 0
    sam2_merge_mapping: Dict[int, List[int]] = {}
    sam2_lookup: Optional[SAM2MaskLookup] = None

    if ENABLE_STEP0_SAM2:
        print("\n[STEP 0] Matching SAM2 masks with tracking trajectories...")
        trajectory_json, detections_json, sam2_merge_mapping, sam2_lookup = match_tracking_with_sam2(
            video_id, trajectory_json, detections_json, savepath,
            override_dims_step0=override_dims_step0,
            vehicle_specs_json=vehicle_specs_json
        )
    else:
        print("\n[STEP 0] SAM2 alignment disabled by configuration")

    # ========================================
    # STEP 1: Read and process JSON
    # ========================================
    print("\n[STEP 1] Reading and processing trajectory JSON...")
    transform_matrices = read_and_process_trajectory(trajectory_json)
    print(f"✓ Got {len(transform_matrices)} transformation matrices")



    # ========================================
    # STEP 2: Apply transformations to ego
    # ========================================
    print("\n[STEP 2] Applying transformations to ego vehicle at origin...")
    ego_bbox = create_ego_bbox_at_origin()
    print(f"  Created ego vehicle bbox: {ego_bbox.shape}")

    objects_at_origin = [ego_bbox]
    transformed_ego = apply_transformations(transform_matrices, objects_at_origin)
    print(f"✓ Got {len(transformed_ego)} transformed ego bboxes in world coordinates")

    # ========================================
    # STEP 3: Visualize ego trajectory
    # ========================================
    if skip_debug_videos:
        print("\n[STEP 3] Skipping ego trajectory debug video")
    else:
        print("\n[STEP 3] Visualizing ego trajectory...")
        output_video_path = os.path.join(savepath, 'step3_ego_trajectory.mp4')
        visualize_transformed_objects(transformed_ego,
                                     output_video=output_video_path,
                                     fps=10, scale=5.0)
        print("✓ Ego visualization complete")

    # ========================================
    # STEP 4: Read and visualize detections
    # ========================================
    print("\n[STEP 4] Reading detection results...")
    print(f"Using detections file: {detections_json}")
    detections = read_detection_results(detections_json)

    if sam2_lookup is None:
        sam2_path = os.path.join(savepath, f"{video_id}_sam2_masks.npz")
        if os.path.exists(sam2_path) and detections:
            try:
                det_min_frame_lookup = min(int(key) for key in detections.keys())
                sam2_lookup = SAM2MaskLookup(sam2_path, det_min_frame_lookup)
                print("  Loaded SAM2 masks for depth correction")
            except Exception as exc:
                print(f"  Warning: Unable to load SAM2 masks for depth correction: {exc}")

    # ========================================
    # TRAJECTORY READING MODE SELECTION
    # ========================================
    # Uncomment/comment ONE of the following two options to switch between modes:

    # # OPTION 1: Normal trajectory reading (without depth correction)
    # detection_bboxes = extract_detection_bboxes(detections)
    # print(f"✓ Extracted bboxes for {len(detection_bboxes)} frames using NORMAL method")

    # OPTION 2: Depth-corrected trajectory reading (with metric depth correction)
    if depth_images_path is None:
        depth_images_path = f'/root/CenterTrack/{video_id}_DEPTH'  # Default path if not provided

    # Read camera matrix from intrinsic file
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        fx = float(lines[0].strip())
        fy = float(lines[1].strip())
        cx = float(lines[2].strip())
        cy = float(lines[3].strip())

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    print(f"✓ Loaded camera matrix from {intrinsic_file}:")
    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    detection_bboxes = extract_detection_bboxes_depth_corrected(detections, depth_images_path, camera_matrix, sam2_lookup=sam2_lookup)
    print(f"✓ Extracted bboxes for {len(detection_bboxes)} frames using DEPTH-CORRECTED method")

    # ========================================

    if skip_debug_videos:
        print("✓ Skipped camera frame debug video")
    else:
        output_video_path = os.path.join(savepath, 'step4_detections_camera.mp4')
        visualize_detections_in_frame(detection_bboxes, detections,
                                     output_video=output_video_path,
                                     fps=10, scale=12.0)
        print("✓ Camera frame visualization complete")

    # ========================================
    # STEP 5: Transform and visualize in world
    # ========================================
    print("\n[STEP 5] Transforming detections to world frame...")

    # Get sorted frame keys
    with open(trajectory_json, 'r') as f:
        trajectory_data = json.load(f)
    frame_keys = sorted(trajectory_data.keys())

    # Transform detections
    transformed_detections = transform_detections_to_world(
        detection_bboxes, transform_matrices, frame_keys)
    print(f"✓ Transformed detections for {len(transformed_detections)} frames")

    if skip_debug_videos:
        print("✓ Skipped world frame debug video")
    else:
        output_video_path = os.path.join(savepath, 'step5_detections_world.mp4')
        visualize_detections_in_world(transformed_detections, transformed_ego,
                                     detections, frame_keys,
                                     output_video=output_video_path,
                                     fps=10, scale=12.0)
        print("✓ World frame visualization complete")

    # ========================================
    # STEP 5.5: Save world frame results as detection format
    # ========================================
    print("\n[STEP 5.5] Saving world frame results to detection format...")

    # Save world frame results in detection format
    world_results_path = os.path.join(savepath, f'world_transformed_{video_id}.mp4_results.json')
    save_world_frame_results_as_detections(transformed_detections, detections, frame_keys,
                                          world_results_path)
    print(f"✓ World frame results saved to: {world_results_path}")

    # ========================================
    # STEP 6: Kalman filtering and RTS smoothing
    # ========================================
    print("\n[STEP 6] Applying Kalman filtering and RTS smoothing...")

    # Single run according to ENABLE_YAW_FILTERING switch
    smoothed_trajectories, merged_mapping = apply_kalman_rts_to_all_trajectories(
        detection_bboxes, detections, transform_matrices, frame_keys,
        dt=0.1, process_noise=0.1, measurement_noise=1.0,
        enable_yaw_filtering=ENABLE_YAW_FILTERING,
        merge_mapping=sam2_merge_mapping if sam2_merge_mapping else None)

    # Single video export (no duplication)
    output_video_path = os.path.join(savepath, 'step6_kalman_rts_smoothed.mp4')
    banner = 'Yaw filtering: ENABLED' if ENABLE_YAW_FILTERING else 'Yaw filtering: DISABLED'
    visualize_smoothed_trajectories(smoothed_trajectories,
                                   output_video=output_video_path,
                                   fps=10, scale=12.0,
                                   banner_text=banner)
    print("✓ Kalman filtering and RTS smoothing complete")

    # ========================================
    # STEP 7: Save Kalman results as detection format
    # ========================================
    print("\n[STEP 7] Saving Kalman smoothed results to detection format...")

    # Save smoothed results in detection format with merged trajectory mapping
    # Add suffix to JSON filename based on yaw filtering switch for easy comparison
    yaw_suffix = 'yaw_on' if ENABLE_YAW_FILTERING else 'yaw_off'
    smoothed_results_path = os.path.join(savepath, f'kalman_smoothed_{video_id}_{yaw_suffix}.mp4_results.json')
    save_kalman_results_as_detections(smoothed_trajectories, detections, frame_keys,
                                    smoothed_results_path, merged_mapping)
    print(f"✓ Kalman smoothed results saved to: {smoothed_results_path}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - 8 STEPS")
    print("=" * 70)


# ============================================================================
# STEP 5.5 & 7: Save Results in Detection Format
# ============================================================================

def save_world_frame_results_as_detections(transformed_detections: Dict[str, List[np.ndarray]],
                                          original_detections: Dict,
                                          frame_keys: List[str],
                                          output_path: str) -> None:
    """
    Save world frame transformed results in original detection format

    Args:
        transformed_detections: World frame transformed detection bboxes
        original_detections: Original detection data to preserve non-position attributes
        frame_keys: Sorted frame keys
        output_path: Path to save the results JSON
    """
    import json
    from copy import deepcopy

    # Create new results dictionary
    world_results = {}

    print(f"Converting world frame results to detection format...")

    # Process each frame
    for frame_idx, frame_key in enumerate(frame_keys):
        frame_num = str(int(frame_key))  # Convert to string for JSON key
        world_results[frame_num] = []

        # Process detections in this frame
        if frame_key in transformed_detections and frame_key in original_detections:
            for det_idx, bbox_3d in enumerate(transformed_detections[frame_key]):
                # Get corresponding original detection
                if det_idx < len(original_detections[frame_key]):
                    original_det = deepcopy(original_detections[frame_key][det_idx])

                    # Calculate center position from transformed bbox
                    center_pos = np.mean(bbox_3d[:4], axis=0)  # Bottom face center

                    # Update position-related fields with world frame values
                    original_det['loc'] = [float(center_pos[0]),
                                         float(center_pos[1]),
                                         float(center_pos[2])]

                    # Update depth value to match z-coordinate
                    original_det['dep'] = [float(center_pos[2])]

                    # Keep all other original attributes (dim, rot_y, class, tracking_id, etc.)
                    world_results[frame_num].append(original_det)

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(world_results, f, indent=2)

    print(f"✓ Saved world frame results for {len(world_results)} frames")

    # Print statistics
    total_detections = sum(len(dets) for dets in world_results.values())
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per frame: {total_detections/len(world_results):.1f}")


# ============================================================================
# STEP 7: Save Kalman Results in Detection Format
# ============================================================================

def save_kalman_results_as_detections(smoothed_trajectories: Dict[int, Dict[str, List[np.ndarray]]],
                                     original_detections: Dict,
                                     frame_keys: List[str],
                                     output_path: str,
                                     merged_trajectory_mapping: Dict[int, List[int]] = None) -> None:
    """
    Save Kalman smoothed trajectory results in original detection format

    Args:
        smoothed_trajectories: Output from apply_kalman_rts_to_all_trajectories
        original_detections: Original detection data to preserve non-position attributes
        frame_keys: Sorted frame keys
        output_path: Path to save the results JSON
        merged_trajectory_mapping: Dict mapping merged_id -> [original_id1, original_id2, ...]
    """
    import json
    from copy import deepcopy

    # Create new results dictionary
    kalman_results = {}

    print(f"Converting {len(smoothed_trajectories)} smoothed trajectories to detection format...")

    # Process each frame
    for frame_idx, frame_key in enumerate(frame_keys):
        frame_num = str(int(frame_key))  # Convert to string for JSON key
        kalman_results[frame_num] = []

        # Find all vehicles that should be in this frame
        for tracking_id, traj_data in smoothed_trajectories.items():
            frame_indices = traj_data['frame_indices']

            # Check if this tracking_id appears in current frame
            if frame_idx in frame_indices:
                # Find the index within this trajectory
                traj_idx = frame_indices.index(frame_idx)

                # Get smoothed position and yaw
                smoothed_pos = traj_data['smoothed'][traj_idx]

                # Get smoothed yaw (or raw yaw if filtering was disabled)
                if 'smoothed_yaw' in traj_data:
                    smoothed_yaw = traj_data['smoothed_yaw'][traj_idx]
                else:
                    # Fallback to raw yaw if yaw filtering was disabled
                    smoothed_yaw = traj_data.get('raw_yaw', [0.0] * len(traj_data['smoothed']))[traj_idx]

                # Find original detection for this tracking_id and frame
                # Consider merged trajectories
                original_det = None
                search_ids = [tracking_id]

                # If this is a merged trajectory, search for original IDs
                if merged_trajectory_mapping and tracking_id in merged_trajectory_mapping:
                    search_ids = merged_trajectory_mapping[tracking_id]

                if frame_key in original_detections:
                    for search_id in search_ids:
                        for det in original_detections[frame_key]:
                            if det.get('tracking_id') == search_id:
                                original_det = deepcopy(det)
                                # Update tracking_id to the merged ID
                                original_det['tracking_id'] = tracking_id
                                break
                        if original_det is not None:
                            break

                if original_det is not None:
                    # Update position-related fields with smoothed values
                    original_det['loc'] = [float(smoothed_pos[0]),
                                         float(smoothed_pos[1]),
                                         float(smoothed_pos[2])]

                    # Update depth value to match z-coordinate
                    original_det['dep'] = [float(smoothed_pos[2])]

                    # Update yaw angle with smoothed value
                    original_det['rot_y'] = float(smoothed_yaw)

                    # Update age field to reflect continuous trajectory
                    # Age should be the position within the merged trajectory + 1
                    original_det['age'] = traj_idx + 1

                    # Update active field to reflect total trajectory length
                    original_det['active'] = len(traj_data['frame_indices'])

                    # Keep all other original attributes (dim, rot_y, class, etc.)
                    # Note: bbox, ct might be slightly inaccurate but we keep them for compatibility

                    kalman_results[frame_num].append(original_det)

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(kalman_results, f, indent=2)

    print(f"✓ Saved Kalman results for {len(kalman_results)} frames")

    # Print statistics
    total_detections = sum(len(dets) for dets in kalman_results.values())
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per frame: {total_detections/len(kalman_results):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eight-step trajectory reconstruction pipeline')
    parser.add_argument('--video_id', type=str, required=True,
                       help='Video ID to process (default: VV_76)')
    parser.add_argument('--intrinsic_file', type=str, required=True,
                       help='Path to camera intrinsic parameters file')
    parser.add_argument('--depth_images_path', type=str, required=True,
                       help='Path to depth images directory (default: {video_id}_DEPTH)')
    parser.add_argument('--trajectories_file', type=str, required=True,
                       help='Path to trajectories JSON file (default: {video_id}_trajectories.json)')
    parser.add_argument('--detections_json', type=str, required=False,
                       help='Path to detection results JSON file (default: /root/CenterTrack/results/default_{video_id}.mp4_results.json)')
    parser.add_argument('--savepath', type=str, required=False,
                       help='Directory path to save all output files (default: current directory)')
    parser.add_argument('--override_dims_step0', action='store_true',
                       help='If set, override bbox dimensions at Step 0 using vehicle specs JSON')
    parser.add_argument('--vehicle_specs_json', type=str, required=False,
                       help='Path to vehicle specs JSON (default: {savepath}/{video_id}_vehicle_specs.json)')
    parser.add_argument('--skip_debug_videos', action='store_true',
                       help='Skip diagnostic step3/step4/step5 videos while keeping JSON outputs and Kalman visualization')
    args = parser.parse_args()

    main(video_id=args.video_id, intrinsic_file=args.intrinsic_file,
         depth_images_path=args.depth_images_path, trajectories_file=args.trajectories_file,
         detections_json=args.detections_json, savepath=args.savepath,
         override_dims_step0=args.override_dims_step0,
         vehicle_specs_json=args.vehicle_specs_json,
         skip_debug_videos=args.skip_debug_videos)
