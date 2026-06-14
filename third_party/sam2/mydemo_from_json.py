import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.5) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    overlay = frame_bgr.copy()
    overlay[mask] = (np.array(color, dtype=np.uint8))
    out = frame_bgr.copy()
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def xywh_to_xyxy(xywh: List[float]) -> List[float]:
    x, y, w, h = xywh
    x2 = x + w - 1
    y2 = y + h - 1
    return [float(x), float(y), float(x2), float(y2)]


def parse_json_bboxes(json_path: str) -> Tuple[int, Dict[int, List[float]]]:
    """Parse JSON file to get anchor frame index and bboxes per car ID.

    Expected JSON fields:
      - bboxes: { car_id_str: [x, y, w, h], ... }
      - frame_start_index or selected_frame_reindexed: integer frame index for prompts
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    bboxes_xywh = data.get("bboxes")
    if not isinstance(bboxes_xywh, dict) or len(bboxes_xywh) == 0:
        raise ValueError(f"Invalid or empty 'bboxes' in {json_path}")

    # Determine which frame index to place the prompts on
    anchor = int(data.get("frame_start_index", data.get("selected_frame_reindexed", 0)))

    bboxes_xyxy: Dict[int, List[float]] = {}
    for k_str, xywh in bboxes_xywh.items():
        try:
            cid = int(k_str)
        except Exception:
            # keep original string if it cannot be parsed as int
            # but we still need a stable numeric obj_id for the tracker mapping; we'll assign later
            cid = int(str(abs(hash(k_str)))[:6])
        if not (isinstance(xywh, (list, tuple)) and len(xywh) == 4):
            raise ValueError(f"BBox for ID {k_str} must be [x,y,w,h], got: {xywh}")
        bboxes_xyxy[cid] = xywh_to_xyxy([float(v) for v in xywh])

    return anchor, bboxes_xyxy


def run_from_json(
    json_path: str,
    video_path: str,
    out_video_path: str,
    out_npz_path: str,
    alpha: float = 0.45,
    out_bbox_img_path: str | None = None,
) -> None:
    try:
        import decord  # noqa: F401
    except Exception as e:
        raise RuntimeError("Please install decord: pip install decord") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"

    anchor_frame, carid_to_bbox = parse_json_bboxes(json_path)

    # If the JSON indicates the video was trimmed starting from the anchor frame,
    # seed on frame 0 of the (trimmed) video. Otherwise use the anchor frame from JSON.
    try:
        with open(json_path, "r") as _f:
            _meta = json.load(_f)
    except Exception:
        _meta = {}
    trimmed = any(k in _meta for k in ("trim_method", "offset_seconds", "out_video_path"))
    anchor_used = 0 if trimmed else int(anchor_frame)
    anchor_used = 0 

    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
    # raise NotImplementedError(video_path)
    state = predictor.init_state(video_path=video_path)

    # Palette (BGR)
    palette = [
        (0, 0, 255),      # red
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 255, 255),    # yellow
        (255, 0, 255),    # magenta
        (255, 255, 0),    # cyan
        (0, 128, 255),    # orange-ish
        (128, 0, 255),    # purple-ish
        (203, 192, 255),  # pink-ish
        (102, 255, 102),  # light green
    ]

    # Map original car IDs to SAM2 obj_ids (1..N)
    car_ids_sorted = sorted(carid_to_bbox.keys())
    car_to_objid: Dict[int, int] = {cid: (i + 1) for i, cid in enumerate(car_ids_sorted)}
    objid_to_car: Dict[int, int] = {v: k for k, v in car_to_objid.items()}

    # Add initial box prompts at the specified anchor frame
    for cid in car_ids_sorted:
        bbox_xyxy = np.array(carid_to_bbox[cid], dtype=np.float32)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=int(anchor_used),
            obj_id=int(car_to_objid[cid]),
            box=bbox_xyxy,
        )

    # Video writer setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    # writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))


    # Save bbox visualization on the anchor frame to disk (no window)
    try:
        cap_bbox = cv2.VideoCapture(video_path)
        if cap_bbox.isOpened():
            total_frames = int(cap_bbox.get(cv2.CAP_PROP_FRAME_COUNT))
            if 0 <= anchor_used < max(1, total_frames):
                cap_bbox.set(cv2.CAP_PROP_POS_FRAMES, anchor_used)
                ret0, frame0 = cap_bbox.read()
            else:
                ret0, frame0 = False, None
        else:
            ret0, frame0 = False, None
    finally:
        try:
            cap_bbox.release()
        except Exception:
            pass

    if ret0 and frame0 is not None:
        vis = frame0.copy()
        for i, cid in enumerate(car_ids_sorted, start=1):
            x1, y1, x2, y2 = [int(round(v)) for v in carid_to_bbox[cid]]
            color = palette[(i - 1) % len(palette)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                f"ID {cid}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        if out_bbox_img_path is None or len(out_bbox_img_path) == 0:
            out_dir = os.path.dirname(video_path) or "."
            base = os.path.splitext(os.path.basename(video_path))[0]
            out_bbox_img_path = os.path.join(out_dir, f"{base}_bboxes_anchor_{anchor_used}.png")
        cv2.imwrite(out_bbox_img_path, vis)
        print(f"Saved bbox image to: {out_bbox_img_path}")

    # Collect masks by frame, keyed by original car ID
    masks_by_frame: Dict[int, Dict[int, np.ndarray]] = {}

    # Also collect per-car lists for NPZ export
    per_car_frames: Dict[int, List[int]] = {cid: [] for cid in car_ids_sorted}
    per_car_masks: Dict[int, List[np.ndarray]] = {cid: [] for cid in car_ids_sorted}

    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else torch.no_grad()
    )
    with torch.inference_mode():
        if device.startswith("cuda"):
            ctx.__enter__()
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            if len(out_obj_ids) == 0:
                continue
            frame_entry: Dict[int, np.ndarray] = {}
            for k, oid in enumerate(out_obj_ids):
                logits = out_mask_logits[k, 0].detach().cpu().float().numpy()
                mask = logits > 0.0
                car_id = objid_to_car.get(int(oid))
                if car_id is None:
                    continue
                frame_entry[car_id] = mask
                per_car_frames[car_id].append(int(out_frame_idx))
                per_car_masks[car_id].append(mask)
            if frame_entry:
                masks_by_frame[int(out_frame_idx)] = frame_entry
        if device.startswith("cuda"):
            ctx.__exit__(None, None, None)

    # Render overlays to the output video
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        m_dict = masks_by_frame.get(fidx)
        if m_dict is not None:
            for i, cid in enumerate(car_ids_sorted, start=1):
                mm = m_dict.get(cid)
                if mm is None:
                    continue
                color = palette[(i - 1) % len(palette)]
                frame = overlay_mask(frame, mm, color=color, alpha=alpha)
                # Put label with original car ID
                # Compute a simple bbox around the mask for placing text
                ys, xs = np.where(mm)
                if xs.size > 0 and ys.size > 0:
                    x1, y1 = int(xs.min()), int(ys.min())
                    cv2.putText(
                        frame,
                        f"ID {cid}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
        writer.write(frame)
        fidx += 1

    writer.release()
    cap.release()

    # Save per-car masks to compressed NPZ
    # For each car, store: frames_<carid>, masks_<carid> (bool), plus some metadata
    npz_save_dict = {
        "video_path": np.array([video_path]),
        "json_path": np.array([json_path]),
        "height": np.array([height], dtype=np.int32),
        "width": np.array([width], dtype=np.int32),
        "car_ids": np.array([str(cid) for cid in car_ids_sorted]),
        "anchor_frame_json": np.array([anchor_frame], dtype=np.int32),
        "anchor_frame_used": np.array([anchor_used], dtype=np.int32),
    }
    for cid in car_ids_sorted:
        frames_arr = np.array(per_car_frames[cid], dtype=np.int32)
        # Stack along axis 0 if any masks exist, otherwise an empty array with correct shape
        if len(per_car_masks[cid]) > 0:
            masks_arr = np.stack(per_car_masks[cid], axis=0).astype(np.bool_)
        else:
            masks_arr = np.zeros((0, height, width), dtype=np.bool_)
        npz_save_dict[f"frames_{cid}"] = frames_arr
        npz_save_dict[f"masks_{cid}"] = masks_arr

    os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)
    np.savez_compressed(out_npz_path, **npz_save_dict)

    print(f"Saved visualization to: {out_video_path}")
    print(f"Saved per-car masks NPZ to: {out_npz_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Track multiple cars from JSON bboxes and save video + NPZ masks")
    p.add_argument("json_path", help="Path to JSON file containing bboxes and anchor frame")
    p.add_argument("video_path", help="Path to input video (original)")
    p.add_argument("out_video_path", help="Path to output visualization .mp4")
    p.add_argument("out_npz_path", help="Path to output masks .npz")
    p.add_argument("--out-bbox-img", dest="out_bbox_img", default=None, help="Optional path to save initial bbox visualization image")
    p.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for masks")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_from_json(
        args.json_path,
        args.video_path,
        args.out_video_path,
        args.out_npz_path,
        alpha=args.alpha,
        out_bbox_img_path=args.out_bbox_img,
    )
