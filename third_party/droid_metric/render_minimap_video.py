import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_trajectories(traj_path: str) -> Dict[str, np.ndarray]:
    """Load trajectory JSON: frame_id -> 4x4 pose matrix."""
    with open(traj_path, "r") as f:
        data = json.load(f)

    trajectories = {}
    for key, mat in data.items():
        arr = np.array(mat, dtype=np.float32)
        if arr.shape == (4, 4):
            trajectories[key] = arr
    return trajectories


def get_frame_list(rgb_dir: str) -> List[str]:
    """Return sorted list of frame ids (without extension) present in rgb_dir."""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [
        f
        for f in os.listdir(rgb_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]
    files.sort()
    frame_ids = [os.path.splitext(f)[0] for f in files]
    return frame_ids


def compute_2d_positions(
    trajectories: Dict[str, np.ndarray],
    frame_ids: List[str],
) -> Tuple[List[str], np.ndarray]:
    """Extract 2D (x, z) positions for frames that have trajectories."""
    valid_ids: List[str] = []
    positions: List[Tuple[float, float]] = []

    for fid in frame_ids:
        pose = trajectories.get(fid)
        if pose is None or pose.shape != (4, 4):
            continue
        # Use translation components; project to (x, z) plane
        tx = float(pose[0, 3])
        tz = float(pose[2, 3])
        valid_ids.append(fid)
        positions.append((tx, tz))

    if not positions:
        raise ValueError("No valid trajectory entries found for given frames.")

    return valid_ids, np.array(positions, dtype=np.float32)


def build_minimap_base(
    positions: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a base minimap image with the full trajectory polyline drawn.

    Returns:
        base_img: HxWx3 uint8 image with global trajectory drawn.
        points_px: Nx2 int32 array of pixel coordinates corresponding to positions.
    """
    base = np.full((height, width, 3), 255, dtype=np.uint8)

    min_x = float(positions[:, 0].min())
    max_x = float(positions[:, 0].max())
    min_z = float(positions[:, 1].min())
    max_z = float(positions[:, 1].max())

    span_x = max_x - min_x
    span_z = max_z - min_z

    if span_x == 0.0 and span_z == 0.0:
        # All positions identical: map everything to center
        cx = width // 2
        cy = height // 2
        points_px = np.repeat([[cx, cy]], positions.shape[0], axis=0).astype(
            np.int32
        )
        return base, points_px

    # Scale to fit into minimap with margins
    max_span = max(span_x, span_z)
    usable_w = width * 0.8
    usable_h = height * 0.8
    scale = min(usable_w, usable_h) / max_span

    # Center the trajectory within the minimap
    traj_w = span_x * scale
    traj_h = span_z * scale
    margin_x = (width - traj_w) / 2.0
    margin_y = (height - traj_h) / 2.0

    pts: List[Tuple[int, int]] = []
    for x, z in positions:
        px = int((x - min_x) * scale + margin_x)
        # Flip z so that increasing z goes "up" or "down" consistently
        py = int((max_z - z) * scale + margin_y)
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        pts.append((px, py))

    points_px = np.array(pts, dtype=np.int32)

    if len(points_px) > 1:
        poly = points_px.reshape(-1, 1, 2)
        cv2.polylines(
            base,
            [poly],
            isClosed=False,
            color=(180, 180, 180),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # Draw outer border
    cv2.rectangle(
        base,
        (0, 0),
        (width - 1, height - 1),
        color=(0, 0, 0),
        thickness=1,
    )

    return base, points_px


def render_video_with_minimap(
    rgb_dir: str,
    trajectories_path: str,
    output_path: str,
    fps: float,
    minimap_fraction: float,
) -> None:
    trajectories = load_trajectories(trajectories_path)
    frame_ids_all = get_frame_list(rgb_dir)

    valid_frame_ids, positions = compute_2d_positions(
        trajectories, frame_ids_all
    )

    if not valid_frame_ids:
        raise ValueError("No frames with valid trajectories found.")

    # Read first frame to get size
    first_frame_path = None
    for fid in valid_frame_ids:
        candidate = os.path.join(rgb_dir, f"{fid}.png")
        if os.path.exists(candidate):
            first_frame_path = candidate
            break
        for ext in (".jpg", ".jpeg", ".bmp"):
            candidate = os.path.join(rgb_dir, f"{fid}{ext}")
            if os.path.exists(candidate):
                first_frame_path = candidate
                break
        if first_frame_path:
            break

    if first_frame_path is None:
        raise FileNotFoundError("Could not find any RGB frame file.")

    frame0 = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
    if frame0 is None:
        raise RuntimeError(f"Failed to read first frame: {first_frame_path}")

    h, w = frame0.shape[:2]
    minimap_w = int(w * minimap_fraction)
    minimap_h = int(h * minimap_fraction)
    minimap_w = max(64, minimap_w)
    minimap_h = max(64, minimap_h)

    minimap_base, points_px = build_minimap_base(positions, minimap_w, minimap_h)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    print(f"Writing video to {output_path}")
    print(f"Frames: {len(valid_frame_ids)}, size: {w}x{h}, FPS: {fps}")

    for idx, fid in enumerate(valid_frame_ids):
        # Read corresponding frame image
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            candidate = os.path.join(rgb_dir, f"{fid}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"Warning: no image file found for frame {fid}, skipping.")
            continue

        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Warning: failed to read image {img_path}, skipping.")
            continue

        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        # Build minimap for this frame
        minimap = minimap_base.copy()
        px, py = points_px[idx]
        cv2.circle(
            minimap,
            (int(px), int(py)),
            radius=4,
            color=(0, 0, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        # Overlay minimap at top-right
        y_offset = 10
        x_offset = w - minimap_w - 10
        y1 = y_offset
        y2 = y_offset + minimap_h
        x1 = x_offset
        x2 = x_offset + minimap_w

        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
            frame[y1:y2, x1:x2] = minimap

        writer.write(frame)

    writer.release()
    print("Finished writing video.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render RGB video with top-right minimap overlay from trajectories."
    )
    parser.add_argument(
        "--rgb_dir",
        type=str,
        required=True,
        help="Directory containing RGB frames (e.g., 000000.png).",
    )
    parser.add_argument(
        "--traj_json",
        type=str,
        required=True,
        help="Trajectory JSON file (frame_id -> 4x4 pose).",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        required=True,
        help="Output video path (e.g., output_minimap.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frames per second for output video.",
    )
    parser.add_argument(
        "--minimap_fraction",
        type=float,
        default=0.25,
        help="Size of minimap as fraction of frame size.",
    )

    args = parser.parse_args()

    render_video_with_minimap(
        rgb_dir=args.rgb_dir,
        trajectories_path=args.traj_json,
        output_path=args.output_video,
        fps=args.fps,
        minimap_fraction=args.minimap_fraction,
    )


if __name__ == "__main__":
    main()

