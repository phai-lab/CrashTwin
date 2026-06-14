#!/usr/bin/env python3
"""
Visualize Kalman filtered trajectory results

This script reads the saved Kalman filtered detection results and visualizes
only the final trajectories without ego vehicle or bounding boxes.
"""

import json
import numpy as np
import cv2
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys
import os
import argparse

# Import from the main trajectory reconstruction file
sys.path.append('.')
# from traj_recon import read_detection_results
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



def extract_trajectories_from_results(results_path: str) -> Tuple[Dict[int, List[np.ndarray]], Dict[str, List]]:
    """
    Extract trajectory data and detection info from saved detection results JSON

    Args:
        results_path: Path to detection results JSON file

    Returns:
        Tuple of (trajectories_by_frame, sorted_frame_ids)
        - trajectories_by_frame: Dictionary mapping frame_id -> list of detections
        - sorted_frame_ids: Sorted list of frame IDs
    """
    with open(results_path, 'r') as f:
        results_data = json.load(f)

    print(f"Loading trajectories from {results_path}...")

    # Sort frame IDs numerically
    sorted_frame_ids = sorted(results_data.keys(), key=lambda x: int(x))

    # Build trajectory data for each tracking ID
    trajectories = {}

    for frame_id in sorted_frame_ids:
        detections = results_data[frame_id]
        for det in detections:
            tracking_id = det.get('tracking_id', -1)
            if tracking_id >= 0:
                if tracking_id not in trajectories:
                    trajectories[tracking_id] = []
                trajectories[tracking_id].append({
                    'frame_id': frame_id,
                    'position': np.array(det['loc']),
                    'detection': det
                })

    print(f"Extracted {len(trajectories)} trajectories")
    for tid, traj_points in trajectories.items():
        frame_ids = [p['frame_id'] for p in traj_points]
        print(f"  Trajectory ID {tid}: {len(traj_points)} points (frames {frame_ids[0]}-{frame_ids[-1]})")

    return trajectories, results_data, sorted_frame_ids


def create_3d_bbox_from_detection(det: Dict) -> np.ndarray:
    """
    Create 3D bounding box corners from detection info using CenterTrack's method

    Args:
        det: Detection dictionary with 'loc', 'dim', 'rot_y' fields

    Returns:
        8x3 array of 3D bbox corners
    """
    loc = np.array(det['loc'])      # [x, y, z] center position
    dim = np.array(det['dim'])      # [h, w, l] dimensions
    rot_y = det['rot_y']            # rotation around Y axis

    # Use CenterTrack's dimension order: l, w, h = dim[2], dim[1], dim[0]
    h, w, l = dim[0], dim[1], dim[2]

    # Create 8 corners using CenterTrack's coordinate system
    # Note: Y coordinates go from 0 (bottom) to -h (top) in CenterTrack
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

    # Rotation matrix around Y axis (same as CenterTrack)
    c, s = np.cos(rot_y), np.sin(rot_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    # Rotate corners and translate to world position
    rotated_corners = np.dot(R, corners).transpose(1, 0)
    bbox_3d = rotated_corners + loc.reshape(1, 3)

    return bbox_3d


def draw_3d_bbox_bev(img: np.ndarray,
                     bbox_3d: np.ndarray,
                     center: Tuple[int, int],
                     scale: float,
                     color: Tuple[int, int, int],
                     thickness: int = 2) -> np.ndarray:
    """
    Draw 3D bounding box in bird's eye view

    Args:
        img: Image to draw on
        bbox_3d: 8x3 array of 3D bbox corners
        center: Image center point (x, y)
        scale: Pixels per meter
        color: BGR color
        thickness: Line thickness

    Returns:
        Image with bbox drawn
    """
    # Convert 3D coordinates to 2D bird's eye view (X-Z plane)
    bbox_2d = []
    for corner in bbox_3d:
        px = int(corner[0] * scale + center[0])
        py = int(-corner[2] * scale + center[1])  # Negative Z for image coordinates
        bbox_2d.append([px, py])
    bbox_2d = np.array(bbox_2d, np.int32)

    # Draw bottom face (first 4 corners)
    cv2.polylines(img, [bbox_2d[:4].reshape((-1, 1, 2))], True, color, thickness)

    # Draw top face (last 4 corners)
    cv2.polylines(img, [bbox_2d[4:].reshape((-1, 1, 2))], True, color, thickness)

    # Draw vertical edges
    for i in range(4):
        cv2.line(img, tuple(bbox_2d[i]), tuple(bbox_2d[i+4]), color, thickness)

    # Draw direction indicator (front face)
    front_center = ((bbox_2d[1] + bbox_2d[2]) // 2).astype(int)
    cv2.circle(img, tuple(front_center), 4, color, -1)

    return img


def visualize_kalman_trajectories(trajectories: Dict[int, List],
                                 results_data: Dict = None,
                                 sorted_frame_ids: List[str] = None,
                                 output_video: str = 'kalman_trajectories.mp4',
                                 fps: int = 10,
                                 img_size: Tuple[int, int] = (1920, 1080),
                                 scale: float = 12.0):
    """
    Visualize Kalman filtered trajectories (simplified version)

    Args:
        trajectories: Dictionary mapping tracking_id -> list of positions
        output_video: Output video file path
        fps: Frames per second
        img_size: Image size (width, height)
        scale: Pixels per meter for visualization
    """
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, img_size)

    center = (img_size[0] // 2, int(img_size[1] * 0.9))  # 70% down from top

    # Total frames is based on sorted_frame_ids
    num_frames = len(sorted_frame_ids) if sorted_frame_ids else 0

    print(f"Creating visualization with {num_frames} frames...")

    # Define colors for different trajectories
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (255, 200, 0), (200, 0, 255),
              (100, 255, 100), (255, 100, 100), (100, 100, 255)]

    # Store trails for visualization
    object_trails = {}

    for frame_idx, frame_id in enumerate(tqdm(sorted_frame_ids, desc="Rendering")):
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
        cv2.putText(img, "X (world)", (center[0] + 90, center[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, "Z (world)", (center[0] + 5, center[1] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Get current frame detections
        current_detections = results_data.get(frame_id, [])

        # Update trails with current detections
        for det in current_detections:
            tracking_id = det.get('tracking_id', -1)
            if tracking_id >= 0:
                position = np.array(det['loc'])
                if tracking_id not in object_trails:
                    object_trails[tracking_id] = []
                object_trails[tracking_id].append(position)

        # Draw all trails
        active_trajectories = 0
        for traj_idx, (tracking_id, trail) in enumerate(object_trails.items()):
            color = colors[traj_idx % len(colors)]

            if len(trail) > 1:
                active_trajectories += 1
                # Draw trajectory line
                for i in range(1, len(trail)):
                    pt1 = trail[i-1]
                    pt2 = trail[i]

                    # Convert to pixel coordinates (bird's eye view: X-Z plane)
                    px1 = int(pt1[0] * scale + center[0])
                    py1 = int(-pt1[2] * scale + center[1])
                    px2 = int(pt2[0] * scale + center[0])
                    py2 = int(-pt2[2] * scale + center[1])

                    # Draw thick trajectory line
                    cv2.line(img, (px1, py1), (px2, py2), color, 4)

        # Draw current frame bounding boxes
        for det in current_detections:
            tracking_id = det.get('tracking_id', -1)
            if tracking_id >= 0:
                color = colors[tracking_id % len(colors)]

                # Create and draw 3D bounding box
                bbox_3d = create_3d_bbox_from_detection(det)
                img = draw_3d_bbox_bev(img, bbox_3d, center, scale, color, thickness=2)

                # Draw trajectory ID on the bounding box
                bbox_center_2d = np.mean(bbox_3d[:4], axis=0)  # Center of bottom face
                px = int(bbox_center_2d[0] * scale + center[0])
                py = int(-bbox_center_2d[2] * scale + center[1])
                cv2.putText(img, f"ID{tracking_id}", (px + 15, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add information text
        info_texts = [
            f"Frame: {frame_id} ({frame_idx+1}/{num_frames})",
            f"Active trajectories: {len(current_detections)}",
            f"Total trajectories: {len(object_trails)}",
            "Kalman filtered results",
            f"Scale: 1 grid square = {grid_spacing/scale:.1f}m"
        ]

        y_offset = 30
        for i, text in enumerate(info_texts):
            cv2.putText(img, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Write frame to video
        out.write(img)

    # Release video writer
    out.release()
    print(f"✓ Kalman trajectory video saved to: {output_video}")

    # Save last frame as preview
    preview_path = output_video.replace('.mp4', '_preview.png')
    cv2.imwrite(preview_path, img)
    print(f"✓ Preview image saved to: {preview_path}")


def main():
    """
    Main function to visualize trajectory results
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize Kalman filtered trajectory results')
    parser.add_argument('--video_id', type=str, default='VV_76',
                       help='Video ID to process (default: VV_76)')
    parser.add_argument('--results_path', type=str, default=None,
                       help='Path to results JSON file (default: auto-generated from video_id)')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Output name prefix (default: kalman_smoothed_{video_id})')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output directory path (default: current directory)')
    args = parser.parse_args()

    # Determine results path and output name
    if args.results_path:
        results_path = args.results_path
        output_name = args.output_name if args.output_name else os.path.splitext(os.path.basename(results_path))[0]
    else:
        results_path = f'/root/CenterTrack/results/kalman_smoothed_{args.video_id}.mp4_results.json'
        output_name = args.output_name if args.output_name else f'kalman_smoothed_{args.video_id}'

    print("=" * 70)
    print("TRAJECTORY VISUALIZATION")
    print("=" * 70)
    print(f"Input file: {results_path}")

    if not os.path.exists(results_path):
        print(f"Error: Results file not found: {results_path}")
        print("\nUsage: python visualize_kalman_results.py --video_id VIDEO_ID")
        print("       python visualize_kalman_results.py --results_path PATH_TO_RESULTS.json")
        print("\nAvailable files in results directory:")
        results_dir = '/root/CenterTrack/results/'
        if os.path.exists(results_dir):
            for f in os.listdir(results_dir):
                if f.endswith('.json'):
                    print(f"  {results_dir}{f}")
        return

    # Extract trajectories and frame data from results
    trajectories, results_data, sorted_frame_ids = extract_trajectories_from_results(results_path)

    if not trajectories:
        print("No trajectories found in results!")
        return

    # Create visualization with bounding boxes
    output_filename = f'{output_name}_trajectories_with_bbox.mp4'
    if args.output_path:
        output_video = os.path.join(args.output_path, output_filename)
        os.makedirs(args.output_path, exist_ok=True)
    else:
        output_video = output_filename

    print(f"Output video path: {output_video}")

    visualize_kalman_trajectories(trajectories,
                                 results_data=results_data,
                                 sorted_frame_ids=sorted_frame_ids,
                                 output_video=output_video,
                                 fps=10,
                                 scale=12.0)

    print(f"\n✓ Visualization complete!")
    print(f"  Output video: {output_video}")
    print(f"  Preview image: {output_video.replace('.mp4', '_preview.png')}")
    if args.output_path:
        print(f"  Files saved to: {args.output_path}")


if __name__ == "__main__":
    main()