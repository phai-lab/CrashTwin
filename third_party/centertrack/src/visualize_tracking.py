#!/usr/bin/env python3
"""
Standalone script to visualize tracking results on video
Reads JSON tracking results and draws bounding boxes with tracking IDs on the input video
Supports both 2D and 3D tracking visualization
"""

import json
import cv2
import numpy as np
import os
import sys
from collections import defaultdict

# Import 3D utils if available
try:
    sys.path.append('/root/CenterTrack/src/lib/utils')
    from ddd_utils import compute_box_3d, project_to_image, draw_box_3d
    HAS_3D = True
except:
    HAS_3D = False
    print("Warning: 3D visualization not available")

def get_color_for_id(track_id, colors_dict):
    """Generate a unique color for each tracking ID"""
    if track_id not in colors_dict:
        # Generate a random but consistent color for each ID
        np.random.seed(track_id * 42)
        colors_dict[track_id] = tuple(map(int, np.random.randint(50, 255, 3)))
    return colors_dict[track_id]

def draw_tracking_box(img, bbox, track_id, color, score=None, cls=None):
    """Draw a single tracking box with ID on the image"""
    # Draw bounding box
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)  # Thicker line
    
    # Prepare text
    text_parts = []
    if track_id is not None:
        text_parts.append(f"ID:{track_id}")
    if cls is not None:
        text_parts.append(f"C:{cls}")
    if score is not None:
        text_parts.append(f"{score:.2f}")
    
    if text_parts:
        text = " ".join(text_parts)
        
        # Get text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(img, 
                     (x1, y1 - text_height - 4),
                     (x1 + text_width + 4, y1),
                     color, -1)
        
        # Draw text
        cv2.putText(img, text,
                   (x1 + 2, y1 - 4),
                   font, font_scale,
                   (255, 255, 255), thickness, cv2.LINE_AA)

def draw_3d_box(img, det, calib_file=None, color=(0, 255, 0), focal_length=1200):
    """Draw 3D bounding box if 3D information is available"""
    if not HAS_3D:
        return False
    
    # Check if we have 3D information
    if 'dim' not in det or 'loc' not in det or 'rot_y' not in det:
        return False

    # h, w = img.shape[:2]
    # raise Exception(f"focal_length: {focal_length}, w: {w}, h: {h}")
    
    try:
        # Get 3D box corners
        dim = np.array(det['dim'])
        loc = np.array(det['loc'])
        rot_y = det['rot_y']
        
        # Create proper 3x4 projection matrix using provided focal length
        h, w = img.shape[:2]
        # raise Exception(f"focal_length: {focal_length}, w: {w}, h: {h}")
        calib = np.array([[focal_length, 0, w/2, 0],    # fx, 0, cx, tx
                         [0, focal_length, h/2, 0],      # 0, fy, cy, ty  
                         [0, 0, 1, 0]])                  # 0, 0, 1, tz
        
        # Compute 3D box corners
        box_3d = compute_box_3d(dim, loc, rot_y)
        box_2d = project_to_image(box_3d, calib)
        
        # Draw the 3D box
        img = draw_box_3d(img, box_2d, color)
        return True
    except Exception as e:
        # Comment out error printing to avoid spam
        # print(f"Error drawing 3D box: {e}")
        return False

def visualize_tracking(json_path, video_path, output_path=None, calib_file=None, class_filter=None, focal_length=None, resize_width=None, resize_height=None):
    """Main function to visualize tracking results on video
    
    Args:
        json_path: Path to JSON tracking results
        video_path: Path to input video
        output_path: Path to output video (optional)
        calib_file: Camera calibration file (optional)
        class_filter: List of class IDs to show, None means show all
        focal_length: Camera focal length (use provided value, no auto-detection)
        resize_width: Target width for video resizing (optional)
        resize_height: Target height for video resizing (optional)
    """
    
    # Define class names for common datasets
    # COCO classes: car=3, motorcycle=4, bus=6, truck=8
    # NuScenes classes: car=1, truck=2, bus=3, trailer=4, construction_vehicle=5
    # KITTI classes: Car=1, Van=2, Truck=3
    
    # Load tracking results from JSON
    print(f"Loading tracking results from: {json_path}")
    with open(json_path, 'r') as f:
        tracking_results = json.load(f)
    
    if class_filter:
        print(f"Filtering for classes: {class_filter}")
    
    # Open input video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original video: {orig_width}x{orig_height} @ {fps}fps, {total_frames} frames")
    
    resize_width = 1920
    resize_height = 1080
    # Use resize dimensions if provided, otherwise use original
    if resize_width is not None and resize_height is not None:
        width, height = resize_width, resize_height
        print(f"Original video: {orig_width}x{orig_height} @ {fps}fps, {total_frames} frames")
        print(f"Resizing to: {width}x{height}")
    else:
        width, height = orig_width, orig_height
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Use provided focal length (no auto-detection)
    if focal_length is not None:
        print(f"Using provided focal length: {focal_length}")
    else:
        # Default focal length if none provided
        focal_length = 1200
        print(f"Using default focal length: {focal_length}")
    
    # Setup output video
    if output_path is None:
        output_path = video_path.replace('.mp4', '_tracked.mp4')
    
    # Use mp4v codec which is widely supported for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Saving output to: {output_path}")
    
    # Color dictionary for consistent colors per track ID
    colors = {}
    
    # Process each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize frame if required
        if resize_width is not None and resize_height is not None:
            frame = cv2.resize(frame, (resize_width, resize_height))
        
        # Get tracking results for this frame
        frame_key = str(frame_count)
        if frame_key in tracking_results:
            detections = tracking_results[frame_key]
            
            # Draw each detection
            for det in detections:
                track_id = det.get('tracking_id', None)
                score = det.get('score', None)
                cls = det.get('class', None)
                
                # Skip if class filter is set and this class is not in the filter
                if class_filter is not None and cls not in class_filter:
                    continue
                
                # Get consistent color for this track ID
                if track_id is not None:
                    color = get_color_for_id(track_id, colors)
                else:
                    color = (0, 255, 0)  # Default green for no ID
                
                # Try to draw 3D box first if available
                drew_3d = draw_3d_box(frame, det, calib_file, color, focal_length)
                
                # If no 3D or 3D failed, draw 2D box
                if not drew_3d and 'bbox' in det:
                    bbox = det['bbox']
                    draw_tracking_box(frame, bbox, track_id, color, score, cls)
                
                # Always draw tracking ID at center point if available
                if 'ct' in det and track_id is not None:
                    ct = det['ct']
                    cv2.putText(frame, f"ID:{track_id}", 
                               (int(ct[0]), int(ct[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               color, 2, cv2.LINE_AA)
        
        # Write frame to output video
        out.write(frame)
        
        # Show progress
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
    
    # Clean up
    cap.release()
    out.release()
    # cv2.destroyAllWindows()  # Not supported in this environment
    
    print(f"Done! Output saved to: {output_path}")
    return True

def main():
    """Main entry point"""
    # Default paths
    json_path = '/root/CenterTrack/results/default_nuscenes_mini.mp4_results.json'
    video_path = '/root/CenterTrack/videos/nuscenes_mini.mp4'
    output_path = '/root/CenterTrack/results/nuscenes_mini_tracked.mp4'
    
    # Class filters for different datasets
    # NuScenes: car=1, truck=2, bus=3, trailer=4, construction_vehicle=5, 
    #           pedestrian=6, motorcycle=7, bicycle=8, traffic_cone=9, barrier=10
    # COCO: car=3, motorcycle=4, bus=6, truck=8
    
    # Default: only show cars for NuScenes
    vehicle_classes = [1]  # NuScenes car class
    
    # Parse parameters
    focal_length = None
    resize_width = None
    resize_height = None
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    if len(sys.argv) > 2:
        video_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    if len(sys.argv) > 4:
        # Parse class filter from command line
        if sys.argv[4] == 'cars':
            vehicle_classes = [1]  # Only cars for NuScenes
        elif sys.argv[4] == 'vehicles':
            vehicle_classes = [1, 2, 3, 4, 5]  # All vehicle types in NuScenes
        elif sys.argv[4] == 'all':
            vehicle_classes = None  # Show all classes
        else:
            # Parse custom class list
            try:
                vehicle_classes = [int(x) for x in sys.argv[4].split(',')]
            except:
                print(f"Invalid class filter: {sys.argv[4]}")
                print("Use 'cars', 'vehicles', 'all', or comma-separated class IDs")
                print("NuScenes classes: 1=car, 2=truck, 3=bus, 4=trailer, 5=construction_vehicle")
                print("                   6=pedestrian, 7=motorcycle, 8=bicycle, 9=traffic_cone, 10=barrier")
                sys.exit(1)
    
    if len(sys.argv) > 5:
        # Parse focal length
        try:
            focal_length = float(sys.argv[5])
        except:
            print(f"Invalid focal length: {sys.argv[5]}")
            print("Usage: python visualize_tracking.py [json] [video] [output] [class_filter] [focal_length] [width] [height]")
            sys.exit(1)
    
    if len(sys.argv) > 6:
        # Parse resize width
        try:
            resize_width = int(sys.argv[6])
        except:
            print(f"Invalid width: {sys.argv[6]}")
            print("Usage: python visualize_tracking.py [json] [video] [output] [class_filter] [focal_length] [width] [height]")
            sys.exit(1)
    
    if len(sys.argv) > 7:
        # Parse resize height
        try:
            resize_height = int(sys.argv[7])
        except:
            print(f"Invalid height: {sys.argv[7]}")
            print("Usage: python visualize_tracking.py [json] [video] [output] [class_filter] [focal_length] [width] [height]")
            sys.exit(1)
    
    # Check if files exist
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Run visualization with class filter, focal length, and resize parameters
    success = visualize_tracking(json_path, video_path, output_path, 
                                class_filter=vehicle_classes, focal_length=focal_length,
                                resize_width=resize_width, resize_height=resize_height)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()