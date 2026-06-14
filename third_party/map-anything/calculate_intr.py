#!/usr/bin/env python3
# Optional config for better memory efficiency
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import sys
import argparse
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate average intrinsics from images')
parser.add_argument('--image_dir', type=str, required=True, help='Path to directory containing RGB images')
parser.add_argument('--output_file', type=str, required=True, help='Path to output intrinsics file')
args = parser.parse_args()

# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init model - This requries internet access or the huggingface hub cache to be pre-downloaded
# For Apache 2.0 license model, use "facebook/map-anything-apache"
model = MapAnything.from_pretrained("facebook/map-anything").to(device)

# Load and preprocess images from a folder or list of paths
images = args.image_dir

views = load_images(images)

import glob
from PIL import Image
img_dir = images
exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
img_paths = []
for ext in exts:
    img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
img_paths = sorted(img_paths)
assert len(img_paths) > 0, f"No images found in {img_dir}"

# 读取每张图的原始尺寸 (W_orig, H_orig)
orig_sizes = []
for p in img_paths:
    with Image.open(p) as im:
        w, h = im.size
        orig_sizes.append((w, h))

# 确认每张图的原始尺寸都是一样的
if not all(s == orig_sizes[0] for s in orig_sizes):
    raise ValueError("All images must have the same original size")
W_orig, H_orig = orig_sizes[0]
print(f"Original image size: {W_orig}x{H_orig}")

# Use deterministic uniform sampling for reproducible benchmark evaluation.
num_samples = min(16, len(views))
sampled = np.linspace(0, len(views) - 1, num_samples, dtype=int).tolist()
views = [views[i] for i in sampled]
# raise NotImplementedError(len(views), type(views))




# Run inference
predictions = model.infer(
    views,                            # Input views
    memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
    use_amp=True,                     # Use mixed precision inference (recommended)
    amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
    apply_mask=True,                  # Apply masking to dense geometry outputs
    mask_edges=True,                  # Remove edge artifacts by using normals and depth
    apply_confidence_mask=False,      # Filter low-confidence regions
    confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
)

# Access results for each view - Complete list of metric outputs
for i, pred in enumerate(predictions):
    # # Geometry outputs
    # pts3d = pred["pts3d"]                     # 3D points in world coordinates (B, H, W, 3)
    # pts3d_cam = pred["pts3d_cam"]             # 3D points in camera coordinates (B, H, W, 3)
    depth_z = pred["depth_z"]                 # Z-depth in camera frame (B, H, W, 1)
    # depth_along_ray = pred["depth_along_ray"] # Depth along ray in camera frame (B, H, W, 1)

    # # Camera outputs
    # ray_directions = pred["ray_directions"]   # Ray directions in camera frame (B, H, W, 3)
    intrinsics = pred["intrinsics"]           # Recovered pinhole camera intrinsics (B, 3, 3)
    # camera_poses = pred["camera_poses"]       # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
    # cam_trans = pred["cam_trans"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world translation in world frame (B, 3)
    # cam_quats = pred["cam_quats"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world quaternion in world frame (B, 4)

    # # Quality and masking
    # confidence = pred["conf"]                 # Per-pixel confidence scores (B, H, W)
    # mask = pred["mask"]                       # Combined validity mask (B, H, W, 1)
    # non_ambiguous_mask = pred["non_ambiguous_mask"]                # Non-ambiguous regions (B, H, W)
    # non_ambiguous_mask_logits = pred["non_ambiguous_mask_logits"]  # Mask logits (B, H, W)

    # # Scaling
    # metric_scaling_factor = pred["metric_scaling_factor"]  # Applied metric scaling (B,)

    # # Original input
    img_no_norm = pred["img_no_norm"]         # Denormalized input images for visualization (B, H, W, 3)


    H_res, W_res = int(img_no_norm.shape[1]), int(img_no_norm.shape[2])

    W_orig, H_orig = orig_sizes[i]

    # 计算从“模型输入”到“原始图像”的缩放比例
    sx = W_orig / float(W_res)
    sy = H_orig / float(H_res)

    # 假设 intr 是标准针孔内参：
    # [ [fx,  0, cx],
    #   [ 0, fy, cy],
    #   [ 0,  0,  1] ]
    print("Original intrinsics:\n", intrinsics.shape)
    intr_scaled = intrinsics[0].cpu().numpy().copy()
    intr_scaled[0, 0] *= sx  # fx
    intr_scaled[1, 1] *= sy  # fy
    intr_scaled[0, 2] *= sx  # cx
    intr_scaled[1, 2] *= sy  # cy

    predictions[i]["intrinsics"] = intr_scaled




    print(f"View {i}: Intrinsics:\n", intr_scaled)
    # print(f"View {i}: Depth shape: ", depth_z.shape)
    # print(f"View {i}: Input image shape: ", img_no_norm.shape)
    # print(f"View {i}: Keys: ", pred.keys())

# average the intrinsics
import numpy as np
intrinsics_all = np.array([pred["intrinsics"] for pred in predictions])
intrinsics_avg = np.mean(intrinsics_all, axis=0)
print("Average intrinsics:\n", intrinsics_avg)

# Write to output file specified in command line
with open(args.output_file, 'w') as f:
    f.write(f"{(intrinsics_avg[0, 0] + intrinsics_avg[1, 1])/2}\n")  # fx
    f.write(f"{(intrinsics_avg[1, 1] + intrinsics_avg[0, 0])/2}\n")  # fy
    f.write(f"{intrinsics_avg[0, 2]}\n")  # cx
    f.write(f"{intrinsics_avg[1, 2]}\n")  # cy
