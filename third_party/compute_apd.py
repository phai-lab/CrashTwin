#!/usr/bin/env python3
"""
Compute APD metric for SAM2-generated vehicle masks using CLIP embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image

AREA_THRESH: float = 0.0005
MAX_FRAMES_PER_CAR: int = 64
MODEL_NAME: str = "openclip_ViT-B-32_laion2b_s34b_b79k"
_MODEL_VARIANT: str = "ViT-B-32"
_MODEL_PRETRAINED: str = "laion2b_s34b_b79k"

_NORMALIZE_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
_NORMALIZE_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _natural_key(path: Path) -> Tuple:
    tokens = re.split(r"(\d+)", path.name)
    key: List = []
    for token in tokens:
        if token.isdigit():
            key.append(int(token))
        elif token:
            key.append(token)
    if not key:
        key.append(path.name)
    return tuple(key)


def _list_frame_paths(frames_dir: Path) -> List[Path]:
    frame_paths = [
        p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]
    return sorted(frame_paths, key=_natural_key)


def _load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _preprocess_crop(image: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Mask has no positive pixels.")

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop[~crop_mask] = 0

    cropped_img = Image.fromarray(crop)
    resized = cropped_img.resize((224, 224), resample=Image.BICUBIC)

    tensor = torch.from_numpy(np.array(resized)).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - _NORMALIZE_MEAN) / _NORMALIZE_STD
    return tensor


def _prepare_embeddings(
    model: torch.nn.Module,
    device: torch.device,
    frames: Sequence[int],
    masks: np.ndarray,
    frame_paths: Sequence[Path],
    image_cache: Dict[int, np.ndarray],
) -> Tuple[List[torch.Tensor], List[int], int]:
    valid_pairs: List[Tuple[int, np.ndarray]] = []
    total_pixels = masks.shape[1] * masks.shape[2]

    for idx_in_car, frame_idx in enumerate(frames):
        mask = masks[idx_in_car]
        area_ratio = float(mask.sum()) / float(total_pixels)
        if area_ratio < AREA_THRESH:
            continue
        if frame_idx < 0 or frame_idx >= len(frame_paths):
            continue
        if frame_idx not in image_cache:
            image_cache[frame_idx] = _load_image_rgb(frame_paths[frame_idx])
        valid_pairs.append((frame_idx, mask))

    pre_sample_count = len(valid_pairs)
    if pre_sample_count < 2:
        return [], [], pre_sample_count

    if len(valid_pairs) > MAX_FRAMES_PER_CAR:
        indices = np.linspace(0, len(valid_pairs) - 1, MAX_FRAMES_PER_CAR, dtype=int)
        valid_pairs = [valid_pairs[i] for i in indices]

    embeddings: List[torch.Tensor] = []
    frame_indices: List[int] = []

    with torch.no_grad():
        for frame_idx, mask in valid_pairs:
            image = image_cache[frame_idx]
            tensor = _preprocess_crop(image, mask).unsqueeze(0).to(device)
            features = model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.squeeze(0).cpu())
            frame_indices.append(int(frame_idx))

    return embeddings, frame_indices, pre_sample_count


def _car_apd(embeddings: Sequence[torch.Tensor]) -> Tuple[Optional[float], List[float], List[float]]:
    if len(embeddings) < 2:
        return None, [], []

    angles: List[float] = []
    cosines: List[float] = []
    for left, right in zip(embeddings[:-1], embeddings[1:]):
        dot = float(torch.dot(left, right))
        dot = max(-1.0, min(1.0, dot))
        cosines.append(dot)
        angles.append(math.acos(dot))

    if not angles:
        return None, [], []
    return float(sum(angles) / len(angles)), angles, cosines


def compute_apd(frames_dir: str, npz_path: str, out_json: str) -> None:
    frames_path = Path(frames_dir)
    npz_file = Path(npz_path)
    out_path = Path(out_json)

    if not frames_path.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_path}")
    if not npz_file.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")

    frame_paths = _list_frame_paths(frames_path)
    if not frame_paths:
        raise RuntimeError(f"No image frames found under: {frames_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(
        _MODEL_VARIANT, pretrained=_MODEL_PRETRAINED
    )
    model = model.eval().to(device)

    total_weight = 0.0
    weighted_sum = 0.0
    image_cache: Dict[int, np.ndarray] = {}
    car_details: List[Dict[str, object]] = []
    skipped_cars: List[Dict[str, object]] = []

    with np.load(npz_file, allow_pickle=True) as data:
        car_ids = data["car_ids"]
        for raw_cid in car_ids:
            car_id = str(raw_cid)
            frames_key = f"frames_{car_id}"
            masks_key = f"masks_{car_id}"

            if frames_key not in data or masks_key not in data:
                skipped_cars.append(
                    {
                        "car_id": car_id,
                        "reason": "missing_keys",
                    }
                )
                continue

            frames = data[frames_key]
            masks = data[masks_key]
            embeddings, frame_indices, filtered_count = _prepare_embeddings(
                model=model,
                device=device,
                frames=frames,
                masks=masks,
                frame_paths=frame_paths,
                image_cache=image_cache,
            )
            original_count = int(len(frames))
            if filtered_count < 2:
                skipped_cars.append(
                    {
                        "car_id": car_id,
                        "reason": "insufficient_valid_frames",
                        "original_frame_count": original_count,
                        "valid_after_area": int(filtered_count),
                    }
                )
                continue
            if not embeddings:
                skipped_cars.append(
                    {
                        "car_id": car_id,
                        "reason": "embedding_generation_failed",
                        "original_frame_count": original_count,
                        "valid_after_area": int(filtered_count),
                    }
                )
                continue

            apd_value, angles, cosines = _car_apd(embeddings)
            if apd_value is None:
                skipped_cars.append(
                    {
                        "car_id": car_id,
                        "reason": "apd_computation_failed",
                        "original_frame_count": original_count,
                        "valid_after_area": int(filtered_count),
                        "sampled_frames": int(len(frame_indices)),
                    }
                )
                continue

            sampled_count = int(len(frame_indices))
            car_weight = float(sampled_count)
            total_weight += car_weight
            weighted_sum += car_weight * apd_value
            car_details.append(
                {
                    "car_id": car_id,
                    "apd": apd_value,
                    "weight": car_weight,
                    "weight_contribution": car_weight * apd_value,
                    "original_frame_count": original_count,
                    "valid_after_area": int(filtered_count),
                    "sampled_frames": sampled_count,
                    "frame_indices": [int(idx) for idx in frame_indices],
                    "pairwise_angles": angles,
                    "pairwise_cosines": cosines,
                }
            )

    video_apd = weighted_sum / total_weight if total_weight > 0 else 0.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "apd": video_apd,
                "total_weight": total_weight,
                "weighted_sum": weighted_sum,
                "cars": car_details,
                "skipped_cars": skipped_cars,
                "config": {
                    "area_thresh": AREA_THRESH,
                    "max_frames_per_car": MAX_FRAMES_PER_CAR,
                    "model_name": MODEL_NAME,
                },
            },
            f,
        )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute APD for SAM2 tracks using CLIP embeddings.",
        usage="%(prog)s <frames_dir> <npz_path> <out_json>",
    )
    parser.add_argument("frames_dir", help="Directory containing RGB frames.")
    parser.add_argument("npz_path", help="SAM2 mask NPZ file.")
    parser.add_argument("out_json", help='Output JSON file (contains {"apd": float}).')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    compute_apd(args.frames_dir, args.npz_path, args.out_json)


if __name__ == "__main__":
    main()
