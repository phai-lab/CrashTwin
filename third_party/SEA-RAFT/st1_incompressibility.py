import os
import json
import argparse
from typing import Dict, List, Tuple, Optional, Iterable
import importlib.util
from importlib.machinery import SourceFileLoader
import sys

import cv2
import numpy as np

# progress bar helper (tqdm if available, otherwise a lightweight fallback)
try:
    from tqdm import tqdm as _tqdm  # type: ignore

    def pbar(iterable: Iterable, desc: str = "", total: Optional[int] = None, disable: bool = False):
        return _tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, disable=disable)

except Exception:  # pragma: no cover
    def pbar(iterable: Iterable, desc: str = "", total: Optional[int] = None, disable: bool = False):
        if disable:
            for x in iterable:
                yield x
            return
        try:
            n = len(iterable)  # type: ignore
        except Exception:
            n = None
        if n is None:
            for i, x in enumerate(iterable, 1):
                if i % 50 == 0:
                    print(f"{desc} {i} steps...", flush=True)
                yield x
        else:
            step = max(1, n // 10)
            for i, x in enumerate(iterable, 1):
                if i == 1 or i % step == 0 or i == n:
                    pct = int(i / n * 100)
                    print(f"{desc} {i}/{n} ({pct}%)", flush=True)
                yield x


_TEMPORAL_MODULE = None


def _load_temporal_module():
    global _TEMPORAL_MODULE
    if _TEMPORAL_MODULE is not None:
        return _TEMPORAL_MODULE
    here = os.path.dirname(os.path.abspath(__file__))
    sea_path = os.path.join(here, "temporal_warping_error.py")
    if not os.path.exists(sea_path):
        raise RuntimeError(f"SEA-RAFT temporal_warping_error.py not found at {sea_path}")
    module_name = "searaft_temporal_warp"
    if module_name in sys.modules:
        _TEMPORAL_MODULE = sys.modules[module_name]
        return _TEMPORAL_MODULE
    loader = SourceFileLoader(module_name, sea_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError("Failed to create import spec for SEA-RAFT module")
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)  # type: ignore
    _TEMPORAL_MODULE = mod
    return mod


def load_sam2_masks(npz_path: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], Optional[str], Tuple[int, int]]:
    d = np.load(npz_path, allow_pickle=True)
    h = int(d["height"][0]) if "height" in d else None
    w = int(d["width"][0]) if "width" in d else None
    video_path = None
    if "video_path" in d:
        vp = d["video_path"][0]
        video_path = str(vp)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    ids = d["car_ids"] if "car_ids" in d else []
    for i in ids:
        key = str(i)
        frames_key = f"frames_{key}"
        masks_key = f"masks_{key}"
        if frames_key in d and masks_key in d:
            out[key] = {
                "frames": d[frames_key],
                "masks": d[masks_key],
            }
    return out, video_path, (h, w)


def get_video_from_npz(npz_path: str, npz_video: Optional[str]) -> Optional[str]:
    if npz_video and os.path.exists(npz_video):
        return npz_video
    base_dir = os.path.dirname(npz_path)
    base_name = os.path.basename(npz_path).replace("_sam2_masks.npz", "")
    cand = os.path.join(base_dir, base_name + ".mp4")
    if os.path.exists(cand):
        return cand
    return None


def read_frame(cap: cv2.VideoCapture, index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _compute_flow_cpu(prev_gray: np.ndarray, next_gray: np.ndarray, method: str) -> np.ndarray:
    if method == "tvl1":
        if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "createOptFlow_DualTVL1"):
            tvl1 = cv2.optflow.createOptFlow_DualTVL1()
            flow = tvl1.calc(prev_gray, next_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    else:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow.astype(np.float32)


def _compute_flow_cuda(prev_gray: np.ndarray, next_gray: np.ndarray, method: str) -> Optional[np.ndarray]:
    try:
        if not hasattr(cv2, "cuda"):
            return None
        if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            return None
        # upload
        gprev = cv2.cuda_GpuMat()
        gnxt = cv2.cuda_GpuMat()
        gprev.upload(prev_gray)
        gnxt.upload(next_gray)
        flow_gpu = None
        if method == "cuda_farneback":
            ctor = getattr(cv2, "cuda_FarnebackOpticalFlow", None)
            if ctor is None or not hasattr(ctor, "create"):
                return None
            of = ctor.create(5, 0.5, False, 15, 3, 5, 1.1, 0)
            flow_gpu = of.calc(gprev, gnxt, None)
        elif method == "cuda_tvl1":
            ctor = getattr(cv2, "cuda_OpticalFlowDual_TVL1", None)
            if ctor is None or not hasattr(ctor, "create"):
                return None
            of = ctor.create()
            flow_gpu = of.calc(gprev, gnxt, None)
        elif method == "cuda_brox":
            ctor = getattr(cv2, "cuda_BroxOpticalFlow", None)
            if ctor is None or not hasattr(ctor, "create"):
                return None
            # parameters: alpha, gamma, scale_factor, inner_iterations, outer_iterations, solver_iterations
            of = ctor.create(0.197, 50.0, 0.8, 10, 77, 10)
            flow_gpu = of.calc(gprev, gnxt, None)
        if flow_gpu is None:
            return None
        flow = flow_gpu.download()
        return flow.astype(np.float32)
    except Exception:
        return None


def compute_flow(prev_bgr: np.ndarray, next_bgr: np.ndarray, method: str = "farneback") -> np.ndarray:
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    if method.startswith("cuda_"):
        flow = _compute_flow_cuda(prev, nxt, method)
        if flow is not None:
            return flow
        # fallback to CPU equivalent
        method = method.replace("cuda_", "")
        return _compute_flow_cpu(prev, nxt, method)
    else:
        return _compute_flow_cpu(prev, nxt, method)


def _load_searaft_estimator(cfg_json: Optional[str], ckpt_path: Optional[str]):
    """Dynamically import SEARAFTEstimator from SEA-RAFT/temporal_warping_error.py.

    This avoids duplicating RAFT wiring and ensures consistency with your existing code.
    """
    mod = _load_temporal_module()
    Est = getattr(mod, "SEARAFTEstimator", None)
    if Est is None:
        raise RuntimeError("SEARAFTEstimator not found in SEA-RAFT module")
    # Provide defaults similar to SEA-RAFT main if not supplied
    here = os.path.dirname(os.path.abspath(__file__))
    if not cfg_json:
        cfg_json = os.path.join(here, "SEA-RAFT", "config", "eval", "kitti-M.json")
    if not ckpt_path:
        ckpt_path = os.path.join(here, "SEA-RAFT", "models", "Tartan-C-T-TSKH-kitti432x960-M.pth")
    return Est(cfg_json=cfg_json, ckpt_path=ckpt_path)


def _create_flow_cache(
    video_path: str,
    flow_method: str,
    searaft_cfg: Optional[str],
    searaft_ckpt: Optional[str],
    cache_frames: bool = False,
):
    mod = _load_temporal_module()
    FlowCache = getattr(mod, "FlowCache", None)
    build_flow_estimator = getattr(mod, "build_flow_estimator", None)
    if FlowCache is None or build_flow_estimator is None:
        raise RuntimeError("Flow cache helpers unavailable from temporal module")
    estimator = build_flow_estimator(flow_method, searaft_cfg, searaft_ckpt)
    return FlowCache(video_path, estimator, cache_frames=cache_frames)


def divergence_from_flow(flow: np.ndarray) -> np.ndarray:
    u = flow[..., 0]
    v = flow[..., 1]
    du_dx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    return du_dx + dv_dy


def median_abs_divergence(div_map: np.ndarray, mask: np.ndarray, erode_iter: int = 1) -> Optional[float]:
    m = mask.astype(np.uint8)
    if erode_iter > 0:
        k = np.ones((3, 3), np.uint8)
        m = cv2.erode(m, k, iterations=erode_iter)
    idx = m > 0
    cnt = int(idx.sum())
    if cnt < 150:
        return None
    vals = np.abs(div_map[idx])
    if vals.size == 0:
        return None
    return float(np.median(vals))


def normalize_series_to_unit(x: List[float], percentile: float = 95.0) -> Tuple[List[float], float]:
    arr = np.array([v for v in x if v is not None], dtype=np.float32)
    if arr.size == 0:
        return [0.0 for _ in x], 1.0
    q = float(np.percentile(arr, percentile))
    q = max(q, 1e-6)
    normed = [float(np.clip(v / q, 0.0, 1.0)) if v is not None else None for v in x]
    return normed, q


def compute_st1_for_instance(
    cap: cv2.VideoCapture,
    frames: np.ndarray,
    masks: np.ndarray,
    flow_method: str = "farneback",
    erode_iter: int = 1,
    norm_percentile: float = 95.0,
    pbar_disable: bool = False,
    pbar_desc: str = "",
) -> Dict:
    e_div_list: List[Optional[float]] = []
    frame_pairs: List[Tuple[int, int]] = []
    iterable = range(1, len(frames))
    for i in pbar(iterable, desc=pbar_desc, total=len(frames) - 1, disable=pbar_disable):
        f0 = int(frames[i - 1])
        f1 = int(frames[i])
        im0 = read_frame(cap, f0)
        im1 = read_frame(cap, f1)
        if im0 is None or im1 is None:
            e_div_list.append(None)
            frame_pairs.append((f0, f1))
            continue
        flow = compute_flow(im0, im1, method=flow_method)
        div_map = divergence_from_flow(flow)
        val = median_abs_divergence(div_map, masks[i - 1], erode_iter=erode_iter)
        e_div_list.append(val)
        frame_pairs.append((f0, f1))

    normed, q = normalize_series_to_unit(e_div_list, percentile=norm_percentile)
    st1_per_pair: List[Optional[float]] = [1.0 - v if v is not None else None for v in normed]
    st1_vals = [v for v in st1_per_pair if v is not None]
    mean_st1 = float(np.mean(st1_vals)) if st1_vals else 0.0
    med_e = float(np.median([v for v in e_div_list if v is not None])) if any(v is not None for v in e_div_list) else 0.0

    per_frame = []
    for (f0, f1), e, s in zip(frame_pairs, e_div_list, st1_per_pair):
        per_frame.append({
            "t0": f0,
            "t1": f1,
            "E_div": None if e is None else float(e),
            "ST1": None if s is None else float(s),
        })

    return {
        "mean_ST1": mean_st1,
        "median_Ediv": med_e,
        "q_norm": q,
        "n_pairs": int(len(frame_pairs)),
        "per_pair": per_frame,
    }


def run_st1(
    npz_path: str,
    video_path: Optional[str],
    ids: Optional[List[str]],
    flow_method: str,
    erode_iter: int,
    norm_percentile: float,
    output: Optional[str],
    no_pbar: bool = False,
    searaft_cfg: Optional[str] = None,
    searaft_ckpt: Optional[str] = None,
    shared_flow_cache: Optional[object] = None,
) -> Dict:
    """Compute ST-1 by pushing optical flow once per unique frame pair and reusing for all instances.

    Steps:
      1) Load tracks (frames, masks) for requested ids
      2) Build union of unique (t0,t1) frame pairs across all tracks and an index map for each instance
      3) Iterate unique pairs: read frames, compute flow and divergence ONCE
      4) For each instance needing the pair, compute median |div| within its mask at t0
      5) After loop, per-instance normalize and aggregate to ST1
    """
    tracks, npz_video, _ = load_sam2_masks(npz_path)
    if not tracks:
        raise RuntimeError("No tracks found in NPZ")
    if ids:
        tracks = {k: v for k, v in tracks.items() if k in set(ids)}
        if not tracks:
            raise RuntimeError("No matching ids in NPZ for given --ids")

    # Build per-instance pair list and a global pair->jobs map
    pair_jobs: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
    inst_pairs: Dict[str, List[Tuple[int, int]]] = {}
    inst_masks = {k: v["masks"] for k, v in tracks.items()}
    for k, data in tracks.items():
        fr = data["frames"].astype(int)
        pairs = [(int(fr[i - 1]), int(fr[i])) for i in range(1, len(fr))]
        inst_pairs[k] = pairs
        for idx, pr in enumerate(pairs):
            pair_jobs.setdefault(pr, []).append((k, idx))

    if not pair_jobs:
        # No frame transitions, return ST1=1 for empty/degenerate case
        return {
            "ST1": 1.0,
            "flow_method": flow_method,
            "norm_percentile": norm_percentile,
            "erode_iter": erode_iter,
            "video": video_path or npz_video,
            "npz": npz_path,
            "per_instance": {k: {
                "mean_ST1": 1.0,
                "median_Ediv": 0.0,
                "q_norm": 1.0,
                "n_pairs": 0,
                "per_pair": [],
            } for k in tracks.keys()},
        }

    vid = video_path or get_video_from_npz(npz_path, npz_video)
    if not vid or not os.path.exists(vid):
        raise RuntimeError("Video not found; provide --video explicitly")
    flow_cache = None
    cap = None
    if shared_flow_cache is not None:
        flow_cache = shared_flow_cache
        cache_video = getattr(flow_cache, "video_path", None)
        if cache_video and os.path.abspath(cache_video) != os.path.abspath(vid):
            raise ValueError("Provided FlowCache video path does not match ST1 video")
    else:
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {vid}")

    # Prepare per-instance E_div storage aligned to inst_pairs order
    inst_e_div: Dict[str, List[Optional[float]]] = {k: [None] * len(inst_pairs[k]) for k in inst_pairs}

    # Prepare flow provider
    searaft_est = None
    effective_flow_method = flow_method
    if flow_cache is not None:
        effective_flow_method = getattr(flow_cache, "flow_name", flow_method)
    elif flow_method == "searaft":
        searaft_est = _load_searaft_estimator(searaft_cfg, searaft_ckpt)

    # Iterate unique pairs once
    unique_pairs = sorted(pair_jobs.keys())
    for (t0, t1) in pbar(unique_pairs, desc="pairs(flow)", total=len(unique_pairs), disable=no_pbar):
        if flow_cache is not None:
            try:
                flow, _ = flow_cache.get_flow(t0, t1)
            except Exception:
                for (iid, idx) in pair_jobs[(t0, t1)]:
                    inst_e_div[iid][idx] = None
                continue
        else:
            im0 = read_frame(cap, t0)
            im1 = read_frame(cap, t1)
            if im0 is None or im1 is None:
                for (iid, idx) in pair_jobs[(t0, t1)]:
                    inst_e_div[iid][idx] = None
                continue
            if flow_method == "searaft":
                f_fw, _ = searaft_est.forward_backward(im0, im1)  # type: ignore
                flow = f_fw
            else:
                flow = compute_flow(im0, im1, method=flow_method)
        div_map = divergence_from_flow(flow)
        # apply to all instances needing this pair
        for (iid, idx) in pair_jobs[(t0, t1)]:
            mask = inst_masks[iid][idx]
            val = median_abs_divergence(div_map, mask, erode_iter=erode_iter)
            inst_e_div[iid][idx] = val

    if cap is not None:
        cap.release()

    # Build per-instance results (normalize per instance)
    per_instance: Dict[str, Dict] = {}
    weights = []
    values = []
    for iid in tracks.keys():
        e_list = inst_e_div[iid]
        normed, q = normalize_series_to_unit(e_list, percentile=norm_percentile)
        st1_per_pair: List[Optional[float]] = [1.0 - v if v is not None else None for v in normed]
        st1_vals = [v for v in st1_per_pair if v is not None]
        mean_st1 = float(np.mean(st1_vals)) if st1_vals else 0.0
        med_e = float(np.median([v for v in e_list if v is not None])) if any(v is not None for v in e_list) else 0.0
        # assemble per_pair entries with frame indices
        pair_entries = []
        for (f0, f1), e, s in zip(inst_pairs[iid], e_list, st1_per_pair):
            pair_entries.append({
                "t0": int(f0),
                "t1": int(f1),
                "E_div": None if e is None else float(e),
                "ST1": None if s is None else float(s),
            })

        per_instance[iid] = {
            "mean_ST1": mean_st1,
            "median_Ediv": med_e,
            "q_norm": q,
            "n_pairs": int(len(inst_pairs[iid])),
            "per_pair": pair_entries,
        }
        # weight by number of valid pairs
        n_valid = max(1, sum(1 for v in st1_per_pair if v is not None))
        weights.append(n_valid)
        values.append(mean_st1)

    weights = np.array(weights, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    overall = float((weights * values).sum() / max(1.0, weights.sum()))

    out = {
        "ST1": overall,
        "flow_method": effective_flow_method,
        "norm_percentile": norm_percentile,
        "erode_iter": erode_iter,
        "video": vid,
        "npz": npz_path,
        "per_instance": per_instance,
        "searaft_cfg": searaft_cfg if effective_flow_method == "searaft" else None,
        "searaft_ckpt": searaft_ckpt if effective_flow_method == "searaft" else None,
    }

    if output:
        with open(output, "w") as f:
            json.dump(out, f, indent=2)
    return out


def compute_metrics_shared(
    npz_path: str,
    video_path: Optional[str],
    ids: Optional[List[str]],
    flow_method: str,
    erode_iter: int,
    norm_percentile: float,
    searaft_cfg: Optional[str],
    searaft_ckpt: Optional[str],
    fb_thr: float,
    fb_rel: float,
    use_fb_consistency: bool,
    no_pbar: bool = False,
    cache_frames: bool = False,
) -> Tuple[Dict, Dict]:
    """Compute ST1 and temporal warping error while sharing optical flow computations."""
    mod = _load_temporal_module()
    compute_temporal = getattr(mod, "compute_temporal_warping_error", None)
    if compute_temporal is None:
        raise RuntimeError("Temporal warping module does not expose compute_temporal_warping_error")

    vid = video_path
    if not vid:
        candidate = get_video_from_npz(npz_path, None)
        if candidate and os.path.exists(candidate):
            vid = candidate
        else:
            npz_video = None
            try:
                with np.load(npz_path, allow_pickle=True) as data:  # type: ignore[call-overload]
                    if "video_path" in data:
                        npz_video = str(data["video_path"][0])
            except Exception:
                npz_video = None
            vid = get_video_from_npz(npz_path, npz_video)
    if not vid or not os.path.exists(vid):
        raise RuntimeError("Video not found; provide --video explicitly")

    flow_cache = _create_flow_cache(
        video_path=vid,
        flow_method=flow_method,
        searaft_cfg=searaft_cfg,
        searaft_ckpt=searaft_ckpt,
        cache_frames=cache_frames,
    )

    try:
        st1_res = run_st1(
            npz_path=npz_path,
            video_path=vid,
            ids=ids,
            flow_method=flow_method,
            erode_iter=erode_iter,
            norm_percentile=norm_percentile,
            output=None,
            no_pbar=no_pbar,
            searaft_cfg=searaft_cfg,
            searaft_ckpt=searaft_ckpt,
            shared_flow_cache=flow_cache,
        )
        mean_err, per_pair = compute_temporal(
            video_path=vid,
            flow_provider=flow_method,
            searaft_cfg=searaft_cfg,
            searaft_ckpt=searaft_ckpt,
            fb_thr=fb_thr,
            fb_rel=fb_rel,
            use_fb_consistency=use_fb_consistency,
            shared_flow_cache=flow_cache,
        )
    finally:
        flow_cache.release()

    temporal_payload = {
        "video_path": os.path.abspath(vid),
        "flow_provider": flow_method,
        "searaft_cfg": os.path.abspath(searaft_cfg) if flow_method == "searaft" and searaft_cfg else None,
        "searaft_ckpt": os.path.abspath(searaft_ckpt) if flow_method == "searaft" and searaft_ckpt else None,
        "use_fb_consistency": use_fb_consistency,
        "fb_thr": fb_thr,
        "fb_rel": fb_rel,
        "num_pairs": len(per_pair),
        "E_warp": mean_err,
        "per_pair": per_pair,
    }
    return st1_res, temporal_payload


def main():
    ap = argparse.ArgumentParser(description="Compute ST-1 Incompressibility Tendency from SAM2 masks and optical flow (depth ignored)")
    ap.add_argument("--sam2_npz", required=True, help="Path to *_sam2_masks.npz")
    ap.add_argument("--video", default=None, help="Video path; default inferred from NPZ")
    ap.add_argument("--ids", nargs="*", default=None, help="Optional subset of instance ids to evaluate (e.g., 26 27)")
    ap.add_argument(
        "--flow",
        default="farneback",
        choices=["farneback", "tvl1", "cuda_farneback", "cuda_tvl1", "cuda_brox", "searaft"],
        help="Optical flow method (CPU/CUDA/SEA-RAFT)",
    )
    ap.add_argument("--searaft-cfg", dest="searaft_cfg", default="/workspace/SEA-RAFT/config/eval/kitti-M.json", help="SEA-RAFT config JSON (when --flow searaft)")
    ap.add_argument("--searaft-ckpt", dest="searaft_ckpt", default="/workspace/SEA-RAFT/models/Tartan-C-T-TSKH-kitti432x960-M.pth", help="SEA-RAFT checkpoint (when --flow searaft)")
    ap.add_argument("--erode", type=int, default=1, help="Erode iterations on ROI mask")
    ap.add_argument("--p", type=float, default=95.0, help="Percentile for normalization of E_div")
    ap.add_argument("--output", default=None, help="Optional JSON output path")
    ap.add_argument("--no-pbar", action="store_true", help="Disable progress bars")
    ap.add_argument(
        "--compute-temporal",
        action="store_true",
        help="Also compute temporal warping error using shared optical flow",
    )
    ap.add_argument("--temporal-output", default=None, help="Optional JSON for temporal warping error result")
    ap.add_argument("--temporal-fb-thr", dest="temporal_fb_thr", type=float, default=1.0, help="Forward-backward absolute threshold for temporal warping")
    ap.add_argument("--temporal-fb-rel", dest="temporal_fb_rel", type=float, default=0.1, help="Forward-backward relative threshold for temporal warping")
    ap.add_argument(
        "--temporal-use-fb",
        dest="temporal_use_fb",
        action="store_true",
        help="Enable forward-backward consistency masking for temporal warping",
    )
    ap.add_argument(
        "--cache-frames",
        action="store_true",
        help="Keep video frames in FlowCache (higher memory, less IO) when sharing flows",
    )
    args = ap.parse_args()

    if args.compute_temporal:
        st1_res, temporal_res = compute_metrics_shared(
            npz_path=args.sam2_npz,
            video_path=args.video,
            ids=args.ids,
            flow_method=args.flow,
            erode_iter=args.erode,
            norm_percentile=args.p,
            searaft_cfg=args.searaft_cfg,
            searaft_ckpt=args.searaft_ckpt,
            fb_thr=args.temporal_fb_thr,
            fb_rel=args.temporal_fb_rel,
            use_fb_consistency=args.temporal_use_fb,
            no_pbar=args.no_pbar,
            cache_frames=args.cache_frames,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(st1_res, f, indent=2)
        if args.temporal_output:
            with open(args.temporal_output, "w") as f:
                json.dump(temporal_res, f, indent=2)
        summary = {
            "ST1": st1_res["ST1"],
            "E_warp": temporal_res["E_warp"],
            "video": st1_res["video"],
            "npz": st1_res["npz"],
        }
        print(json.dumps(summary, indent=2))
    else:
        res = run_st1(
            npz_path=args.sam2_npz,
            video_path=args.video,
            ids=args.ids,
            flow_method=args.flow,
            erode_iter=args.erode,
            norm_percentile=args.p,
            output=args.output,
            no_pbar=args.no_pbar,
            searaft_cfg=args.searaft_cfg,
            searaft_ckpt=args.searaft_ckpt,
        )
        print(json.dumps({"ST1": res["ST1"], "video": res["video"], "npz": res["npz"]}, indent=2))


if __name__ == "__main__":
    main()
