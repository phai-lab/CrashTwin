import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor_image(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert BGR uint8 HxWx3 to torch float32 tensor [1,3,H,W] in [0,1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def _to_tensor_flow(flow_hw2: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert HxWx2 (u,v) to torch float32 tensor [1,2,H,W]."""
    return torch.from_numpy(flow_hw2).permute(2, 0, 1).unsqueeze(0).float().to(device)


def _grid_from_flow(flow: torch.Tensor) -> torch.Tensor:
    """Build sampling grid from pixel flow.

    flow: [1,2,H,W] in pixels
    return: grid [1,H,W,2] with absolute pixel coordinates x,y
    """
    n, c, h, w = flow.shape
    assert n == 1 and c == 2
    y, x = torch.meshgrid(
        torch.arange(h, device=flow.device),
        torch.arange(w, device=flow.device),
        indexing="ij",
    )
    base = torch.stack([x, y], dim=-1).float()  # [H,W,2]
    grid = base.unsqueeze(0) + flow.permute(0, 2, 3, 1)  # [1,H,W,2]
    return grid


def _grid_sample(img: torch.Tensor, grid_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample img at absolute pixel grid using bilinear sampling.

    img: [1,C,H,W], grid_xy: [1,H,W,2] with x,y pixel coords
    returns: (sampled [1,C,H,W], inside_mask [1,1,H,W])
    """
    h, w = img.shape[-2:]
    # normalize to [-1,1]
    x = grid_xy[..., 0]
    y = grid_xy[..., 1]
    xn = 2 * (x / (w - 1)) - 1
    yn = 2 * (y / (h - 1)) - 1
    grid = torch.stack([xn, yn], dim=-1)
    sampled = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    inside = (xn > -1) & (xn < 1) & (yn > -1) & (yn < 1)
    return sampled, inside.unsqueeze(1).float()


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
            of = ctor.create(0.197, 50.0, 0.8, 10, 77, 10)
            flow_gpu = of.calc(gprev, gnxt, None)
        if flow_gpu is None:
            return None
        flow = flow_gpu.download()
        return flow.astype(np.float32)
    except Exception:
        return None


class FlowEstimator:
    def forward_backward(self, img0_bgr: np.ndarray, img1_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class FarnebackEstimator(FlowEstimator):
    def __init__(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
        self.params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flags,
        )

    def _flow(self, g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, **self.params)
        return flow.astype(np.float32)  # HxWx2 (u,v)

    def forward_backward(self, img0_bgr: np.ndarray, img1_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        f_fw = self._flow(g0, g1)
        f_bw = self._flow(g1, g0)
        return f_fw, f_bw


class OpenCVEstimator(FlowEstimator):
    """Generic OpenCV optical flow estimator supporting CPU and CUDA variants."""

    def __init__(self, method: str = "farneback"):
        self.method = method

    def _flow(self, g_prev: np.ndarray, g_next: np.ndarray) -> np.ndarray:
        if self.method.startswith("cuda_"):
            flow = _compute_flow_cuda(g_prev, g_next, self.method)
            if flow is None:
                fallback = self.method.replace("cuda_", "")
                return _compute_flow_cpu(g_prev, g_next, fallback)
            return flow
        return _compute_flow_cpu(g_prev, g_next, self.method)

    def forward_backward(self, img0_bgr: np.ndarray, img1_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        f_fw = self._flow(g0, g1)
        f_bw = self._flow(g1, g0)
        return f_fw.astype(np.float32), f_bw.astype(np.float32)


class SEARAFTEstimator(FlowEstimator):
    def __init__(self, cfg_json: str, ckpt_path: str):
        # Load SEA-RAFT without importing its parser; directly parse JSON
        with open(cfg_json, "r") as f:
            cfg = json.load(f)
        args = SimpleNamespace(**cfg)
        # set required runtime keys if missing
        if not hasattr(args, "iters"):
            args.iters = 12
        if not hasattr(args, "radius"):
            args.radius = 4
        if not hasattr(args, "dim"):
            args.dim = 128
        if not hasattr(args, "scale"):
            args.scale = 0
        if not hasattr(args, "use_var"):
            args.use_var = True
        if not hasattr(args, "var_min"):
            args.var_min = 0
        if not hasattr(args, "var_max"):
            args.var_max = 10

        # import RAFT from SEA-RAFT
        import sys
        root = os.path.dirname(os.path.abspath(__file__))
        # When this script lives in SEA-RAFT/, core is SEA-RAFT/core
        sea_core = os.path.join(root, "core")
        if sea_core not in sys.path:
            sys.path.append(sea_core)
        from raft import RAFT  # type: ignore
        from utils.utils import load_ckpt  # type: ignore

        # Always use CUDA if available, otherwise fallback silently to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RAFT(args).to(self.device).eval()
        if not ckpt_path:
            raise ValueError("SEARAFTEstimator requires a checkpoint path.")
        load_ckpt(self.model, ckpt_path)

        self.iters = args.iters
        self.scale = args.scale

    @torch.no_grad()
    def _run(self, img0_bgr: np.ndarray, img1_bgr: np.ndarray) -> np.ndarray:
        # Build tensors expected by SEA-RAFT: [1,3,H,W] in 0..255 (model normalizes internally)
        img0 = torch.from_numpy(cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0)
        img1 = torch.from_numpy(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0)
        img0 = img0.to(self.device)
        img1 = img1.to(self.device)
        # Optional scaling per SEA-RAFT convention
        if self.scale != 0:
            s_up = float(2 ** self.scale)
            s_down = float(0.5 ** self.scale)
            img0_s = F.interpolate(img0, scale_factor=s_up, mode="bilinear", align_corners=False)
            img1_s = F.interpolate(img1, scale_factor=s_up, mode="bilinear", align_corners=False)
            out = self.model(img0_s, img1_s, iters=self.iters, test_mode=True)
            flow = out["flow"][-1]
            flow = F.interpolate(flow, scale_factor=s_down, mode="bilinear", align_corners=False) * s_down
        else:
            out = self.model(img0, img1, iters=self.iters, test_mode=True)
            flow = out["flow"][-1]
        flow_np = flow[0].permute(1, 2, 0).detach().cpu().numpy()
        return flow_np

    def forward_backward(self, img0_bgr: np.ndarray, img1_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f_fw = self._run(img0_bgr, img1_bgr)
        f_bw = self._run(img1_bgr, img0_bgr)
        return f_fw.astype(np.float32), f_bw.astype(np.float32)


def build_flow_estimator(
    flow_provider: str,
    searaft_cfg: Optional[str] = None,
    searaft_ckpt: Optional[str] = None,
) -> FlowEstimator:
    """Factory returning a FlowEstimator instance for the requested provider."""
    name = flow_provider.lower()
    if name == "opencv":
        name = "farneback"
    if name in {"farneback", "tvl1", "cuda_farneback", "cuda_tvl1", "cuda_brox"}:
        return OpenCVEstimator(name)
    if name == "searaft":
        if searaft_cfg is None or searaft_ckpt is None:
            raise ValueError("searaft requires both cfg_json and ckpt_path.", searaft_cfg is None)
        return SEARAFTEstimator(searaft_cfg, ckpt_path=searaft_ckpt)
    raise ValueError(f"Unsupported flow provider: {flow_provider}")


@torch.no_grad()
def _compute_mask_and_warp(
    frame_t_bgr: np.ndarray,
    frame_tp1_bgr: np.ndarray,
    flow_fw_hw2: np.ndarray,
    flow_bw_hw2: np.ndarray,
    fb_thr: float,
    fb_rel: float,
    use_fb_consistency: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute visibility mask, warped next frame, and per-pixel squared error skeleton inputs.

    Returns:
      vt: [1,3,H,W] in [0,1]
      vtp1_warp: [1,3,H,W] in [0,1]
      mask: [1,1,H,W] in {0,1}
    """
    vt = _to_tensor_image(frame_t_bgr, device)  # [1,3,H,W]
    vtp1 = _to_tensor_image(frame_tp1_bgr, device)
    f_fw = _to_tensor_flow(flow_fw_hw2, device)  # [1,2,H,W]
    f_bw = _to_tensor_flow(flow_bw_hw2, device)

    # Build sampling grid for warping t+1 -> t
    grid_tp1 = _grid_from_flow(f_fw)  # [1,H,W,2]
    # warp frame t+1
    vtp1_warp, inside1 = _grid_sample(vtp1, grid_tp1)

    if use_fb_consistency:
        # forward-backward consistency: sample backward flow at x' = x + f_fw(x)
        f_bw_warp, inside2 = _grid_sample(f_bw, grid_tp1)
        cycle = f_fw + f_bw_warp  # [1,2,H,W]
        cycle_norm = torch.linalg.norm(cycle, dim=1, keepdims=True)  # [1,1,H,W]
        # Threshold: max(abs threshold, relative to min(H,W)) similar to RAFT's check
        H, W = frame_t_bgr.shape[:2]
        thr_abs = fb_thr
        thr_rel = fb_rel * float(min(H, W))
        thr = max(thr_abs, thr_rel)
        mask = (cycle_norm <= thr).float() * inside1 * inside2
    else:
        # ECCV'18 strict: single-direction warping + visibility mask only
        mask = inside1
    return vt, vtp1_warp, mask


@torch.no_grad()
def _pair_warp_error(
    frame_t_bgr: np.ndarray,
    frame_tp1_bgr: np.ndarray,
    flow_fw_hw2: np.ndarray,
    flow_bw_hw2: np.ndarray,
    fb_thr: float = 1.0,
    fb_rel: float = 0.1,
    use_fb_consistency: bool = False,
    device: Optional[torch.device] = None,
) -> float:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    vt, vtp1_warp, mask = _compute_mask_and_warp(
        frame_t_bgr,
        frame_tp1_bgr,
        flow_fw_hw2,
        flow_bw_hw2,
        fb_thr,
        fb_rel,
        use_fb_consistency,
        device,
    )
    # Per-pixel squared L2 over RGB channels
    diff = vt - vtp1_warp  # [1,3,H,W]
    sq = (diff * diff).sum(dim=1, keepdim=True)  # [1,1,H,W], sum over channels
    denom = mask.sum().item()
    if denom < 1:
        return float("nan")
    err = (sq * mask).sum().item() / denom
    return float(err)


class FlowCache:
    """Cache optical flow (and optionally frames) so multiple metrics reuse computations."""

    def __init__(
        self,
        video_path: str,
        estimator: FlowEstimator,
        cache_frames: bool = False,
    ):
        self.video_path = os.path.abspath(video_path)
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self.estimator = estimator
        self.cache_frames = cache_frames
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count: Optional[int] = frame_count if frame_count > 0 else None
        self._flow_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._frame_cache: Dict[int, np.ndarray] = {}
        self._last_frame_idx: Optional[int] = None
        self._last_frame: Optional[np.ndarray] = None
        self.flow_name = getattr(estimator, "method", getattr(estimator, "__class__.__name__", "unknown"))
        self.estimator_device = getattr(estimator, "device", None)

    def __enter__(self) -> "FlowCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None  # type: ignore

    def _read_frame(self, index: int) -> np.ndarray:
        if index < 0:
            raise ValueError("Frame index must be non-negative")
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to re-open video: {self.video_path}")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {index} from {self.video_path}")
        return frame

    def get_frame(self, index: int) -> np.ndarray:
        if self.cache_frames and index in self._frame_cache:
            return self._frame_cache[index]
        if not self.cache_frames and self._last_frame_idx == index and self._last_frame is not None:
            return self._last_frame
        frame = self._read_frame(index)
        if self.cache_frames:
            self._frame_cache[index] = frame
        else:
            self._last_frame_idx = index
            self._last_frame = frame
        return frame

    def get_flow(
        self,
        t0: int,
        t1: int,
        frames: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = (t0, t1)
        if key in self._flow_cache:
            return self._flow_cache[key]
        if frames is None:
            frame0 = self.get_frame(t0)
            frame1 = self.get_frame(t1)
        else:
            frame0, frame1 = frames
        f_fw, f_bw = self.estimator.forward_backward(frame0, frame1)
        self._flow_cache[key] = (f_fw.astype(np.float32), f_bw.astype(np.float32))
        return self._flow_cache[key]

    def consecutive_pair_count(self) -> Optional[int]:
        if self.frame_count is None or self.frame_count <= 1:
            return None
        return self.frame_count - 1

    def iter_consecutive_pairs(self) -> Iterable[Tuple[int, int]]:
        count = self.consecutive_pair_count()
        if count is not None:
            for idx in range(count):
                yield idx, idx + 1
            return
        idx = 0
        while True:
            try:
                _ = self.get_frame(idx)
                _ = self.get_frame(idx + 1)
            except RuntimeError:
                break
            yield idx, idx + 1
            idx += 1


def compute_temporal_warping_error(
    video_path: str,
    flow_provider: str = "searaft",
    searaft_cfg: Optional[str] = None,
    searaft_ckpt: Optional[str] = None,
    fb_thr: float = 1.0,
    fb_rel: float = 0.1,
    use_fb_consistency: bool = False,
    shared_flow_cache: Optional[FlowCache] = None,
) -> Tuple[float, List[float]]:
    """Compute ECCV'18 Temporal Warping Error for a single video.

    Returns (E_warp_over_video, per_pair_errors).
    - E_warp_over_video: arithmetic mean over valid frame pairs (ignores NaNs if a pair has 0 valid pixels).
    - per_pair_errors: each pair's masked MSE.
    """
    if shared_flow_cache is not None:
        flow_cache = shared_flow_cache
        if os.path.abspath(video_path) != flow_cache.video_path:
            raise ValueError("Shared FlowCache video path mismatch.")
        estimator = flow_cache.estimator
        torch_device = flow_cache.estimator_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        own_cache = False
    else:
        estimator = build_flow_estimator(flow_provider, searaft_cfg, searaft_ckpt)
        flow_cache = FlowCache(video_path, estimator)
        own_cache = True
        if hasattr(estimator, "device"):
            torch_device = estimator.device  # type: ignore[attr-defined]
        else:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_pair: List[float] = []
    total_pairs = flow_cache.consecutive_pair_count()

    def _show_progress(i: int, total: Optional[int]):
        # i: processed pairs so far (1-based when called after appending)
        bar_len = 30
        if total and total > 0:
            frac = min(max(i / total, 0.0), 1.0)
            filled = int(bar_len * frac)
            bar = "#" * filled + "-" * (bar_len - filled)
            percent = 100.0 * frac
            sys.stdout.write(f"\r[{bar}] {i}/{total} ({percent:.1f}%)")
        else:
            sys.stdout.write(f"\rProcessed {i} pairs")
        sys.stdout.flush()

    processed = 0
    try:
        for (t0, t1) in flow_cache.iter_consecutive_pairs():
            try:
                frame0 = flow_cache.get_frame(t0)
                frame1 = flow_cache.get_frame(t1)
            except RuntimeError:
                per_pair.append(float("nan"))
                processed += 1
                _show_progress(processed, total_pairs)
                continue
            f_fw, f_bw = flow_cache.get_flow(t0, t1, frames=(frame0, frame1))

            # compute pair error
            err = _pair_warp_error(
                frame0,
                frame1,
                f_fw,
                f_bw,
                fb_thr=fb_thr,
                fb_rel=fb_rel,
                use_fb_consistency=use_fb_consistency,
                device=torch_device,
            )
            per_pair.append(err)
            processed += 1
            _show_progress(processed, total_pairs)
    finally:
        if own_cache:
            flow_cache.release()

    if processed > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
    # Average over valid pairs
    valid = [e for e in per_pair if not (np.isnan(e) or np.isinf(e))]
    mean_err = float(np.mean(valid)) if len(valid) > 0 else float("nan")
    return mean_err, per_pair


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Robust defaults relative to this script's directory
    def_video = os.path.join(script_dir, "..", "Phy_metric_dir_", "VV_1", "VV_1.mp4")
    def_video = os.path.normpath(def_video)
    def_cfg = os.path.join(script_dir, "config", "eval", "kitti-M.json")
    def_ckpt = os.path.join(script_dir, "models", "Tartan-C-T-TSKH-kitti432x960-M.pth")

    parser = argparse.ArgumentParser(description="Compute Temporal Warping Error (ECCV'18)")
    parser.add_argument(
        "--video_path",
        type=str,
        default=def_video,
        help="Path to video file",
    )
    parser.add_argument(
        "--flow",
        type=str,
        default="searaft",
        choices=["opencv", "searaft"],
        help="Optical flow provider",
    )
    parser.add_argument(
        "--searaft-cfg",
        dest="searaft_cfg",
        type=str,
        default=def_cfg,
        help="SEA-RAFT config JSON path",
    )
    parser.add_argument(
        "--searaft-ckpt",
        dest="searaft_ckpt",
        type=str,
        default=def_ckpt,
        help="SEA-RAFT checkpoint path",
    )
    parser.add_argument("--fb-thr", dest="fb_thr", type=float, default=1.0, help="Absolute forward-backward consistency threshold in pixels (used only if --use-fb is set)")
    parser.add_argument("--fb-rel", dest="fb_rel", type=float, default=0.1, help="Relative threshold as a fraction of min(H,W) (used only if --use-fb is set)")
    parser.add_argument(
        "--use-fb",
        dest="use_fb",
        action="store_true",
        help="Enable forward-backward consistency masking (disabled by default to match ECCV'18)",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        type=str,
        default=None,
        help="If set, write results to this JSON file",
    )

    args = parser.parse_args()
    mean_err, per_pair = compute_temporal_warping_error(
        video_path=args.video_path,
        flow_provider=args.flow,
        searaft_cfg=args.searaft_cfg,
        searaft_ckpt=args.searaft_ckpt,
        fb_thr=args.fb_thr,
        fb_rel=args.fb_rel,
        use_fb_consistency=args.use_fb,
    )

    print(f"E_warp(video) = {mean_err}")
    print("Per-pair E_warp:")
    for i, e in enumerate(per_pair, start=1):
        print(f"  pair {i}: {e}")

    # Optional JSON output
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        payload = {
            "video_path": os.path.abspath(args.video_path),
            "flow_provider": args.flow,
            "searaft_cfg": os.path.abspath(args.searaft_cfg) if args.flow == "searaft" and args.searaft_cfg else None,
            "searaft_ckpt": os.path.abspath(args.searaft_ckpt) if args.flow == "searaft" and args.searaft_ckpt else None,
            "use_fb_consistency": args.use_fb,
            "fb_thr": args.fb_thr,
            "fb_rel": args.fb_rel,
            "num_pairs": len(per_pair),
            "E_warp": mean_err,
            "per_pair": per_pair,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
