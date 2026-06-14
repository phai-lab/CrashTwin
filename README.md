# CrashTwin

Official code for **CrashTwin**, a physics-grounded evaluation framework for
world-model collision rollouts. Given generated videos on **CrashTwin-Eval**,
the evaluator reconstructs metric-scale physical attributes from monocular
videos and reports the diagnostic scores used in the paper.

CrashTwin-Eval contains 300 synthetic and 44 real-world collision videos sampled
from the held-out CrashTwin test split. Each clip provides the two collision
actors needed by the evaluation protocol.

## Installation

Clone the repository and pull the released Docker environments. The host machine
only needs Bash, Docker, and NVIDIA Container Toolkit; all CrashTwin Python code
runs inside Docker.

```bash
git clone https://github.com/phai-lab/CrashTwin.git
cd CrashTwin

docker pull nuochen1203/crashtwin-preprocess:v1.0.0
docker pull nuochen1203/crashtwin-reconstruct:v1.0.0
```

Download the CrashTwin-Eval files and model checkpoints from Hugging Face:
https://huggingface.co/datasets/nnuochen/crashtwin_eval

From the repository root:

```bash
huggingface-cli download nnuochen/crashtwin_eval \
  --repo-type dataset \
  --local-dir .
```

## Data Layout

Place the downloaded benchmark files, checkpoints, and one model's generated
videos under the repository root:

```text
CrashTwin/
├── benchmark/
│   ├── crashtwin_eval.csv
│   ├── auto_json/
│   │   └── <video_id>_auto.json
│   └── vehicle_specs/
│       └── <video_id>_vehicle_specs.json
├── checkpoints/
│   ├── metric_depth_vit_giant2_800k.pth
│   ├── droid.pth
│   ├── nuScenes_3Dtracking.pth
│   └── searaft/
│       └── Tartan-C-T-TSKH-kitti432x960-M.pth
└── predictions/
    └── <model_name>/
        ├── <video_id>.mp4
        ├── <video_id>.mp4
        └── ...
```

The expected `video_id` values are listed in `benchmark/crashtwin_eval.csv`.
Generated videos may use any resolution or frame rate; the evaluator normalizes
them internally. Camera intrinsics are estimated from each input video, so users
do not need to provide fixed camera parameters. SEA-RAFT config files are
included in this repository under `third_party/SEA-RAFT/config/`; the SEA-RAFT
checkpoint is provided with the downloaded `checkpoints/` files.

## Run Evaluation

Evaluate one model from the host with:

```bash
bash scripts/evaluate.sh \
  --method-name <model_name> \
  --predictions predictions/<model_name> \
  --output outputs/<model_name> \
  --gpus 0,1,2,3
```

`scripts/evaluate.sh` starts the released Docker images with `docker run`. It
mounts the repository root to `/crashtwin` in each container, so `benchmark/`,
`checkpoints/`, `predictions/`, and `outputs/` must all live inside the cloned
repository. The script also mounts `.cache/` to `/cache` so downloaded model hub
files are reused across runs. Use `--gpus 0` for a single GPU, or pass a
comma-separated list such as `--gpus 0,1,2,3` for multiple GPUs. For multi-GPU
runs, the script splits the benchmark rows across GPUs and starts one Docker
container per GPU; logs are written to `outputs/<model_name>/logs/`.

## Outputs

The main result files are:

```text
outputs/<model_name>/
├── per_video_metrics.csv
├── summary_metrics.csv
└── failed_videos.csv
```

`summary_metrics.csv` reports the aggregate CrashTwin-Eval scores for all,
synthetic, and real-world subsets. Per-video intermediate reconstruction files
are stored under `outputs/<model_name>/per_video/`.

## Acknowledgements

This code builds on several open-source projects: DROID-SLAM and Metric3D for
camera motion and metric depth reconstruction, MapAnything for intrinsic
estimation, SAM 2 for video object segmentation, SEA-RAFT for optical flow,
CenterTrack for monocular 3D vehicle detection and tracking, and OpenCLIP for
appearance-feature extraction. We thank the authors of these projects for
releasing their code and models. Please refer to the corresponding files under
`third_party/` for the original licenses and notices.
