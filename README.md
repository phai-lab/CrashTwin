# CrashTwin: Physics-Grounded Evaluation for Collision Rollouts

[![Benchmark Assets](https://img.shields.io/badge/Benchmark-CrashTwin--Eval-yellow?logo=huggingface)](https://huggingface.co/datasets/nnuochen/crashtwin_eval)
[![Preprocess Docker](https://img.shields.io/badge/Docker-preprocess-blue?logo=docker)](https://hub.docker.com/r/nuochen1203/crashtwin-preprocess)
[![Reconstruct Docker](https://img.shields.io/badge/Docker-reconstruct-blue?logo=docker)](https://hub.docker.com/r/nuochen1203/crashtwin-reconstruct)

Official code for **CrashTwin**, a physics-grounded evaluation framework for
world-model collision rollouts. Given a model's generated collision videos on
**CrashTwin-Eval**, the evaluator reconstructs metric-scale physical attributes
from monocular videos and reports the diagnostic scores used in the paper.

CrashTwin-Eval contains 300 synthetic and 44 real-world collision videos sampled
from the held-out CrashTwin test split. Each clip contains two collision actors
and is evaluated with the same public reconstruction-and-scoring protocol.

## Release Scope

This repository provides the public CrashTwin evaluation pipeline:

1. normalize generated videos and estimate camera intrinsics,
2. run the reconstruction stack inside the released Docker environments,
3. compute per-video and aggregate CrashTwin-Eval metrics.

The release does not include generated videos from any evaluated method. To score
a method, place that method's generated videos under `predictions/<model_name>/`
using the benchmark `video_id` names.

## Installation

The host machine needs Bash, Docker, NVIDIA Container Toolkit, and
`huggingface-cli`. All CrashTwin Python code runs inside Docker.

```bash
git clone https://github.com/phai-lab/CrashTwin.git
cd CrashTwin

docker pull nuochen1203/crashtwin-preprocess:v1.0.0
docker pull nuochen1203/crashtwin-reconstruct:v1.0.0
```

The Docker images are hosted on Docker Hub:

- Preprocessing image: https://hub.docker.com/r/nuochen1203/crashtwin-preprocess
- Reconstruction image: https://hub.docker.com/r/nuochen1203/crashtwin-reconstruct

## Required Assets

Download the benchmark metadata and required checkpoints from Hugging Face:

https://huggingface.co/datasets/nnuochen/crashtwin_eval

From the repository root, run:

```bash
huggingface-cli download nnuochen/crashtwin_eval \
  --repo-type dataset \
  --local-dir .
```

The downloaded files include the benchmark CSV, per-video metadata, and model
checkpoints required by the public evaluator.

## Data Layout

After downloading the benchmark assets and adding one model's generated videos,
the repository should look like this:

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
them internally. Camera intrinsics are estimated from each input video, so fixed
camera parameters are not required.

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
mounts the repository root to `/crashtwin` inside each container, so
`benchmark/`, `checkpoints/`, `predictions/`, and `outputs/` must all live
inside the cloned repository. The script also mounts `.cache/` to `/cache` so
downloaded model-hub files are reused across runs.

Use `--gpus 0` for a single GPU, or pass a comma-separated list such as
`--gpus 0,1,2,3` for multiple GPUs. For multi-GPU runs, the script splits the
benchmark rows across GPUs and starts one Docker container per GPU. Logs are
written to `outputs/<model_name>/logs/`.

## Outputs

The main result files are:

```text
outputs/<model_name>/
├── per_video_metrics.csv
├── summary_metrics.csv
├── failed_videos.csv
├── logs/
└── per_video/
```

`summary_metrics.csv` reports the aggregate CrashTwin-Eval scores.
`per_video_metrics.csv` contains one row per evaluated video, and
`failed_videos.csv` records clips that did not complete. Intermediate
preprocessing and reconstruction outputs are stored under
`outputs/<model_name>/per_video/`.

## Acknowledgements

CrashTwin builds on the following open-source projects:

- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) for camera motion estimation.
- [Metric3D](https://github.com/YvanYin/Metric3D) for metric depth reconstruction.
- [MapAnything](https://github.com/facebookresearch/map-anything) for intrinsic estimation.
- [SAM 2](https://github.com/facebookresearch/sam2) for video object segmentation.
- [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) for optical flow.
- [CenterTrack](https://github.com/xingyizhou/CenterTrack) for monocular 3D vehicle detection and tracking.
- [OpenCLIP](https://github.com/mlfoundations/open_clip) for appearance-feature extraction.

We thank the authors of these projects for releasing their code and models.
Please refer to the corresponding files under `third_party/` for upstream
licenses and notices.
