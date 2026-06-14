# CrashTwin

Official code for **CrashTwin**, a physics-grounded evaluation framework for
world-model collision rollouts. Given generated videos on **CrashTwin-Eval**,
the evaluator reconstructs metric-scale physical attributes from monocular
videos and reports the diagnostic scores used in the paper.

CrashTwin-Eval contains 300 synthetic and 44 real-world collision videos sampled
from the held-out CrashTwin test split. Each clip provides the two collision
actors needed by the evaluation protocol.

## Installation

Clone the repository and pull the released Docker environments:

```bash
git clone https://github.com/phai-lab/CrashTwin.git
cd CrashTwin

docker pull nuochen1203/crashtwin-preprocess:draft-20260614-env
docker pull nuochen1203/crashtwin-reconstruct:draft-20260613
```

Download the released CrashTwin-Eval files and model checkpoints, then place
them using the layout below. For the public release, Hugging Face is preferred
over Google Drive because it provides stable versioned files and command-line
downloads; Google Drive is best kept as a temporary mirror.

## Data Layout

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
│   └── nuScenes_3Dtracking.pth
└── predictions/
    └── <model_name>/
        ├── <video_id>.mp4
        ├── <video_id>.mp4
        └── ...
```

The expected `video_id` values are listed in `benchmark/crashtwin_eval.csv`.
Generated videos may use any resolution or frame rate; the evaluator normalizes
them internally. Camera intrinsics are estimated from each input video, so users
do not need to provide fixed camera parameters.

## Run Evaluation

Evaluate one model with:

```bash
python scripts/evaluate.py \
  --method-name <model_name> \
  --predictions predictions/<model_name> \
  --output outputs/<model_name> \
  --gpus 0,1,2,3
```

Use `--gpus 0` for a single GPU, or pass a comma-separated list such as
`--gpus 0,1,2,3` for multiple GPUs.

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
