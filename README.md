# CrashTwin

Official code for **CrashTwin**, a physics-grounded evaluation framework for
world-model video rollouts. Given generated collision videos on the
**CrashTwin-Eval** split, this repository reconstructs metric-scale physical
attributes from monocular videos and reports the seven diagnostic metrics used in
the paper.

CrashTwin evaluates three dimensions of physical plausibility: spatio-temporal
consistency, momentum and kinetic-energy conservation, and interaction-dynamics
integrity. Users provide generated videos and the released CrashTwin-Eval
benchmark files; the toolkit handles camera processing, 3D reconstruction, and
metric computation.

## Requirements

- Linux server with NVIDIA GPU support
- Docker with NVIDIA Container Toolkit
- Python 3.10 or newer for the lightweight launcher scripts
- Released CrashTwin-Eval benchmark files
- Required model checkpoints

No local Docker build is required for normal evaluation.

## Quick Start

```bash
git clone https://github.com/phai-lab/CrashTwin.git
cd CrashTwin

# Download and unpack the released CrashTwin-Eval benchmark files at the repo root.
unzip crashtwin_eval_files.zip -d .

# Place required model checkpoints under checkpoints/.

python scripts/validate_benchmark_files.py
python scripts/validate_inputs.py --predictions predictions/<model_name>

docker pull nuochen1203/crashtwin-preprocess:draft-20260614-env
docker pull nuochen1203/crashtwin-reconstruct:draft-20260613

python scripts/evaluate.py \
  --method-name <model_name> \
  --predictions predictions/<model_name> \
  --output outputs/<model_name> \
  --gpus 0
```

## Repository Layout

```text
CrashTwin/
├── assets/                         # README figures and examples
├── benchmark/                      # CrashTwin-Eval benchmark files
│   ├── crashtwin_eval.csv           # evaluation video IDs and split labels
│   ├── auto_json/                   # per-video two-vehicle initialization files
│   └── vehicle_specs/               # per-video vehicle roles, dimensions, and masses
├── crashtwin/                       # evaluation package
│   ├── preprocess.py                # video normalization, masks, intrinsics, depth, SLAM
│   ├── reconstruct.py               # vehicle detection and 3D trajectory reconstruction
│   ├── metrics.py                   # paper metric computation
│   ├── scoring.py                   # per-video and aggregate score tables
│   └── io.py                        # input/output validation helpers
├── scripts/
│   ├── evaluate.py                  # one-command evaluation entry point
│   ├── validate_inputs.py           # check generated videos before evaluation
│   ├── validate_benchmark_files.py  # check benchmark files after download
│   ├── export_report.py             # optional report/table exporter
│   └── smoke_test.py                # maintainer/user environment smoke test
├── configs/
│   └── default.yaml                 # official evaluation settings
├── docker/
│   └── docker-compose.yaml          # uses released Docker images
├── third_party/                     # vendored algorithm code used by the pipeline
├── examples/
│   └── predictions/                 # tiny example input layout
├── LICENSE
└── README.md
```

## Generated Videos

Place all generated videos for one model under a single folder:

```text
predictions/
└── <model_name>/
    ├── <video_id>.mp4
    ├── <video_id>.mp4
    └── ...
```

The expected video IDs are listed in:

```text
benchmark/crashtwin_eval.csv
```

Rules:

1. Provide one `.mp4` for each `video_id` in the CrashTwin-Eval manifest.
2. Keep the `video_id` unchanged.
3. Videos may use any resolution or frame rate; the evaluator normalizes them internally.
4. Camera intrinsics are estimated from each input video. No fixed intrinsics are required from the user.

## Benchmark Files And Checkpoints

Download the released CrashTwin-Eval benchmark files and unpack them at the
repository root:

```bash
unzip crashtwin_eval_files.zip -d .
```

After extraction, the repository should contain:

```text
benchmark/auto_json/
benchmark/vehicle_specs/
```

Validate the benchmark files:

```bash
python scripts/validate_benchmark_files.py
```

These files define the two evaluated collision actors in each clip, their roles,
and the physical parameters used by the diagnostic metrics. Do not modify them
when comparing models.

Place model checkpoints under:

```text
checkpoints/
```

Expected checkpoint files:

```text
checkpoints/metric_depth_vit_giant2_800k.pth
checkpoints/droid.pth
checkpoints/nuScenes_3Dtracking.pth
```

SAM2, MapAnything, SEA-RAFT, and OpenCLIP may also use their upstream Hugging
Face or package caches unless those checkpoints are pre-bundled into the Docker
image.

## Docker Images

Use the released Docker images:

```bash
docker pull nuochen1203/crashtwin-preprocess:draft-20260614-env
docker pull nuochen1203/crashtwin-reconstruct:draft-20260613
```

The Docker images provide the runtime environments. The evaluation code is in
this repository and is mounted into the containers at runtime.

No local Docker build is required for normal evaluation.

## Run Evaluation

Evaluate one model with a single command:

```bash
python scripts/evaluate.py \
  --method-name <model_name> \
  --predictions predictions/<model_name> \
  --output outputs/<model_name> \
  --gpus 0,1,2,3
```

This command runs the complete evaluation pipeline:

```text
input videos
  -> video normalization
  -> two-vehicle mask estimation
  -> camera intrinsic estimation
  -> metric depth estimation
  -> camera trajectory estimation
  -> appearance and temporal consistency metrics
  -> vehicle detection
  -> 3D trajectory reconstruction
  -> collision metrics
  -> final score tables
```

All intermediate files and final results are written under:

```text
outputs/<model_name>/
```

## Output Files

After evaluation, the output folder contains:

```text
outputs/<model_name>/
├── per_video/
│   └── <video_id>/
│       ├── normalized.mp4
│       ├── masks.npz
│       ├── intrinsics.txt
│       ├── depth/
│       ├── camera_trajectory.json
│       ├── detections_2d3d.json
│       ├── trajectories_3d.json
│       ├── contact.json
│       └── metrics.json
├── per_video_metrics.csv
├── summary_metrics.csv
├── failed_videos.csv
└── logs/
```

`per_video_metrics.csv` uses the paper metric names:

```text
video_id,split,E_flow,E_warp,J_p,J_H,J_E,S_ID,D_ad,status
```

`summary_metrics.csv` reports aggregate scores:

```text
split,E_flow,E_warp,J_p,J_H,J_E,S_ID,D_ad,num_videos,num_failed
all,...
synthetic,...
real,...
```

## Metrics

| Metric | Meaning | Direction |
|---|---|---|
| `E_flow` | local flow/deformation consistency | lower is better |
| `E_warp` | temporal warping consistency | lower is better |
| `J_p` | linear momentum residual | lower is better |
| `J_H` | angular momentum residual | lower is better |
| `J_E` | kinetic energy residual | lower is better |
| `S_ID` | instance identity stability | higher is better |
| `D_ad` | appearance drift | lower is better |

## Validate Inputs

Before running the full evaluation, check that the input folder matches
CrashTwin-Eval:

```bash
python scripts/validate_inputs.py \
  --predictions predictions/<model_name> \
  --benchmark benchmark/crashtwin_eval.csv
```

The validator reports missing, duplicated, or incorrectly named videos.

If a clip fails during evaluation, it is listed in:

```text
outputs/<model_name>/failed_videos.csv
```

## Troubleshooting

If benchmark-file validation fails, re-download and unpack the released
CrashTwin-Eval files at the repository root. The expected files are:

```text
benchmark/auto_json/<video_id>_auto.json
benchmark/vehicle_specs/<video_id>_vehicle_specs.json
```

If Docker cannot see the GPU, verify that `nvidia-smi` works on the host and
that NVIDIA Container Toolkit is installed.

If evaluation is interrupted, rerun the same command with the same output
folder. Existing intermediate files can be reused by passing
`--skip-preprocess` or `--skip-reconstruction` when appropriate.

## Reproducibility

All reported comparisons should use the same released CrashTwin-Eval files,
config, and Docker image versions. The public output format uses only the metric
names from the paper.
