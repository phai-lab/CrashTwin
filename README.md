# CrashTwin

Official code for **CrashTwin**. Given generated crash videos on the fixed
CrashTwin-344 benchmark split, this repository reconstructs metric-scale
collision dynamics and reports the seven physics-grounded metrics used in the
paper.

The evaluator is designed for comparing video generation methods. Users provide
generated videos and the released benchmark metadata package; camera processing,
3D reconstruction, and metric computation are handled by the toolkit.

## Requirements

- Linux server with NVIDIA GPU support
- Docker with NVIDIA Container Toolkit
- Python 3.10 or newer for the lightweight launcher scripts
- The released CrashTwin-344 metadata package and model artifacts

No local Docker build is required for normal evaluation.

## Quick Start

```bash
git clone https://github.com/phai-lab/CrashTwin.git
cd CrashTwin

# Download the metadata package, then:
unzip crashtwin_344_metadata_draft-20260613.zip -d .

# Place required model files under artifacts/weights/.

python scripts/validate_metadata.py
python scripts/validate_inputs.py --predictions predictions/<method_name>

docker pull nuochen1203/crashtwin-preprocess:draft-20260614-env
docker pull nuochen1203/crashtwin-reconstruct:draft-20260613

python scripts/evaluate.py \
  --method-name <method_name> \
  --predictions predictions/<method_name> \
  --output outputs/<method_name> \
  --gpus 0
```

## Repository Layout

```text
CrashTwin/
├── assets/                         # README figures and examples
├── benchmark/                      # fixed CrashTwin-344 benchmark files
│   ├── crashtwin_344.csv            # evaluation video IDs and split labels
│   ├── auto_json/                   # per-video two-vehicle initialization metadata
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
│   ├── validate_metadata.py         # check benchmark metadata after download
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

## Input Videos

Place all generated videos for one method under a single folder:

```text
predictions/
└── <method_name>/
    ├── VV_7107.mp4
    ├── VV_7109.mp4
    ├── ...
    └── Video035clip0033.mp4
```

The expected video IDs are listed in:

```text
benchmark/crashtwin_344.csv
```

Rules:

1. Provide one `.mp4` for each video ID in CrashTwin-344.
2. Keep benchmark IDs unchanged, for example `VV_7107` and `Video031clip0029`.
3. Videos may use any resolution or frame rate; the evaluator normalizes them internally.
4. Camera intrinsics are estimated from each input video. No fixed intrinsics are required from the user.

The canonical public input name is:

```text
<video_id>.mp4
```

The evaluator also accepts generated outputs that keep the long-form raw name:

```text
<video_id>__<six_digit_index>_<real|syn>_test_<video_id>__output.mp4
```

Examples:

```text
<video_id>__000000_real_test_<video_id>__output.mp4
<video_id>__000001_syn_test_<video_id>__output.mp4
```

Long-form names are normalized internally to the benchmark video ID before
scoring.

## Benchmark Metadata And Model Files

Download the CrashTwin-344 metadata package and unzip it at the repository root:

```bash
unzip crashtwin_344_metadata_draft-20260613.zip -d .
```

After extraction, the repository should contain:

```text
benchmark/auto_json/
benchmark/vehicle_specs/
```

Validate the metadata:

```bash
python scripts/validate_metadata.py
```

These files define the two evaluated vehicles in each clip, their roles, and the
physical parameters used by the collision metrics. Do not modify these files when
comparing methods.

Place model files under:

```text
artifacts/weights/
```

Expected model files:

```text
artifacts/weights/metric_depth_vit_giant2_800k.pth
artifacts/weights/droid.pth
artifacts/weights/nuScenes_3Dtracking.pth
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

The compose file points to these images:

```text
docker/docker-compose.yaml
```

The Docker images provide the runtime environments. The evaluation code is in
this repository and is mounted into the containers at runtime.

No local Docker build is required for normal evaluation.

## Run Evaluation

Evaluate one method with a single command:

```bash
python scripts/evaluate.py \
  --method-name <method_name> \
  --predictions predictions/<method_name> \
  --output outputs/<method_name> \
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
outputs/<method_name>/
```

## Output Files

After evaluation, the output folder contains:

```text
outputs/<method_name>/
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
CrashTwin-344:

```bash
python scripts/validate_inputs.py \
  --predictions predictions/<method_name> \
  --benchmark benchmark/crashtwin_344.csv
```

The validator reports missing, duplicated, or incorrectly named videos.

If a clip fails during evaluation, it is listed in:

```text
outputs/<method_name>/failed_videos.csv
```

## Troubleshooting

If metadata validation fails, re-download and unzip the metadata package at the
repository root. The expected files are:

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

All reported comparisons should use the same released benchmark files, config,
and Docker image versions. The public output format uses only the metric names
from the paper.
