# Third-Party Code

This directory vendors the code paths used by the CrashTwin evaluation pipeline:

- `sam2`: two-vehicle mask propagation from metadata-seeded boxes.
- `map-anything`: camera intrinsic estimation.
- `droid_metric`: metric depth and camera trajectory estimation.
- `SEA-RAFT`: flow and temporal warping metrics.
- `centertrack`: 2D/3D vehicle detection and trajectory reconstruction.

Large checkpoints, generated outputs, datasets, and local caches are excluded
from GitHub. See `ARTIFACTS.md` for the expected artifact layout.
