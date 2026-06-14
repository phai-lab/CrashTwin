# standard library
from pathlib import Path
from typing import *
import shutil, os
# third party
import argparse
# sample


import sys
sys.path.append("/workspace/droid_metric")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.utils import sample_from_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="video file to process")
    parser.add_argument("--out-dir", type=str, required=True, help="directory containing images")
    parser.add_argument("--sample-fps", type=float, default=30, help="sample fps")
    parser.add_argument("--limit", type=int, default=None, help="limit number of frames")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    print(out_dir)
    # remove if output directory exists
    # shutil.rmtree(str(out_dir), ignore_errors=True)
    os.makedirs(str(out_dir), exist_ok=True)
    
    sample_from_video(
        video_path=args.video,
        output_dir=out_dir,
        sample_fps=args.sample_fps,
        limit=args.limit
    )
