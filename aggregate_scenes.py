#!/usr/bin/env python3
"""Aggregate example scenes from multiple sources into ./data_sample."""

import json
import shutil
from pathlib import Path

INDEX_PATH = Path("/mnt/fillipo/yaowei/Sceneverse/web_net_viewer/src/best_index.json")
SCENE_DATA_ROOT = Path("/mnt/fillipo/yaowei/svpp_release_data/svpp_data")
VIDEO_ROOT = Path("/mnt/fillipo/junfeng/BIGAI-demo/youtube-raw-data")
OUTPUT_ROOT = Path(__file__).resolve().parent / "data_sample"

REQUIRED_VIDEOS = [
    "crop_images.mp4",
    "bbox_overlay_final_mix_label.mp4",
    "rendered_rgb_white.mp4",
    "rendered_seg.mp4",
]


def main():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Found {len(entries)} scenes in best_index.json")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    success = 0
    for entry in entries:
        scene_id = entry["id"]
        # Extract exp_num from seg path: /data/exp2/scene_id/mesh_segmented.glb
        seg_parts = entry["seg"].split("/")
        exp_num = seg_parts[2]  # e.g. "exp2"

        print(f"\n--- {scene_id} (exp={exp_num}) ---")
        dst = OUTPUT_ROOT / scene_id

        # 1. Copy scene data
        src_scene = SCENE_DATA_ROOT / scene_id
        if not src_scene.is_dir():
            print(f"  [WARN] Scene data not found: {src_scene}")
            continue

        if dst.exists():
            print(f"  [SKIP] Destination already exists, updating videos only")
        else:
            print(f"  Copying scene data from {src_scene}")
            shutil.copytree(src_scene, dst)

        # 2. Copy videos
        video_src = VIDEO_ROOT / exp_num / scene_id / "videos"
        if not video_src.is_dir():
            print(f"  [WARN] Video dir not found: {video_src}")
            continue

        for vname in REQUIRED_VIDEOS:
            vpath = video_src / vname
            if vpath.is_file():
                shutil.copy2(vpath, dst / vname)
                print(f"  Copied {vname}")
            else:
                print(f"  [WARN] Video not found: {vpath}")

        success += 1

    print(f"\n=== Done: {success}/{len(entries)} scenes aggregated ===")


if __name__ == "__main__":
    main()
