#!/usr/bin/env python3
"""Compress mesh.ply and instance_seg_mesh.ply to .drc (Draco) for web delivery."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import trimesh
import DracoPy

ROOT = Path(__file__).resolve().parent / "data_sample"
SCENES = json.loads((ROOT / "scenes.json").read_text())
TARGETS = ["mesh.ply", "instance_seg_mesh.ply"]
QBITS = 14
LEVEL = 7


def encode(src: Path, dst: Path):
    m = trimesh.load(str(src), force="mesh")
    if not hasattr(m, "vertices"):
        raise RuntimeError(f"{src} did not load as a mesh")
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    colors = None
    if getattr(m.visual, "kind", None) in ("vertex", "face"):
        c = m.visual.vertex_colors
        if c is not None and len(c) == len(verts):
            colors = np.asarray(c[:, :3], dtype=np.uint8)
    buf = DracoPy.encode(verts, faces=faces, colors=colors,
                         quantization_bits=QBITS, compression_level=LEVEL)
    dst.write_bytes(buf)
    return src.stat().st_size, dst.stat().st_size


def main():
    total_before = total_after = 0
    for scene in SCENES:
        sdir = ROOT / scene
        if not sdir.is_dir():
            print(f"  [skip] {scene}: dir missing")
            continue
        for name in TARGETS:
            src = sdir / name
            if not src.is_file():
                print(f"  [skip] {scene}/{name}: missing")
                continue
            dst = src.with_suffix(".drc")
            t0 = time.time()
            try:
                a, b = encode(src, dst)
            except Exception as e:
                print(f"  [FAIL] {scene}/{name}: {e}")
                continue
            total_before += a
            total_after += b
            print(f"  {scene}/{name}: {a/1e6:5.1f} MB -> {b/1e6:4.2f} MB  ({time.time()-t0:.1f}s)")
    print(f"\nTotal: {total_before/1e6:.1f} MB -> {total_after/1e6:.1f} MB "
          f"({100*(1-total_after/total_before):.1f}% smaller)")


if __name__ == "__main__":
    main()
