#!/usr/bin/env python3
"""One-time script to pre-compute instances.json for static site serving."""

import json
from pathlib import Path

import numpy as np
import trimesh

HIDDEN_CLASS_NAMES = {"wall", "floor", "ceiling"}
DATA_SAMPLE = Path(__file__).resolve().parent / "data_sample"


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(geometries)
    return mesh


def compute_instances(metadata: dict, vertices: np.ndarray) -> list[dict]:
    instances = []
    for instance_id, payload in metadata.items():
        class_name = str(payload.get("pred_class_name", "unknown")).strip()
        if class_name.lower() in HIDDEN_CLASS_NAMES:
            continue

        point_ids = np.asarray(payload.get("point_ids", []), dtype=np.int64)
        point_ids = point_ids[(point_ids >= 0) & (point_ids < len(vertices))]
        if point_ids.size == 0:
            continue

        points = vertices[point_ids]
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = ((mins + maxs) / 2.0).tolist()
        size = (maxs - mins).tolist()

        instances.append({
            "instance_id": str(instance_id),
            "class_name": class_name or "unknown",
            "describe": payload.get("pred_describe", ""),
            "class_id": payload.get("pred_class_id"),
            "bbox_min": mins.tolist(),
            "bbox_max": maxs.tolist(),
            "bbox_center": center,
            "bbox_size": size,
        })

    instances.sort(key=lambda x: int(x["instance_id"]) if x["instance_id"].isdigit() else x["instance_id"])
    return instances


def main():
    for scene_dir in sorted(DATA_SAMPLE.iterdir()):
        if not scene_dir.is_dir():
            continue
        mesh_path = scene_dir / "mesh.ply"
        meta_path = scene_dir / "metadata.json"
        if not mesh_path.exists() or not meta_path.exists():
            print(f"Skipping {scene_dir.name}: missing mesh.ply or metadata.json")
            continue

        print(f"Processing {scene_dir.name}...")
        mesh = load_mesh(mesh_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        instances = compute_instances(metadata, np.asarray(mesh.vertices))
        out_path = scene_dir / "instances.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(instances, f, indent=2)
        print(f"  Wrote {len(instances)} instances to {out_path}")


if __name__ == "__main__":
    main()
