#!/usr/bin/env python3
from pathlib import Path
import trimesh

root = Path(__file__).resolve().parents[1]
for rel in ["assets/meshes/Drawer_original.stl", "assets/meshes/muri_bottle_original.stl", "assets/meshes/drawer_visual_m.stl", "assets/meshes/bottle_visual_m.stl"]:
    path = root / rel
    mesh = trimesh.load(path, force="mesh")
    print(f"{rel}")
    print(f"  vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, watertight={mesh.is_watertight}")
    print(f"  bounds=\n{mesh.bounds}")
    print(f"  extents={mesh.extents}\n")
