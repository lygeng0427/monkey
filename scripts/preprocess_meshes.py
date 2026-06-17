#!/usr/bin/env python3
"""Generate articulated part meshes from the preprocessed CAD meshes.

The uploaded bottle (``bottle_visual_m.stl``) is a single STL but is made of two
disconnected components: the bottle body and a wide flat cap brim. This script
splits those two components and rescales the whole bottle uniformly so the cap
rim becomes pinch-graspable by the Panda gripper (~0.08 m opening). The raw cap
(diameter ~0.135 m) and body (~0.090 m) are far too large to grasp at CAD scale.

Outputs (consumed by assets/objects/bottle_articulated.xml):
    assets/meshes/bottle_body_m.stl
    assets/meshes/bottle_cap_m.stl

Conventions preserved:
    - millimeters were already converted to meters in *_visual_m.stl
    - xy-centered, bottom of the bottle at z=0
    - the two parts stay in the SAME coordinate frame (NOT re-centered
      independently) so the cap keeps its z offset above the body and the hinge
      axis (the z-axis through the origin) is the correct pivot.

The drawer needs no preprocessing: its CAD mesh is used as-is as the sliding
drawer body, and the cabinet frame it pulls out of is built from primitives in
drawer_articulated.xml.

Run from the project root:

    python scripts/preprocess_meshes.py
    python scripts/preprocess_meshes.py --scale 0.45
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "assets" / "meshes"

# Default uniform scale: cap 0.135 m -> ~0.061 m, body 0.090 m -> ~0.041 m.
DEFAULT_SCALE = 0.45


def _radius(mesh: trimesh.Trimesh) -> float:
    return float(max(mesh.extents[0], mesh.extents[1]) / 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Uniform scale factor applied to the bottle (about the origin).",
    )
    args = parser.parse_args()

    src = MESH_DIR / "bottle_visual_m.stl"
    mesh = trimesh.load(src, force="mesh")
    components = mesh.split(only_watertight=False)
    if len(components) != 2:
        raise RuntimeError(
            f"Expected bottle mesh to have 2 connected components (body + cap), "
            f"got {len(components)}. Cannot split automatically."
        )

    # Body = the component whose center of mass sits lower in z.
    components = sorted(components, key=lambda c: c.centroid[2])
    body, cap = components[0], components[1]

    # Uniform scale about the origin. Because the mesh is xy-centered with its
    # bottom at z=0, origin scaling keeps it xy-centered with bottom at z=0.
    scale_matrix = np.diag([args.scale, args.scale, args.scale, 1.0])
    body.apply_transform(scale_matrix)
    cap.apply_transform(scale_matrix)

    body_out = MESH_DIR / "bottle_body_m.stl"
    cap_out = MESH_DIR / "bottle_cap_m.stl"
    body.export(body_out)
    cap.export(cap_out)

    print(f"scale = {args.scale}")
    for name, m, path in [("body", body, body_out), ("cap", cap, cap_out)]:
        b = np.round(m.bounds, 4).tolist()
        print(
            f"{name:4s} -> {path.name}: faces={len(m.faces)} "
            f"diameter~{2 * _radius(m):.4f} m  z=[{m.bounds[0, 2]:.4f}, {m.bounds[1, 2]:.4f}]  bounds={b}"
        )

    cap_d = 2 * _radius(cap)
    if cap_d > 0.078:
        print(
            f"WARNING: cap diameter {cap_d:.4f} m is close to / above the "
            f"~0.08 m gripper opening; consider a smaller --scale."
        )


if __name__ == "__main__":
    main()
