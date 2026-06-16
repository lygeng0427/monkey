# Franka + drawer + bottle starter scene

This starter package uses **robosuite + MuJoCo** to visualize a Franka/Panda arm, a table, and the two uploaded CAD meshes. It also moves the arm with smooth random actions so you can check that the scene, controller, camera, and scale are working.

This is intentionally a first scene-preview environment. The drawer and bottle are loaded as whole-object visual meshes with simple collision proxies. For the actual tasks, the CAD must be split into articulated parts:

- drawer frame fixed + drawer body / handle on a slide joint
- bottle body fixed/clamped + cap on a hinge or hinge+slide screw joint

## Files

```text
assets/meshes/Drawer_original.stl
assets/meshes/muri_bottle_original.stl
assets/meshes/drawer_visual_m.stl      # processed: mm -> meters, xy centered, bottom at z=0
assets/meshes/bottle_visual_m.stl      # processed: mm -> meters, xy centered, bottom at z=0
assets/objects/drawer_visual.xml       # robosuite-compatible MJCF object
assets/objects/bottle_visual.xml       # robosuite-compatible MJCF object
franka_drawer_bottle/env.py            # custom robosuite environment
scripts/run_random_robosuite_env.py    # viewer + random Panda motion
scripts/inspect_assets.py              # print mesh bounds
```

## Install

I recommend a fresh conda env:

```bash
conda create -n monkey python=3.10 -y
conda activate monkey
pip install -r requirements.txt
```

For Linux rendering problems, first test robosuite itself:

```bash
python -m robosuite.demos.demo_random_action --environment Lift --robots Panda
```

## Run

From this folder:

```bash
python scripts/run_random_robosuite_env.py
```

Optional:

```bash
python scripts/run_random_robosuite_env.py --steps 5000 --seed 1
python scripts/run_random_robosuite_env.py --show-collision
```

## Next development step

Once this scene opens correctly, create separate MJCF files for:

1. `drawer_articulated.xml`: fixed frame, moving drawer body, slide joint, handle collision capsule.
2. `bottle_articulated.xml`: fixed bottle body, rotating cap hinge, optional vertical slide coupled to cap rotation.

Then replace `CADVisualObject` with task-specific articulated XML objects and add reward / success logic.
