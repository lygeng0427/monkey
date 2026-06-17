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

## Task environments

Two articulated task environments build on the static scene (the static
`FrankaDrawerBottleScene` is kept for CAD scale/placement checks):

- **`FrankaDrawerOpen`** — `assets/objects/drawer_articulated.xml`: the CAD drawer
  mesh slides out of a primitive cabinet on a slide joint (a grasp handle is added
  as a primitive, since the CAD handle is a flush recess). Success = slide distance
  past a threshold.
- **`FrankaBottleUntwist`** — `assets/objects/bottle_articulated.xml`: the cap
  rotates about the vertical axis on a hinge joint. The bottle is rescaled (~0.45×,
  via `scripts/preprocess_meshes.py`) so the cap is graspable by the Panda gripper.
  Success = cap angle past a threshold.

```bash
python scripts/preprocess_meshes.py                      # generate bottle part meshes (once)
python scripts/run_random_robosuite_env.py --env FrankaDrawerOpen
python scripts/run_random_robosuite_env.py --env FrankaBottleUntwist
python scripts/verify_envs.py                            # headless joint/success checks
```

### Collecting scripted demonstrations

One hardcoded-trajectory collector per task writes a robomimic-style HDF5
(`data/demo_i` with `states`, `actions`, `rewards`, `dones`, `final_state`, and
`model_file` / `env_args`). Shared infra lives in `scripts/demo_common.py`.

```bash
python scripts/collect_drawer_demos.py --out data/drawer.hdf5 --n 20
python scripts/collect_bottle_demos.py --out data/bottle.hdf5 --n 20

# Options: --render (watch), --seed, --n, --noise-scale (approach jitter), --keep-failures
```

Replay by **setting `states`** (the robomimic convention), not by re-applying
`actions` open-loop — the OSC controller's goal state isn't captured in the sim
state. Each demo's `final_state` is the terminal success state.

### Further work

- Tune grasp/contact: handle and cap-rim friction, joint `damping`/`frictionloss`,
  and `success_thresh` are constructor / XML knobs.
- Optionally make the cap a hinge+slide screw joint (cap rises as it unscrews).
- Add demonstrations / an RL training script using the `reward_shaping=True` reward.
