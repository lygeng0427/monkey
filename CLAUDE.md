# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **robosuite + MuJoCo** project with a Franka/Panda arm at a table and two CAD objects (a drawer and a bottle). There are three registered environments:

- **`FrankaDrawerBottleScene`** (`env.py`) — the original *static scene-preview*: both meshes loaded as whole-object visuals with box/cylinder collision proxies, no joints, `reward=0`/`_check_success=False`. Kept for CAD scale/placement/camera checks.
- **`FrankaDrawerOpen`** (`drawer_env.py`) — articulated drawer task: the CAD drawer mesh slides out of a primitive cabinet on a **slide joint**; success = slide displacement past a threshold.
- **`FrankaBottleUntwist`** (`bottle_env.py`) — articulated bottle task: a wide cap rotates about the vertical axis on a **hinge joint**; success = cap angle past a threshold.

## Setup & run

```bash
conda create -n monkey python=3.10 -y && conda activate monkey
pip install -r requirements.txt

# Regenerate the bottle part meshes (split + rescale) — needed before the bottle task:
python scripts/preprocess_meshes.py                        # writes bottle_body_m.stl, bottle_cap_m.stl

# Run from the project root so relative asset paths resolve:
python scripts/run_random_robosuite_env.py                 # default: FrankaDrawerBottleScene
python scripts/run_random_robosuite_env.py --env FrankaDrawerOpen
python scripts/run_random_robosuite_env.py --env FrankaBottleUntwist
python scripts/run_random_robosuite_env.py --steps 5000 --seed 1
python scripts/inspect_assets.py                           # print mesh bounds/extents

# Headless correctness check (drives each joint, asserts qpos moves + success flips):
python scripts/verify_envs.py

# Collect scripted demonstrations (one collector per task) -> robomimic-style HDF5:
python scripts/collect_drawer_demos.py --out data/drawer.hdf5 --n 20
python scripts/collect_bottle_demos.py --out data/bottle.hdf5 --n 20
python scripts/collect_drawer_demos.py --out /tmp/d.hdf5 --n 1 --render   # watch one
```

## Architecture

- **`franka_drawer_bottle/__init__.py`** — importing the package registers all three environments with robosuite's global env registry. Any script that calls `suite.make("Franka...")` **must** `import franka_drawer_bottle` first (subclassing a `robosuite` env auto-registers it by class name).

- **All three envs subclass `ManipulationEnv`** and share the same override order: `_load_model` (builds `TableArena`, sets Panda base xpos, places object(s), assembles `ManipulationTask`) → `_setup_references` (resolves MuJoCo joint/site/body ids) → `_setup_observables` (registers `@sensor` `Observable`s) → `_reset_internal` (set joint qpos to closed, `sim.forward()`) → `reward` / `_check_success`. The task envs (`drawer_env.py`, `bottle_env.py`) read the articulation **joint qpos** (`get_joint_qpos_addr`) for observables/reward/success and offer door-style shaping gated by `reward_shaping`; thresholds, placement, and `reward_scale` are constructor args.

- **`CADVisualObject`** (in `env.py`, used only by the static scene) — a `MujocoXMLObject` subclass that returns empty `visual_geoms`/`contact_geoms`/`sites` to dodge robosuite 1.5.x double-prefixing of the temporary CAD XML names. The **articulated** objects (`DrawerArticulatedObject`, `BottleArticulatedObject`) do **not** use this hack — they need their joints/geoms/sites registered; name-resolution instead goes through `franka_drawer_bottle/utils.py:find_name` (fuzzy substring lookup over `sim.model.*_names`), which absorbs the prefixing.

- **`assets/objects/*.xml`** — robosuite `MujocoXMLObject` MJCF: outer body wrapping a `body name="object"` plus `bottom_site`/`top_site`/`horizontal_radius_site`. `*_visual.xml` are the static single-mesh objects; `*_articulated.xml` are the task objects. **An object with no joint on its `object` body is welded to the world** — that is how the static meshes, the cabinet frame, and the bottle body stay anchored; only the child body carrying the slide/hinge joint moves. Visual geoms are `contype=0 conaffinity=0 group=1`; collision proxies are primitives with default contact. XML `file=` paths are relative to the XML (`../meshes/...`).
  - *Drawer*: the CAD mesh has no separate frame and only a flush recessed handle, so `drawer_articulated.xml` adds a primitive **cabinet** (open on +y) and a primitive **D-handle** to grasp; the env rotates the object (`set_euler([0,0,π/2])`, radians) so +y faces the robot.
  - *Bottle*: the cap collision disk sits in a ~0.3 mm gap **above** the body collision so the two proxies never touch — the hinge alone holds the cap, keeping it free to spin (no rotational jamming).

- **Scripted demo collection** — `scripts/demo_common.py` holds the shared infra (Panda BASIC/OSC_POSE controller via `make_env`; a `Recorder` that steps the env while recording sim states + actions, with `reach`/`reach_until` servo helpers; `save_hdf5`). Each task has its own hardcoded-trajectory collector: `collect_drawer_demos.py` (above-handle → descend → grasp → pull −x) and `collect_bottle_demos.py` (above-cap → descend → grasp → press-and-twist about z, with a release/counter-rotate/regrasp **ratchet** fallback). Both servo the eef toward the env's `handle_site`/`cap_site` (`*_site_id`) and stop on `_check_success`. Output is a robomimic-style HDF5 (`data/demo_i` with `states`, `actions`, `rewards`, `dones`, plus `model_file` + `env_args`); `states[t]` is the state **before** `actions[t]`, so the terminal success state is stored separately per demo as `final_state` (setting the sim to it reproduces success). Action layout is `[dx,dy,dz, drx,dry,drz, gripper]`, gripper −1 open / +1 close. Note: open-loop **action** replay does not reproduce a trajectory (OSC controller goal state isn't in the sim state) — replay by setting `states`, the robomimic convention.

- **`assets/meshes/*`** — `*_original.stl` are the raw uploads; `*_visual_m.stl` are preprocessed (**mm → meters, xy-centered, bottom at z=0**). `bottle_body_m.stl` / `bottle_cap_m.stl` are generated by `scripts/preprocess_meshes.py`, which splits the bottle's two connected components and applies one uniform ~0.45× scale (raw cap Ø0.135 / body Ø0.090 are too big for the Panda's ~0.08 m gripper) — the parts are **not** re-centered independently, so the cap keeps its z offset above the body.

## Conventions that bite

- **Quaternions**: MuJoCo uses `wxyz`; robosuite observables convert to `xyzw` via `convert_quat(..., to="xyzw")`. Body free-joint quats are `wxyz`.
- **Table height**: `table_offset[2]` (0.80) is the tabletop z in robosuite; objects are placed slightly above it.
- **robosuite version compatibility** is handled defensively in `scripts/run_random_robosuite_env.py`: `load_basic_osc_controller()` tries the 1.5 composite-controller API then falls back to the older `OSC_POSE` config, and `step_env()` handles both the 4-tuple and Gymnasium 5-tuple `step()` returns. Preserve these shims when editing.
- **Body/joint/site-name resolution**: the static scene uses `find_body_id` (in `env.py`); the task envs use `utils.py:find_name`. Both do fuzzy substring matching to absorb robosuite 1.5.x object-name prefixing — don't hard-code prefixed names.
