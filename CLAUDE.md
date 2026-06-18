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
# ...or one config: --object-scale 1.15 --px 0.0 --py 0.05

# Collect across a SIZE x POSITION grid (for size/position generalization). Default
# grid = sizes {0.85,1.0,1.15} x 3x3 positions (+/-0.05 m) x 5 demos = 135 demos/task.
# Each demo records its object_scale + placement_xy as per-demo HDF5 attrs.
python scripts/collect_drawer_demos.py --out data/drawer.hdf5 --grid --per-config 5
python scripts/collect_bottle_demos.py --out data/bottle.hdf5 --grid --per-config 5

# Pilot/validate the size knob before a big sweep:
python scripts/pilot_size.py        # static: drive joints across sizes (no jam)
python scripts/pilot_collect.py     # closed-loop: scripted demos succeed at size extremes
MUJOCO_GL=egl python scripts/framing_check.py   # cameras keep object in frame across grid

# Offline state->image pass (robomimic): add agentview+sideview 84x84 obs from stored states:
MUJOCO_GL=egl python scripts/render_obs.py --in data/drawer.hdf5 --out data/drawer_img.hdf5

# Render a scripted demo to an mp4 (offscreen; reuses the collector's exact
# trajectory via Recorder.frame_cb). Output is git-ignored (videos/, *.mp4).
# Needs an EGL offscreen GL context, e.g. MUJOCO_GL=egl on a headless box:
MUJOCO_GL=egl python scripts/record_drawer_video.py --camera sideview   # -> videos/drawer_open_sideview.mp4
MUJOCO_GL=egl python scripts/record_bottle_video.py --camera frontview  # -> videos/bottle_untwist_frontview.mp4
```

## Architecture

- **`franka_drawer_bottle/__init__.py`** — importing the package registers all three environments with robosuite's global env registry. Any script that calls `suite.make("Franka...")` **must** `import franka_drawer_bottle` first (subclassing a `robosuite` env auto-registers it by class name).

- **All three envs subclass `ManipulationEnv`** and share the same override order: `_load_model` (builds `TableArena`, sets Panda base xpos, places object(s), assembles `ManipulationTask`) → `_setup_references` (resolves MuJoCo joint/site/body ids) → `_setup_observables` (registers `@sensor` `Observable`s) → `_reset_internal` (set joint qpos to closed, `sim.forward()`) → `reward` / `_check_success`. The task envs (`drawer_env.py`, `bottle_env.py`) read the articulation **joint qpos** (`get_joint_qpos_addr`) for observables/reward/success and offer door-style shaping gated by `reward_shaping`; thresholds, placement, and `reward_scale` are constructor args.

- **`CADVisualObject`** (in `env.py`, used only by the static scene) — a `MujocoXMLObject` subclass that returns empty `visual_geoms`/`contact_geoms`/`sites` to dodge robosuite 1.5.x double-prefixing of the temporary CAD XML names. The **articulated** objects (`DrawerArticulatedObject`, `BottleArticulatedObject`) do **not** use this hack — they need their joints/geoms/sites registered; name-resolution instead goes through `franka_drawer_bottle/utils.py:find_name` (fuzzy substring lookup over `sim.model.*_names`), which absorbs the prefixing.

- **`assets/objects/*.xml`** — robosuite `MujocoXMLObject` MJCF: outer body wrapping a `body name="object"` plus `bottom_site`/`top_site`/`horizontal_radius_site`. `*_visual.xml` are the static single-mesh objects; `*_articulated.xml` are the task objects. **An object with no joint on its `object` body is welded to the world** — that is how the static meshes, the cabinet frame, and the bottle body stay anchored; only the child body carrying the slide/hinge joint moves. Visual geoms are `contype=0 conaffinity=0 group=1`; collision proxies are primitives with default contact. XML `file=` paths are relative to the XML (`../meshes/...`).
  - *Drawer*: the CAD mesh has no separate frame, so `drawer_articulated.xml` adds a primitive **cabinet** (open on +y) for it to slide out of; the env rotates the object (`set_euler([0,0,π/2])`, radians) so +y faces the robot. The whole object is **scaled 2×** (mesh `scale="2 2 2"` + doubled primitive dims); `FrankaDrawerOpen.success_thresh` is `0.18` accordingly. The **cabinet has both `group="0"` collision geoms and `group="1"` visual twins** (`contype=0 conaffinity=0`): with the default `render_collision_mesh=False`, robosuite hides geomgroup 0 (`base.py` sets `vopt.geomgroup[0]=0`), so without the twins the cabinet is *invisible* and the drawer looks like it slides out into empty space. The drawer is grasped by its **real CAD handle** — a bag-style **loop** on the front face (no D-handle is added; an earlier version added one in front of the real handle and it was removed). The loop's graspable **front bar** (local y≈0.090, z≈0.064, world y after the yaw) gets a thin invisible cylinder collision (`handle_bar`); the body collision (`tray_collision`) is a **low slab covering only the bottom of the block** — it tops out *below* the handle bar (tray-local z≈0.045 vs bar at ≈0.064) so the tilted gripper descending onto the bar never grazes/catches the block top (verified: gripper↔body contacts = 0 over an episode; the drawer is on a 1-DOF slide so a short collision still keeps it seated, and the upper block is visual-only); and the **cabinet top is pulled back** (ends at the body front, not overhanging the protruding handle) — otherwise the lip blocks the grasp. The drawer slides along world −x; the **slide joint is low-friction** (`damping=0.5 frictionloss=0.1`) so a modest grip can pull it open. The whole object is **welded to the world** (no joint on the `object` body), so it needs no support: the env places it at `table_offset[2] + 0.27` and it simply **floats** there with **nothing below it** (a `cab_riser` pedestal was tried but is purely cosmetic and was removed as unwanted). This puts the **handle at world z≈1.15** so the Panda wrist can hold a **tilted ~45° grasp** (between top-down and horizontal): at the table-height handle (z≈0.88) the wrist can only point horizontal with the arm fully extended *upward* (~0.6 m from the handle), so a true horizontal grasp is kinematically infeasible — verified by `scripts/probe_horizontal.py` (height sweep) and `scripts/probe_tilt_grasp.py` (reachability + grip-axis check at the raised handle). A **tall table** (raising `table_offset` instead) does **not** work: with the robot floor-mounted at a fixed base z≈0.91, a work surface at z≈1.07 sits above the gripper's reset height (z≈1.01) and traps the gripper under the tabletop (the robot base height is independent of table height). Raising the handle does **not** change `success_thresh` (still `0.18`; success is slide displacement, height-independent).
  - *Bottle*: kept at **full CAD size** (scale 1.0: cap Ø~0.135 m, body Ø~0.090 m) — large because the cap is turned by **non-prehensile pushing**, not grasping. The cap collision sits in a ~1 mm gap **above** the body collision so the two never touch — the hinge alone holds the cap, free to spin (no jamming). The cap mesh already has 8 flat **radial tabs/handles** at its brim; the collision models them (a small `cap_core` disk of r=0.030 + 8 flat radial `cap_petal_*` boxes — *not* added upright nubs). The core is deliberately **smaller than the body** so the valleys between tabs are open and the closed gripper can drop down *beside* a tab to push its side. `tab_site` marks the tab to push; `cap_site` is the cap center (arc pivot). Placed at `placement_xy=(0.10,0)` (further out than the drawer) for arm reach around the larger object.

- **Scripted demo collection** — `scripts/demo_common.py` holds the shared infra (Panda BASIC/OSC_POSE controller via `make_env`; a `Recorder` that steps the env while recording sim states + actions, with `reach`/`reach_until` servo helpers; `save_hdf5`). Each task has its own hardcoded-trajectory collector: `collect_drawer_demos.py` (**tilted ~45° cross-bar grasp** of the real loop handle's front bar → pull −x. The target orientation is a *fixed* quaternion = pitch `PITCH=−45°` (toward horizontal, about world y) on top of a 90° yaw (about world z) applied to the reset top-down pose; it's held every step by a full-quaternion servo (`T.get_orientation_error(target_quat, cur_quat)` → `action[3:6]`). The yaw makes the fingers close in the x-z plane = **across** the world-y bar (default grasp closes along world y = *parallel* = weak); the pitch is the in-between angle the raised handle makes reachable. Trajectory: pre-grasp backed off `BACKOFF` along −approach → seat `SEAT_FWD` along +approach onto the *true* bar → close → pull `PULL_DX` along −x until success. The seat/close are biased `GRASP_XBIAS=−0.016` toward the robot (world −x) of the bar center so the lower/forward finger descends in the clear corridor **in front of** the bar instead of catching on the drawer-body top (the actual noisy-failure mode: gripper stuck ~5 cm high on the body, bar never seated). `make_env` pins `initialization_noise=None` so a seed is reproducible. Robustness: **8/8** at no noise, ~**7/8** at `--noise-scale 0.01`, ~**6/8** at 0.02 — the per-episode jitter perturbs only the *approach start* (the seat converges onto the true bar). A *fully* horizontal grasp remains infeasible even at the raised handle, so 45° is the most-horizontal pose used) and `collect_bottle_demos.py` (**non-prehensile, single tab**: gripper closed throughout → drive down into a valley beside one tab → push it around an arc about the cap center via `Recorder.servo`. The hand's commanded angle is locked to the *measured* cap angle — `hand_angle = tab_angle0 + cap_angle + DRIVE` — so it never outruns the tab and slips to the next one; an open-loop fixed-rate sweep does, which was a real bug). Both servo the eef toward the env's `handle_site`/`cap_site` (`*_site_id`) and stop on `_check_success`. Output is a robomimic-style HDF5 (`data/demo_i` with `states`, `actions`, `rewards`, `dones`, plus `model_file` + `env_args`); `states[t]` is the state **before** `actions[t]`, so the terminal success state is stored separately per demo as `final_state` (setting the sim to it reproduces success). Action layout is `[dx,dy,dz, drx,dry,drz, gripper]`, gripper −1 open / +1 close. Note: open-loop **action** replay does not reproduce a trajectory (OSC controller goal state isn't in the sim state) — replay by setting `states`, the robomimic convention.

- **Size & position parameterization (for multi-config datasets)** — both task envs take `object_scale=1.0` and `placement_xy` constructor kwargs. **Size** uses robosuite 1.5's built-in `MujocoXMLObject(scale=)` (compounds onto the drawer's baked `2 2 2` mesh scale; scales geom `size`/`pos`, body `pos`, sites — but **not** geom `fromto`, which is why the drawer `handle_bar` is written in `pos`+`euler` form, not `fromto`). No per-size mesh regeneration: the bottle scales via the `<mesh scale>` attr. Size-coupled details: the **drawer placement-z is computed so the handle's world-z stays fixed at ~1.153 m across sizes** (the handle floats; scaling the 0.27 offset would push it out of reach), `success_thresh = 0.18 × object_scale` (slide displacement scales; the slide *range* 0.32 is unscaled so keep `thresh < 0.32`), and the **cabinet-top lip is pulled back** (front edge local-y ≈ +0.031) so the fixed-size gripper still seats at small scale (the cab-top→handle clearance is ~0.070·scale and shrinks as the object shrinks — at 0.85 the gripper caught the lip until it was pulled back). The bottle stays on the table at any size; its `success_thresh` is an **angle** (`π/2`), size-invariant. The **collectors are size-aware**: drawer `BACKOFF/SEAT_FWD/GRASP_XBIAS × env.object_scale`; bottle reads its push radius **live** from `‖tab_site − cap_site‖` (no hardcode) and scales `PUSH_Z × object_scale`. **Grid collection**: `--grid` (in both collectors, driven by `demo_common.collect_grid` + `grid_positions`) builds a **fresh env per (size, position)** config (sidesteps `hard_reset=False` re-placement), runs `--per-config` demos each, and writes one combined HDF5 with per-demo `object_scale`/`placement_xy` attrs. Demos are collected only at **seen** configs; "unseen" sizes/positions are just eval-time env kwargs (no demos needed). Validated by `scripts/pilot_collect.py` (scripted success at sizes {0.85,1.0,1.15}, drawer gripper↔body contacts = 0) and `scripts/pilot_size.py` (static joint-drive, no jam).

- **Image observations are rendered OFFLINE** (`scripts/render_obs.py`), not stored at collection time — the collectors store sim **states + the scaled `model_file`** (robomimic convention), so `render_obs.py` rebuilds the env per demo from that `model_file` (right size/position), replays the states with `set_state_from_flattened`, and grabs **`agentview` + `sideview`** at 84×84 into a new HDF5 (`obs/{agentview,sideview}_image`). Needs `MUJOCO_GL=egl`. The **drawer repoints its `agentview` to look straight DOWN at the handle** (`FrankaDrawerOpen._aim_agentview_topdown`, run each reset, tracking the live handle xpos) — the default agentview sits on the +x side and only sees the cabinet's closed back wall, since the drawer's working face points at the robot (−x). The bottle keeps the default agentview (already a good top-down on the cap). Both recorders (`record_*_video.py`) also accept `--object-scale/--px/--py` for per-config mp4s.

- **`assets/meshes/*`** — `*_original.stl` are the raw uploads; `*_visual_m.stl` are preprocessed (**mm → meters, xy-centered, bottom at z=0**). `bottle_body_m.stl` / `bottle_cap_m.stl` are generated by `scripts/preprocess_meshes.py`, which splits the bottle's two connected components and applies one uniform `--scale` (default **1.0** = full CAD size, since the cap is pushed not grasped) — the parts are **not** re-centered independently, so the cap keeps its z offset above the body.

## Conventions that bite

- **Quaternions**: MuJoCo uses `wxyz`; robosuite observables convert to `xyzw` via `convert_quat(..., to="xyzw")`. Body free-joint quats are `wxyz`.
- **Table height**: `table_offset[2]` (0.80) is the tabletop z in robosuite; objects are placed slightly above it.
- **robosuite version compatibility** is handled defensively in `scripts/run_random_robosuite_env.py`: `load_basic_osc_controller()` tries the 1.5 composite-controller API then falls back to the older `OSC_POSE` config, and `step_env()` handles both the 4-tuple and Gymnasium 5-tuple `step()` returns. Preserve these shims when editing.
- **Body/joint/site-name resolution**: the static scene uses `find_body_id` (in `env.py`); the task envs use `utils.py:find_name`. Both do fuzzy substring matching to absorb robosuite 1.5.x object-name prefixing — don't hard-code prefixed names.
