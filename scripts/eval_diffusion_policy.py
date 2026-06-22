#!/usr/bin/env python3
"""Evaluate a trained robomimic Diffusion Policy on a task, across SEEN + UNSEEN configs.

Loads the trained policy from a robomimic checkpoint and rolls it out in OUR env
(built directly via suite.make, NOT robomimic's EnvRobosuite -- which doesn't speak
robosuite 1.5's composite-controller API). For each (object_scale, placement_xy) it
runs N stochastic rollouts and reports success rate, aggregated by category:

  * SEEN        : the 3 sizes x 9 positions the policy was trained on
  * UNSEEN_SIZE : held-out sizes INSIDE [0.85,1.15] at seen positions
  * UNSEEN_POS  : seen size, held-out positions INSIDE +/-0.05
  * UNSEEN_BOTH : held-out sizes x held-out positions (in-range corners)
  * UNSEEN_OOB  : mild extrapolation just OUTSIDE the seen min-max boundary

Per config we also save the first SUCCESS rollout as a <=5 s mp4 and the first FAILURE
rollout as a <=10 s mp4 (sped up by subsampling to a fixed fps) under videos/eval/<task>/.

Per step we build the obs dict EXACTLY as render_obs.py did at train time -- the two
84x84 cameras via np.flipud(sim.render(...)) plus robot0 proprio from the env's own
observables -- so train/eval pixel distributions match. The RolloutPolicy buffers the
observation horizon and chunks actions internally, so we feed one obs dict per step.

Needs an EGL offscreen context. Run from repo root:

    MUJOCO_GL=egl python scripts/eval_diffusion_policy.py --task drawer
    MUJOCO_GL=egl python scripts/eval_diffusion_policy.py --task bottle \
        --ckpt runs/diffusion_bottle/.../models/model_epoch_600.pth --n-rollouts 20
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import robosuite as suite

import franka_drawer_bottle  # noqa: F401  (registers both task envs)
from scripts.demo_common import load_controller_config

CAMERAS = ("agentview", "sideview")
LOWDIM_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")

# --- Demonstration video settings -------------------------------------------------
# Per config we save the first SUCCESS rollout (<=5 s) and the first FAILURE rollout
# (<=10 s). The rollout runs at control_freq=20 and can be hundreds/thousands of steps
# (10-50 s real time), so we speed it up with ONE universal rule: write at a fixed
# VIDEO_FPS and SUBSAMPLE frames down to the time cap (success 5 s, failure 10 s).
VIDEO_CAMERA = "sideview"   # side scene view (shows arm + drawer slide / bottle cap)
# Cameras stitched (left->right) into each rollout mp4. Default = the single side
# view (backward-compatible); --video-cameras overrides, e.g. "agentview,sideview"
# for the drawer, whose handle/slide face the robot and are largely hidden in the
# pure sideview. All listed cameras must be in CAMERAS (so they're already rendered).
VIDEO_CAMERAS = [VIDEO_CAMERA]
VIDEO_SIZE = 256            # render res for the mp4 (obs still rendered at 84 for the policy)
VIDEO_FPS = 30
SUCCESS_SEC, FAILURE_SEC = 5, 10

TASKS = {
    "drawer": dict(env_name="FrankaDrawerOpen", nominal_xy=(0.05, 0.0), max_horizon=400),
    "bottle": dict(env_name="FrankaBottleUntwist", nominal_xy=(0.10, 0.0), max_horizon=1000),
}

# Seen = trained grid. ALL unseen configs INTERPOLATE strictly INSIDE the seen
# min-max hull (no extrapolation): sizes within [0.85, 1.15], offsets within
# [-0.05, 0.05]. Each unseen value is held-out (off the seen grid) but in-range.
SEEN_SIZES = [0.85, 1.0, 1.15]
SEEN_OFFSETS = [-0.05, 0.0, 0.05]        # 3x3 grid the policy trained on
# Held-out sizes, all inside [0.85, 1.15] (one in the lower cell, two in the upper).
UNSEEN_SIZES = [0.925, 1.05, 1.10]
# Held-out position offsets inside [-0.05, 0.05], none equal to a seen offset
# {-0.05, 0, 0.05} -> every 3x3 combo is a genuinely unseen, in-range position.
UNSEEN_OFFSETS = [-0.04, 0.01, 0.04]
# In-range corner offsets for the size x position (UNSEEN_BOTH) probe.
UNSEEN_CORNERS = [(-0.04, -0.04), (0.04, 0.04), (0.04, -0.04)]
# EXTRAPOLATION configs OUTSIDE the seen hull (sizes < 0.85 or > 1.15, offsets beyond
# +/-0.05). Sized to MIRROR the 27 in-range unseen configs (9 size + 9 pos + 9 both) so
# the OOB and in-boundary unseen sets are directly comparable.
OOB_SIZES = [0.78, 1.20, 1.25]                  # 1 below 0.85, 2 above 1.15 (all OOB)
OOB_OFFSETS = [-0.075, 0.075, 0.10]             # position offsets, all beyond +/-0.05
OOB_CORNERS = [(-0.075, -0.075), (0.075, 0.075), (0.075, -0.075)]  # OOB size x OOB pos


def build_configs(nominal_xy, categories):
    """Return list of dicts {scale, xy, category}. nominal_xy is the task's center."""
    nx, ny = nominal_xy
    seen_pos = [(nx + dx, ny + dy) for dx in SEEN_OFFSETS for dy in SEEN_OFFSETS]
    cfgs = []
    if "seen" in categories:
        for s in SEEN_SIZES:
            for xy in seen_pos:
                cfgs.append(dict(scale=s, xy=xy, category="SEEN"))
    # --- 27 unseen configs (9 per sub-category), mirroring the seen grid's size. ---
    if "unseen-size" in categories:
        # Held-out SIZE at in-distribution (seen) x-positions -> isolates size generalization.
        for s in UNSEEN_SIZES:
            for dx in SEEN_OFFSETS:
                cfgs.append(dict(scale=s, xy=(nx + dx, ny), category="UNSEEN_SIZE"))
    if "unseen-pos" in categories:
        # Seen SIZE (1.0) at a 3x3 grid of held-out positions -> isolates position generalization.
        for dx in UNSEEN_OFFSETS:
            for dy in UNSEEN_OFFSETS:
                cfgs.append(dict(scale=1.0, xy=(nx + dx, ny + dy), category="UNSEEN_POS"))
    if "unseen-both" in categories:
        # Held-out SIZE x held-out (beyond-grid) corner positions -> hardest, both shifted.
        for s in UNSEEN_SIZES:
            for dx, dy in UNSEEN_CORNERS:
                cfgs.append(dict(scale=s, xy=(nx + dx, ny + dy), category="UNSEEN_BOTH"))
    if "unseen-oob" in categories:
        # EXTRAPOLATION outside the seen hull, mirroring the 27 in-range unseen configs:
        # 9 size-OOB + 9 position-OOB + 9 both-OOB = 27.
        for s in OOB_SIZES:                                  # OOB size, in-range x-positions
            for dx in SEEN_OFFSETS:
                cfgs.append(dict(scale=s, xy=(nx + dx, ny), category="UNSEEN_OOB"))
        for dx in OOB_OFFSETS:                               # OOB position (3x3), in-range size
            for dy in OOB_OFFSETS:
                cfgs.append(dict(scale=1.0, xy=(nx + dx, ny + dy), category="UNSEEN_OOB"))
        for s in OOB_SIZES:                                  # OOB size AND OOB position (corners)
            for dx, dy in OOB_CORNERS:
                cfgs.append(dict(scale=s, xy=(nx + dx, ny + dy), category="UNSEEN_OOB"))
    return cfgs


def make_eval_env(env_name, scale, xy, max_horizon, init_noise):
    return suite.make(
        env_name=env_name,
        robots="Panda",
        gripper_types="default",
        controller_configs=load_controller_config(),
        has_renderer=False,
        has_offscreen_renderer=True,
        render_visual_mesh=True,
        render_collision_mesh=False,
        use_camera_obs=False,            # we render the cameras ourselves (flip-match train)
        camera_names=list(CAMERAS),
        camera_heights=VIDEO_SIZE,       # offscreen buffer >= video res; obs still rendered at 84
        camera_widths=VIDEO_SIZE,
        control_freq=20,
        horizon=max_horizon,
        ignore_done=True,
        hard_reset=False,
        initialization_noise=init_noise,  # small arm-start jitter -> stochastic rollouts
        object_scale=scale,
        placement_xy=tuple(xy),
    )


def build_obs(env, raw_obs):
    """Obs dict matching the training keys: 2 RGB (flipped like render_obs) + proprio."""
    ob = {}
    for cam in CAMERAS:
        img = env.sim.render(width=84, height=84, camera_name=cam)
        ob[f"{cam}_image"] = np.flipud(img).copy()           # HWC uint8, MuJoCo is bottom-up
    for k in LOWDIM_KEYS:
        ob[k] = np.asarray(raw_obs[k], dtype=np.float32)
    return ob


def step_env(env, action):
    """robosuite version shim: handle 4-tuple and Gymnasium 5-tuple step() returns."""
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, (terminated or truncated), info
    return out  # (obs, reward, done, info)


def use_ddim_fast_sampler(policy, n_steps):
    """Swap the policy's sampler to DDIM for fast few-step inference.

    The policy was trained with a 100-step DDPM. DDPM degrades badly with few steps,
    but the same epsilon-prediction model samples near-losslessly with a DDIM scheduler
    in ~10-20 steps (standard Diffusion Policy fast inference) -- ~6x faster for eval.
    We build a DDIM scheduler from the trained DDPM's beta/prediction settings and flip
    the algo's ddpm/ddim flags so get_action reads the new step count + scheduler.
    """
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    algo = policy.policy
    d = algo.algo_config.ddpm
    algo.noise_scheduler = DDIMScheduler(
        num_train_timesteps=d.num_train_timesteps,
        beta_schedule=d.beta_schedule,
        clip_sample=d.clip_sample,
        prediction_type=d.prediction_type,
        set_alpha_to_one=True,
        steps_offset=0,
    )
    with algo.algo_config.values_unlocked():
        algo.algo_config.ddpm.enabled = False
        algo.algo_config.ddim.enabled = True
        algo.algo_config.ddim.num_inference_timesteps = n_steps
        algo.algo_config.ddim.num_train_timesteps = d.num_train_timesteps
        algo.algo_config.ddim.beta_schedule = d.beta_schedule
        algo.algo_config.ddim.clip_sample = d.clip_sample
        algo.algo_config.ddim.prediction_type = d.prediction_type
    print(f"[eval] fast sampler: DDIM x{n_steps} (was DDPM x{d.num_inference_timesteps})")


def find_checkpoint(task, explicit):
    if explicit:
        return Path(explicit)
    root = REPO_ROOT / "runs" / f"diffusion_{task}"
    ckpts = list(root.rglob("models/*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {root}; pass --ckpt explicitly.")
    # Prefer a best-validation ckpt; else the most recently modified.
    best = [c for c in ckpts if "best" in c.name]
    pool = best or ckpts
    return max(pool, key=lambda c: c.stat().st_mtime)


def _stack_history(hist):
    """Concatenate the per-key deque of [1, ...] frames into [To, ...] (robomimic
    FrameStackWrapper convention)."""
    return {k: np.concatenate(list(hist[k]), axis=0) for k in hist}


def _video_frame(env):
    """One upright RGB frame for the rollout mp4. Renders each camera in
    VIDEO_CAMERAS (MuJoCo is bottom-up -> flipud) and stitches them left-to-right
    with a thin divider, so a single clip can carry e.g. agentview + sideview."""
    panels = []
    for cam in VIDEO_CAMERAS:
        img = np.flipud(env.sim.render(width=VIDEO_SIZE, height=VIDEO_SIZE,
                                       camera_name=cam)).copy()
        panels.append(img)
    if len(panels) == 1:
        return panels[0]
    sep = np.full((panels[0].shape[0], 4, 3), 60, dtype=np.uint8)  # thin gray divider
    stitched = [panels[0]]
    for p in panels[1:]:
        stitched += [sep, p]
    return np.hstack(stitched)


def _save_video(frames, path, max_seconds):
    """Write frames to mp4, sped up by subsampling to a fixed VIDEO_FPS so the clip
    lasts at most `max_seconds` (universal scale: long rollouts get more sped up)."""
    import imageio
    target = max(1, int(round(VIDEO_FPS * max_seconds)))
    if len(frames) > target:
        idx = np.linspace(0, len(frames) - 1, target).round().astype(int)
        frames = [frames[i] for i in idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=VIDEO_FPS, macro_block_size=16)


def rollout(env, policy, max_horizon, obs_horizon, record=False):
    """One stochastic rollout. The diffusion policy expects obs stacked over the
    observation horizon (To frames) along a leading time axis -- robomimic normally
    supplies this via FrameStackWrapper; we replicate it (pad the initial history
    with To copies of the first obs, then slide the window).

    Returns (success, frames); `frames` is the per-step side-camera video (empty unless
    `record`, captured only when this config still needs a success/failure clip)."""
    raw = env.reset()
    policy.start_episode()
    ob = build_obs(env, raw)
    frames = [_video_frame(env)] if record else []
    # Pad initial history with To copies of the first observation.
    hist = {k: deque([v[None]] * obs_horizon, maxlen=obs_horizon) for k, v in ob.items()}
    for _ in range(max_horizon):
        act = policy(ob=_stack_history(hist))     # RolloutPolicy adds batch -> [1, To, ...]
        raw, _, _, _ = step_env(env, np.asarray(act))
        if record:
            frames.append(_video_frame(env))
        if env._check_success():
            return True, frames
        ob = build_obs(env, raw)
        for k, v in ob.items():
            hist[k].append(v[None])
    return env._check_success(), frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--ckpt", default=None, help="Checkpoint .pth (default: newest best under runs/diffusion_<task>).")
    p.add_argument("--n-rollouts", type=int, default=10, help="Rollouts per config.")
    p.add_argument("--categories", default="seen,unseen-size,unseen-pos,unseen-both,unseen-oob",
                   help="Comma list: seen,unseen-size,unseen-pos,unseen-both,unseen-oob.")
    p.add_argument("--video-dir", default=None,
                   help="Dir for per-config success/failure mp4s (default videos/eval/<task>).")
    p.add_argument("--no-videos", action="store_true", help="Disable saving demonstration videos.")
    p.add_argument("--video-cameras", default=None,
                   help="Comma list of cameras stitched L->R into each rollout mp4 "
                        "(default 'sideview'; e.g. 'agentview,sideview'). Must be in CAMERAS.")
    p.add_argument("--max-horizon", type=int, default=None, help="Override per-task default.")
    p.add_argument("--init-noise-mag", type=float, default=0.02,
                   help="Arm-start joint noise magnitude for stochastic rollouts (0 = deterministic).")
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="If set, swap to a DDIM sampler with this many steps (fast eval; e.g. 16).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON results path (default eval/<task>_eval.json).")
    args = p.parse_args()

    cfg = TASKS[args.task]
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    max_horizon = args.max_horizon or cfg["max_horizon"]
    init_noise = None if args.init_noise_mag <= 0 else {"magnitude": args.init_noise_mag, "type": "gaussian"}
    if args.video_cameras:
        global VIDEO_CAMERAS
        cams = [c.strip() for c in args.video_cameras.split(",") if c.strip()]
        bad = [c for c in cams if c not in CAMERAS]
        if bad:
            raise SystemExit(f"--video-cameras {bad} not in rendered CAMERAS={CAMERAS}")
        VIDEO_CAMERAS = cams
        print(f"[eval] rollout video cameras = {VIDEO_CAMERAS}")
    np.random.seed(args.seed)

    import torch  # noqa: F401
    import robomimic.utils.torch_utils as TorchUtils
    import robomimic.utils.file_utils as FileUtils

    ckpt = find_checkpoint(args.task, args.ckpt)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"[eval] task={args.task} ckpt={ckpt}\n[eval] device={device} n={args.n_rollouts} horizon={max_horizon}")
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(ckpt), device=device, verbose=False)
    if args.num_inference_steps:
        use_ddim_fast_sampler(policy, args.num_inference_steps)
    try:
        obs_horizon = int(policy.policy.algo_config.horizon.observation_horizon)
    except Exception:
        obs_horizon = 2
    print(f"[eval] observation_horizon (To) = {obs_horizon}")

    video_dir = Path(args.video_dir) if args.video_dir else REPO_ROOT / "videos" / "eval" / args.task
    save_videos = not args.no_videos

    def _clip_path(c, kind):
        name = (f"{c['category']}_s{c['scale']:.3f}_"
                f"x{c['xy'][0]:.3f}_y{c['xy'][1]:.3f}_{kind}.mp4")
        return video_dir / name

    configs = build_configs(cfg["nominal_xy"], categories)
    results = []
    for i, c in enumerate(configs):
        env = make_eval_env(cfg["env_name"], c["scale"], c["xy"], max_horizon, init_noise)
        n_succ = 0
        need_succ_vid = need_fail_vid = save_videos   # capture first success + first failure
        for _ in range(args.n_rollouts):
            record = need_succ_vid or need_fail_vid
            ok, frames = rollout(env, policy, max_horizon, obs_horizon, record=record)
            n_succ += int(ok)
            if ok and need_succ_vid:
                _save_video(frames, _clip_path(c, "success"), SUCCESS_SEC); need_succ_vid = False
            elif (not ok) and need_fail_vid:
                _save_video(frames, _clip_path(c, "failure"), FAILURE_SEC); need_fail_vid = False
        env.close()
        rate = n_succ / args.n_rollouts
        results.append(dict(**c, n=args.n_rollouts, n_success=n_succ, rate=rate))
        print(f"  [{i+1}/{len(configs)}] {c['category']:<11} scale={c['scale']:<5} "
              f"xy=({c['xy'][0]:.3f},{c['xy'][1]:.3f}) : {n_succ}/{args.n_rollouts}  ({rate:.0%})")

    # Aggregate by category + overall.
    print("\n=== SUMMARY ===")
    summary = {}
    for cat in sorted(set(r["category"] for r in results)):
        rs = [r for r in results if r["category"] == cat]
        succ = sum(r["n_success"] for r in rs)
        tot = sum(r["n"] for r in rs)
        summary[cat] = dict(success=succ, total=tot, rate=succ / tot)
        print(f"  {cat:<11}: {succ}/{tot}  ({succ/tot:.1%})  over {len(rs)} configs")
    all_succ = sum(r["n_success"] for r in results)
    all_tot = sum(r["n"] for r in results)
    summary["OVERALL"] = dict(success=all_succ, total=all_tot, rate=all_succ / all_tot)
    print(f"  {'OVERALL':<11}: {all_succ}/{all_tot}  ({all_succ/all_tot:.1%})")

    out = Path(args.out) if args.out else REPO_ROOT / "eval" / f"{args.task}_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(task=args.task, ckpt=str(ckpt), per_config=results, summary=summary), f, indent=2)
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
