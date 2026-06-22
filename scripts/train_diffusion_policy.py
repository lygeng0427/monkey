#!/usr/bin/env python3
"""Train an image-based robomimic Diffusion Policy on a collected task dataset.

One policy per task (`--task drawer` or `--task bottle`), trained on the offline
state->image dataset `data/<task>_img.hdf5` (obs = agentview + sideview 84x84 RGB
plus robot0 proprio; actions = 7-d OSC_POSE deltas).

We drive robomimic's own training loop (`robomimic.scripts.train.train`) with a
`diffusion_policy` config built from the bundled template, overriding only what we
need. Two deliberate choices:
  * **Rollouts during training are DISABLED** (`experiment.rollout.enabled=False`):
    robomimic's `EnvRobosuite` wrapper does not speak robosuite 1.5's composite-
    controller API, so we evaluate separately with scripts/eval_diffusion_policy.py
    (which builds OUR env directly). Training only needs the HDF5.
  * A **train/val split** is added in-place to the dataset (robomimic's
    `split_train_val_from_hdf5`, creating mask/train + mask/valid) if not present,
    and we validate on it (best-val checkpoint is saved).

    MUJOCO_GL=egl is NOT needed for training (no rendering). Run from repo root:

    python scripts/train_diffusion_policy.py --task drawer
    python scripts/train_diffusion_policy.py --task bottle --num-epochs 600 --batch-size 64
    # quick smoke test (a couple tiny epochs, just checks the pipeline runs):
    python scripts/train_diffusion_policy.py --task drawer --num-epochs 2 --epoch-steps 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# RGB cameras + low-dim proprio that render_obs.py wrote into <task>_img.hdf5.
RGB_KEYS = ["agentview_image", "sideview_image"]
LOWDIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def ensure_train_val_split(dataset_path, val_ratio, train_key="train", valid_key="valid"):
    """Add mask/train + mask/valid filter keys in-place if absent (robomimic util)."""
    with h5py.File(dataset_path, "r") as f:
        has_masks = "mask" in f and train_key in f["mask"] and valid_key in f["mask"]
    if has_masks:
        print(f"[split] mask/{train_key} + mask/{valid_key} already present.")
        return
    from robomimic.scripts.split_train_val import split_train_val_from_hdf5
    print(f"[split] creating mask/{train_key} + mask/{valid_key} (val_ratio={val_ratio})")
    split_train_val_from_hdf5(dataset_path, val_ratio=val_ratio, filter_key=None)


def build_config(task, data_path, output_dir, num_epochs, epoch_steps, batch_size,
                 val_ratio, name, seed, num_workers, cache_mode="all",
                 obs_horizon=2, action_horizon=8, pred_horizon=16):
    from robomimic.config import config_factory

    # Start from the diffusion_policy defaults (correct encoder/horizon/ddpm), then
    # override only the dataset-, obs-, and run-specific pieces.
    config = config_factory("diffusion_policy")
    with config.values_unlocked():
        # --- experiment: validate on the val split, NO env rollouts (eval separately)
        config.experiment.name = name
        config.experiment.validate = True
        config.experiment.epoch_every_n_steps = epoch_steps
        config.experiment.validation_epoch_every_n_steps = max(1, epoch_steps // 10)
        config.experiment.rollout.enabled = False           # <- key: no EnvRobosuite
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = max(10, num_epochs // 5)
        config.experiment.save.on_best_validation = True
        config.experiment.save.on_best_rollout_success_rate = False
        config.experiment.logging.log_tb = True

        # --- train: point at our HDF5 + the filter keys we just created
        config.train.data = str(data_path)
        config.train.output_dir = str(output_dir)
        config.train.num_epochs = num_epochs
        config.train.batch_size = batch_size
        config.train.num_data_workers = num_workers
        # cache ALL obs (incl. images) in RAM (default): the image datasets are ~3-5GB
        # uncompressed and fit, eliminating per-batch disk reads (training was ~97%
        # data-loading bound). 'low_dim' is the old, slow path (images read from disk).
        config.train.hdf5_cache_mode = None if cache_mode == "none" else cache_mode
        config.train.hdf5_filter_key = "train"
        config.train.hdf5_validation_filter_key = "valid"
        config.train.dataset_keys = ["actions", "rewards", "dones"]
        config.train.seed = seed
        config.train.cuda = True

        # --- diffusion horizons (template default To=2/Ta=8/Tp=16). Only To and Tp
        # affect TRAINING: process_batch slices obs to `frame_stack`=To and actions to
        # `seq_length`=Tp (action_horizon Ta is INFERENCE-only -- receding-horizon
        # execution -- so it does not change the loss). Keep frame_stack==To and
        # seq_length==Tp in lockstep with the algo horizons or the data loader and the
        # net's global_cond_dim disagree.
        config.algo.horizon.observation_horizon = obs_horizon
        config.algo.horizon.action_horizon = action_horizon
        config.algo.horizon.prediction_horizon = pred_horizon
        config.train.frame_stack = obs_horizon
        config.train.seq_length = pred_horizon

        # --- observation: two RGB cameras + proprio; ResNet18 + crop randomization
        config.observation.modalities.obs.rgb = list(RGB_KEYS)
        config.observation.modalities.obs.low_dim = list(LOWDIM_KEYS)
        config.observation.encoder.rgb.core_class = "VisualCore"
        config.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"
        config.observation.encoder.rgb.obs_randomizer_kwargs = dict(
            crop_height=76, crop_width=76, num_crops=1, pos_enc=False,
        )
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["drawer", "bottle"])
    p.add_argument("--data", default=None, help="Dataset HDF5 (default data/<task>_img.hdf5).")
    p.add_argument("--output", default=None, help="Output dir (default runs/diffusion_<task>).")
    p.add_argument("--num-epochs", type=int, default=600)
    p.add_argument("--epoch-steps", type=int, default=100, help="Gradient steps per robomimic 'epoch'.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-mode", default="all", choices=["all", "low_dim", "none"],
                   help="robomimic hdf5_cache_mode: 'all' caches images in RAM (fast); "
                        "'low_dim' reads images from disk per batch (old, slow).")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=1)
    # Diffusion horizons. Defaults reproduce the original runs (To=2/Ta=8/Tp=16).
    p.add_argument("--obs-horizon", type=int, default=2, help="To: stacked obs frames (=frame_stack).")
    p.add_argument("--action-horizon", type=int, default=8, help="Ta: executed steps/replan (inference-only).")
    p.add_argument("--pred-horizon", type=int, default=16, help="Tp: predicted action chunk (=seq_length).")
    args = p.parse_args()

    data_path = Path(args.data) if args.data else REPO_ROOT / "data" / f"{args.task}_img.hdf5"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path} (run render_obs.py first).")
    # Resolve to ABSOLUTE: robomimic's get_exp_dir resolves a *relative* output_dir
    # against the robomimic install dir (not CWD), so a relative --output would write
    # checkpoints under site-packages and miss our repo runs/.
    output_dir = (Path(args.output).resolve() if args.output
                  else REPO_ROOT / "runs" / f"diffusion_{args.task}")
    name = args.name or f"diffusion_{args.task}"

    ensure_train_val_split(data_path, args.val_ratio)

    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.scripts.train import train

    config = build_config(
        args.task, data_path, output_dir, args.num_epochs, args.epoch_steps,
        args.batch_size, args.val_ratio, name, args.seed, args.num_workers, args.cache_mode,
        obs_horizon=args.obs_horizon, action_horizon=args.action_horizon, pred_horizon=args.pred_horizon,
    )
    # Assert the 5 coupled horizon values actually propagated (a horizon change is a
    # classic source of frame-stack shape mismatches at the first batch).
    assert config.algo.horizon.observation_horizon == args.obs_horizon == config.train.frame_stack
    assert config.algo.horizon.prediction_horizon == args.pred_horizon == config.train.seq_length
    assert config.algo.horizon.action_horizon == args.action_horizon
    config.lock()
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    print(f"[train] task={args.task} data={data_path} device={device} "
          f"epochs={args.num_epochs}x{args.epoch_steps}steps batch={args.batch_size} "
          f"To={args.obs_horizon} Ta={args.action_horizon} Tp={args.pred_horizon}")
    train(config, device=device)
    print(f"[train] DONE. checkpoints under {output_dir}/{name}/")


if __name__ == "__main__":
    main()
