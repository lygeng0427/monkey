"""FrankaBottleUntwist: rotate the bottle cap with a Panda arm.

The bottle object (assets/objects/bottle_articulated.xml) is a static bottle body
with a wide cap on a revolute (hinge) joint about the vertical axis. Success = the
cap is rotated past a threshold angle.
"""
from __future__ import annotations

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

from .utils import ASSET_ROOT, find_name


class BottleArticulatedObject(MujocoXMLObject):
    """Bottle body + cap. The cap hinge joint is declared in the XML."""

    def __init__(self, name: str, scale=None):
        super().__init__(
            fname=str(ASSET_ROOT / "objects" / "bottle_articulated.xml"),
            name=name,
            joints=None,  # joints are declared inside the XML
            obj_type="all",
            duplicate_collision_geoms=False,
            scale=scale,  # robosuite scales geom size/pos, mesh scale, body pos, sites
        )


class FrankaBottleUntwist(ManipulationEnv):
    """Panda arm + table + articulated bottle. Task: untwist (rotate) the cap.

    Reward (when ``reward_shaping``): a reaching term toward the cap plus a
    rotation term proportional to the cap angle; a sparse 1.0 on success.
    """

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.90, 0.80, 0.05),
        table_friction=(1.0, 0.005, 0.0001),
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        success_thresh=np.pi / 2.0,
        placement_xy=(0.10, 0.0),
        object_scale=1.0,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.80))
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.object_scale = float(object_scale)
        # Success is a cap *angle*, which is size-invariant, so the threshold does
        # NOT scale with object_scale (unlike the drawer's displacement threshold).
        self.success_thresh = success_thresh
        self.placement_xy = np.array(placement_xy, dtype=float)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ------------------------------------------------------------------ model
    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.bottle = BottleArticulatedObject(
            name="bottle", scale=(self.object_scale if self.object_scale != 1.0 else None)
        )
        # The bottle scales about its base (z=0), so it stays resting on the table
        # at any size; only the cap height grows (the cap_site is read live by the
        # scripted pusher, so the push adapts automatically).
        self.bottle.set_pos(
            [self.placement_xy[0], self.placement_xy[1], float(self.table_offset[2] + 0.001)]
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.bottle],
        )

    # -------------------------------------------------------------- references
    def _setup_references(self):
        super()._setup_references()

        self.cap_joint = find_name(self.sim.model.joint_names, "cap_joint")
        self.cap_qpos_addr = self.sim.model.get_joint_qpos_addr(self.cap_joint)
        self.cap_site_id = self.sim.model.site_name2id(
            find_name(self.sim.model.site_names, "cap_site")
        )
        # Reference at one cap tab/handle (rotates with the cap) for grasping.
        self.tab_site_id = self.sim.model.site_name2id(
            find_name(self.sim.model.site_names, "tab_site")
        )

    # ----------------------------------------------------------------- helpers
    @property
    def _cap_angle(self):
        return float(self.sim.data.qpos[self.cap_qpos_addr])

    @property
    def _cap_xpos(self):
        return np.array(self.sim.data.site_xpos[self.cap_site_id])

    @property
    def _gripper_to_cap(self):
        dists = [
            np.linalg.norm(self._cap_xpos - np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]]))
            for arm in self.robots[0].arms
        ]
        return min(dists)

    # ------------------------------------------------------------- observables
    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cap_angle(obs_cache):
                return np.array([self._cap_angle])

            @sensor(modality=modality)
            def cap_pos(obs_cache):
                return self._cap_xpos

            @sensor(modality=modality)
            def gripper_to_cap(obs_cache):
                return np.array([self._gripper_to_cap])

            sensors = [cap_angle, cap_pos, gripper_to_cap]
            for s in sensors:
                observables[s.__name__] = Observable(
                    name=s.__name__, sensor=s, sampling_rate=self.control_freq
                )

        return observables

    # ----------------------------------------------------------------- dynamics
    def _reset_internal(self):
        super()._reset_internal()
        # Start with the cap fully seated (unturned).
        self.sim.data.qpos[self.cap_qpos_addr] = 0.0
        self.sim.forward()

    def reward(self, action=None):
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            reaching = 0.25 * (1 - np.tanh(10.0 * self._gripper_to_cap))
            turning = 0.75 * np.clip(abs(self._cap_angle) / self.success_thresh, 0.0, 1.0)
            reward = reaching + turning
        else:
            reward = 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _check_success(self):
        return abs(self._cap_angle) > self.success_thresh
