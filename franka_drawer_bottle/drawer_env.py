"""FrankaDrawerOpen: pull the articulated drawer open with a Panda arm.

The drawer object (assets/objects/drawer_articulated.xml) is a static primitive
cabinet plus the uploaded CAD drawer mesh on a slide joint, with an added grasp
handle. Success = the slide joint travels past a threshold distance.
"""
from __future__ import annotations

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

from .utils import ASSET_ROOT, find_name


class DrawerArticulatedObject(MujocoXMLObject):
    """Cabinet + sliding drawer. The slide joint is declared in the XML."""

    def __init__(self, name: str, scale=None):
        super().__init__(
            fname=str(ASSET_ROOT / "objects" / "drawer_articulated.xml"),
            name=name,
            joints=None,  # joints are declared inside the XML
            obj_type="all",
            duplicate_collision_geoms=False,
            scale=scale,  # robosuite scales geom size/pos, mesh scale, body pos, sites
        )


class FrankaDrawerOpen(ManipulationEnv):
    """Panda arm + table + articulated drawer. Task: open the drawer.

    Reward (when ``reward_shaping``): a reaching term toward the handle plus an
    opening term proportional to slide displacement; a sparse 1.0 on success.
    """

    # Object-local z of handle_site at object_scale=1.0: tray body pos z (0.018) +
    # handle_site local z (0.064). Scales linearly with object_scale; used to hold the
    # handle's world height fixed across sizes (see _load_model).
    _HANDLE_LOCAL_Z = 0.082

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
        success_thresh=0.18,
        placement_xy=(0.05, 0.0),
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
        # Success is slide *displacement*, so it scales with object size (the whole
        # drawer/handle grow uniformly). The slide joint range (0.32) is unscaled, so
        # keep success_thresh * scale comfortably below it (0.18*1.15=0.207 < 0.32).
        self.success_thresh = float(success_thresh) * self.object_scale
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

        self.drawer = DrawerArticulatedObject(
            name="drawer", scale=(self.object_scale if self.object_scale != 1.0 else None)
        )
        # Rotate so the +y opening/handle faces the robot (local +y -> world -x).
        self.drawer.set_euler([0.0, 0.0, np.pi / 2.0])
        # The object is welded to the world (no joint on the object body), so place
        # it ABOVE the table and it simply FLOATS there with nothing below it. The
        # whole object scales about its base (z=0), so the handle's object-local
        # height is HANDLE_LOCAL_Z * object_scale. We pick the placement z so the
        # handle's WORLD z stays at the reachable height (~1.15) the ~45 deg tilted
        # grasp is tuned for, REGARDLESS of size -- scaling the 0.27 offset instead
        # would push the handle out of reach at the size extremes.
        # (A pedestal under it is purely cosmetic and was removed; a tall table does
        # not work -- it traps the gripper under the raised surface.)
        handle_world_z = self.table_offset[2] + 0.271 + self._HANDLE_LOCAL_Z
        drawer_z = handle_world_z - self._HANDLE_LOCAL_Z * self.object_scale
        self.drawer.set_pos([self.placement_xy[0], self.placement_xy[1], float(drawer_z)])

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.drawer],
        )

    # -------------------------------------------------------------- references
    def _setup_references(self):
        super()._setup_references()

        self.slide_joint = find_name(self.sim.model.joint_names, "slide")
        self.slide_qpos_addr = self.sim.model.get_joint_qpos_addr(self.slide_joint)
        self.handle_site_id = self.sim.model.site_name2id(
            find_name(self.sim.model.site_names, "handle_site")
        )

    # ----------------------------------------------------------------- helpers
    @property
    def _slide_pos(self):
        return float(self.sim.data.qpos[self.slide_qpos_addr])

    @property
    def _handle_xpos(self):
        return np.array(self.sim.data.site_xpos[self.handle_site_id])

    @property
    def _gripper_to_handle(self):
        dists = [
            np.linalg.norm(self._handle_xpos - np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]]))
            for arm in self.robots[0].arms
        ]
        return min(dists)

    # ------------------------------------------------------------- observables
    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def drawer_slide(obs_cache):
                return np.array([self._slide_pos])

            @sensor(modality=modality)
            def handle_pos(obs_cache):
                return self._handle_xpos

            @sensor(modality=modality)
            def gripper_to_handle(obs_cache):
                return np.array([self._gripper_to_handle])

            sensors = [drawer_slide, handle_pos, gripper_to_handle]
            for s in sensors:
                observables[s.__name__] = Observable(
                    name=s.__name__, sensor=s, sampling_rate=self.control_freq
                )

        return observables

    # ----------------------------------------------------------------- dynamics
    def _reset_internal(self):
        super()._reset_internal()
        # Start closed.
        self.sim.data.qpos[self.slide_qpos_addr] = 0.0
        self.sim.forward()
        self._aim_agentview_topdown()

    # ------------------------------------------------------------------ cameras
    def _aim_agentview_topdown(self):
        """Repoint 'agentview' to look straight DOWN at the handle.

        The drawer's opening/handle faces the robot (world -x), while the default
        TableArena agentview sits on the +x side and only sees the cabinet's closed
        back wall. A top-down camera over the handle shows the loop handle, the
        gripper, and the -x slide clearly. We read the live handle xpos so the view
        auto-tracks placement_xy and object_scale; the drawer slides within frame.
        """
        try:
            cid = self.sim.model.camera_name2id("agentview")
        except Exception:
            return  # no agentview camera in this model build; nothing to do
        h = self._handle_xpos
        eye = np.array([h[0], h[1], h[2] + 0.47])
        self.sim.model.cam_pos[cid] = eye
        self.sim.model.cam_quat[cid] = self._lookat_quat_wxyz(eye, h, up=(1.0, 0.0, 0.0))

    @staticmethod
    def _lookat_quat_wxyz(eye, target, up=(0.0, 0.0, 1.0)):
        """MuJoCo camera quat (wxyz) so the camera at ``eye`` looks at ``target``.

        MuJoCo cameras view along -z of the camera frame, +y up. ``up`` is the world
        direction that should map to image-up (use +x for a top-down view).
        """
        import robosuite.utils.transform_utils as T

        eye = np.asarray(eye, float)
        fwd = np.asarray(target, float) - eye
        fwd /= np.linalg.norm(fwd)
        cam_z = -fwd
        cam_x = np.cross(np.asarray(up, float), cam_z)
        cam_x /= np.linalg.norm(cam_x)
        cam_y = np.cross(cam_z, cam_x)
        R = np.stack([cam_x, cam_y, cam_z], axis=1)
        q = T.mat2quat(R)  # xyzw
        return np.array([q[3], q[0], q[1], q[2]])  # wxyz

    def reward(self, action=None):
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            reaching = 0.25 * (1 - np.tanh(10.0 * self._gripper_to_handle))
            opening = 0.75 * np.clip(self._slide_pos / self.success_thresh, 0.0, 1.0)
            reward = reaching + opening
        else:
            reward = 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _check_success(self):
        return self._slide_pos > self.success_thresh
