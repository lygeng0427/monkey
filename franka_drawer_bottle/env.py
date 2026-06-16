from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"


def _quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    """MuJoCo free-joint quaternion convention: w, x, y, z."""
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


class CADVisualObject(MujocoXMLObject):
    """Temporary CAD object for the starter viewer scene.

    The XML is still merged into MuJoCo, so the STL visual mesh and primitive
    collision proxy remain visible / physical. But for this first scene we do
    not want robosuite to build object-level geom/site ID mappings, because the
    temporary XML names can get double-prefixed in robosuite 1.5.x.

    Later, for the real drawer/cap task, we should replace this with a proper
    articulated robosuite object.
    """

    def __init__(self, name: str, xml_name: str):
        xml_path = ASSET_ROOT / "objects" / xml_name
        super().__init__(
            fname=str(xml_path),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def visual_geoms(self):
        return []

    @property
    def contact_geoms(self):
        return []

    @property
    def sites(self):
        return []


class FrankaDrawerBottleScene(ManipulationEnv):
    """Minimal scene: Panda arm + table + uploaded drawer and bottle meshes.

    This is intentionally a *starter scene*, not the final articulated drawer / cap task.
    It is useful for checking CAD scale, placement, camera, and controller setup before
    splitting the CAD into movable parts.
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
        self.table_offset = np.array((0.0, 0.0, 0.80))  # robosuite convention: table top height
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        # Fixed starter placements on top of the table. z is set in _reset_internal.
        self.drawer_xy_yaw = np.array([0.20, -0.18, 0.0])
        self.bottle_xy_yaw = np.array([0.20, 0.18, 0.0])

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

    def reward(self, action=None):
        # Placeholder. Later: drawer opening distance or bottle cap angle.
        return 0.0

    def _load_model(self):
        super()._load_model()

        # Put Panda in a standard table-top position.
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.drawer = CADVisualObject(name="drawer", xml_name="drawer_visual.xml")
        self.bottle = CADVisualObject(name="bottle", xml_name="bottle_visual.xml")

        # Static starter placements. table_offset[2] is the tabletop height in robosuite.
        # These are fixed/welded objects now, not free-falling dynamic bodies.
        self.drawer.set_pos([0.20, -0.18, float(self.table_offset[2] + 0.002)])
        self.bottle.set_pos([0.20,  0.18, float(self.table_offset[2] + 0.002)])

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.drawer, self.bottle],
        )

    def _setup_references(self):
        super()._setup_references()

        body_names = list(self.sim.model.body_names)

        def find_body_id(obj, base_name):
            candidates = [
                obj.root_body,
                f"{base_name}_main",
                f"{base_name}_{base_name}_main",
                f"{base_name}_object",
                f"{base_name}_{base_name}_object",
            ]

            for name in candidates:
                if name in body_names:
                    return self.sim.model.body_name2id(name)

            fuzzy = [
                n for n in body_names
                if base_name in n.lower() and ("main" in n.lower() or "object" in n.lower())
            ]
            if len(fuzzy) >= 1:
                return self.sim.model.body_name2id(fuzzy[0])

            raise ValueError(
                f"Could not find body for {base_name}.\n"
                f"Candidates tried: {candidates}\n"
                f"Fuzzy matches: {fuzzy}\n"
                f"All body names: {body_names}"
            )

        self.drawer_body_id = find_body_id(self.drawer, "drawer")
        self.bottle_body_id = find_body_id(self.bottle, "bottle")

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def drawer_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.drawer_body_id])

            @sensor(modality=modality)
            def drawer_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.drawer_body_id]), to="xyzw")

            @sensor(modality=modality)
            def bottle_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.bottle_body_id])

            @sensor(modality=modality)
            def bottle_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.bottle_body_id]), to="xyzw")

            sensors = [drawer_pos, drawer_quat, bottle_pos, bottle_quat]
            for s in sensors:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        # Objects are fixed in the starter scene, so there is no free-joint qpos to reset.
        # Later, replace this with articulated drawer / bottle-cap joint initialization.
        pass

    def _check_success(self):
        return False

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
