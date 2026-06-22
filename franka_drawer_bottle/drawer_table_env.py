"""FrankaDrawerOpenTable: the articulated drawer resting ON the table.

This is a sibling of ``FrankaDrawerOpen`` (drawer_env.py) that does NOT float the
cabinet in the air. The cabinet bottom sits flush on the tabletop, so the handle
lives at the usual table-pick height (world z ~= table_offset_z + 0.082*scale ~=
0.882 at scale 1.0). At that height a *horizontal* grasp is infeasible (the Panda
wrist can only point horizontal with the arm fully extended up -- the reason the
original task floats the drawer), so the matching collector
(scripts/collect_drawer_table_demos.py) grasps the protruding loop handle bar with
a TOP-DOWN grip instead and pulls the drawer toward the robot.

Everything else (slide joint, handle site, success = slide displacement past
``success_thresh``, the top-down agentview repoint) is inherited unchanged from
FrankaDrawerOpen; only the placement z differs. Importing this module registers
the env with robosuite by class name.
"""
from __future__ import annotations

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

from .drawer_env import DrawerArticulatedObject, FrankaDrawerOpen


class FrankaDrawerOpenTable(FrankaDrawerOpen):
    """Drawer resting on the table (cabinet bottom flush on the tabletop)."""

    def _load_model(self):
        # Skip FrankaDrawerOpen._load_model (which floats the object) and rebuild
        # the scene with the cabinet resting on the tabletop instead.
        ManipulationEnv._load_model(self)

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
        # Cabinet bottom geom bottom face is at object-local z=0, so placing the
        # object base at the tabletop z rests the cabinet flush on the table. The
        # object is welded to the world (no joint), so it stays put with no support.
        # A 1 mm lift keeps the cabinet-bottom and tabletop from being exactly
        # coplanar (avoids render z-fighting flicker at the seam).
        drawer_z = self.table_offset[2] + 0.001
        self.drawer.set_pos([self.placement_xy[0], self.placement_xy[1], float(drawer_z)])

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.drawer],
        )
