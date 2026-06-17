"""Starter + task robosuite environments for the uploaded drawer + bottle CAD meshes.

Importing this package registers all environments with robosuite's env registry:
    - FrankaDrawerBottleScene : static scene-preview (CAD scale/placement check)
    - FrankaDrawerOpen        : open the articulated drawer (slide joint)
    - FrankaBottleUntwist     : untwist the bottle cap (hinge joint)
"""
from .env import FrankaDrawerBottleScene
from .drawer_env import FrankaDrawerOpen
from .bottle_env import FrankaBottleUntwist

__all__ = ["FrankaDrawerBottleScene", "FrankaDrawerOpen", "FrankaBottleUntwist"]
