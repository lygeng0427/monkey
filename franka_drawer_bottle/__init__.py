"""Starter robosuite environments for the uploaded drawer + bottle CAD meshes.

Importing this package registers FrankaDrawerBottleScene with robosuite's env registry.
"""
from .env import FrankaDrawerBottleScene

__all__ = ["FrankaDrawerBottleScene"]
