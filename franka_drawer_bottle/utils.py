"""Small helpers shared by the articulated task environments.

robosuite 1.5.x prefixes object body/joint/site names with the object name, and
the exact prefixing can be inconsistent. These fuzzy lookups resolve names by
substring instead of hard-coding the prefixed form.
"""
from __future__ import annotations

from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"


def find_name(names, *required: str) -> str:
    """Return the first entry of ``names`` containing all ``required`` substrings.

    Matching is case-insensitive. Raises ValueError if nothing matches so failures
    are loud rather than silently selecting the wrong element.
    """
    req = [r.lower() for r in required]
    for n in names:
        ln = n.lower()
        if all(r in ln for r in req):
            return n
    raise ValueError(f"No name containing all of {required} found in {list(names)}")
