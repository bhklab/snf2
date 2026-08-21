"""SNF2: a modern Python implementation of Similarity Network Fusion."""

from snf2.affinity import make_affinity
from snf2.fusion import fuse

__all__ = ["fuse", "make_affinity"]
