"""SNF2: a modern Python implementation of Similarity Network Fusion."""

from snf2.affinity import affinity_matrix, make_affinity
from snf2.fusion import fuse

__all__ = ["affinity_matrix", "fuse", "make_affinity"]
