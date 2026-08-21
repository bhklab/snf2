"""Fuse multiple sample affinity matrices with SNF."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from snf2._validation import (
    as_affinity_matrices,
    validate_n_iter,
    validate_n_neighbors,
)


def _normalize_affinity(affinity: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the transition normalization used by SNFtool."""

    normalized = affinity.copy()
    off_diagonal_sums = normalized.sum(axis=1) - np.diag(normalized)
    off_diagonal_sums[off_diagonal_sums == 0] = 1
    normalized /= 2 * off_diagonal_sums[:, None]
    np.fill_diagonal(normalized, 0.5)

    return np.asarray((normalized + normalized.T) / 2, dtype=np.float64)


def _top_k_transition(
    affinity: NDArray[np.float64], n_neighbors: int
) -> NDArray[np.float64]:
    """Retain and row-normalize exactly the strongest K entries per row."""

    transition = np.zeros_like(affinity)
    n_samples = affinity.shape[0]
    for row_index in range(n_samples):
        order = np.argsort(affinity[row_index], kind="stable")
        keep = order[n_samples - n_neighbors :]
        transition[row_index, keep] = affinity[row_index, keep]

    row_sums = transition.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("top-K affinity rows must have positive sums")
    transition /= row_sums[:, None]

    return transition


def fuse(
    affinities: Sequence[ArrayLike],
    *,
    n_neighbors: int = 20,
    n_iter: int = 20,
) -> NDArray[np.float64]:
    """Fuse sample affinity matrices using Similarity Network Fusion.

    Parameters
    ----------
    affinities
        Sequence containing at least two square, symmetric, nonnegative
        sample-by-sample affinity matrices with identical shapes.
    n_neighbors
        Number of strongest entries retained in each local transition row.
        This follows SNFtool and therefore includes the diagonal when it is
        among the strongest entries.
    n_iter
        Positive number of diffusion iterations.

    Returns
    -------
    numpy.ndarray
        A symmetric ``float64`` fused affinity matrix.

    Raises
    ------
    TypeError
        If the matrices or parameters have incompatible types.
    ValueError
        If the matrices or parameters have invalid values.
    """
    networks = as_affinity_matrices(affinities)
    neighbors = validate_n_neighbors(n_neighbors, networks[0].shape[0])
    iterations = validate_n_iter(n_iter)

    probabilities = [_normalize_affinity(network) for network in networks]
    transitions = [
        _top_k_transition(probability, neighbors) for probability in probabilities
    ]

    for _ in range(iterations):
        total = np.add.reduce(probabilities)
        updated = [
            transition
            @ ((total - probability) / (len(probabilities) - 1))
            @ transition.T
            for probability, transition in zip(probabilities, transitions, strict=True)
        ]
        probabilities = [_normalize_affinity(network) for network in updated]

    fused = np.add.reduce(probabilities) / len(probabilities)

    return _normalize_affinity(fused)
