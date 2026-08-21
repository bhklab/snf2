"""Private input validation helpers for SNF2."""

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Constants for symmetry checks
SYMMETRY_RTOL = 1e-7  # Relative tolerance
SYMMETRY_ATOL = 1e-12  # Absolute tolerance


def as_feature_matrix(data: ArrayLike) -> NDArray[np.float64]:
    """Return validated sample-by-feature data as an owned float64 array."""
    try:
        raw = np.asarray(data)
    except (TypeError, ValueError) as error:
        raise TypeError("data must be a rectangular real-valued array") from error

    if raw.ndim != 2:
        raise ValueError("data must be a two-dimensional sample-by-feature array")
    if raw.shape[0] < 2:
        raise ValueError("data must contain at least two samples")
    if raw.shape[1] < 1:
        raise ValueError("data must contain at least one feature")
    is_boolean = np.issubdtype(raw.dtype, np.bool_)
    is_real = np.issubdtype(raw.dtype, np.number) and not np.issubdtype(
        raw.dtype, np.complexfloating
    )
    if not (is_boolean or is_real):
        raise TypeError("data must contain real numeric values")

    matrix = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("data must contain only finite values")
    return matrix


def validate_n_neighbors(n_neighbors: int, n_samples: int) -> int:
    """Validate and return the requested neighborhood size."""
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, Integral):
        raise TypeError("n_neighbors must be an integer")

    value = int(n_neighbors)
    if not 1 <= value < n_samples:
        raise ValueError(f"n_neighbors must satisfy 1 <= n_neighbors < {n_samples}")
    return value


def validate_positive_float(value: float, *, name: str) -> float:
    """Validate and return a positive finite real parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    validated = float(value)
    if not np.isfinite(validated) or validated <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return validated


def validate_n_iter(n_iter: int) -> int:
    """Validate and return the number of fusion iterations."""
    if isinstance(n_iter, bool) or not isinstance(n_iter, Integral):
        raise TypeError("n_iter must be an integer")

    value = int(n_iter)
    if value < 1:
        raise ValueError("n_iter must be at least 1")
    return value


def as_affinity_matrices(
    affinities: Sequence[ArrayLike],
) -> list[NDArray[np.float64]]:
    """Return validated affinity matrices as owned float64 arrays."""
    if isinstance(affinities, np.ndarray) and affinities.ndim == 2:
        raise TypeError("affinities must be a sequence of at least two matrices")

    try:
        supplied = list(affinities)
    except TypeError as error:
        raise TypeError("affinities must be a sequence of matrices") from error

    if len(supplied) < 2:
        raise ValueError("at least two affinity matrices are required")

    matrices: list[NDArray[np.float64]] = []
    expected_shape: tuple[int, int] | None = None
    for index, affinity in enumerate(supplied):
        try:
            raw = np.asarray(affinity)
        except (TypeError, ValueError) as error:
            raise TypeError(f"affinity matrix {index} must be rectangular") from error

        if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
            raise ValueError(f"affinity matrix {index} must be square")
        if raw.shape[0] < 2:
            raise ValueError(f"affinity matrix {index} must have at least two samples")
        if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
            raw.dtype, np.complexfloating
        ):
            raise TypeError(f"affinity matrix {index} must contain real numeric values")

        matrix = np.array(raw, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"affinity matrix {index} must contain only finite values")
        if np.any(matrix < 0):
            raise ValueError(f"affinity matrix {index} must be nonnegative")
        if not np.allclose(
            matrix,
            matrix.T,
            rtol=SYMMETRY_RTOL,
            atol=SYMMETRY_ATOL,
        ):
            raise ValueError(f"affinity matrix {index} must be symmetric")

        matrix = (matrix + matrix.T) / 2
        if expected_shape is None:
            expected_shape = matrix.shape
        elif matrix.shape != expected_shape:
            raise ValueError("all affinity matrices must have the same shape")
        matrices.append(matrix)

    return matrices
