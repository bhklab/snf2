"""Construct sample affinity matrices from feature data."""

from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

from snf2._validation import (
    as_distance_matrix,
    as_feature_matrix,
    validate_n_neighbors,
    validate_positive_float,
)


def make_affinity(
    data: ArrayLike,
    *,
    metric: str = "sqeuclidean",
    metric_kwargs: Mapping[str, object] | None = None,
    n_neighbors: int = 20,
    scale: float = 0.5,
) -> NDArray[np.float64]:
    """Construct an SNF affinity matrix from feature data.

    Parameters
    ----------
    data
        Two-dimensional sample-by-feature data. SNF2 does not standardize or
        otherwise preprocess features.
    metric
        Named distance metric supported by :func:`scipy.spatial.distance.pdist`.
    metric_kwargs
        Optional metric-specific keyword arguments forwarded to
        :func:`scipy.spatial.distance.pdist`, such as ``p``, ``w``, ``V``, or
        ``VI``. The ``out`` argument is not accepted.
    n_neighbors
        Number of nearest neighbors used to estimate each local scale.
    scale
        Positive multiplier applied to the locally estimated kernel width.

    Returns
    -------
    numpy.ndarray
        A symmetric ``float64`` sample-by-sample affinity matrix.

    Raises
    ------
    TypeError
        If the data or parameters have incompatible types.
    ValueError
        If the data or parameters have invalid values.
    """

    matrix = as_feature_matrix(data)
    neighbors = validate_n_neighbors(n_neighbors, matrix.shape[0])
    kernel_scale = validate_positive_float(scale, name="scale")

    if not isinstance(metric, str):
        raise TypeError("metric must be the name of a SciPy distance metric")
    if not metric:
        raise ValueError("metric must not be empty")

    if metric_kwargs is None:
        distance_kwargs: dict[str, object] = {}
    elif isinstance(metric_kwargs, Mapping):
        distance_kwargs = dict(metric_kwargs)
    else:
        raise TypeError("metric_kwargs must be a mapping or None")

    if not all(isinstance(key, str) for key in distance_kwargs):
        raise TypeError("metric_kwargs keys must be strings")
    if "out" in distance_kwargs:
        raise ValueError("metric_kwargs must not contain 'out'")

    pdist_with_named_metric = cast(Callable[..., NDArray[np.float64]], pdist)
    distances = squareform(
        pdist_with_named_metric(matrix, metric=metric, **distance_kwargs)
    )
    if not np.all(np.isfinite(distances)):
        raise ValueError(
            f"metric {metric!r} produced non-finite pairwise distances; "
            "check whether the data satisfy that metric's requirements"
        )
    if np.any(distances < 0):
        raise ValueError(f"metric {metric!r} produced negative pairwise distances")

    return affinity_matrix(
        distances,
        n_neighbors=neighbors,
        scale=kernel_scale,
    )


def affinity_matrix(
    distances: ArrayLike,
    *,
    n_neighbors: int = 20,
    scale: float = 0.5,
) -> NDArray[np.float64]:
    """Construct an SNF affinity matrix from pairwise distances.

    Parameters
    ----------
    distances
        Square, finite, nonnegative, symmetric pairwise-distance matrix with a
        zero diagonal. Similarities must first be converted to distances using
        a transformation appropriate to the similarity measure.
    n_neighbors
        Number of nearest neighbors used to estimate each local scale.
    scale
        Positive multiplier applied to the locally estimated kernel width.

    Returns
    -------
    numpy.ndarray
        A symmetric ``float64`` sample-by-sample affinity matrix.

    Raises
    ------
    TypeError
        If the distances or parameters have incompatible types.
    ValueError
        If the distances or parameters have invalid values.
    """
    diff = as_distance_matrix(distances)
    n = diff.shape[0]
    k = validate_n_neighbors(n_neighbors, n)
    kernel_scale = validate_positive_float(scale, name="scale")
    eps = np.finfo(np.float64).eps

    # Symmetrize and remove self-distances.
    diff = np.asarray((diff + diff.T) / 2, dtype=np.float64)
    np.fill_diagonal(diff, 0.0)

    # For each row, sort distances and take the first k non-self values.
    sorted_rows = np.sort(diff, axis=1)
    nearest = sorted_rows[:, 1 : k + 1]

    # R's mean(x[is.finite(x)]), applied row-wise.
    finite = np.isfinite(nearest)
    counts = finite.sum(axis=1)
    means = np.where(finite, nearest, 0.0).sum(axis=1) / counts
    means += eps

    # Equivalent to:
    # outer(means, means, avg) / 3 * 2 + Diff / 3 + eps
    sig = (2.0 / 3.0) * ((means[:, None] + means[None, :]) / 2)
    sig += diff / 3.0
    sig += eps
    sig = np.maximum(sig, eps)

    # Gaussian density: dnorm(Diff, mean=0, sd=scale * Sig)
    densities = np.asarray(
        norm.pdf(diff, loc=0.0, scale=kernel_scale * sig),
        dtype=np.float64,
    )

    # Ensure the resulting affinity matrix is symmetric.
    affinities = np.asarray((densities + densities.T) / 2, dtype=np.float64)
    if not np.all(np.isfinite(affinities)):
        raise ValueError("affinity computation produced non-finite values")
    return affinities
