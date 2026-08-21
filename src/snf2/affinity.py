"""Construct sample affinity matrices from feature data."""

from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist, squareform

from snf2._validation import (
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
    distances = np.asarray((distances + distances.T) / 2, dtype=np.float64)
    np.fill_diagonal(distances, 0)

    if not np.all(np.isfinite(distances)):
        raise ValueError(
            f"metric {metric!r} produced non-finite pairwise distances; "
            "check whether the data satisfy that metric's requirements"
        )
    if np.any(distances < 0):
        raise ValueError(f"metric {metric!r} produced negative pairwise distances")

    epsilon = np.finfo(np.float64).eps
    sorted_distances = np.sort(distances, axis=1)
    neighborhood_means = sorted_distances[:, 1 : neighbors + 1].mean(axis=1) + epsilon
    local_widths = (
        neighborhood_means[:, None] + neighborhood_means[None, :] + distances
    ) / 3 + epsilon
    local_widths = np.maximum(local_widths, epsilon)

    kernel_widths = kernel_scale * local_widths
    affinities = np.exp(-(distances**2) / (2 * kernel_widths**2))
    affinities /= np.sqrt(2 * np.pi) * kernel_widths
    affinities = np.asarray((affinities + affinities.T) / 2, dtype=np.float64)

    if not np.all(np.isfinite(affinities)):
        raise ValueError("affinity computation produced non-finite values")

    return affinities
