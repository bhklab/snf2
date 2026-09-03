# SPDX-License-Identifier: LGPL-3.0-only

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from snf2 import affinity_matrix, make_affinity

DATA_DIR = Path(__file__).parent / "reference" / "data"
REFERENCE_RTOL = 1e-10
REFERENCE_ATOL = 1e-12

CONTINUOUS_DATA = np.array(
    [
        [0.10, 0.25, 0.65, 0.30],
        [0.20, 0.15, 0.55, 0.45],
        [0.45, 0.30, 0.10, 0.15],
        [0.55, 0.20, 0.15, 0.10],
        [0.30, 0.50, 0.10, 0.10],
        [0.25, 0.45, 0.20, 0.10],
    ],
)
BINARY_DATA = CONTINUOUS_DATA > 0.25
PROBABILITY_DATA = CONTINUOUS_DATA / CONTINUOUS_DATA.sum(axis=1, keepdims=True)

SCIPY_METRIC_CASES = [
    ("braycurtis", "continuous"),
    ("canberra", "continuous"),
    ("chebyshev", "continuous"),
    ("cityblock", "continuous"),
    ("correlation", "continuous"),
    ("cosine", "continuous"),
    ("dice", "binary"),
    ("euclidean", "continuous"),
    ("hamming", "binary"),
    ("jaccard", "binary"),
    ("jensenshannon", "probability"),
    ("mahalanobis", "continuous"),
    ("matching", "binary"),
    ("minkowski", "continuous"),
    ("rogerstanimoto", "binary"),
    ("russellrao", "binary"),
    ("seuclidean", "continuous"),
    ("sokalsneath", "binary"),
    ("sqeuclidean", "continuous"),
    ("yule", "binary"),
]


def load_fixture(name: str) -> np.ndarray:
    """Load a human-readable SNFtool reference fixture."""
    return np.loadtxt(DATA_DIR / name, delimiter=",")


@pytest.mark.parametrize(
    ("features_name", "affinity_name"),
    [
        ("features_1.csv", "affinity_1.csv"),
        ("features_2.csv", "affinity_2.csv"),
    ],
)
def test_make_affinity_matches_snftool(features_name: str, affinity_name: str) -> None:
    features = load_fixture(features_name)
    expected = load_fixture(affinity_name)

    actual = make_affinity(
        features,
        metric="euclidean",
        n_neighbors=3,
        scale=0.5,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )


@pytest.mark.parametrize(
    ("features_name", "affinity_name"),
    [
        ("features_1.csv", "affinity_1.csv"),
        ("features_2.csv", "affinity_2.csv"),
    ],
)
def test_affinity_matrix_matches_snftool(
    features_name: str, affinity_name: str
) -> None:
    features = load_fixture(features_name)
    distances = squareform(pdist(features, metric="euclidean"))
    expected = load_fixture(affinity_name)

    actual = affinity_matrix(
        distances,
        n_neighbors=3,
        scale=0.5,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )


def test_affinity_matrix_returns_owned_symmetric_float64_array() -> None:
    distances = squareform(pdist(CONTINUOUS_DATA, metric="euclidean"))
    original = distances.copy()

    affinity = affinity_matrix(distances, n_neighbors=3)

    np.testing.assert_array_equal(distances, original)
    assert affinity.dtype == np.float64
    assert affinity.shape == distances.shape
    assert np.all(np.isfinite(affinity))
    assert np.all(affinity > 0)
    np.testing.assert_allclose(affinity, affinity.T, rtol=0, atol=0)
    assert not np.shares_memory(affinity, distances)


@pytest.mark.parametrize(
    ("distances", "error", "message"),
    [
        ([0.0, 1.0], ValueError, "square"),
        (np.zeros((2, 3)), ValueError, "square"),
        (np.zeros((1, 1)), ValueError, "at least two"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), ValueError, "finite"),
        (np.array([[0.0, np.inf], [np.inf, 0.0]]), ValueError, "finite"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), ValueError, "nonnegative"),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), ValueError, "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), ValueError, "zero diagonal"),
        (np.array([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]]), TypeError, "real"),
        (np.array([["0", "1"], ["1", "0"]]), TypeError, "real"),
    ],
)
def test_affinity_matrix_rejects_invalid_distances(
    distances: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        affinity_matrix(cast(Any, distances), n_neighbors=1)


@pytest.mark.parametrize("n_neighbors", [0, 2, 1.5, True])
def test_affinity_matrix_rejects_invalid_neighborhood(n_neighbors: object) -> None:
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises((TypeError, ValueError)):
        affinity_matrix(
            distances,
            n_neighbors=cast(Any, n_neighbors),
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan, True])
def test_affinity_matrix_rejects_invalid_scale(scale: object) -> None:
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises((TypeError, ValueError)):
        affinity_matrix(
            distances,
            n_neighbors=1,
            scale=cast(Any, scale),
        )


def test_make_affinity_defaults_to_squared_euclidean() -> None:
    features = load_fixture("features_1.csv")
    expected = load_fixture("affinity_sqeuclidean_1.csv")

    actual = make_affinity(
        features,
        n_neighbors=3,
        scale=0.5,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )


def test_make_affinity_returns_owned_symmetric_float64_array() -> None:
    features = load_fixture("features_1.csv")
    original = features.copy()

    affinity = make_affinity(features, n_neighbors=3)

    np.testing.assert_array_equal(features, original)
    assert affinity.dtype == np.float64
    assert affinity.shape == (features.shape[0], features.shape[0])
    assert np.all(np.isfinite(affinity))
    assert np.all(affinity > 0)
    np.testing.assert_allclose(affinity, affinity.T, rtol=0, atol=0)
    assert not np.shares_memory(affinity, features)


@pytest.mark.parametrize(("metric", "data_kind"), SCIPY_METRIC_CASES)
def test_make_affinity_supports_each_scipy_metric(metric: str, data_kind: str) -> None:
    data_by_kind = {
        "binary": BINARY_DATA,
        "continuous": CONTINUOUS_DATA,
        "probability": PROBABILITY_DATA,
    }
    data = data_by_kind[data_kind]
    metric_kwargs: dict[str, object] = {}
    if metric == "minkowski":
        metric_kwargs = {"p": 3.5, "w": np.array([1.0, 2.0, 1.5, 0.5])}
    elif metric == "seuclidean":
        metric_kwargs = {"V": np.var(data, axis=0, ddof=1)}
    elif metric == "mahalanobis":
        metric_kwargs = {"VI": np.linalg.inv(np.cov(data, rowvar=False))}

    affinity = make_affinity(
        data,
        metric=metric,
        metric_kwargs=metric_kwargs,
        n_neighbors=3,
    )

    assert affinity.dtype == np.float64
    assert affinity.shape == (data.shape[0], data.shape[0])
    assert np.all(np.isfinite(affinity))
    assert np.all(affinity > 0)
    np.testing.assert_allclose(affinity, affinity.T, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ([1.0, 2.0], ValueError),
        (np.empty((1, 2)), ValueError),
        (np.empty((2, 0)), ValueError),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), ValueError),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), ValueError),
        (np.array([[1 + 2j], [2 + 3j]]), TypeError),
        (np.array([["1"], ["2"]]), TypeError),
    ],
)
def test_make_affinity_rejects_invalid_data(
    data: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        make_affinity(cast(Any, data), n_neighbors=1)


@pytest.mark.parametrize("n_neighbors", [0, 2, 1.5, True])
def test_make_affinity_rejects_invalid_neighborhood(n_neighbors: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_affinity(
            np.array([[0.0], [1.0]]),
            n_neighbors=cast(Any, n_neighbors),
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan, True])
def test_make_affinity_rejects_invalid_scale(scale: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_affinity(
            np.array([[0.0], [1.0]]),
            n_neighbors=1,
            scale=cast(Any, scale),
        )


def test_make_affinity_rejects_unsupported_metric() -> None:
    with pytest.raises(ValueError, match="Metric|metric"):
        make_affinity(
            np.array([[0.0], [1.0]]),
            metric="not-a-scipy-metric",
            n_neighbors=1,
        )


@pytest.mark.parametrize("metric_kwargs", [[("p", 2)], {1: 2}, {"out": None}])
def test_make_affinity_rejects_invalid_metric_kwargs(metric_kwargs: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_affinity(
            CONTINUOUS_DATA,
            metric="minkowski",
            metric_kwargs=cast(Any, metric_kwargs),
            n_neighbors=3,
        )
