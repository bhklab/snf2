# SPDX-License-Identifier: LGPL-3.0-only

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from snf2 import fuse, make_affinity
from snf2.fusion import _normalize_affinity, _top_k_transition

DATA_DIR = Path(__file__).parent / "reference" / "data"
REFERENCE_RTOL = 1e-10
REFERENCE_ATOL = 1e-12


def load_fixture(name: str) -> np.ndarray:
    """Load a human-readable SNFtool reference fixture."""
    return np.loadtxt(DATA_DIR / name, delimiter=",")


@pytest.mark.parametrize(
    ("n_iter", "expected_name"),
    [(1, "fused_t1.csv"), (20, "fused_t20.csv")],
)
def test_fuse_matches_snftool(n_iter: int, expected_name: str) -> None:
    affinities = [load_fixture("affinity_1.csv"), load_fixture("affinity_2.csv")]
    expected = load_fixture(expected_name)

    actual = fuse(affinities, n_neighbors=3, n_iter=n_iter)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )


def test_fuse_preserves_inputs_and_output_invariants() -> None:
    features_1 = load_fixture("features_1.csv")
    features_2 = load_fixture("features_2.csv")
    affinities = [
        make_affinity(features_1, n_neighbors=3),
        make_affinity(features_2, n_neighbors=3),
    ]
    originals = [affinity.copy() for affinity in affinities]

    fused = fuse(affinities, n_neighbors=3, n_iter=2)

    for affinity, original in zip(affinities, originals, strict=True):
        np.testing.assert_array_equal(affinity, original)
    assert fused.dtype == np.float64
    assert fused.shape == affinities[0].shape
    assert np.all(np.isfinite(fused))
    assert np.all(fused >= 0)
    np.testing.assert_allclose(fused, fused.T, rtol=0, atol=0)
    np.testing.assert_allclose(np.diag(fused), 0.5, rtol=0, atol=0)


def test_public_defaults_support_twenty_neighbors() -> None:
    samples = np.column_stack([np.linspace(0, 1, 21), np.linspace(1, 0, 21) ** 2])
    first = make_affinity(samples)
    second = make_affinity(samples[:, ::-1])

    fused = fuse([first, second])

    assert fused.shape == (21, 21)
    np.testing.assert_allclose(np.diag(fused), 0.5, rtol=0, atol=0)


def test_top_k_transition_retains_exactly_k_entries_with_ties() -> None:
    affinity = np.ones((4, 4), dtype=np.float64)

    transition = _top_k_transition(affinity, 2)

    expected = np.zeros((4, 4), dtype=np.float64)
    expected[:, 2:] = 0.5
    np.testing.assert_array_equal(transition, expected)
    np.testing.assert_array_equal(np.count_nonzero(transition, axis=1), 2)


def test_normalize_affinity_uses_snftool_diagonal_convention() -> None:
    affinity = np.array(
        [[2.0, 1.0, 3.0], [1.0, 4.0, 2.0], [3.0, 2.0, 5.0]],
    )

    normalized = _normalize_affinity(affinity)

    np.testing.assert_allclose(normalized, normalized.T, rtol=0, atol=0)
    np.testing.assert_allclose(np.diag(normalized), 0.5, rtol=0, atol=0)


def valid_affinities() -> list[np.ndarray]:
    """Return two small valid affinity matrices."""
    first = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    second = np.array([[1.0, 0.2, 0.5], [0.2, 1.0, 0.4], [0.5, 0.4, 1.0]])
    return [first, second]


def test_fuse_accepts_negligible_symmetry_noise() -> None:
    affinities = valid_affinities()
    affinities[0][0, 1] += 1e-13

    fused = fuse(affinities, n_neighbors=2, n_iter=1)

    np.testing.assert_allclose(fused, fused.T, rtol=0, atol=0)


@pytest.mark.parametrize(
    "affinities",
    [
        [],
        [np.eye(3)],
        np.eye(3),
        [np.ones((2, 3)), np.ones((2, 3))],
        [np.eye(3), np.eye(4)],
        [np.array([[1.0, 0.5], [0.2, 1.0]]), np.eye(2)],
        [np.array([[1.0, -0.1], [-0.1, 1.0]]), np.eye(2)],
        [np.array([[1.0, np.nan], [np.nan, 1.0]]), np.eye(2)],
        [np.array([[1.0, np.inf], [np.inf, 1.0]]), np.eye(2)],
        [np.array([[1 + 1j, 0], [0, 1 + 1j]]), np.eye(2)],
    ],
)
def test_fuse_rejects_invalid_affinities(affinities: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        fuse(cast(Any, affinities), n_neighbors=1, n_iter=1)


@pytest.mark.parametrize("n_neighbors", [0, 3, 1.5, True])
def test_fuse_rejects_invalid_neighborhood(n_neighbors: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        fuse(
            valid_affinities(),
            n_neighbors=cast(Any, n_neighbors),
            n_iter=1,
        )


@pytest.mark.parametrize("n_iter", [0, -1, 1.5, True])
def test_fuse_rejects_invalid_iterations(n_iter: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        fuse(
            valid_affinities(),
            n_neighbors=2,
            n_iter=cast(Any, n_iter),
        )
