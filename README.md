# SNF2

**Authors:** [Michael Tran](https://github.com/mtran-code), James Bannon

**Contact:** [bhklab.michaeltran@gmail.com](mailto:bhklab.michaeltran@gmail.com)

**Description:** SNF2 is a modern Python implementation of Similarity Network Fusion.

--------------------------------------

[![pixi-badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square)](https://github.com/prefix-dev/pixi)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)

![GitHub last commit](https://img.shields.io/github/last-commit/bhklab/snf2?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/bhklab/snf2?style=flat-square)
![GitHub pull requests](https://img.shields.io/github/issues-pr/bhklab/snf2?style=flat-square)
![GitHub contributors](https://img.shields.io/github/contributors/bhklab/snf2?style=flat-square)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/bhklab/snf2?style=flat-square)

## Usage

```python
import numpy as np

from snf2 import affinity_matrix, fuse, make_affinity

modality_a = np.array(
    [[0.0, 1.0], [0.2, 0.8], [1.0, 0.1], [0.9, 0.2]],
)
modality_b = np.array(
    [[1.0, 0.0], [0.8, 0.1], [0.1, 1.0], [0.2, 0.9]],
)

affinities = [
    make_affinity(modality_a, n_neighbors=2),
    make_affinity(modality_b, n_neighbors=2),
]
fused_network = fuse(affinities, n_neighbors=2)
```

Rows are samples and columns are features. SNF2 does not standardize or align
inputs: callers must preprocess each modality and ensure identical sample
ordering before constructing affinities.

Affinity construction defaults to squared Euclidean distance and accepts every
named metric supported by
[`scipy.spatial.distance.pdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html).
Use `metric_kwargs` for metric-specific arguments:

```python
correlation_network = make_affinity(
    modality_a,
    metric="correlation",
    n_neighbors=2,
)

minkowski_network = make_affinity(
    modality_a,
    metric="minkowski",
    metric_kwargs={"p": 3.5},
    n_neighbors=2,
)
```

Metric-specific data requirements follow SciPy. SNF2 raises an error if a
metric produces non-finite or negative pairwise distances for the supplied
data.

For precomputed distances, use `affinity_matrix` directly:

```python
distances = np.array(
    [
        [0.0, 0.3, 1.2, 1.0],
        [0.3, 0.0, 1.0, 0.8],
        [1.2, 1.0, 0.0, 0.2],
        [1.0, 0.8, 0.2, 0.0],
    ],
)
precomputed_network = affinity_matrix(
    distances,
    n_neighbors=2,
)
```

`affinity_matrix` accepts distances, not similarities. Convert a similarity
matrix with a transformation appropriate to that measure before calling it;
for a similarity bounded to `[0, 1]`, that may be `1 - similarity`.

## Development setup

[Pixi](https://pixi.sh/latest/) manages the development environments and lock
file.

```console
pixi install
pixi run -e dev check
```

Build the documentation locally with:

```console
pixi run -e docs docs-build
```

The published documentation is available at
[bhklab.github.io/snf2](https://bhklab.github.io/snf2/).

## License

SNF2 is licensed under the [MIT License](LICENSE).
