# SNF2

[![PyPI](https://img.shields.io/pypi/v/snf2?style=flat-square)](https://pypi.org/project/snf2/)
[![Python](https://img.shields.io/pypi/pyversions/snf2?style=flat-square)](https://pypi.org/project/snf2/)
[![Documentation](https://img.shields.io/badge/docs-MkDocs-blue?style=flat-square)](https://bhklab.github.io/snf2/)

SNF2 is a modern Python implementation of Similarity Network Fusion for
combining multiple data modalities into one sample-similarity network.

SNF2 requires Python 3.12 or newer.

## Installation

Install the released package from PyPI:

```console
pip install snf2
```

## Quick start

```python
import numpy as np

from snf2 import fuse, make_affinity

modality_a = np.array(
    [[0.0, 1.0], [0.2, 0.8], [1.0, 0.1], [0.9, 0.2]],
)
modality_b = np.array(
    [[1.0, 0.0], [0.8, 0.1], [0.1, 1.0], [0.2, 0.9]],
)

networks = [
    make_affinity(modality_a, n_neighbors=2),
    make_affinity(modality_b, n_neighbors=2),
]
fused_network = fuse(networks, n_neighbors=2)
```

Rows are samples and columns are features. SNF2 does not standardize features,
align samples, or impute missing values. Preprocess each modality and place its
samples in the same order before constructing affinity matrices.

## Public API

- `make_affinity(data, ...)` computes pairwise distances from a
  sample-by-feature matrix and constructs an affinity matrix.
- `affinity_matrix(distances, ...)` constructs an affinity matrix from a
  precomputed distance matrix.
- `fuse(affinities, ...)` combines two or more affinity matrices.

`make_affinity` defaults to squared Euclidean distance and accepts the named
metrics supported by
[`scipy.spatial.distance.pdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html).
Metric-specific arguments can be supplied through `metric_kwargs`:

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

Metric-specific data requirements follow SciPy. SNF2 rejects non-finite or
negative pairwise distances.

Use `affinity_matrix` when distances have already been computed:

```python
from snf2 import affinity_matrix

distances = np.array(
    [
        [0.0, 0.3, 1.2, 1.0],
        [0.3, 0.0, 1.0, 0.8],
        [1.2, 1.0, 0.0, 0.2],
        [1.0, 0.8, 0.2, 0.0],
    ],
)
precomputed_network = affinity_matrix(distances, n_neighbors=2)
```

The input to `affinity_matrix` is a distance matrix, not a similarity matrix.
Convert similarities using a transformation appropriate to the similarity
measure first. For similarities bounded to `[0, 1]`, that transformation may
be `1 - similarity`. Pairwise-complete correlations and other missing-value
policies must be handled upstream.

## Documentation

The full documentation is available at
[bhklab.github.io/snf2](https://bhklab.github.io/snf2/). Report problems through
the [GitHub issue tracker](https://github.com/bhklab/snf2/issues).

## Development

[Pixi](https://pixi.sh/latest/) manages the development environments and lock
file:

```console
pixi install
pixi run -e dev check
pixi run -e docs docs-build
```

See the [developer notes](https://bhklab.github.io/snf2/devnotes/) for the
implementation provenance, compatibility boundaries, and release procedure.

## Citation and provenance

Similarity Network Fusion was introduced in:

> Wang B, Mezlini AM, Demir F, Fiume M, Tu Z, Brudno M, Haibe-Kains B,
> Goldenberg A. Similarity network fusion for aggregating data types on a
> genomic scale. *Nature Methods*. 2014;11:333–337.
> [doi:10.1038/nmeth.2810](https://doi.org/10.1038/nmeth.2810)

SNF2 was informed by pinned versions of
[SNFpy](https://github.com/rmarkello/snfpy) and
[SNFtool](https://github.com/cran/SNFtool). Complete attribution and reference
commits are recorded in [NOTICE](https://github.com/bhklab/snf2/blob/main/NOTICE).

## Authors and license

SNF2 is developed by [Michael Tran](https://github.com/mtran-code) and James
Bannon at BHKLab. Contact
[bhklab.michaeltran@gmail.com](mailto:bhklab.michaeltran@gmail.com).

SNF2 is distributed under the
[MIT License](https://github.com/bhklab/snf2/blob/main/LICENSE).
