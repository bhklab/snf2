# SNF2

SNF2 is a modern Python implementation of Similarity Network Fusion.

The initial API separates affinity construction from network fusion:

```python
import numpy as np

from snf2 import affinity_matrix, fuse, make_affinity

rna = np.array([[0.0, 1.0], [0.2, 0.8], [1.0, 0.1], [0.9, 0.2]])
protein = np.array([[1.0, 0.0], [0.8, 0.1], [0.1, 1.0], [0.2, 0.9]])

networks = [
    make_affinity(rna, n_neighbors=2),
    make_affinity(protein, n_neighbors=2),
]
fused = fuse(networks, n_neighbors=2)
```

Each input must use rows for samples and columns for features. SNF2 performs no
automatic feature standardization, sample alignment, or missing-value
handling. Preprocess modalities and place samples in the same order before
calling `make_affinity`.

`make_affinity` defaults to squared Euclidean distance and accepts every named
metric supported by
[`scipy.spatial.distance.pdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html).
Metric-specific arguments such as Minkowski `p`, standardized Euclidean `V`,
and Mahalanobis `VI` can be supplied through `metric_kwargs`. `fuse` requires
at least two finite, nonnegative, symmetric affinity matrices with the same
shape. Metric-specific data requirements follow SciPy; SNF2 rejects
non-finite or negative pairwise distances before constructing affinities.

Use `affinity_matrix` when distances have already been computed:

```python
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
Convert similarities with a transformation appropriate to the similarity
measure first; for a similarity bounded to `[0, 1]`, that may be
`1 - similarity`.

For setup and development commands, see the
[project README](https://github.com/bhklab/snf2#readme).
