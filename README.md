# SNF2

SNF2 is a modern Python implementation of Similarity Network Fusion.

SNF2 currently provides the two core algorithm stages: constructing an
affinity matrix from one feature matrix and fusing affinity matrices across
modalities.

## Usage

```python
import numpy as np

from snf2 import fuse, make_affinity

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
