# Developer Notes

This page will record implementation decisions and compatibility constraints
that are important to future SNF2 contributors.

## Design decisions

- `make_affinity` computes pairwise distances from sample-by-feature data and
  delegates the shared SNF kernel to `affinity_matrix`.
- `affinity_matrix` accepts precomputed distances rather than similarities, so
  callers remain responsible for choosing a scientifically appropriate
  similarity-to-distance transformation.

## Compatibility notes

- Missing-value handling belongs in preprocessing or in the distance
  calculation. SNF2 does not define a generic `nan_policy` for SciPy metrics.

## Open questions

No entries yet.
