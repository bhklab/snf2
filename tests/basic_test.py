import sys
import numpy as np

from snf2 import fuse, make_affinity, affinity_matrix
from scipy.spatial.distance import pdist
import pandas as pd

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
# print(fused_network)


correlation_network = make_affinity(
    modality_a,
    metric="correlation",
    n_neighbors=2,
    
)

# print(correlation_network)
minkowski_network = make_affinity(
    modality_a,
    metric="minkowski",
    metric_kwargs={"p": 3.5},
    n_neighbors=2,
)



modality_c = np.array(
    [[1, 1.0,0.3], [0.2, 0.8,0.9], [1.0, 0.1,0.5], [0.9, 0.2,np.nan]],
)

c = make_affinity(modality_c, metric='correlation',n_neighbors=2)
print(c)

# df = pd.DataFrame(modality_c)
# print(df.corr(min_periods=1))
# print(df.corr(numeric_only=True))
# sys.exit()
# precomputed_aff = 1- df.corr(numeric_only=True).fillna(0)

# aff = 1-precomputed_aff

# print(aff)
# aff = affinity_matrix(aff,k=2)
# print(aff)