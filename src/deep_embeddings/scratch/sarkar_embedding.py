#%%
import pandas as pd
import numpy as np

from hyperlib.embedding.treerep import treerep
from hyperlib.embedding.sarkar import sarkar_embedding

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

#%%
def make_X_y(df):
    """
    Split into gene expression values (X) and perturbagen labels (y)
    """
    df = df.drop(['pert_dose', 
    'pert_dose_unit', 
    'pert_id', 
    'pert_mfc_id',
    'pert_time', 
    'pert_time_unit', 
    'pert_type', 
    'rna_plate', 
    'rna_well',
    'cell_id',
    'det_plate',
    'det_well'
    ], axis=1)

    X = df.loc[:, df.columns != 'pert_iname']
    y = df['pert_iname']
    return (X, y)

#%%
df = pd.read_parquet("data/clean/clean_sample_small.parquet")
df.shape

#%%
X, y = make_X_y(df)
X_scaler = StandardScaler()
X_scaler.fit(X)
X_scale = X_scaler.transform(X)

#%%
# Get 10 nearest neighbours of each data point
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(X_scale)

#%%
# Create adjency matrix from NN
adjacency_matrix = neigh.kneighbors_graph(mode='distance')
adjacency_matrix = csr_matrix.toarray(adjacency_matrix)

#%%
# Create distance matrix between each data point
dist_matrix, predecessors = shortest_path(csgraph=adjacency_matrix, 
                                method='FW', 
                                directed=False, 
                                return_predecessors=True)

#%%
# outputs a weighted networkx Graph
tree = treerep(dist_matrix, return_networkx=True)

#%%
root = 0
embs = sarkar_embedding(tree, root, tau=0.01, precision=50)
embs