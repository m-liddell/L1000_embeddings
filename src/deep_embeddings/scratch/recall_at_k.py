#%%
import time

import numba as nb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.model_selection import GroupShuffleSplit

import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(123)

# %%
embedded = np.load("../../../data/embedded.npy")
labs = np.load("../../../data/y_test.npy", allow_pickle=True)

# %%
def slice_by_index(X, idx):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    elif isinstance(X, np.ndarray):
        return np.take(X, idx, axis=0)
    else:
        raise(TypeError("Please pass pandas dataframe or numpy array"))

def grouped_train_test_split(X, y, groups, test_size=0.2):
    """
    Train/test split which doesn't split up groups
    https://stackoverflow.com/questions/61337373/split-on-train-and-test-separating-by-group
    """  
    gs = GroupShuffleSplit(n_splits=2, test_size=test_size)
    train_ix, test_ix = next(gs.split(X, y, groups=groups))
    return slice_by_index(X, train_ix), slice_by_index(X, test_ix), slice_by_index(y, train_ix), slice_by_index(y, test_ix)

#%%
@nb.jit(parallel=True)
def argsort_parallel(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i, :] = np.argsort(a[i, :])
    return b 

def recall_at_k(embedded_sample, embedded, labs):
    """
    Make cumulative distribution for predicted perturbagen ranks
    for comparison with ideal to generate ROC
    Takes embedded gene expression profiles and pertubagen labels
    Embedded profiles and labels must be in the same order
    """
    cos_sim = cosine_similarity(embedded_sample, embedded)
    idx_sorted = argsort_parallel(-1*cos_sim) #-1* to get descending
    ranked_perts = np.take_along_axis(np.tile(labs, (idx_sorted.shape[0], 1)), idx_sorted, axis=1) #apply pert labels
    
    query_pert = ranked_perts[:, 0] #first col must be query pert, as it is a comparison with itself
    pert_match = ranked_perts[:, 1:] == query_pert[:, None] #compare elements in each column for pert label match
    if any(np.all(pert_match, axis=1)): 
        raise ValueError("Every pert should have at least another to compare. Check train/test data splitting")
    
    sum_matches = normalize(pert_match, axis=1, norm='l1').sum(axis=0)
    return np.cumsum(sum_matches)/sum_matches.sum()
  
# %%
embedded = np.load("../../../data/embedded.npy", allow_pickle=True)
labs = np.load("../../../data/y_test.npy", allow_pickle=True)

_, embedded_sample, _, labs_sample = grouped_train_test_split(embedded, labs, labs, test_size=100)

# %%
recall = recall_at_k(embedded_sample, embedded, labs)
quantile = minmax_scale(np.arange(1, embedded.shape[0]), feature_range=(0, 1))

auc = np.trapz(recall, quantile)
auc_lab = f"AUC {auc:.2f}"

fig, ax = plt.subplots()
ax.plot(quantile, recall)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.73, 0.1, auc_lab, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)
plt.title("Compound Retrieval for Embedded Signatures in Test Set")
plt.xlabel("Proportion of Results Included")
plt.ylabel("Proportion of Compound Instances Identified")
plt.show()

# %%
a = np.array([[4.54, 5.7454, 2.456, 3.2343], [7.56, 1, 2, 9]])
b = np.array([[7.54, 8.7454, 1.456, 8.2343], [2.56, 7, 7, 2]])
c_s = cosine_similarity(a, b)

srt = argsort_parallel(c_s)
srt_np = np.argsort(c_s)
# %%
