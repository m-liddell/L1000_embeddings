import numba as nb
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

seed = 42
np.random.seed(seed)

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