#%%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt

#%%
def cum_sum_pred(embedded_sample, embedded, perts):
    """
    Make cumulative distribution for predicted perturbagen ranks
    for comparison with ideal to generate ROC
    Takes embedded gene expression profiles and pertubagen labels
    Embedded profiles and labels must be in the same order
    """
    cos_sim = cosine_similarity(embedded_sample, embedded)
    idx_sorted = np.argsort(-1*cos_sim, axis=1) #-1* to get descending
    ranked_perts = np.take_along_axis(np.tile(perts, (idx_sorted.shape[0], 1)), idx_sorted, axis=1) #apply pert labels

    query_pert = ranked_perts[:, 0] #first col must be query pert, as it is a comparison with itself
    pert_match = ranked_perts[:, 1:] == query_pert[:, None] #compare elements in each column for pert label match
    print(pert_match.shape)
    if any(np.all(pert_match, axis=1)): 
        raise ValueError("Every pert should have at least another to compare. Check train/test data splitting")
    
    count_matches = np.count_nonzero(pert_match, axis=0)
    return np.cumsum(count_matches) #last value should be no of perts-1 (remove itself)
   
# %%
embedded = np.load("../../../data/X_test_scale_embedded.npy", allow_pickle=True)
labs = np.load("../../../data/y_test.npy", allow_pickle=True)

# %%
def sample_test(embedded, labs, n_test, n_samples):
    """
    Take sample from test set
    """
    embedded_sample = None
    labs_sample = None
    if n_test > n_samples:
        idx = np.random.randint(0, n_test, n_samples) #TODO: use group split here
        embedded_sample = np.take(embedded, idx, 0)
        labs_sample = np.take(labs, idx, 0)
    else:
        embedded_sample = embedded
        labs_sample = labs
    return labs_sample, embedded_sample

# %%
def sum_ideal_matches(labs, labs_sample):
    #make df of 
    #   - unique perturbagens in test set
    #   - now many times they are in full test set embeds 
    #   how many times in sampled test set embeds
    labs_value_counts = pd.value_counts(pd.Series(labs))
    sample_value_counts = pd.value_counts(pd.Series(labs_sample))

    ideal_df = pd.concat([labs_value_counts, sample_value_counts], axis=1).fillna(0).astype('int32')
    ideal_df.columns = ['lab_n', 'sample_n']

    n_comparisons = len(labs)-1 #don't compare with pert itself
    running_sum = np.zeros(n_comparisons, dtype=int)
    for row in ideal_df.itertuples():
        matches = np.ones(row.lab_n-1, dtype=int) #-1 to not compare with pert itself
        unmatches = np.zeros(n_comparisons-len(matches), dtype=int)
        ideal_matches = np.concatenate((matches, unmatches), axis=0, dtype=int)
        running_sum = np.add(running_sum, ideal_matches*row.sample_n, dtype=int)
        np.add(running_sum, ideal_matches, dtype=int)

    return np.cumsum(running_sum)

# %%
n_embedded = embedded.shape[0]

labs_sample, embedded_sample = sample_test(embedded, labs, n_embedded, int(1e3))
ideal = sum_ideal_matches(labs, labs_sample)

# %%
pred = cum_sum_pred(embedded_sample, embedded, labs)
recall = pred / ideal
quantile = minmax_scale(np.arange(1, n_embedded), feature_range=(1, 100))

# %%
auc = np.trapz(recall, quantile)
auc_lab = f"AUC {auc:.2f}"

fig, ax = plt.subplots()
ax.plot(quantile, recall)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.73, 0.1, auc_lab, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)
plt.title("Quantile/Recall for test set pertubagens")
plt.xlabel("Quantile")
plt.ylabel("Recall")
plt.show()

# %%
