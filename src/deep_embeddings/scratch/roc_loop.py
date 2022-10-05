#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit

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

# %%
#embedded = np.array([
#    [-1, 2, 6, 8], 
#    [7, -3, -5, 8], 
#    [3, -7, 6, 2],
#    [6, -3, -4, 9], 
#    [3, -5, 4, 2],
#    [1, -1, -4, 2], 
#    [1, 0, -6, 0], 
#    [0, 1, -2, 1]
#])
#
#labs = np.array([
#    "chem1", 
#    "chem1", 
#    "chem1", 
#    "chem2", 
#    "chem2", 
#    "chem3",
#    "chem3", 
#    "chem3"
#])

# %%
_, embedded_sample, _, labs_sample = grouped_train_test_split(embedded, labs, labs, test_size=410)
df_sample = pd.DataFrame(embedded_sample, labs_sample)

df = pd.DataFrame(embedded, labs)

# %%
def get_pos_in_negs(cos_sim_neg, cos_sim_pos):
    """
    Rank quantile of the similarity score of the each positive 
    within all negative scores
    """
    pos_in_negs = np.atleast_1d(np.zeros(cos_sim_pos.shape, cos_sim_pos.dtype))
    for i, p in enumerate(np.nditer(cos_sim_pos)):
        pos_in_negs[i] = np.sum((np.sign(cos_sim_neg-p)+1)/2)/len(cos_sim_neg)
    return pos_in_negs

def get_pos_greater_than_quant(quant, pos):
    """
    Fraction of positives ranked higher than each quantile
    """
    pos_greater_than_quant = np.atleast_1d(np.zeros(quant.shape, quant.dtype))
    for i, q in enumerate(np.nditer(quant)):
        pos_greater_than_quant[i] = np.sum((np.sign(q-pos)+1)/2)/len(pos)
    return pos_greater_than_quant

# %%
#n_counts = df_sample.index.value_counts()
#df_sample = df_sample[~df_sample.index.isin(n_counts[n_counts <= 1].index)]

# %%
#itertuples is faster
all_pos_in_neg = np.array([])
for index, query in df_sample.iterrows():

    pos_with_self = df.loc[query.name]
    pos = pos_with_self[(pos_with_self != query).any(axis=1)]
    neg = df.loc[df.index.difference(pos.index)]

    pos_profiles = pos.values.reshape(1, -1) if pos.ndim == 1 else pos.values #deal with array shape with only one positive
    query_profile = query.values.reshape(1, -1)
    cos_sim_neg = np.squeeze(cosine_similarity(query_profile, neg.values))
    cos_sim_pos = np.squeeze(cosine_similarity(query_profile, pos_profiles))

    #rank quantile of each positive similarity score in array of negative similarity scores
    pos_in_neg = get_pos_in_negs(cos_sim_neg, cos_sim_pos)
    all_pos_in_neg = np.concatenate((all_pos_in_neg, pos_in_neg))

# %%
#fraction of positives ranked higher than quantile
incriment = 0.1
quants = np.arange(0, 1+incriment, incriment)
pos_quant = get_pos_greater_than_quant(quants, all_pos_in_neg)
pos_quant

# %%
auc = np.trapz(pos_quant, quants)
fig, ax = plt.subplots()
ax.plot(quants, pos_quant)

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
auc_lab = f"AUC {auc:.2f}"
ax.text(0.73, 0.1, auc_lab, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)

plt.title("Quantile/Recall for test set pertubagens")
plt.xlabel("Quantile")
plt.ylabel("Recall")
# %%

