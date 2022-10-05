#%%
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np

#%%
df = pd.DataFrame({"x": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
                   "id": [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                   "label": ["a", "a", "a", "b", "a", "b", "b", "b", "a", "b", "a", "b"]})

#%%
X = df.drop('label', 1)
y = df.label

#%%
X_n = X.to_numpy()
y_n = y.to_numpy()

#%%
gs = GroupShuffleSplit(n_splits=2, test_size=.5, random_state=0)
train_ix, test_ix = next(gs.split(X_n, y_n, groups=y_n))

#%%
X_train = X.loc[train_ix]
y_train = y.loc[train_ix]

X_test = X.loc[test_ix]
y_test = y.loc[test_ix]

# %%
if isinstance()
np.take(X, train_ix)
# %%
def slice_by_index(X, idx):
    if isinstance(X, pd.DataFrame):
        return X.loc[idx]
    elif isinstance(X, np.ndarray):
        return np.take(X, idx)
    else:
        raise(TypeError("Please pass pandas dataframe or numpy array"))
# %%

# %%
