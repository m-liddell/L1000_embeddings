# %%
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def grouped_train_test_split(X, y, groups, test_size=0.2):
    """
    Train/test split which doesn't split up groups
    https://stackoverflow.com/questions/61337373/split-on-train-and-test-separating-by-group
    """  
    gs = GroupShuffleSplit(n_splits=2, test_size=test_size)
    train_ix, test_ix = next(gs.split(X, y, groups=groups))

    return X.iloc[train_ix].values, X.iloc[test_ix].values, y.iloc[train_ix].values, y.iloc[test_ix].values

# %%
df_t = pd.DataFrame({"x": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
                   "id": [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                   "label": ["a", "a", "a", "b", "a", "b", "b", "b", "a", "b", "a", "c"]})

X_t = df_t.drop('label', axis=1)
y_t = df_t['label']

X_train, X_test, y_train, y_test = grouped_train_test_split(X_t, y_t, y_t)

# %%
df = pd.read_parquet("../../../data/clean/clean_sample_small.parquet")

# %%
X = df.drop(['pert_iname'], axis=1)
y = df['pert_iname']

X_train, X_test, y_train, y_test = grouped_train_test_split(X, y, y)

# %%
n_counts = df['pert_iname'].value_counts() < 5
df = df[~df['pert_iname'].isin(n_counts[n_counts < 5].index)]
# %%
