import pandas as pd
import numpy as np
from hyperlib.manifold.poincare import Poincare

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

def get_all_pos_in_neg(df_sample, df):
    all_pos_in_neg = np.array([])
    for index, query in df_sample.iterrows():

        pos_with_self = df.loc[query.name]
        if isinstance(pos_with_self, pd.Series): #skip if only has itself as positive
            continue
        pos = pos_with_self[(pos_with_self != query).any(axis=1)]
        neg = df.loc[df.index.difference(pos.index)]

        pos_profiles = pos.values.reshape(1, -1) if pos.ndim == 1 else pos.values #deal with array shape with only one positive
        query_profile = query.values.reshape(1, -1)

        p = Poincare()
        cos_sim_neg = np.squeeze(p.dist(query_profile, neg.values, 1))
        cos_sim_pos = np.squeeze(p.dist(query_profile, pos_profiles, 1))

        #rank quantile of each positive similarity score in array of negative similarity scores
        pos_in_neg = get_pos_in_negs(cos_sim_neg, cos_sim_pos)
        all_pos_in_neg = np.concatenate((all_pos_in_neg, pos_in_neg))
    return all_pos_in_neg