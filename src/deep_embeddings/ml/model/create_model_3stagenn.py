import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, metrics

from keras import backend as K
#import tensorflow_addons as tfa
import tensorflow_probability as tfp

from model.arcface_loss import ArcFace

import numpy as np

def create_model(num_classes, batch_size, input_length=978, embedding_size=32, hidden_size=1024, max_m=0.25):
    """
    3-stagenn
    https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold//blob/main/final/Best%20LB/Training/3-stagenn-train.ipynb
    https://www.kaggle.com/c/lish-moa/discussion/201510
    """
    gene_input = layers.Input(shape=(input_length,), name="gene_expression_vector")

    x = layers.BatchNormalization()(gene_input)
    x = layers.Dropout(0.15)(x)
    #x = tfa.layers.WeightNormalization(layers.Dense(hidden_size))(x) #equiv of pytorch linear
    x = tfp.layers.weight_norm.WeightNorm(layers.Dense(hidden_size), data_init=True)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(hidden_size)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = tfp.layers.weight_norm.WeightNorm(layers.Dense(embedding_size, name="embedding"))(x) #equiv of pytorch linear

    l2norm_embedding = layers.Lambda(lambda t: K.l2_normalize(t, axis=1))(x) #https://stackoverflow.com/questions/53960965/normalized-output-of-keras-layer

    labels = layers.Input(shape=(1,), dtype = np.int32, name="labels") 
    x = ArcFace(num_classes, batch_size, max_m=max_m)([l2norm_embedding, labels]) 
    output = layers.Activation('softmax')(x)

    return Model([gene_input, labels], output)