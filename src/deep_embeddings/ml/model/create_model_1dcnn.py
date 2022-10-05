import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, metrics

from keras import backend as K
import tensorflow_addons as tfa

from model.arcface_loss import ArcFace

import numpy as np

def create_model(num_classes, batch_size, input_length=978, embedding_size=32, hidden_size=4096, max_m=0.25):
    """
    Create keras model for entire network: 1D-CNN with (modified) ArcFace loss
    https://www.kaggle.com/c/lish-moa/discussion/202256
    """
    gene_input = layers.Input(shape=(input_length,), name="gene_expression_vector")

    x = layers.BatchNormalization()(gene_input)
    x = layers.Dropout(0.2)(x)
    x = tfa.layers.WeightNormalization(layers.Dense(hidden_size, activation=None))(x) #equiv of pytorch linear

    x = layers.Reshape((256, 16))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ZeroPadding1D(padding=2)(x)
    x = tfa.layers.WeightNormalization(layers.Conv1D(filters=16, kernel_size=5, strides=1, activation='relu', use_bias=False))(x)

    x = tfa.layers.AdaptiveAveragePooling1D(8)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ZeroPadding1D(padding=1)(x)
    x = tfa.layers.WeightNormalization(layers.Conv1D(filters=16, kernel_size=3, strides=1, activation='relu'))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.ZeroPadding1D(padding=1)(x)
    x = tfa.layers.WeightNormalization(layers.Conv1D(filters=8, kernel_size=3, strides=1, activation='relu'))(x)

    x_s = x

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ZeroPadding1D(padding=2)(x)
    x = tfa.layers.WeightNormalization(layers.Conv1D(filters=8, kernel_size=5, strides=1, activation='relu'))(x)

    x = layers.Multiply()([x, x_s])

    x = layers.MaxPooling1D(pool_size=4, strides=2)(x)
    x = layers.Flatten()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(2048, activation='relu')(x)

    x = layers.Dense(embedding_size, name="embedding")(x)
    l2norm_embedding = layers.Lambda(lambda t: K.l2_normalize(t, axis=1))(x) #https://stackoverflow.com/questions/53960965/normalized-output-of-keras-layer

    labels = layers.Input(shape=(1,), dtype = np.int32, name="labels") 
    x = ArcFace(num_classes, batch_size, max_m=max_m)([l2norm_embedding, labels]) 
    output = layers.Activation('softmax')(x)

    return Model([gene_input, labels], output)


