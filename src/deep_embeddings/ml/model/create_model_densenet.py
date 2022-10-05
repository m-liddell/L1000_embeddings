import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, metrics

from model.densenet import create_densenet
from model.arcface_loss import ArcFace

import numpy as np

def create_model(num_classes, batch_size, input_length=978, dense_blocks_no=1, embedding_size=32, max_m=0.25):
    """
    Create keras model for entire network: DenseNet with (modified) ArcFace loss
    """
    densenet = create_densenet(input_length, dense_blocks_no, embedding_size)

    labels = layers.Input(shape=(1,), dtype = np.int32, name="labels") 
    x = ArcFace(num_classes, batch_size, max_m=max_m)([densenet.output, labels]) 
    output = layers.Activation('softmax')(x)

    return Model([densenet.input, labels], output)