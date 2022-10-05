import math

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers

def _resolve_training(layer, training):
    if training is None:
        training = K.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        training = False
    return training

class ArcFace(Layer):
    """
    Modified implementation of ArcFace layer with
        - scale factor as a learned param
        - margin linearly increased with each batch up to a maximum 

    Arguments:
      num_classes: number of classes to classify
      m: margin
      regularizer: weights regularizer
    """
    def __init__(self,
                 num_classes,
                 batch_size,
                 max_m=0.5,
                 regularizer=None,
                 name='arcface',
                 **kwargs):
        
        super().__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self.max_m = float(max_m)
        self.batch_size = batch_size
        self._regularizer = regularizer

    def build(self, input_shapes):
        embedding_shape, label_shape = input_shapes
        self.count = tf.Variable(0, dtype=tf.float32, trainable=False)

        self._s = self.add_weight(name='scale',
                                 shape=(1),
                                 initializer=tf.keras.initializers.Constant(30.0),
                                 dtype='float32',
                                 trainable=True)

        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer,
                                  name='cosine_weights')
        
    def call(self, inputs, training=None):
        """
        During training, requires 2 inputs: embedding (after backbone+pool+dense),
        and ground truth labels. The labels should be sparse (and use
        sparse_categorical_crossentropy as loss).
        """
        embedding, label = inputs
        
        #increase margin param by 0.0002 for each batch up to a maximum
        _m = self.count.assign(
            tf.math.maximum(
                (tf.math.floordiv(self.count, self.batch_size))*0.0002, self.max_m
                )
            )

        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        label = tf.reshape(label, [-1], name='label_shape_correction')

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')

        training = _resolve_training(self, training)
        if not training:
            # We don't have labels if we're not in training mode
            return self._s * cosine_sim
        else:
            one_hot_labels = tf.one_hot(label,
                                        depth=self._n_classes,
                                        name='one_hot_labels')
            theta = tf.math.acos(K.clip(
                    cosine_sim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            selected_labels = tf.where(tf.greater(theta, math.pi - _m),
                                       tf.zeros_like(one_hot_labels),
                                       one_hot_labels,
                                       name='selected_labels')
            final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                                   theta + _m,
                                   theta,
                                   name='final_theta')
            output = tf.math.cos(final_theta, name='cosine_sim_with_margin')
            return self._s * output

    def get_config(self):
        config = {
            'n_classes': self._n_classes,
            'max_m': self.max_m,
            'batch_size': self.batch_size,
            'regularizer': self._regularizer
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))   

    def compute_output_shape(self, input_shape):
        return (None, self._n_classes)

