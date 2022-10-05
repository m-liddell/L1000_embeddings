#%%
import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset

#python3 -m pip install git+https://github.com/nalexai/hyperlib.git@main
from hyperlib.manifold.poincare import Poincare

#%%
a = tf.constant([[5.0,9.4,3.0],[2.0,5.2,8.9],[4.0,7.2,8.9]])
b = tf.constant([[4.8,1.0,2.3]])


#%%
a_dataset = np.array([[4.0,9.4,3.0],[2.0,5.2,8.9],[4.0,7.2,8.9]])
b_dataset = np.array([[4.8,1.0,2.3]])

#%%
p = Poincare()

curvature = 1
p.dist(a_dataset, b_dataset, curvature)

# %%
