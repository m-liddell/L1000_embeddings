import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, metrics
from keras import backend as K

def dense_block(x, blocks):
	"""
    Generate n DenseNet dense blocks
	# Arguments
		x: input tensor.
		blocks: integer, the number of building blocks.
	# Returns
		output tensor for the block.
	"""
	for i in range(blocks):
		x = building_block(x)
	return x

def building_block(x):
	"""
    Generate a DenseNet dense block which contains (BN + DENSE + selu)*3
	# Arguments
		x: input tensor.
		name: string, block label.
	# Returns
		Output tensor for the block.
	"""
	x1 = layers.Dense(x.shape[1], activation='selu')(x)
	x1 = layers.Dense(x1.shape[1], activation='selu')(x1)
	x1 = layers.Dense(x1.shape[1], activation='selu')(x1)
	x = layers.Concatenate()([x, x1])
	return x
	
def create_densenet(input_length, blocks, embedding_size):
    """
    Generates a DenseNet architecture
    # Arguments
        blocks: numbers of building blocks for the four dense layers
    # Returns
        A Keras model instance
    """
    # input layer
    densenet_input = layers.Input(shape=(input_length,), name="gene_expression_vector")
    x = layers.GaussianNoise(0.3)(densenet_input)
    x = layers.Dense(input_length, activation='selu')(x)

    #densely connected parts
    x = dense_block(x, blocks)

    #output layer
    x = layers.Dense(embedding_size, name="embedding")(x)
    l2norm_embedding = layers.Lambda(lambda t: K.l2_normalize(t, axis=1))(x) #https://stackoverflow.com/questions/53960965/normalized-output-of-keras-layer
    model = Model(inputs=densenet_input, outputs=l2norm_embedding, name='densenet')
    return model