import os

from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import add, Activation, LSTM, Conv1D, InputSpec, Input
from tensorflow.keras.layers import MaxPooling1D, SpatialDropout1D, Bidirectional, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

os.environ['KERAS_BACKEND']='tensorflow'

def make_resblock1d(layer, filters=64, kernel_size=11, activation="relu", rate=0.1):
    """Creates a residual network block.

    Parameters:
        layer: the input layer to the block
        filters (list of ints): number of filters in the conv layers
        kernel_size (int): size of the convolution kernel
        activation (str or tf.keras.activation): activation type
        rate (float): spatial dropout rate

    Returns:
        A residual block
    """
    x = BatchNormalization()(layer)
    x = Activation(activation)(x)
    x = SpatialDropout1D(rate=rate)(x)
    x = Conv1D(filters,kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = SpatialDropout1D(rate=rate)(x)
    x = Conv1D(filters,kernel_size,padding="same")(x)

    return x


def make_network(input_shape, kernel_size=11):
    """Create the neural network model.

    Parameters:
        input_shape (tuple of ints): the shape of the input tensor
        kernel_size (int): size of convolution kernel
    Returns:
        Keras model
    """
    first = True

    input = Input(shape=input_shape)

    # first conv layers
    for nfilters in [8, 16,16,32,32,64,64]:
        if first:
            first = False
            x = Conv1D(filters=nfilters, kernel_size=kernel_size, padding="same")(input)
            x = MaxPooling1D()(x)
        else:
            x = Conv1D(filters=nfilters, kernel_size=kernel_size, padding="same")(x)
            x = MaxPooling1D()(x)

    # add residual blocks
    for i in range(7):
        resid = make_resblock1d(x)
        x = add([x, resid])

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    return Model(input,x)