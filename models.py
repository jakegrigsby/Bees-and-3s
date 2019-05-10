import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.regularizers import l2

def build_model():
    inp = Input(shape=(28,28,1))
    conv1 = Conv2D(filters=64, kernel_size=(4,4), padding="same", activation='relu', data_format='channels_last')(inp)
    conv1 = Dropout(.5)(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu', data_format='channels_last')(conv1)
    conv2 = Dropout(.5)(conv2)
    conv3 = Conv2D(filters=16, kernel_size=(2,2), padding="same", activation='relu', data_format='channels_last')(conv2)
    conv3 = Dropout(.5)(conv3)
    flatten = Flatten()(conv3)
    dense = Dense(256, kernel_regularizer=l2(.01), activation='relu')(flatten)
    out = Dense(1, activation='sigmoid')(dense)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model