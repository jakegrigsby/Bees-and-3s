import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input

def build_model():
    inp = Input(shape=(100,100,1))
    conv1 = Conv2D(filters=3, kernel_size=(8,8), padding="same", activation='relu', data_format='channels_last')(inp)
    flatten = Flatten()(conv1)
    dense = Dense(3, activation='relu')(flatten)
    out = Dense(1, activation='sigmoid')(dense)
    return tf.keras.models.Model(inputs=inp, outputs=out)