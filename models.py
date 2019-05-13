import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Dropout, MaxPool2D
from tensorflow.keras.regularizers import l2

def build_model(training=False):
    inp = Input(shape=(100,100,1))
    conv1 = Conv2D(filters=64, kernel_size=(4,4), padding="same", activation='relu', data_format='channels_last')(inp)
    conv1 = Dropout(.5)(conv1)
    conv1 = MaxPool2D(2)(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu', data_format='channels_last')(conv1)
    conv2 = Dropout(.5)(conv2)
    conv2 = MaxPool2D(2)(conv2)

    conv3 = Conv2D(filters=16, kernel_size=(2,2), padding="same", activation='relu', data_format='channels_last')(conv2)
    conv3 = Dropout(.5)(conv3)
    conv3 = MaxPool2D(2)(conv3)

    flatten = Flatten()(conv3)
    dense1 = Dense(128, kernel_regularizer=l2(.01), activation='relu')(flatten)
    dense2 = Dense(32, kernel_regularizer=l2(.01), activation='relu')(dense1)
    out = Dense(1, activation='sigmoid')(dense2)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    if training: model.summary()
    return model