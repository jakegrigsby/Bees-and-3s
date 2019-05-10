import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import models

#suppress annoying tf warnings
tf.logging.set_verbosity(tf.logging.FATAL)

#load data
assert os.path.exists('data/data.npz'), "Run setup.py first"
data = np.load('data/data.npz')
images, labels = data['arr_0'], data['arr_1']
train_test_split_idx = int(.7*images.shape[0])
images, test_images = images[:train_test_split_idx], images[train_test_split_idx:]
labels, test_labels = labels[:train_test_split_idx], labels[train_test_split_idx:]

if not os.path.exists('models'):
    os.mkdir('models')
    os.mkdir('models/log_dir')

#build and compile neural network
model = models.build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                loss = 'binary_crossentropy',
                metrics=['accuracy'])
    
#image augmentation to expand the small dataset
augmentor = ImageDataGenerator(rotation_range=30, zoom_range=.2,
                                width_shift_range=.3, height_shift_range=.3,
                                shear_range=.25, horizontal_flip=False, 
                                fill_mode='nearest')

#callbacks to increase validation accuracy
#not used (for now) because we only need to run for 1 epoch...
callbacks = [EarlyStopping(monitor='val_acc', patience=2), 
                ReduceLROnPlateau(monitor='val_acc', factor=.5, patience=1)]

#train 
model.fit_generator(augmentor.flow(images, labels, batch_size=32), validation_data=(test_images, test_labels), 
                    steps_per_epoch=int(images.shape[0]/32), epochs=1, callbacks=callbacks)
model.save_weights('models/model_save.h5')