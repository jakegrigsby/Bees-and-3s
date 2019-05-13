import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import models, random_noise

#suppress annoying tf warnings
tf.logging.set_verbosity(tf.logging.FATAL)

#load data
assert os.path.exists('data/data.npz'), "Run setup.py first"
data = np.load('data/data.npz')
images, labels = data['arr_0'], data['arr_1']
train_test_split_idx = int(.8*images.shape[0])
images, test_images = images[:train_test_split_idx], images[train_test_split_idx:]
labels, test_labels = labels[:train_test_split_idx], labels[train_test_split_idx:]

if not os.path.exists('weights'):
    os.mkdir('weights')

#build and compile neural network
model = models.build_model(training=True)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                loss = 'binary_crossentropy',
                metrics=['accuracy'])
    
#image augmentation to expand the small dataset
base_augmentor = ImageDataGenerator(rotation_range=20, zoom_range=.25,
                                width_shift_range=.2, height_shift_range=.2,
                                shear_range=.15, horizontal_flip=False, 
                                fill_mode='nearest').flow(images, labels, batch_size=32)

def random_noise_generator(batches):
    while True:
        batch_images, batch_labels = next(batches)
        for image_idx in range(batch_images.shape[0]):
            batch_images[image_idx,...] = random_noise.bg_noise_tf(batch_images[image_idx,...])
        yield (batch_images, batch_labels)


augmented_augmentor = random_noise_generator(base_augmentor)

#callbacks to increase validation accuracy
callbacks = [EarlyStopping(monitor='val_acc', patience=2), 
                ReduceLROnPlateau(monitor='val_acc', factor=.5, patience=1)]

#train 
model.fit_generator(augmented_augmentor, validation_data=(test_images, test_labels), 
                    steps_per_epoch=int(images.shape[0]/32), epochs=5, callbacks=callbacks)