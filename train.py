import os
import numpy as np
import tensorflow as tf
import models

assert os.path.exists('data/data.npz'), "Run setup.py first"
data = np.load('data/data.npz')
images, labels = data['arr_0'], data['arr_1']

if not os.path.exists('models'):
    os.mkdir('models')
    os.mkdir('models/log_dir')

model = models.build_model()
model.compile(optimizer=tf.train.AdamOptimizer(0.002),
                loss = 'binary_crossentropy',
                metrics=['binary_accuracy'])
model.fit(images, labels, batch_size=32, epochs=1,
                    validation_split=.3, verbose=1)
model.save_weights('models/model_save.h5')