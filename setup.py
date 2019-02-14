import os
import csv
import urllib
import numpy as np
import cv2
import tensorflow as tf

print("Loading MNIST images from tensorflow...")
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = tf.keras.datasets.mnist.load_data()
train_3_idxs, test_3_idxs = mnist_train_y==3, mnist_test_y==3
mnist_train_x, mnist_test_x = mnist_train_x[train_3_idxs], mnist_test_x[test_3_idxs]
threes = np.concatenate((mnist_train_x, mnist_test_x), axis=0)

print("Resizing MNIST 3s...")
progbar = tf.keras.utils.Progbar(threes.shape[0])
resized_buffer = []
for i in range(threes.shape[0]):
    try:
        resized_buffer.append(cv2.resize(threes[i,...], (100,100)) / 255)
    except:
        continue
    progbar.add(1)
threes = np.asarray(resized_buffer, dtype=np.float32)
threes_labels = np.zeros(threes.shape[0])

print("Loading bee images...")
DATA_PATH = 'data/bee_imgs'
progbar = tf.keras.utils.Progbar(len(os.listdir('data/bee_imgs')))
bees = []
for image_filename in os.listdir(DATA_PATH):
    bee = cv2.imread(os.path.join(DATA_PATH, image_filename), cv2.IMREAD_GRAYSCALE)
    try:
        bee = cv2.resize(bee, (100,100)) / 255
    except:
        continue
    else:
        bees.append(bee)
    progbar.add(1)

bees = np.asarray(bees, dtype=np.float32)
print('Saving images to data/data.npz')
bees_labels = np.ones(bees.shape[0])
x = np.concatenate((threes, bees), axis=0)
x = np.expand_dims(x, 3)
y = np.append(threes_labels, bees_labels)
y = np.expand_dims(y, 1)
if not os.path.exists('data'):
    os.mkdir('data')
np.savez('data/data.npz', x, y)