import random
import sys

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
tf.enable_eager_execution()
import predict

def bg_noise(image):
    bg = np.logical_or(image < .02, image > .98)
    bg = bg.astype(int)
    noise = random.uniform(-.4,.4) * np.random.rand(*image.shape)
    bg_noise = bg * noise
    image += bg_noise
    return image

def bg_noise_tf(image):
    bg = tf.logical_or(image < .02, image > .98)
    bg = tf.dtypes.cast(bg, tf.float32)
    noise = random.uniform(-.4, .4) * tf.random.uniform(image.shape, dtype=tf.float32)
    bg_noise = bg * noise
    image += bg_noise
    return image.numpy()

def noise_tf(image):
    image += random.uniform(-.4,.4) * tf.random.uniform(image.shape, dtype=tf.float32)
    return image.numpy()

if __name__ == "__main__":
    image, raw_image = predict.url_to_image(sys.argv[1])
    image = np.squeeze(np.squeeze(image, axis=0))
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image, cmap='gray')
    axarr[1].imshow(bg_noise_tf(image), cmap='gray')
    plt.show()