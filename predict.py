import sys
import urllib

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import models, random_noise

tf.logging.set_verbosity(tf.logging.FATAL)

def url_to_image(url, add_noise=False):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    raw_image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(raw_image, (100,100), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 255
    image = np.expand_dims(np.asarray(image, dtype=np.float32), -1)
    image = np.expand_dims(image, 0)
    if add_noise: image = random_noise.bg_noise_tf(image)
    return image, raw_image


if __name__ == "__main__":
    image, raw_image = url_to_image(sys.argv[1])
    model = models.build_model()
    model.compile(optimizer=tf.train.AdamOptimizer(0.003),
                    loss = 'binary_crossentropy',
                    metrics=['binary_accuracy'])
    model.load_weights('weights/model_save.h5')
    prediction = model.predict(image)[0]
    url_classification = "THREE" if prediction < .5 else "BEE"
    url_confidence = max(prediction, 1-prediction) * 100
    plt.imshow(raw_image, cmap='gray')
    plt.title("{} with {:.3f}% confidence...".format(url_classification, url_confidence[0]))
    plt.show()