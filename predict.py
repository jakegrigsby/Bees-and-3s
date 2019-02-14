import numpy as np
import tensorflow as tf
import models
import sys
import cv2
import urllib

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100,100)) / 255
    image = np.expand_dims(np.asarray(image, dtype=np.float32), -1)
    return np.expand_dims(image, 0)


image = url_to_image(sys.argv[1])
model = models.build_model()
model.compile(optimizer=tf.train.AdamOptimizer(0.003),
                loss = 'binary_crossentropy',
                metrics=['binary_accuracy'])
model.load_weights('models/model_save.h5')
prediction = model.predict(image)[0]
classification = "THREE" if prediction < .5 else "BEE"
confidence = (abs(prediction - .5)/.5)*100
print("{} with {}% confidence.".format(classification, confidence))

