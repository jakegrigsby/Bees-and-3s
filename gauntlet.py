import math

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

import predict, models

if __name__ == "__main__":
    image_set = []
    with open('data/test_set.csv') as test_file:
        urls = test_file.readlines()
    for url in urls:
        image, _= predict.url_to_image(url, add_noise=True)
        image = np.squeeze(image, 0)
        image_set.append(image)
        """
        plt.imshow(np.squeeze(image), cmap='gray')
        print(url)
        plt.show()
        """

    model = models.build_model()
    model.compile(optimizer=tf.train.AdamOptimizer(0.003),
                        loss = 'binary_crossentropy',
                        metrics=['binary_accuracy'])
    model.load_weights('weights/model_save.h5')
    predictions = model.predict(np.asarray(image_set))
    labels = ["THREE" if pred < .5 else "BEE" for pred in predictions]

    columns = 4
    rows = math.ceil(len(urls)/columns)
    f, axarr = plt.subplots(rows, columns)
    for row in range(rows):
        for column in range(columns):
            idx = columns*(row) + (column)
            if idx >= len(urls): break
            image = np.squeeze(image_set[idx])
            conf = max(predictions[idx][0], 1-predictions[idx][0]) * 100
            label = labels[idx]
            axarr[row, column].imshow(image, cmap='gray')
            title = '{} with {:.1f} confidence'.format(label, conf)
            axarr[row, column].set_title(title)
    plt.show()