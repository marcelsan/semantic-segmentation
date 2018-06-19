from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras.applications.vgg16 import VGG16
from keras import initializers
from keras import optimizers
from keras.models import model_from_json
from keras.losses import binary_crossentropy
from keras import backend as K

import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    json_file = open('weights/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights("weights/model.h5")
    print("[INFO] Loaded model from disk")

    test_image = cv2.imread(
        'datasets/HUMANS/train/jpge/2007_000170.jpg') * 1./255
    
    shape_ = (test_image.shape[0], test_image.shape[1])
    test_image = cv2.resize(test_image, (480, 480))

    batch = np.expand_dims(test_image, axis=0)
    label = loaded_model.predict(batch, 1)[0]


    # image_pred = K.variable(label)
    # true_image = cv2.imread(
    #     'datasets/HUMANS/train/segmented/2007_001430.png', 0) * 1.
    # true_image = cv2.resize(true_image, (480, 480))
    # true_image2 = true_image * 255
    # true_image = np.expand_dims(true_image, axis=2)
    # true_image = K.variable(true_image)

    #loss = binary_crossentropy(true_image, image_pred)
    print("[INFO] Segmentation predicted")
    #a = K.eval(loss)
    #print(a)

    label = cv2.resize(label, shape_) * 255
    #_, label = cv2.threshold(label, 128, 255, cv2.THRESH_BINARY)
    #plt.imshow(label, cmap='gray')
    #plt.imshow(true_image2, cmap='gray')
    #plt.show()
    cv2.imwrite('test2.jpg', label)

if __name__ == '__main__':
    main()
