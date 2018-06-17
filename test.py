from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras.applications.vgg16 import VGG16
from keras import initializers
from keras import optimizers
from keras.models import model_from_json

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

    test_image = cv2.imread('datasets/test_images/2.png') * 1./255
    #shape_ = (test_image.shape[0], test_image.shape[1])
    #test_image = cv2.resize(test_image, (128, 128))

    batch = np.expand_dims(test_image, axis=0)
    label = loaded_model.predict(batch, 1)

    #print(label)

    print("[INFO] Segmentation predicted")

    label = label[0] * 255
    #label = cv2.resize(label, shape_) * 255
    _, label = cv2.threshold(label, 25, 255, cv2.THRESH_BINARY)
    #plt.imshow('test.jpg', label)
    cv2.imwrite('test.jpg', label)

if __name__ == '__main__':
    main()
