from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras.applications.vgg16 import VGG16
from keras import initializers
from keras import optimizers

import glob
import os
import cv2
import numpy as np

SAVE_MODEL = False
TRAIN_SET_IMAGES_DIR = 'datasets/HUMANS/train/jpge'
TRAIN_SET_LABELS_DIR = 'datasets/HUMANS/train/segmented'
VALIDATION_SET_IMAGES_DIR = 'datasets/HUMANS/validation/jpge'
VALIDATION_SET_LABELS_DIR = 'datasets/HUMANS/validation/segmented'

def create_model(input_w=480, input_h=480):
    input_ = layers.Input(shape=(input_h, input_w, 3))

    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input_)

    set_trainable = False
    for layer in vgg.layers:
        if layer.name == 'block5_pool':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    x = layers.Conv2D(4096, kernel_size=(3, 3), use_bias=False, activation='relu',
                    padding='same', kernel_initializer=initializers.glorot_uniform(42))(vgg.output)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, kernel_size=(1, 1), use_bias=False, activation='relu',
                    padding='same', kernel_initializer=initializers.glorot_uniform(42))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(1, kernel_size=(1, 1), use_bias=False, activation='relu',
                    padding='same', kernel_initializer=initializers.glorot_uniform(42))(x)
    x = layers.Conv2DTranspose(1, kernel_size=(64, 64), strides=(
        32, 32), use_bias=False, activation='sigmoid', padding='same', kernel_initializer=initializers.glorot_uniform(42))(x)

    model = models.Model(input_, x)

    return model

def data_gen(images_dir, labels_dir, batch_size=16, image_size=(480, 480)):
    """
    Generator to yield batches of two inputs (per sample) with shapes top_dim and 
    bot_dim along with their labels.
    """

    images = glob.glob(os.path.join(images_dir, '*.jpg'))
    labels = glob.glob(os.path.join(labels_dir, '*.png'))
    assert(len(images) == len(labels))

    data_size = len(images)

    i = 0

    while True:
        top_batch = []
        batch_labels = []

        for _ in range(batch_size):
            image = cv2.imread(images[i]) * 1./255
            label = cv2.imread(labels[i], 0)
            label.astype(np.uint8)
            label = cv2.resize(label, image_size)
            label = np.expand_dims(label, axis=2)
            image = cv2.resize(image, image_size)

            top_batch.append(image)
            batch_labels.append(label)

            i += 1

            if i >= data_size:
                i = 0

        yield np.array(top_batch), np.array(batch_labels)

def main():
    model = create_model()

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, verbose=1, min_lr=1e-6)

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=10,
                               verbose=0, mode='auto')

    callbacks_list = [reduce_lr, early_stop]

    train_generator = data_gen(TRAIN_SET_IMAGES_DIR, TRAIN_SET_LABELS_DIR)
    validation_generator = data_gen(VALIDATION_SET_IMAGES_DIR, VALIDATION_SET_LABELS_DIR)

    history = model.fit_generator(train_generator, steps_per_epoch=1500, epochs=10,
                                  validation_data=validation_generator, validation_steps=800,
                                  callbacks=callbacks_list)

    if SAVE_MODEL:
        # serialize model to JSON
        model_json = model.to_json()
        with open("weights/model.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights("weights/model.h5")
        print("[INFO] Saved model to disk.")

if __name__ == '__main__':
    main()
