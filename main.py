import datetime
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from PIL import Image

from keras import optimizers, callbacks
from keras.utils import np_utils
from models.SegNet import SegNet
from initialize import FLAGS

def data_gen(images_dir, labels_dir, nb_classes=21, batch_size=8, image_size=(320, 320)):
    """
    Generator to yield batches of two inputs (per sample) with shapes top_dim and 
    bot_dim along with their labels.
    """
    images = glob.glob(os.path.join(images_dir, '*.jpg'))
    data_size = len(images)
    i = 0

    while True:
        top_batch = []
        batch_labels = []

        for _ in range(batch_size):
            # Input
            image = cv2.imread(images[i]) * 1./255
            image = cv2.resize(image, image_size, cv2.INTER_NEAREST)

            # Label
            label_file = os.path.join(labels_dir, 
                                      os.path.splitext(os.path.basename(images[i]))[0] + '.png')

            Y = np.array(Image.open(label_file))
            Y[Y == 255] = 0
            Y = np_utils.to_categorical(Y, nb_classes)
            Y = cv2.resize(Y, image_size)
            label = Y.reshape(image_size[0] * image_size[1], nb_classes).astype(np.int8)
    
            top_batch.append(image)
            batch_labels.append(label)

            i += 1

            if i >= data_size:
                # Shuffle??
                i = 0

        yield np.array(top_batch), np.array(batch_labels)

def main():
    # Create the model.
    model = SegNet(input_shape=(320,320), classes=FLAGS.numClasses)
    model.summary()
    # model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.001, momentum=0.9), metrics=['acc'])

    # Read the dataset.
    train_generator = data_gen(FLAGS.trainImageDir, FLAGS.trainLabelsDir, FLAGS.numClasses)
    validation_generator = data_gen(FLAGS.valImagesDir, FLAGS.valLabelsDir, FLAGS.numClasses)

    # Create the CSV Logger callback
    csv_logger = callbacks.CSVLogger('logs/training_%s.log' % (datetime.datetime.now()))

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
                                            patience=4, min_lr=0.00003, verbose=1)

    # Train the model.
    print('[INFO] Started training.')
    start_time = time.time()
    history = model.fit_generator(train_generator, steps_per_epoch=1000, epochs=30,
                                  validation_data=validation_generator, validation_steps=500,
                                  callbacks=[csv_logger, reduce_lr])

    print('[INFO] Train took: %s' % (time.time() - start_time))

    if FLAGS.saveModel:
        # serialize model to JSON
        model_json = model.to_json()
        with open("weights/model.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights("weights/model.h5")
        print("[INFO] Saved model to disk.")

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(acc) + 1)
    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # plt.figure()
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()
