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
from models.SegNet import SegNet, SegNetSkip
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
                i = 0

        yield np.array(top_batch), np.array(batch_labels)

def main():
    # Create the model.
    model = SegNetSkip(input_shape=(320,320), classes=FLAGS.numClasses)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.01), metrics=['acc'])

    if FLAGS.saveModel:
        # serialize model to JSON
        model_json = model.to_json()
        with open("weights/%s_model.json" % (FLAGS.experimentName), "w") as json_file:
            json_file.write(model_json)

    # Read the dataset.
    train_generator = data_gen(FLAGS.trainImageDir, FLAGS.trainLabelsDir, FLAGS.numClasses)
    validation_generator = data_gen(FLAGS.valImagesDir, FLAGS.valLabelsDir, FLAGS.numClasses)

    # Create the CSV Logger callback
    csv_logger = callbacks.CSVLogger('logs/%s_training_%s.log' % (FLAGS.experimentName, datetime.datetime.now()))

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
                                            patience=4, min_lr=0.00003, verbose=1)

    checkpoint = callbacks.ModelCheckpoint('weights/%s_weights.{epoch:02d}-{val_loss:.2f}.hdf5' % (FLAGS.experimentName), 
                                            monitor='val_loss', verbose=1, period=5)

    # Train the model.
    print('Started training.')
    start_time = time.time()
    history = model.fit_generator(train_generator, steps_per_epoch=400, epochs=60,
                                  validation_data=validation_generator, validation_steps=150,
                                  callbacks=[csv_logger, reduce_lr, checkpoint])

    print('Train took: %s' % (time.time() - start_time))

    if FLAGS.saveFinalWeights:
        # serialize weights to HDF5
        model.save_weights("weights/%s_model.h5" % (FLAGS.experimentName))
        print("Saved model to disk.")

if __name__ == '__main__':
    main()
