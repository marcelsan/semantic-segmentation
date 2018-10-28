import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

from keras import optimizers
from models.SegNet import SegNet
from initialize import FLAGS

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
            image = cv2.resize(image, image_size, cv2.INTER_NEAREST)

            label = cv2.imread(labels[i]) * 1.0
            label = cv2.resize(label, image_size, cv2.INTER_NEAREST)
            label = np.expand_dims(label, axis=2)
            label = label.reshape(image_size[0] * image_size[1], 3)

            top_batch.append(image)
            batch_labels.append(label)

            i += 1

            if i >= data_size:
                i = 0

        yield np.array(top_batch), np.array(batch_labels)

def main():
    model = SegNet(input_shape=(480,480))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.1), metrics=['acc'])

    train_generator = data_gen(FLAGS.trainImageDir, FLAGS.trainLabelsDir)
    validation_generator = data_gen(FLAGS.valImagesDir, FLAGS.valLabelsDir)

    history = model.fit_generator(train_generator, steps_per_epoch=1000, epochs=50,
                                  validation_data=validation_generator, validation_steps=500)

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
