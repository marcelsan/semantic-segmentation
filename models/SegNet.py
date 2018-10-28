from keras import initializers
from keras import layers
from keras import models
from keras import optimizers

from keras.applications.vgg16 import VGG16

def SegNet(input_shape, classes=3):
    input_ = layers.Input(shape=input_shape+(3, ))

    # Encoder
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.glorot_uniform(42))(input_)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.glorot_uniform(42))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.glorot_uniform(42))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.glorot_uniform(42))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Decoder
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.glorot_uniform(42))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.glorot_uniform(42))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.glorot_uniform(42))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.glorot_uniform(42))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(classes, (1, 1), padding='valid', kernel_initializer=initializers.glorot_uniform(42))(x)
    x = layers.Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = layers.Activation("softmax")(x)

    model = models.Model(input_, x)

    return model
