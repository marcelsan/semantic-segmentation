from keras import initializers
from keras import layers
from keras import models
from keras import optimizers

from keras.applications.vgg16 import VGG16

def SegNet(input_shape, classes=3):
    # Input layer.
    input_ = layers.Input(shape=input_shape+(3, ))

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(input_)     # (320,320)
    batch1 = layers.BatchNormalization()(conv1)
    relu1 = layers.LeakyReLU(alpha=0.1)(batch1)
    pool1 = layers.MaxPooling2D()(relu1)

    conv2 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(pool1)     # (160,160)
    batch2 = layers.BatchNormalization()(conv2)
    relu2 = layers.LeakyReLU(alpha=0.1)(batch2)
    pool2 = layers.MaxPooling2D()(relu2)

    conv3 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(pool2)     # (80,80)
    batch3 = layers.BatchNormalization()(conv3)
    relu3 = layers.LeakyReLU(alpha=0.1)(batch3)
    pool3 = layers.MaxPooling2D()(relu3)

    conv4 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(pool3)     # (40,40)
    batch4 = layers.BatchNormalization()(conv4)
    relu4 = layers.LeakyReLU(alpha=0.1)(batch4)

    # Decoder
    conv5 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(relu4)     # (40,40)  
    batch5 = layers.BatchNormalization()(conv5)
    relu5 = layers.LeakyReLU(alpha=0.1)(batch5)

    up1 = layers.UpSampling2D()(relu5)
    conv6 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(up1)     # (80,80) 
    batch6 = layers.BatchNormalization()(conv6)
    relu6 = layers.LeakyReLU(alpha=0.1)(batch6)

    up2 = layers.UpSampling2D()(relu6)
    conv7 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(up2)     # (160,160)
    batch7 = layers.BatchNormalization()(conv7)
    relu7 = layers.LeakyReLU(alpha=0.1)(batch7)

    up3 = layers.UpSampling2D()(relu7)
    conv8 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(up3)          # (320,320)
    batch8 = layers.BatchNormalization()(conv8)
    relu8 = layers.LeakyReLU(alpha=0.1)(batch8)

    out = layers.Conv2D(classes, (1, 1), padding='valid', kernel_initializer=initializers.he_normal(42))(relu8)
    out = layers.Reshape((input_shape[0]*input_shape[1], classes))(out)
    out = layers.Activation("softmax")(out)

    model = models.Model(input_, out)

    return model
