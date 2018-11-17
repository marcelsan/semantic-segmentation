from keras import initializers
from keras import layers
from keras import models
from keras import optimizers

from keras.applications.vgg16 import VGG16

def SegNet(input_shape, classes=3):
    # Input layer.
    input_ = layers.Input(shape=input_shape+(3, ))

    # VGG-like encoder
    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input_)

    set_trainable = False
    for layer in vgg.layers:
        if layer.name == 'block5_pool':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # Decoder
    conv = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(vgg.output)        # (10,10)  
    batch = layers.BatchNormalization()(conv)
    relu = layers.LeakyReLU(alpha=0.1)(batch)

    up1 = layers.UpSampling2D()(relu)
    conv1 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(up1)             # (20,20) 
    batch1 = layers.BatchNormalization()(conv1)
    relu1 = layers.LeakyReLU(alpha=0.1)(batch1)

    up2 = layers.UpSampling2D()(relu1)
    conv2 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(up2)              # (40,40)
    batch2 = layers.BatchNormalization()(conv2)
    relu2 = layers.LeakyReLU(alpha=0.1)(batch2)

    up3 = layers.UpSampling2D()(relu2)
    conv3 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(up3)               # (80,80)
    batch3 = layers.BatchNormalization()(conv3)
    relu3 = layers.LeakyReLU(alpha=0.1)(batch3)

    up4 = layers.UpSampling2D()(relu3)
    conv4 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(up4)               # (160,160)
    batch4 = layers.BatchNormalization()(conv4)
    relu4 = layers.LeakyReLU(alpha=0.1)(batch4)

    up5 = layers.UpSampling2D()(relu4)
    conv5 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(up5)               # (320,320)
    batch5 = layers.BatchNormalization()(conv5)
    relu5 = layers.LeakyReLU(alpha=0.1)(batch5)

    out = layers.Conv2D(classes, (1, 1), padding='valid', kernel_initializer=initializers.he_normal(42))(relu5)
    out = layers.Reshape((input_shape[0]*input_shape[1], classes))(out)
    out = layers.Activation("softmax")(out)

    model = models.Model(input_, out)

    return model


def SegNetSkip(input_shape, classes=3):
    # Input layer.
    input_ = layers.Input(shape=input_shape+(3, ))

    # VGG-like encoder
    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input_)

    set_trainable = False
    for layer in vgg.layers:
        if layer.name == 'block4_pool':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # Decoder
    conv = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(vgg.output)        # (10,10)  
    batch = layers.BatchNormalization()(conv)
    relu = layers.LeakyReLU(alpha=0.1)(batch)

    up1 = layers.UpSampling2D()(relu)
    conc1 = layers.Concatenate()([vgg.get_layer('block5_conv2').output, up1])
    conv1 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(conc1)             # (20,20) 
    batch1 = layers.BatchNormalization()(conv1)
    relu1 = layers.LeakyReLU(alpha=0.1)(batch1)

    up2 = layers.UpSampling2D()(relu1)
    conc2 = layers.Concatenate()([vgg.get_layer('block4_conv2').output, up2])
    conv2 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(conc2)              # (40,40)
    batch2 = layers.BatchNormalization()(conv2)
    relu2 = layers.LeakyReLU(alpha=0.1)(batch2)

    up3 = layers.UpSampling2D()(relu2)
    conc3 = layers.Concatenate()([vgg.get_layer('block3_conv2').output, up3])
    conv3 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(conc3)               # (80,80)
    batch3 = layers.BatchNormalization()(conv3)
    relu3 = layers.LeakyReLU(alpha=0.1)(batch3)

    up4 = layers.UpSampling2D()(relu3)
    conc4 = layers.Concatenate()([vgg.get_layer('block2_conv2').output, up4])
    conv4 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(conc4)               # (160,160)
    batch4 = layers.BatchNormalization()(conv4)
    relu4 = layers.LeakyReLU(alpha=0.1)(batch4)

    up5 = layers.UpSampling2D()(relu4)
    conc5 = layers.Concatenate()([vgg.get_layer('block1_conv2').output, up5])
    conv5 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(42))(conc5)               # (320,320)
    batch5 = layers.BatchNormalization()(conv5)
    relu5 = layers.LeakyReLU(alpha=0.1)(batch5)

    out = layers.Conv2D(classes, (1, 1), padding='valid', kernel_initializer=initializers.he_normal(42))(relu5)
    out = layers.Reshape((input_shape[0]*input_shape[1], classes))(out)
    out = layers.Activation("softmax")(out)

    model = models.Model(input_, out)

    return model

