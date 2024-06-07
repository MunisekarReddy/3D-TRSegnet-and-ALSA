import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
import cv2
import numpy as np


def residual_block(x, filters, kernel_size=3, strides=1):
    y = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization()(y)

    # Skip connection
    if strides != 1 or x.shape[-1] != filters:
        x = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)

    y = Add()([x, y])
    y = Activation('relu')(y)
    return y

def RSegNet(input_shape, num_channels):
    inputs = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=128, strides=2)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=256, strides=2)
    x = residual_block(x, filters=256)

    # Transposed convolutional layers for upsampling
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Output layer
    outputs = Conv2D(num_channels, kernel_size=3, activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def Model_RSegNet(Data, Gt):
    input_shape = (Data.shape[1], Data.shape[2], Data.shape[3])
    if len(Gt.shape) <= 3:
        gt = []
        for i in range(Gt.shape[0]):
            gtimg = cv2.cvtColor(Gt[i], cv2.COLOR_GRAY2RGB)
            gt.append(gtimg)
        Gt = np.asarray(gt)
    num_classes = Gt.shape[-1]
    model = RSegNet(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Data, Gt, steps_per_epoch=2, epochs=5)
    pridicted = model.predict(Data)
    return pridicted


