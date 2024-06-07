import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, concatenate
import cv2
import numpy as np


def CNN(input_shape, num_classes):
    # Encoder
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Decoder
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def Model_CNN(Data, Gt):
    input_shape = (Data.shape[1], Data.shape[2], Data.shape[3])
    if len(Gt.shape) <= 3:
        gt = []
        for i in range(Gt.shape[0]):
            gtimg = cv2.cvtColor(Gt[i], cv2.COLOR_GRAY2RGB)
            gt.append(gtimg)
        Gt = np.asarray(gt)
    num_classes = Gt.shape[-1]
    model = CNN(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Data, Gt, steps_per_epoch=2, epochs=5)
    pridicted = model.predict(Data)
    return pridicted

