import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, concatenate
import cv2
import numpy as np


def Interactive_Segmentation(input_shape, num_classes):
    # Image input
    image_input = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(image_input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv2), conv1], axis=-1)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(up3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(conv3)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(conv3)

    model = tf.keras.Model(inputs=image_input, outputs=outputs)
    return model


def Model_Interactive_segmentation(image, Gt):
    input_shape = (image.shape[1], image.shape[2], image.shape[3])

    if len(Gt.shape) <= 3:
        gt = []
        for i in range(Gt.shape[0]):
            gtimg = cv2.cvtColor(Gt[i], cv2.COLOR_GRAY2RGB)
            gt.append(gtimg)
        Gt = np.asarray(gt)
    num_classes = Gt.shape[-1]
    model = Interactive_Segmentation(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(image, Gt, steps_per_epoch=2, epochs=5)
    pridicted = model.predict(image)
    return pridicted

