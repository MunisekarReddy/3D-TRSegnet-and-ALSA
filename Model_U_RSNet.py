import cv2
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, concatenate

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

    y = tf.keras.layers.add([x, y])
    y = Activation('relu')(y)
    return y

def U_RSNet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = residual_block(inputs, filters=64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_block(pool1, filters=128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv2), conv1], axis=-1)
    conv3 = residual_block(up3, filters=64)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv3)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def Model_U_RSNet(image, Gt):
    input_shape = (image.shape[1], image.shape[2], image.shape[3])

    if len(Gt.shape) <= 3:
        gt = []
        for i in range(Gt.shape[0]):
            gtimg = cv2.cvtColor(Gt[i], cv2.COLOR_GRAY2RGB)
            gt.append(gtimg)
        Gt = np.asarray(gt)
    num_classes = Gt.shape[-1]
    model = U_RSNet(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(image, Gt, steps_per_epoch=2, epochs=5)
    pridicted = model.predict(image)
    return pridicted



import numpy as np

if __name__ == '__main__':
    # image = np.random.uniform(0, 255, (10, 256, 256, 3))
    # gt = np.random.uniform(0, 255, (10, 256, 256, 3))
    Orig = np.load('CT_GT_Image_1.npy', allow_pickle=True)[:5]
    Feat_1 = np.load('CT_Registered_Images.npy', allow_pickle=True)[:5]
    pred = Model_U_RSNet(Feat_1, Orig)
    yhtrgb = 435
