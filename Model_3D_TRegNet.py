import numpy as np
import cv2 as cv
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate


def A3D_TRegnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    up2 = UpSampling3D(size=(2, 2, 2))(pool1)
    up2 = Conv3D(64, (2, 2, 2), activation='relu', padding='same')(up2)
    merge2 = concatenate([conv1, up2], axis=4)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge2)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    output = Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv2)
    model = Model(inputs=inputs, outputs=output)
    return model


def Model_3D_TRegNet(Data, Target):
    input_shape = (32, 32, 32, 3)
    num_classes = Target.shape[3]
    tar = []
    for i in range(len(Target)):
        if len(Target[i].shape) != 3:
            targ = cv.cvtColor(Target[i], cv.COLOR_GRAY2RGB)
            tar.append(targ)
        else:
            tar.append(Target[i])
    tar = np.asarray(tar)

    IMG_SIZE = 32
    Train_X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, IMG_SIZE, 3))

    Train_Y = np.zeros((tar.shape[0], IMG_SIZE, IMG_SIZE, IMG_SIZE, 3))
    for i in range(tar.shape[0]):
        temp_1 = np.resize(tar[i], (IMG_SIZE * IMG_SIZE * IMG_SIZE, 3))
        Train_Y[i] = np.reshape(temp_1, (IMG_SIZE, IMG_SIZE, IMG_SIZE, 3))

    model = A3D_TRegnet(input_shape, num_classes)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, Train_Y, epochs=10, batch_size=32)
    predictions = model.predict(Train_X)
    Pred = np.zeros((predictions.shape[0], Data.shape[1], Data.shape[2], 3))
    for i in range(predictions.shape[0]):
        temp_1 = np.resize(predictions[i], (Data.shape[1] * Data.shape[2], 3))
        Pred[i] = np.reshape(temp_1, (Data.shape[1], Data.shape[2], 3))
    return Pred

