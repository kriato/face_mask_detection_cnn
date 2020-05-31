# pylint: disable=undefined-variable, no-member

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D 
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

import config as cfg

def get_model(which):
    if which == cfg.MODEL.CLEAN_V1:
        return clean_v1()
    elif which == cfg.MODEL.MOBILEV2TOP_V1:
        return mobilev2top_v1()
    else:
        print(f'{which} not found')
        return None

def clean_v1():
    data_mask_path = '../dataset/AFDB/train_data_clean_model/2203_faces_with_mask_(224, 224, 3)_clean.npy'
    data_no_mask_path = '../dataset/AFDB/train_data_clean_model/2203_faces_without_mask_(224, 224, 3)_clean.npy'
    data_mask = np.load(data_mask_path) 
    data_nomask = np.load(data_no_mask_path)

    model=Sequential()

    model.add(Conv2D(100,(3,3),input_shape=cfg.IMG_SHAPE, padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #The first CNN layer followed by Relu and MaxPooling layers

    model.add(Conv2D(200, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #The second convolution layer followed by Relu and MaxPooling layers

    model.add(Dropout(0.5))
    #Flatten layer to stack the output convolutions from second convolution layer

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1))
    #The Final layer with the output

    return model, data_mask, data_nomask

def mobilev2top_v1():
    data_mask_path = '../dataset/AFDB/train_data_mobilev2_top/2203_faces_with_mask_(224, 224, 3)_mobilev2.npy'
    data_no_mask_path = '../dataset/AFDB/train_data_mobilev2_top/2203_faces_without_mask_(224, 224, 3)_mobilev2.npy'
    data_mask = np.load(data_mask_path) 
    data_nomask = np.load(data_no_mask_path)

    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(1)(headModel) #Dense 1 without activation

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    return model, data_mask, data_nomask