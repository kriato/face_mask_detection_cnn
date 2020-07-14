# pylint: disable=undefined-variable, no-member

from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

import sys
sys.path.append('../')
from face_mask_detection import config as cfg
from face_mask_detection import models

# LOAD TRAINING DATA
# _, data_mask, data_nomask = models.get_model_and_data(cfg.target)
# data = np.concatenate((data_mask ,data_nomask), axis = 0)

model = ResNet50V2(weights='imagenet', include_top=False)

x = preprocess_input(data)

features = model.predict(x)
print(features.shape)