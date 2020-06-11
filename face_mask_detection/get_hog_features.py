

import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import config as cfg
import pickle
import cv2

# data_mask_path = '../dataset/AFDB/train_data_clean_model/2203_faces_with_mask_(224, 224, 3)_clean.npy'
# data_no_mask_path = '../dataset/AFDB/train_data_clean_model/2203_faces_without_mask_(224, 224, 3)_clean.npy'
# data_mask = np.load(data_mask_path)
# data_nomask = np.load(data_no_mask_path)

# data_mask_path = f'{cfg.TEST_PATH}/testing_set_data_mask.npy'
# data_no_mask_path = f'{cfg.TEST_PATH}/testing_set_data_nomask.npy'
# data_mask = np.load(data_mask_path) 
# data_nomask = np.load(data_no_mask_path)
data = np.concatenate((data_mask ,data_nomask), axis = 0)

from skimage import exposure
from skimage import feature
hog_features = []
for face in data:
    face = np.float32(face)
    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    H = feature.hog(face, orientations=18, pixels_per_cell=(16, 16),
                cells_per_block=(8, 8), transform_sqrt=True, block_norm="L1")
    hog_features.append(H)

np.save('testing_set_hog_features.npy', np.asarray(hog_features))