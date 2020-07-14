# pylint: disable=no-member

import numpy as np

from face_mask_detection import config as cfg
import pickle

data_mask_path = '../dataset/AFDB/train_data_clean_model/2203_faces_with_mask_(224, 224, 3)_clean.npy'
data_no_mask_path = '../dataset/AFDB/train_data_clean_model/2203_faces_without_mask_(224, 224, 3)_clean.npy'
data_mask = np.load(data_mask_path) 
data_nomask = np.load(data_no_mask_path)

# UNCOMMENT FOR ALL PIXELS
# data = np.concatenate((data_mask ,data_nomask), axis = 0)

# UNCOMMENT FOR HOG FEATURES
# data = np.load('../dataset/AFDB/training_features/training_set_hog_features.npy')

# UNCOMMENT FOR CONV. FEATURES 
# data = np.load('../dataset/AFDB/training_features/training_data_features.npy')

# RESHAPE OF DATA
# data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))

label = np.concatenate((np.ones(data_mask.shape[0]), np.zeros(data_nomask.shape[0])), axis = 0)


from sklearn.svm import SVC

kernel = 'linear'
max_iteration = 500

model = SVC(kernel=kernel, max_iter=max_iteration, verbose=True) 

model.fit(data, label)

with open(f'{cfg.MODEL_PATH}/svm_model.pkl', 'wb') as fid:
    pickle.dump(model, fid)