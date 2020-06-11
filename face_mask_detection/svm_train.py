# optimization finished, #iter = 2000
# obj = -0.000009, rho = 3.813227
# nSV = 712, nBSV = 0
# Total nSV = 712

# pylint: disable=no-member

import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import config as cfg
import pickle

data_mask_path = '../dataset/AFDB/train_data_clean_model/2203_faces_with_mask_(224, 224, 3)_clean.npy'
data_no_mask_path = '../dataset/AFDB/train_data_clean_model/2203_faces_without_mask_(224, 224, 3)_clean.npy'
data_mask = np.load(data_mask_path) 
data_nomask = np.load(data_no_mask_path)

# data = np.concatenate((data_mask ,data_nomask), axis = 0)

# data = np.load('features.npy')
# # CREATE LABELS
# data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))
data = np.load('training_set_hog_features.npy')
label = np.concatenate((np.ones(data_mask.shape[0]), np.zeros(data_nomask.shape[0])), axis = 0)
data_mask = None
data_nomask = None
# SPLIT DATA INTO TRAINING AND VALIDATION AND CREATE TF DATASET

# (x_train, x_test, y_train, y_test) = train_test_split(data, label, test_size=0.20, random_state=cfg.TRAIN_TEST_SPLIT_SEED)

# print(x_train.shape)
# # print(y_train.shape)
# r_train = np.reshape(x_train[:,:,:,0], (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
# g_train = np.reshape(x_train[:,:,:,1], (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
# b_train = np.reshape(x_train[:,:,:,2], (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3]))
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))

# plt.scatter(x_train[:,:], x_train[:,:])
# plt.figure()
# plt.scatter(r_train[:,0], r_train[:,1], c=y_train)
# plt.figure()
# plt.scatter(g_train[:,0], g_train[:,1], c=y_train)
# plt.figure()
# plt.scatter(b_train[:,0], b_train[:,1], c=y_train)
# plt.figure()
# plt.show()
# exit()

from sklearn.svm import SVC

kernel = 'linear'
max_iteration = 500

model = SVC(kernel=kernel, max_iter=max_iteration, verbose=True) 

model.fit(data, label)

with open(f'{cfg.MODEL_PATH}/svm_model.pkl', 'wb') as fid:
    pickle.dump(model, fid)