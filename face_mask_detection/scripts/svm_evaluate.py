# pylint: disable=import-error, no-member

import sys
sys.path.append('../')
from face_mask_detection import config as cfg
from face_mask_detection import models

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from tqdm import tqdm

# LOAD TESTING DATA
data_mask_path = f'{cfg.TEST_PATH}/testing_set_data_mask.npy'
data_no_mask_path = f'{cfg.TEST_PATH}/testing_set_data_nomask.npy'
data_mask = np.load(data_mask_path) 
data_nomask = np.load(data_no_mask_path)
# test_data = np.concatenate((data_mask, data_nomask), axis = 0)
test_labels = np.concatenate((np.ones(data_mask.shape[0]), np.zeros(data_nomask.shape[0])), axis = 0)
data_mask = None
data_nomask = None
test_data = np.load('testing_set_features.npy')
# test_data = np.load('testing_set_hog_features.npy')

print(test_data.shape)
print(test_labels.shape)

with open(f'{cfg.MODEL_PATH}/svm_model_cnn_features.pkl', 'rb') as fid:
    model = pickle.load(fid)

# PREDICTION
predictions = []
for face in tqdm((test_data)):
    face = np.reshape(face, (1, face.shape[0] * face.shape[1] * face.shape[2]))
    # face = np.reshape(face, (1, face.shape[0]))
    predicted = model.predict(face)
    if predicted > 0.5:
        predictions.append(1)
    else:
        predictions.append(0)

# CONFUSION MATRICES
cm_norm = confusion_matrix(test_labels, predictions, normalize='all')
cm = confusion_matrix(test_labels, predictions)

# EVALUATE MODEL
accuracy = np.sum(cm.diagonal())/np.sum(cm)
precision_mask = cm[0,0]/ np.sum(cm[:,0])
precision_nomask = cm[1,1]/ np.sum(cm[:,1])
recall_mask = cm[0,0]/ np.sum(cm[0,:])
recall_nomask = cm[1,1]/ np.sum(cm[1,:])

# OUTPUT
print('Accuracy: ' + "{0:.2f}".format(accuracy*100) + '%')
print('Mask precision: ' + "{0:.2f}".format(precision_mask))
print('No_mask precision: ' + "{0:.2f}".format(precision_nomask))
print('Mask recall: ' + "{0:.2f}".format(recall_mask))
print('No_mask recall: ' + "{0:.2f}".format(recall_nomask))

disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['mask', 'no mask'])

disp_norm = disp_norm.plot(cmap=plt.cm.Blues)
plt.savefig('svm_cnn_cm.png', dpi=300)
plt.show()