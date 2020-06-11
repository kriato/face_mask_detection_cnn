# pylint: disable=import-error, no-member

import config as cfg
import models

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.activations import sigmoid
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix

from tqdm import tqdm

# LOAD TESTING DATA
data_mask_path = f'{cfg.TEST_PATH}/testing_set_data_mask.npy'
data_no_mask_path = f'{cfg.TEST_PATH}/testing_set_data_nomask.npy'
data_mask = np.load(data_mask_path) 
data_nomask = np.load(data_no_mask_path)
test_data = np.concatenate((data_mask, data_nomask), axis = 0)
test_labels = np.concatenate((np.ones(data_mask.shape[0]), np.zeros(data_nomask.shape[0])), axis = 0)

print(test_data.shape)
print(test_labels.shape)

# LOAD MODEL
model = models.get_model(cfg.target)

# PREDICTION
predictions = []
for face in tqdm((test_data)):
    face = np.expand_dims(face, axis=0)

    logit = model.predict(face)
    mask_prob = sigmoid(logit)
    if mask_prob > 0.5:
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

disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                              display_labels=['mask', 'no mask'])

disp_norm = disp_norm.plot(cmap=plt.cm.Blues)
plt.show()