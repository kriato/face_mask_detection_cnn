# pylint: disable=undefined-variable, import-error

import sys
sys.path.append('../../')
from face_mask_detection import config as cfg

import os
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

if cfg.preprocess:
    SUBFOLDER = 'train_data_mobilev2_top'
    LABEL = 'mobilev2'
else:
    SUBFOLDER = 'train_data_clean_model'
    LABEL = 'clean'

if not os.path.exists(SUBFOLDER):
    os.makedirs(SUBFOLDER)

# WITH MASK
matches = []
files = []
for root, dirnames, filenames in os.walk(os.getcwd() + '/AFDB_masked_face_dataset'):
    for filename in filenames:
        if filename.endswith(cfg.IMG_FORMATS):
            matches.append(os.path.join(root, filename))
            files.append(filename)
        else:
            print(filename)

N = len(matches)
FILENAME = str(N) + f'_faces_with_mask_{cfg.IMG_SHAPE}_{LABEL}'

data = np.zeros((N,) + cfg.IMG_SHAPE)

with open(f'{SUBFOLDER}/{FILENAME}.txt', 'w') as f:
    for i in range(N):
        img = load_img(matches[i], target_size=cfg.IMG_SHAPE)
        img = img_to_array(img)
        if not cfg.preprocess:
            img = preprocess_input(img)
        data[i] = img
        name = matches[i].replace(os.getcwd() + "/AFDB_masked_face_dataset","")
        print(f'{i};{name}', file=f)

np.save(f'{SUBFOLDER}/{FILENAME}.npy', data)

# WITHOUT MASK
matches = []
files = []
for root, dirnames, filenames in os.walk(os.getcwd() + '/AFDB_face_dataset'):
    for filename in filenames:
        if filename.endswith(cfg.IMG_FORMATS):
            matches.append(os.path.join(root, filename))
            files.append(filename)
        else:
            print(filename)


FILENAME = str(N) + f'_faces_without_mask_{cfg.IMG_SHAPE}_{LABEL}'
data = np.zeros((N,) + cfg.IMG_SHAPE)

with open(f'{SUBFOLDER}/{FILENAME}.txt', 'w') as f:
    for i in range(N):
        img = load_img(matches[i], target_size=cfg.IMG_SHAPE)
        img = img_to_array(img)
        if not cfg.preprocess:
            img = preprocess_input(img)
        data[i] = img
        name = matches[i].replace(os.getcwd() + "/AFDB_face_dataset","")
        print(f'{i};{name}', file=f)

np.save(f'{SUBFOLDER}/{FILENAME}.npy', data)
