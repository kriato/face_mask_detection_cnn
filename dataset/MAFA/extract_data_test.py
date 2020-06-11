import sys
sys.path.append('../../')
from face_mask_detection import config as cfg

import numpy as np
import os

from tensorflow.keras.preprocessing.image import img_to_array, load_img

matches = []
files = []
for root, dirnames, filenames in os.walk(os.getcwd() + '/test_data/subset'):
    for filename in filenames:
        if filename.endswith(cfg.IMG_FORMATS):
            matches.append(os.path.join(root, filename))
            files.append(filename)
        else:
            print(filename)

N = len(matches)
print(N)
data = np.zeros((N,) + cfg.IMG_SHAPE)
for index, match in enumerate(matches):
    img = load_img(match, target_size=cfg.IMG_SHAPE)
    img = img_to_array(img)
    data[index] = img

np.save('testing_set_data_mask.npy', data)