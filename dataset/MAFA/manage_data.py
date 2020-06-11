import numpy as np
import csv
from cv2 import cv2
from PIL import Image

from tensorflow.keras.preprocessing.image import img_to_array, load_img

N = 1100
w = 224
h = 224
c = 3

roi_coords = np.loadtxt('test_label.csv', delimiter=';', usecols=(1,2,3,4))

print(roi_coords.shape)

filenames = []
labels = []
difficulties = []
with open('test_label.csv') as csvfile:
    f = csv.reader(csvfile, delimiter=';')
    for row in f:
        filenames.append(row[0])
        labels.append(row[5])
        difficulties.append(row[10])

# SKIP IF LABEL == 3
# SKIP IF FACE < 224,224
# SKIP IF DIFF IS 2-3?

noface_counter = 0
nomask_counter = 0
mask_counter = 0

easy_counter = 0
hard_counter = 0
body_counter = 0

too_small_counter = 0
ok_counter = 0

ins = 0
data = np.zeros((N, w, h, c))
final_labels = np.zeros(N)
for index, (name, roi, label, diff) in enumerate(zip(filenames, roi_coords, labels, difficulties)):

    # CHECKING LABELS
    if label == '3':
        noface_counter += 1
        continue
    
    if label == '1':
        mask_counter += 1
    elif label == '2':
        nomask_counter += 1
    else:
        print(f'ERROR IN LABELS {index}')
        continue

    # CHECKING DIFFICULTIES
    if diff == '3':
        body_counter += 1
        continue
    
    if diff == '1':
        easy_counter += 1
    elif diff == '2':
        hard_counter += 1
    else:
        # print(f'ERROR IN DIFF {index}')
        continue
    
    # CHECK FACE SIZE
    if roi[2] < w or roi[3] < h:
        too_small_counter += 1
        continue
    
    ok_counter += 1

    image = cv2.imread(f'test_data/{name}')
    face = image[int(roi[1]):int(roi[1])+int(roi[3]), int(roi[0]):int(roi[0])+int(roi[2]), :]
    face = cv2.resize(face, (h, w))

    # cv2.imshow('', face)
    # cv2.waitKey(0)
    cv2.imwrite(f'test_data/subset/{name}', face)
    
    data[ins] = img_to_array(face)

    ins += 1
    if N == ins:
        break

for x in data:
    if not x.any():
        print('error')
        
print(np.count_nonzero(final_labels))
print(mask_counter, nomask_counter, noface_counter)
print(easy_counter, hard_counter, body_counter)
print(ok_counter, too_small_counter)

