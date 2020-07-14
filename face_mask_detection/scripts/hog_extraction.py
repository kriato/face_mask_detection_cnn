# pylint: disable=no-member, import-error

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

import sys
sys.path.append('../')
from face_mask_detection import config as cfg
from face_mask_detection import models

from skimage import exposure
from skimage import feature

def hog_extractor(img):
    detector = MTCNN()
    image = cv2.imread(img)
  
    w, h, _ = image.shape
    if w < 224 and h < 224:
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # pass the blob through the network and obtain the face detections
    faces = detector.detect_faces(image)
    if faces is None:
        return
        
    # loop over the detections
    for face in faces:
        confidence = face["confidence"]

        # compute the (x, y)-coordinates of the bounding box for
        # the object
        box = face["box"]
        (startX, startY, width, height) = box

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.85 and width >= cfg.WIDTH_THRESHOLD and height >= cfg.HEIGHT_THRESHOLD:
            face = image[max(0,startY):startY+height, max(0,startX):startX+width]
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            face = cv2.resize(face, (224, 224))
            rgb = face
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

            # H = feature.hog(face, orientations=9, pixels_per_cell=(8, 8),
            #     cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

            (H, hogImage) = feature.hog(face, orientations=18, pixels_per_cell=(16, 16),
                cells_per_block=(8, 8), transform_sqrt=True, block_norm="L1",
                visualize=True)

            hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
            hogImage = hogImage.astype("uint8")
            
            cv2.imshow("HOG Image", np.hstack((cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), cv2.cvtColor(hogImage, cv2.COLOR_GRAY2RGB))))
            # hog_features = hog.compute(face)
            
            # print(hog_features.shape)
            # cv2.imshow('', hog_features)
            cv2.waitKey(0)
           
import os
li = os.listdir(cfg.TEST_IMAGES_PATH)
for img in li:
    hog_extractor(f'{cfg.TEST_IMAGES_PATH}/{img}')
