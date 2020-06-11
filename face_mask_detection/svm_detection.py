# pylint: disable=no-member

from mtcnn.mtcnn import MTCNN
from cv2 import cv2
import numpy as np

import config as cfg
import pickle

with open(f'{cfg.MODEL_PATH}/svm_model.pkl', 'rb') as fid:
    model = pickle.load(fid)

def mask_detect(img):
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
        print(face)
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = face["confidence"]

        # compute the (x, y)-coordinates of the bounding box for
        # the object
        box = face["box"]
        (startX, startY, width, height) = box

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.85 and width >= cfg.WIDTH_THRESHOLD and height >= cfg.HEIGHT_THRESHOLD:
            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[max(0,startY):startY+height, max(0,startX):startX+width]
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face = cv2.resize(face, (224, 224))

            face = np.reshape(face, (1, face.shape[0] * face.shape[1] * face.shape[2]))

            predicted = model.predict(face)
            
            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if predicted == 1 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(image, f'{label}-{predicted}', (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color)
            cv2.rectangle(image, (startX, startY), (startX+width, startY+height), color, 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imshow('', image)
    cv2.waitKey(0)

import os
li = os.listdir(cfg.TEST_IMAGES_PATH)
for img in li:
    mask_detect(f'{cfg.TEST_IMAGES_PATH}/{img}')