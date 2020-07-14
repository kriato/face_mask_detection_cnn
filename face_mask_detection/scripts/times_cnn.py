# pylint: disable=import-error, no-member

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn.mtcnn import MTCNN

import numpy as np
import argparse
import imutils
import time
from cv2 import cv2
import os

import sys
sys.path.append('../')
from face_mask_detection import config as cfg
from face_mask_detection import models

def mask_detect(img):
    global counter

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    w, h, _ = img.shape
    if w < cfg.SIZE_Y and h < cfg.SIZE_Y:
        return

    locs = []
    logit = []

    # pass the image through the MTCNN and obtain the face detections
    faces = detector.detect_faces(img)

    face_counter = 0
    # loop over the detection

    for face in faces:
        # extract the confidence associated with the detection
        confidence = face["confidence"]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.9:
            face_counter += 1
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = face["box"]
            (startX, startY, width, height) = box

            # extract the face, resize it to 224x224
            face = img[max(0, startY):startY+height, max(0, startX):startX+width]
            face = cv2.resize(face, (cfg.SIZE_X, cfg.SIZE_Y))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            # start = time.time()
            logit.append(model.predict(face))
            # end = time.time()
            # times.append(end-start)
            # print(end-start)
            counter += 1

            locs.append((startX, startY, startX+width, startY+height))

    return (locs, logit)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2_imshow(image)


times = []
counter = 0
detector = MTCNN()
cam = cv2.VideoCapture(0)

model = models.get_model(cfg.target)
time.sleep(2.0)

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = cam.read()

    if not ret or frame is None:
        continue
        
    (locs, preds) = mask_detect(frame)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if pred > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
