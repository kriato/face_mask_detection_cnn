from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.activations import sigmoid

from mtcnn.mtcnn import MTCNN
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

if own:
  model_path = "own.model.weights.best.hdf5"
  
else:
  model_path = "mobilev2.model.weights.best.hdf5"

model = load_model(model_path)
print(model)

def mask_detect(img):
  detector = MTCNN() 
  image = plt.imread(img) # COME NON KEKKARE

  # pass the blob through the network and obtain the face detections
  faces = detector.detect_faces(image)

  # loop over the detections
  for face in faces:
    print(face)
    # extract the confidence (i.e., probability) associated with
    # the detection
    confidence = face["confidence"]

    # filter out weak detections by ensuring the confidence is
    # greater than the minimum confidence
    if confidence > 0.75:
      # compute the (x, y)-coordinates of the bounding box for
      # the object
      box = face["box"]
      (startX, startY, width, height) = box

      # extract the face ROI, convert it from BGR to RGB channel
      # ordering, resize it to 224x224, and preprocess it
      face = image[startY:startY+height, startX:startX+width]
      face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
      face = cv2.resize(face, (224, 224))
      face = img_to_array(face)

      if not own:
        face = preprocess_input(face)
      face = np.expand_dims(face, axis=0)

      # pass the face through the model to determine if the face
      # has a mask or not
      logit = model.predict(face)
      mask_prob = sigmoid(logit)
      
      # determine the class label and color we'll use to draw
      # the bounding box and text
      label = "Mask" if mask_prob > 0.5 else "No Mask"
      color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

      # display the label and bounding box rectangle on the output frame
      cv2.putText(image, label + f' {mask_prob}', (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color)
      cv2.rectangle(image, (startX, startY), (startX+width, startY+height), color, 2)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  cv2.imshow('', image)
  cv2.waitKey(0)

import os
li = os.listdir('examples')
for img in li:
    mask_detect(f'examples/{img}')