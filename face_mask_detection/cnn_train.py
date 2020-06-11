# pylint: disable=undefined-variable, no-member

import tensorflow as tf
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

import config as cfg
import models

# CREATE THE MODEL AND LOAD ITS DATA
model, data_mask, data_nomask = models.get_model_and_data(cfg.target)
data = np.concatenate((data_mask ,data_nomask), axis = 0)

# CREATE LABELS
label = np.concatenate((np.ones(data_mask.shape[0]), np.zeros(data_nomask.shape[0])), axis = 0)

# SPLIT DATA INTO TRAINING AND VALIDATION AND CREATE TF DATASET
tf.random.set_seed(cfg.TF_SEED)
np.random.seed(cfg.TF_SEED)

(trainX, valX, trainY, valY) = train_test_split(data, label, test_size=0.20, random_state=cfg.TRAIN_TEST_SPLIT_SEED, stratify=True)
train = tf.data.Dataset.from_tensor_slices((trainX, trainY))
val = tf.data.Dataset.from_tensor_slices((valX, valY))
train_dataset = train.repeat().shuffle(cfg.SHUFFLE_BUFFER_SIZE).batch(cfg.BATCH_SIZE)
val_dataset = val.batch(cfg.BATCH_SIZE)

trainX, valX, trainY, valY = [None] * 4
train = None
val = None

# SET OPTIMIZER
optimizer = Adam(lr=cfg.LR)
model.compile(loss=BinaryCrossentropy(from_logits=True), 
              optimizer=optimizer, metrics=['accuracy'])

if cfg.DEBUG_NET:
    model.summary()

# CALLBACKS FOR MONITORING AND SAVING 
checkpointer = ModelCheckpoint(filepath=f'{cfg.MODEL_PATH}/{cfg.target}.model.weights.hdf5', verbose = 1, save_best_only=True)
log_dir = f'{cfg.LOG_DIR}/{cfg.target}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# ACTUAL TRAIN FUNCTION
history = model.fit(train_dataset, epochs=cfg.EPOCHS, 
                    callbacks = [checkpointer, tensorboard_callback], 
                    validation_data=val_dataset,
                    steps_per_epoch=110)

# FINAL PLOTS
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title("Loss during the training process")
plt.xlabel('# epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{log_dir}/loss.png')
if cfg.SHOW_PLOT:
    plt.show()

plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title("Accuracy during the training process")
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'{log_dir}/accuracy.png')
if cfg.SHOW_PLOT:
    plt.show()