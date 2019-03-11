import numpy as np
import os
import pickle
import glob
import random
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import keras
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adadelta, RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import multi_gpu_model

from model import *
from dataset import ImageDataGenerator 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

list_files = []
list_file_pattern = ["/data/training/part1/*.jpg", "/data/training/part2/*.jpg",]
for file_pattern in list_file_pattern:
    sub_list_files = glob.glob(file_pattern)
    list_files = [*list_files, *sub_list_files]

# def get_file():
#     data = pd.read_csv('shape.csv')
#     fname, x, y = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
#     list_files = []
#     for i in range(len(fname)):
#         rate = y[i] / x[i]
#         if 1.0 < rate and rate < 1.4:
#             list_files.append(fname[i])
#     return list_files
# list_files = get_file()
random.seed(10)
random.shuffle(list_files)

N_TEST_SAMPLES = int(0.1 * len(list_files))
N_TRAIN_SAMPLES= len(list_files) - N_TEST_SAMPLES
list_files_train, list_files_val = list_files[:N_TRAIN_SAMPLES], list_files[N_TRAIN_SAMPLES:]

print("Total number of image:", len(list_files))
print("Total number of training:", N_TRAIN_SAMPLES)
print("Total number of testing:", N_TEST_SAMPLES)

N_EPOCHS = 100
BATCH_SIZE = 16 * 2
BETA = 0.8
params = {
    'dim': (100, 120, 3),
    'std': np.array([0.229, 0.224, 0.225]),
    'mean': np.array([0.485, 0.456, 0.406]),
    'batch_size': BATCH_SIZE
}

# Generators
training_generator = ImageDataGenerator(list_files_train, **params, shuffle=True)
validation_generator = ImageDataGenerator(list_files_val, **params, shuffle=True)

# config to run with multi-GPU
with tf.device("/cpu:0"):
    model = net(params['dim'])
model = multi_gpu_model(model, gpus=2)

# model.load_weights("models/CNN/weights-127-1.19.hdf5")
model.summary()
model.compile(
    loss={'hidding_net_out': 'mse', 'reveal_net_out': 'mse'},
    optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
    loss_weights={'hidding_net_out': 1, 'reveal_net_out': BETA},
    metrics=None
)


model_file = "./models/CNN/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1)
tbCallBack = TensorBoard(log_dir='./tensorboard/CNN', write_graph=True, write_images=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, 
                                            min_lr=0.000001)
callbacks_list = [checkpoint, tbCallBack, learning_rate_reduction]

model.fit_generator(
        training_generator,
        steps_per_epoch=min(N_TRAIN_SAMPLES // BATCH_SIZE, 500),
        initial_epoch=0,
        epochs=N_EPOCHS,
        validation_data=validation_generator,
        validation_steps=min(N_TEST_SAMPLES // BATCH_SIZE, 50),
        callbacks=callbacks_list,
        verbose=1,
        max_queue_size=20,
        workers=10,
        use_multiprocessing=True,
    )

