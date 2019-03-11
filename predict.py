import numpy as np
import os
import cv2
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

def read_image(fname, size):
    img = cv2.imread(fname)
    img = cv2.resize(img, size)
    return img
# list_files = []
# list_file_pattern = ["/data/tuanpv2/VT50201707_reviewed/*/*.jpg"]
# for file_pattern in list_file_pattern:
#     sub_list_files = glob.glob(file_pattern)
#     list_files = [*list_files, *sub_list_files]
def get_file():
    data = pd.read_csv('shape.csv')
    fname, x, y = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
    list_files = []
    for i in range(len(fname)):
        rate = y[i] / x[i]
        if 1.0 < rate and rate < 1.4:
            list_files.append(fname[i])
    return list_files

list_files = get_file()
random.seed(10)
random.shuffle(list_files)

N_TEST_SAMPLES = int(0.1 * len(list_files))
N_TRAIN_SAMPLES= len(list_files) - N_TEST_SAMPLES
list_files_train, list_files_val = list_files[:N_TRAIN_SAMPLES], list_files[N_TRAIN_SAMPLES:]

print("Total number of image:", len(list_files))
print("Total number of training:", N_TRAIN_SAMPLES)
print("Total number of testing:", N_TEST_SAMPLES)

BATCH_SIZE = 16
params = {
    'dim': (100, 120, 3),
    'std': np.array([0.229, 0.224, 0.225]),
    'mean': np.array([0.485, 0.456, 0.406]),
    'batch_size': BATCH_SIZE
}

# Generators
testing_generator = ImageDataGenerator(list_files_val[:64], **params, shuffle=False)

# Restore model
with tf.device("/cpu:0"):
    model = net(params['dim'])
model = multi_gpu_model(model, gpus=2)
model.load_weights("models/CNN/weights-127-1.19.hdf5")
print("Done loaded model!")


pair_prediction = model.predict_generator(testing_generator, max_queue_size=10, workers=10, verbose=1, use_multiprocessing=True)
cover_pred = testing_generator.denormalize_batch(pair_prediction[0])
secret_pred = testing_generator.denormalize_batch(pair_prediction[1])

for i in range(len(cover_pred)):
    idx_secret = int(i / BATCH_SIZE) * 2 * BATCH_SIZE + i % BATCH_SIZE
    idx_cover = idx_secret + BATCH_SIZE
    secret = read_image(list_files_val[idx_secret], (params['dim'][1], params['dim'][0]))
    cover = read_image(list_files_val[idx_cover], (params['dim'][1], params['dim'][0]))
    # concat all images and save
    concat_all = np.concatenate([secret, cover, cover_pred[i], secret_pred[i]], axis=1)
    cv2.imwrite('result/' + str(i) + '.jpg', concat_all)
    
