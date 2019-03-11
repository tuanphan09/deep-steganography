import keras
import numpy as np
import os
import pickle

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, concatenate, Input
from keras.layers.noise import GaussianNoise
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop



def prepare_network(input_shape, input_tensor):
    # first block
    first_3x3_conv = Sequential([
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu', input_shape = input_shape),
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu')
    ])(input_tensor)

    first_4x4_conv = Sequential([
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu', input_shape = input_shape),
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu')
    ])(input_tensor)

    first_5x5_conv = Sequential([
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu', input_shape = input_shape),
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu')
    ])(input_tensor)
    
    first_tensor = concatenate([first_3x3_conv, first_4x4_conv, first_5x5_conv], axis=3)

    # second blockGaussianNoise
    second_3x3_conv = Conv2D(filters = 50, kernel_size = (3, 3), 
                            padding = 'Same', activation = 'relu')(first_tensor)

    second_4x4_conv = Conv2D(filters = 50, kernel_size = (4, 4), 
                            padding = 'Same', activation = 'relu')(first_tensor)

    second_5x5_conv = Conv2D(filters = 50, kernel_size = (5, 5), 
                            padding = 'Same', activation = 'relu')(first_tensor)                                                

    second_tensor = concatenate([second_3x3_conv, second_4x4_conv, second_5x5_conv], axis=3, name='prepare_net_out')

    return second_tensor


def hidding_network(input_shape, input_tensor):
    # first block
    first_3x3_conv = Sequential([
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu', input_shape = input_shape),
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu')
    ])(input_tensor)

    first_4x4_conv = Sequential([
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu', input_shape = input_shape),
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu')
    ])(input_tensor)

    first_5x5_conv = Sequential([
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu', input_shape = input_shape),
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu')
    ])(input_tensor)
    
    first_tensor = concatenate([first_3x3_conv, first_4x4_conv, first_5x5_conv], axis=3)

    # second block
    second_3x3_conv = Conv2D(filters = 50, kernel_size = (3, 3), 
                            padding = 'Same', activation = 'relu')(first_tensor)

    second_4x4_conv = Conv2D(filters = 50, kernel_size = (4, 4), 
                            padding = 'Same', activation = 'relu')(first_tensor)

    second_5x5_conv = Conv2D(filters = 50, kernel_size = (5, 5), 
                            padding = 'Same', activation = 'relu')(first_tensor)                                                

    second_tensor = concatenate([second_3x3_conv, second_4x4_conv, second_5x5_conv], axis=3)

    # convert to 3 channels image
    out = Conv2D(filters = 3, kernel_size = (1, 1), padding = 'Same', 
                activation = 'relu', name='hidding_net_out')(second_tensor) 
                
    # add noise
    out_noise = GaussianNoise(stddev=0.1, name='hidding_net_out_noise')(out)

    return out, out_noise

def reveal_network(input_shape, input_tensor):
    # first block
    first_3x3_conv = Sequential([
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu', input_shape = input_shape),
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (3, 3),padding = 'Same', activation = 'relu')
    ])(input_tensor)

    first_4x4_conv = Sequential([
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu', input_shape = input_shape),
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (4, 4),padding = 'Same', activation = 'relu')
    ])(input_tensor)

    first_5x5_conv = Sequential([
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu', input_shape = input_shape),
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu'),
        Conv2D(filters = 50, kernel_size = (5, 5),padding = 'Same', activation = 'relu')
    ])(input_tensor)
    
    first_tensor = concatenate([first_3x3_conv, first_4x4_conv, first_5x5_conv], axis=3)

    # second block
    second_3x3_conv = Conv2D(filters = 50, kernel_size = (3, 3), 
                            padding = 'Same', activation = 'relu')(first_tensor)

    second_4x4_conv = Conv2D(filters = 50, kernel_size = (4, 4), 
                            padding = 'Same', activation = 'relu')(first_tensor)

    second_5x5_conv = Conv2D(filters = 50, kernel_size = (5, 5), 
                            padding = 'Same', activation = 'relu')(first_tensor)                                                

    second_tensor = concatenate([second_3x3_conv, second_4x4_conv, second_5x5_conv], axis=3)

    # convert to 3 channels image
    out = Conv2D(filters = 3, kernel_size = (1, 1), padding = 'Same', 
                activation = 'relu', name = 'reveal_net_out')(second_tensor) 
    
    return out

def net(img_shape):
    # prepare network
    secret_img = Input(shape=img_shape)
    prepare_net_input_shape = img_shape
    prepare_net_out = prepare_network(prepare_net_input_shape, secret_img)

    # hidding network
    cover_img = Input(shape=img_shape)
    hidding_net_input = concatenate([cover_img, prepare_net_out], axis=3)
    hidding_net_input_shape = (*img_shape[:2], 153)
    hidding_net_out, hidding_net_out_noise = hidding_network(hidding_net_input_shape, hidding_net_input)

    reveal_net_input_shape = img_shape
    reveal_net_out = reveal_network(reveal_net_input_shape, hidding_net_out)

    model = Model(inputs=[secret_img, cover_img], outputs=[hidding_net_out, reveal_net_out])
    return model