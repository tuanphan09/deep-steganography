
import numpy as np
import cv2
import glob
from multiprocessing import Pool
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os, sys
from PIL import Image
list_files = glob.glob("./data/image/*.jpg")

def get_shape(fname):
    try:
        img = cv2.imread(fname)
        h, w, c = img.shape
        return fname, h, w, c 
    except:
        print(fname)

def get_shape_parallel(save_file):
    p = Pool(48)
    data = p.map(get_shape, list_files)
    print(len(data))

    with open(save_file,'w') as out:
        for row in data:
            out.write("{},{},{},{}\n".format(row[0], row[1], row[2], row[3]))

def stattistic(save_file, h_mean, w_mean):
    data = pd.read_csv(save_file)
    fname, size = data.iloc[:, 0], data.iloc[:, 1:3]
   
    print(np.mean(size, axis=0))
    # for i in idx:
    #     h = x[i]
    #     w = y[i]
    #     w_scale = h_mean * w / h
    #     if w_scale > w_mean - 30 and w_scale < w_mean + 30:
    #         cnt += 1
    # print(cnt)


h_mean = 190
w_mean = 233
save_file = 'shape.csv'
# get_shape_parallel(save_file)
# stattistic(save_file, h_mean, w_mean)

img = cv2.imread('origin.jpg')
new_img = cv2.resize(img, (100, 200))
cv2.imwrite('new.jpg', new_img)

