import os
import csv

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import cv2

from keras.preprocessing.image import ImageDataGenerator

def load_data_5fold(dirname, filename, val_fold, tst_fold):
    trn, vld, tst = None, None, None
    for i in range(1,6):
        path = dirname + str(i) + '/' + filename
        if i == val_fold:
            vld = np.load(path)
        elif i == tst_fold:
            tst = np.load(path)
        else:
            if trn is None:
                trn = np.load(path)
            else:
                trn = np.concatenate((trn, np.load(path)))
    return (trn, vld, tst)

def resize_data(data, n, row, col, ch):
    ret = np.zeros((n, row, col, ch))
    for i in range(len(data)):
        img = data[i,:,:,0]
        img = cv2.resize(img,(row, col),interpolation=cv2.INTER_NEAREST)
        for j in range(ch):
            ret[i,...,j] = img
    return ret

def shuffle_resize_encode_data(X, Y1, seed, row, col):
    Y = []
    for i in range(3):
        X[i], Y1[i] = shuffle(X[i], Y1[i], random_state=seed)
        
        X[i] = resize_data(X[i], len(X[i]), row, col, 3)
        Y1[i] = resize_data(Y1[i], len(Y1[i]), row, col, 1)
        
        encoded = np.ones((len(Y1[i]), row, col, 2), dtype=np.uint8)
        encoded[...,1] = (Y1[i][...,0] != 0) * 1
        encoded[...,0] = encoded[...,0] - encoded[...,1]
        Y.append(encoded)
    
    return X, Y

def augment_data(X, Y, batch_size, seed, datagen_args=None):
    if datagen_args is None:
        datagen = ImageDataGenerator()
    else:
        datagen = ImageDataGenerator(**datagen_args)

    X_tmp = datagen.flow(X, batch_size=batch_size, shuffle=False, seed=seed)
    Y_tmp = datagen.flow(Y, batch_size=batch_size, shuffle=False, seed=seed)

    data_generator = zip(X_tmp, Y_tmp)
    
    return data_generator


