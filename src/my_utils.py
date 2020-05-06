import os
import csv

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import cv2

from keras import optimizers
from keras.models import Model
from keras.utils.training_utils import multi_gpu_model

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):

        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
    
def save_imgs(save_dir, img_arr):
    for i in range(len(img_arr)):
        path = save_dir + '/' + str(i) + '.png'
        cv2.imwrite(path, img_arr[i,...])
