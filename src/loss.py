import numpy as np
import cv2
import tensorflow.keras.backend as K
import keras
from scipy import ndimage

def dice_coef(y_true, y_pred, smooth=1e-08):
    y_pred = K.clip(y_pred, 0, 1)
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
##################################################################
def generalized_dice_coef(y_true, y_pred):
    y_pred = K.clip(y_pred, 0, 1)
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    
    n_cl = K.shape(y_true)[-1]
    w = K.zeros(shape=(n_cl,))
    w = K.sum(y_true_f, axis=-1)
    w = 1 / (w ** 2 + 0.000001)

    numerator = y_true_f * y_pred_f
    numerator = w * K.sum(numerator,axis=-1)
    numerator = K.sum(numerator)
    
    denominator = y_true_f + y_pred_f
    denominator = w * K.sum(denominator,axis=-1)
    denominator = K.sum(denominator)
    
    return 2. * numerator / denominator
def generalized_dice_coef_loss(y_true, y_pred):
    return 1 - generalized_dice_coef(y_true, y_pred)
##################################################################
'''
Keras implementation of clDice(arXiv:2003.07311 [cs.CV])
'''
def soft_erode(img):
    p1 = -K.pool2d(-img, pool_size=(3,1), strides=(1, 1), padding='same', data_format=None, pool_mode='max')
    p2 = -K.pool2d(-img, pool_size=(1,3), strides=(1, 1), padding='same', data_format=None, pool_mode='max')
    return K.minimum(p1, p2)
def soft_dilate(img):
    return K.pool2d(img, pool_size=(3,3), strides=(1, 1), padding='same', data_format=None, pool_mode='max')
def soft_open(img):
    return soft_dilate(soft_erode(img))
def soft_skel(img, iter):
    img1 = soft_open(img)
    skel = K.relu(img - img1)
    for i in range(iter):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = K.relu(img - img1)
        skel = skel + K.relu(delta-skel*delta)
    return skel
def soft_clDice(y_true, y_pred, iter=50, smooth=1):
    v_p = y_pred[..., 1:]
    v_t = y_true[..., 1:]
    s_p = soft_skel(v_p, iter)
    s_t = soft_skel(v_t, iter)
    
    axes=(0,1,2,3)
    tprec = (K.sum(s_p * v_t, axes) + smooth) / (K.sum(s_p, axes) + smooth)
    tsens = (K.sum(s_t * v_p, axes) + smooth) / (K.sum(s_t, axes) + smooth)
    return 2 * tprec * tsens / (tprec + tsens)
def soft_dice(y_true, y_pred, smooth=1e-6):
    y_true_o = y_true[..., 1:]
    y_pred_o = y_pred[..., 1:]
    axes=(0,1,2,3)
    numerator = 2. * K.sum(y_true_o * y_pred_o, axes)
    denominator = K.sum(K.square(y_pred_o) + K.square(y_true_o), axes)  
    return K.mean(numerator / (denominator + smooth))
def soft_dice_loss(y_true, y_pred):
    return 1 - soft_dice(y_true, y_pred)
def shit_loss(y_true, y_pred, alpha=0.5):
    return alpha * (1 - soft_dice(y_true, y_pred)) + (1 - alpha) * soft_dice_loss(y_true, y_pred)
##################################################################
