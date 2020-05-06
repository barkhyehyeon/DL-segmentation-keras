import csv
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import distance

def dice_coef(y_true, y_pred, smooth=1e-08):
    product = np.multiply(y_true, y_pred)
    intersection = np.sum(product)
    union = np.sum(y_true) + np.sum(y_pred)
    coefficient = (2. * intersection + smooth) / (union + smooth)
    return (np.round(coefficient, 4))

def CLD(y_true, y_pred):
    ts = skeletonize(y_true)
    ps = skeletonize(y_pred)
    ts_c = np.argwhere(ts > 0)
    ps_c = np.argwhere(ps > 0)
    n = len(ps_c)
    dist_sum = 0
    for i in range(n):
        dist_sum += np.amin(distance.cdist(ps_c[i:i+1], ts_c, 'euclidean'))
    return (np.round(dist_sum / n, 4))

def calculate_score(y_true, y_pred, func, save_path=None):
    if func == 'Dice':
        f = dice_coef
    elif func == 'CLD':
        f = CLD
        
    score = np.zeros(len(y_pred), dtype=np.float32)
    if save_path is None:
        for i in range(len(y_pred)):
            score[i] = f(y_true[i,...,1], y_pred[i,...,1])
        mean = np.round(np.mean(score), 4)
        std = np.round(np.std(score), 4)
    else:
        file = open(save_path, 'w')
        writer = csv.writer(file)
        for i in range(len(y_pred)):
            score[i] = f(y_true[i,...,1], y_pred[i,...,1])
            writer.writerow([i, score[i]])

        mean = np.round(np.mean(score), 4)
        std = np.round(np.std(score), 4)

        writer.writerow(['Mean', mean])
        writer.writerow(['SD', std])
        file.close()

    print(func + ')', 'Mean=' + str(mean) +', SD=' + str(std))
