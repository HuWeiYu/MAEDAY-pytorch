from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt

def get_anomaly_map(query_img, rec_img_list):
    N = len(rec_img_list)
    E = None
    for each_rec in rec_img_list:
        diff_bgr = query_img - each_rec
        b,g,r = np.square(diff_bgr[:,:,0]), np.square(diff_bgr[:,:,1]), np.square(diff_bgr[:,:,2])
        b = gaussian_filter(b, sigma=1.4)
        g = gaussian_filter(g, sigma=1.4)
        r = gaussian_filter(r, sigma=1.4)
        Ei = np.sum([b,g,r],axis=0)
        if E is None:
            E = Ei
        else:
            E += Ei
    E = E/N
    return E

def calculate_Img_roc_auc(gt_list, score_list):
    score_list = np.asarray(score_list)
    max_anomaly_score = score_list.max()
    min_anomaly_score = score_list.min()
    scores = (score_list - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    img_roc_auc = roc_auc_score(gt_list, img_scores)

    return img_roc_auc

