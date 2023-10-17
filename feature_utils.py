import torch
import torch.nn.functional as F
# import kornia as K
# import kornia.feature as KF
import cv2
import os
import h5py
import numpy as np
from utils import *
#from kornia_moons.feature import *


def load_h5(filename):
    """Loads dictionary from hdf5 file."""
    dict_to_load = {}

    if not os.path.exists(filename):
        print('Cannot find file {}'.format(filename))
        return dict_to_load

    with h5py.File(filename, 'r') as f:
        keys = [key for key in f.keys()]
        for key in keys:
            dict_to_load[key] = f[key][()]
    return dict_to_load


def normalize_keypoints(keypoints, K):
    """Normalize keypoints using the calibration data."""

    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

    return keypoints


def normalize_keypoints_tensor(keypoints, K):
    """Normalize keypoints using the calibration data."""

    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - torch.as_tensor([[C_x, C_y]], device=keypoints.device)) / torch.as_tensor([[f_x, f_y]], device=keypoints.device)

    return keypoints

# A function to convert the point ordering to probabilities used in NG-RANSAC's sampler or AR-Sampler.
def get_probabilities(len_tentatives):
    probabilities = []
    # Since the correspondences are assumed to be ordered by their SNN ratio a priori,
    # we just assign a probability according to their order.
    for i in range(len_tentatives):
        probabilities.append(1.0 - i / len_tentatives)
    return probabilities
