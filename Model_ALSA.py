import cv2 as cv
import numpy as np
from lv_set.find_lsf import find_lsf
from lv_set.potential_func import *

# use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model
DOUBLE_WELL = 'double-well'
# use double-well potential in Eq. (16), which is good for both edge and region based models
SINGLE_WELL = 'single-well'


def gourd_params(Image, sol):
    # img = imread('gourd.bmp', True)
    img = Image
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[24:35, 19:25] = -c0
    initial_lsf[24:35, 39:50] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,  # time step
        'iter_inner': 50,  # 10   20
        'iter_outer': 150,  # 30  70
        'lmda': 3,  # coefficient of the weighted length term L(phi)  5
        'alfa': -2,  # coefficient of the weighted area term A(phi)  -3
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': sol[0],  # 0.8,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }


def two_cells_params(Image, sol):
    img = Image[0]
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[9:55, 9:75] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 5,  # time step
        'iter_inner': 10,  # 5
        'iter_outer': 40,  # 40
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': 1.5,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': sol[0],  # 1.5,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }


def Model_ALSA(img, gt, sol):
    Segmented = []
    for i in range(len(gt)):
        print(i)
        image = img[i]
        Groumd_truth = gt[i]
        if len(Groumd_truth.shape) == 3:
            Groumd_truth = cv.cvtColor(Groumd_truth, cv.COLOR_RGB2GRAY)
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        params = gourd_params(image, sol)
        phi = find_lsf(**params)
        Segmented.append(phi)
    return Segmented

