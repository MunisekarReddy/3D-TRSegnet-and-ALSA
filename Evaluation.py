import numpy as np
import math
import cv2 as cv
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import mutual_info
from sklearn.feature_selection import mutual_info_classif as MIC
from scipy import ndimage
EPS = np.finfo(float).eps


def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
              / np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
              - np.sum(s2 * np.log(s2)))
    return mi


def correlation_coefficient(img1, img2):
    # Convert the images to NumPy arrays.
    img1 = np.array(img1)
    img2 = np.array(img2)
    # Calculate the means of the images.
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    # Calculate the standard deviations of the images.
    std1 = np.std(img1)
    std2 = np.std(img2)
    # Calculate the correlation coefficient.
    correlation_coefficient = np.sum((img1 - mean1) * (img2 - mean2)) / (std1 * std2)
    return correlation_coefficient


def net_evaluation(sp, act):
    # dice = dice_coef(sp, act)
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(p.shape[0]):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    Dice = (2 * tp) / ((2 * tp) + fp + fn)
    Jaccard = tp / (tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    FPR = fp / (fp + tn)
    FNR = fn / (tp + fn)
    NPV = tn / (tn + fp)
    FDR = fp / (tp + fp)
    F1_score = (2 * tp) / (2 * tp + fp + fn)
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, Dice, Jaccard, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score,
            MCC]
    return EVAL


def Evaluate_Image(Pred, Orig):
    RMSE = rmse(Pred, Orig)
    SSIM = ssim(Pred, Orig)
    MI = mutual_information_2d(Pred.ravel(), Orig.ravel())
    corr_coeff = np.corrcoef(Pred.flatten(), Orig.flatten())
    Correlation_Coefficient = np.mean(corr_coeff)
    EVAL = [SSIM[0], MI, RMSE, Correlation_Coefficient]
    return EVAL
