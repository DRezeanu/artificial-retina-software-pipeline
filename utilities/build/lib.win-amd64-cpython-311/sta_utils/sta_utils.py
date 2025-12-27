'''
Library of functions to process raw STAs from Vision's white noise analysis.

author: Alex Gogliettino
date: 2020-05-01
'''

import numpy as np
import scipy as sp
from scipy.io import loadmat
from scipy import linalg
import os
import matplotlib.pyplot as plt
import statsmodels.robust as robust
#from scipy.stats import median_absolute_deviation
from scipy.optimize import *
import math
import warnings
import pdb
import h5py
import re
import visionloader as vl

# Constants used in multiple functions.
NUM_PHOSPHORS = 3 # RGB.

def _norm_sta(sta_tensor: np.ndarray) -> np.ndarray:
    """
    Arguments: 
        sta_tensor: sta to be normalized
    Returns: 
        normalized sta 

    Normalizes STA such that the pixels are in the range 0,1.
    """
    # Constants.
    MIN_PIX_VAL = -1
    MAX_PIX_VAL = 1
    MID_PIX_VAL = 0.5

    # Normalize by maximum, set in the range of 0,1.
    sta_tensor /= np.max(np.abs(sta_tensor))
    less_inds = np.argwhere(sta_tensor < MIN_PIX_VAL)
    greater_inds = np.argwhere(sta_tensor > MAX_PIX_VAL)

    if len(less_inds) > 0:
        for ind in less_inds: 
            sta_tensor[ind] = MIN_PIX_VAL

    if len(greater_inds) > 0:
        for ind in greater_inds: 
            sta_tensor[ind] = MAX_PIX_VAL

    return MID_PIX_VAL + (sta_tensor / 2)


def get_sta_tensor(sta_container: "visionloader.STAContainer",
                   normalize=True) -> np.ndarray:
    """
    Arguments: 
        sta_container: sta object for a single cell
        normalize: boolean flag to normalize or not.

    Returns: 
        sta tensor constucted from the sta object.

    Constructs a 4D tensor (w x h x RGB x depth) of the STA for a 
    given cell. If indicated, the STA will be normalized such that
    pixel values are in the range 0,1 (reccomended).
    """


    # Initialize tensor, based on red phosphor dimensions.
    sta_shape = sta_container.red.shape
    sta_tensor = np.zeros((sta_shape[0],sta_shape[1],
                           NUM_PHOSPHORS,sta_shape[2]))

    # Write to tensor such that dim 2 is R,G,B.
    sta_tensor[:,:,0,:] = sta_container.red
    sta_tensor[:,:,1,:] = sta_container.green
    sta_tensor[:,:,2,:] = sta_container.blue

    if normalize:
        return _norm_sta(sta_tensor)

    return sta_tensor

def _is_sig_stixel(sig_stixels: np.ndarray, stixel: np.ndarray) -> bool:
    """
    Arguments:
        sig_stixels: significant stixels in STA
        stixel: comparison stixel, within the original set.
    Return: 
        True if stixel is in sig_stixels, False otherwise.

    Checks to see if a given stixel has at least one neighboring significant
    stixel.
    """
    x = stixel[0]
    y = stixel[1]

    for i in np.arange(x-1,x+2):
        for j in np.arange(y-1,y+2):

            if i == x and j == y: continue
            neighbor = np.array([i,j])

            for k in range(sig_stixels.shape[0]):
                if np.all(sig_stixels[k] == neighbor):
                    return True

    return False


def _clean_stixels(sig_stixels: np.ndarray) -> np.ndarray or None:
    """
    Arguments:
        sig_stixels: significant stixels in STA
    Returns:
        Significant stixels with single "outlier" values removed.

    Searches through the set of significant stixels, and filters out any
    "lone wolf" values. Assumes spatially contiguous STA (i.e. hotspot
    structure STAs will NOT work here). For a stixel to be kept, it must have
    AT LEAST one neighboring stixel.
    """
    sig_stixels_clean = []

    # Check for at least one neighbor, skipping itself of course. 
    for stixel in sig_stixels:

        if _is_sig_stixel(sig_stixels,stixel):
            sig_stixels_clean.append(stixel)

    if len(sig_stixels_clean) == 0:
        return None

    return np.array(sig_stixels_clean)


def get_sig_stixels(sta: np.ndarray, dec_val: float=.5) -> np.ndarray or None:
    """
    Arguments: 
        sta: tensor of size w x h x t, must be grayscaled.
        dec_val: value by which to decrement the number of sigmas
                 (default=.5).
    Returns:
        Significant stixels of the STA (0-indexed), or None if none are found.
    Raises: 
        Exception: if a non-grayscaled STA is passed.

    Implements an iterative algorithm to find the significant stixels above 
    a threshold value. The robust STD is first computed, and frames of the STA
    are scanned for significant stixels. The stixels are chosen greedily, i.e.
    the largest set of stixels found are returned. First, 4 x robust STD is the 
    threshold. If no stixels are found, the threshold then only 3.5,3. If
    none are found above a threshold of 2 x robust STD, the algorithm returns
    None. 

    Note: assumes a normalized STA, i.e. mean .5, in range 0,1.
    """

    # Constants.
    NUM_SIGMAS_MAX = 4.5
    NUM_SIGMAS_MIN = 3
    MID_PIX_VAL = .50

    if len(sta.shape) < 3:
        raise Exception("Only grayscaled STAs supported.")

    # Intitialize thresholds, iteratively scan the STA.
    num_sigmas = NUM_SIGMAS_MAX
    robust_std = robust.scale.mad(sta.flatten())
    max_snr = -np.inf

    while True:
        sig_stixels = np.array([])
        thresh_upper = MID_PIX_VAL + (robust_std * num_sigmas)
        thresh_lower = MID_PIX_VAL - (robust_std * num_sigmas)

        for i in range(sta.shape[2]):
            greater_inds = np.argwhere(sta[:,:,i] > thresh_upper)
            less_inds = np.argwhere(sta[:,:,i] < thresh_lower)

            # Assign the stixels, greedly (RF-center will dominate).
            if greater_inds.shape[0] > less_inds.shape[0]:
                sig_stixels_tmp = greater_inds
            else:
                sig_stixels_tmp = less_inds

            # Update stixels if more were found.
            if sig_stixels_tmp.shape[0] > sig_stixels.shape[0]:
                sig_stixels = sig_stixels_tmp

        # If at least 2 sig stixels, return it.
        if sig_stixels.shape[0] > 1:
            return _clean_stixels(sig_stixels)

        num_sigmas -= dec_val
        if num_sigmas < NUM_SIGMAS_MIN:
            return None


def mask_sta(sta: np.ndarray, sig_stixels: np.ndarray) -> np.ndarray:
    """
    Arguments: 
        sta: tensor of size w x h x t, must be grayscaled.
        sig_stixels: array of stixel indices (0-indexed)
    Returns: STA with non-significant stixels set to 0.

    Sets all non significant stixels in an STA to zero.
    """

    masked_sta = np.zeros(sta.shape)

    for i in range(sig_stixels.shape[0]):
        y_stix = sig_stixels[i][1]
        x_stix = sig_stixels[i][0]
        masked_sta[x_stix,y_stix,:] = sta[x_stix,y_stix,:]

    return masked_sta


def get_timecourse_matrix_for_cell(vcd: "visionloader.VisionCellDataTable",
                                   cell: int) -> np.ndarray:
    """
    Arguments:
        vcd: Vision data table, with include_params set to True
        cell: cell id of interest
    Returns: matrix containing the RGB, or 3 repeated gray, phosphors of the
             STA time course.
    """

    # Write the individual phoshpor time courses to an array.
    timecourse_list = []
    timecourse_list.append(vcd.get_data_for_cell(cell,"RedTimeCourse"))
    timecourse_list.append(vcd.get_data_for_cell(cell,"GreenTimeCourse"))
    timecourse_list.append(vcd.get_data_for_cell(cell,"BlueTimeCourse"))

    return np.c_[[phosphor for phosphor in timecourse_list]]


def ttf_lp(t: int or np.ndarray , p1: float, p2: float,
           tau1: float, tau2: float, n1: int or float, 
           n2: int or float) -> int or np.ndarray:
    '''
    Arguments:
        t: Frame of the time course (sample number)
        p1: Proportional to one of the peaks
        p2: Proportional to one of the peaks
        tau1: Time constant to one of the peaks
        tau2: Time constant to one of the peaks
        n1: Width of one of the peaks
        n2: Width of one of the peaks
    Returns: Function evaluated at some t and the above parameters.

    Function used to fit an STA time course. This is a difference of two 
    cascades of low pass filters, as described in 
    Chichilnisky and Kalmar, 2002, J Neurosci.
    '''
    return p1 * ((t / tau1)**n1) * np.exp(- n1 *((t / tau1) - 1)) -\
           p2 * ((t / tau2)**n2) * np.exp(- n2 * ((t / tau2) - 1))

def fit_ttf(ttf: np.ndarray) -> np.ndarray and np.ndarray:
    '''
    Arguments:
        ttf: Timecourse vector (array)
    Returns:
        The fit to the function, evaluated at the data points from ttf
        Optimal parameters found.

    Function to fit an STA time course. Makes use of ttf_lp in this module. Uses 
    least squares (scipy.optimize.curve_fit). Initialization of the parameters is 
    critical for a good fit. Here, the parameters are initialized in a similar way
    to a MATLAB implementation of this function, written by GDF. This Python 
    version was based on code originally written by bhaishahster.
    '''

    # Constants.
    N1_INIT = 15
    N2_INIT = 6
    MAX_ITER = 100000000
    P1_SCALAR = 1.05
    P2_SCALAR = .95
    TAU1_SCALAR = 1.1
    MIN_TROUGH_PEAK_RATIO = .15
    TAU2_SCALAR = 1.25

    # Initialize parameters, based on time course. 
    n1 = N1_INIT
    n2 = N2_INIT
    max_tc = np.max(ttf)
    max_frame = np.argmax(ttf)
    min_tc = np.min(ttf)
    min_frame = np.argmin(ttf)

    ## Set things up, depending on the polarity of the cell. 
    if min_frame < max_frame:
        peak_frame = min_frame + 1
        trough_frame = max_frame + 1
        peak_scale = min_tc
        trough_scale = max_tc
    else:
        peak_frame = max_frame + 1
        trough_frame = min_frame + 1
        peak_scale = max_tc
        trough_scale = min_tc

    p1 = peak_scale * P1_SCALAR
    p2 = trough_scale * P2_SCALAR
    tau1 = peak_frame * TAU1_SCALAR

    if np.abs(trough_scale / peak_scale) > MIN_TROUGH_PEAK_RATIO:
        tau2 = trough_frame
    else:
        tau2 = trough_frame * TAU2_SCALAR

    x0 = [p1, p2, tau1, tau2, n1, n2]

    # Fit function.
    popt, pcov = curve_fit(ttf_lp, np.arange(ttf.shape[0]), 
                           ttf, p0=x0, maxfev=MAX_ITER)
    ttf_fit = np.flipud(ttf_lp(np.arange(ttf.shape[0]), *popt))

    return ttf_fit,popt
