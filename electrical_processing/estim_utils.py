'''
Module of utility functions for various things related to analysis of
electrical stimulation data.

author: Alex Gogliettino
date: 2020-05-15
'''
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit


def sigmoid(xs: np.ndarray or int,m: float,b: float) -> np.ndarray or float:
    '''
    Args:
        xs: x values to evaluate (array or scalar).
        m: slope of curve.
        b: intercept of the curve.

    Returns:
        Sigmoid evaluated at each xs (either array or scalar). 

    Functional form of sigmoid. Useful for fitting activation curves.
    '''
    return 1/ (1 + np.exp(-m*(xs - b)))

def fit_sigmoid(stim_amps: np.ndarray,spike_probs: np.ndarray,
                param_guess=None,maxfev: int=10000) -> tuple:
    '''
    Args:
        stim_amps: array of stimulation amplitudes provided during a single 
            electrode scan.
        spike_probs: array of probababilties in response to each delivered 
            amplitude.
        param_guess: initial guess of the parameters (default=None)
        maxfev: maximum number of iterations (default=10000)

    Returns:
        sigmoid_fit: a sigmoid fitted to the activation curve using non-linear 
            least squares.
        popt: the optimal parameters from fitting procedure.

    Raises:
        ValueError: 
            If the stim_amps and spike_probs are different sizes.

    Function to fit a sigmoidal curve using nonlinear least squares
    (scipy.optimize.curve_fit) to an activation curve from a single electrode 
    scan experiment.
    '''

    # Check input sizes, otherwise fit the data.
    if np.shape(stim_amps) != np.shape(spike_probs):
        raise ValueError("Amplitude and probability arrays must have the"
                         " same dimensions.")
        
    popt,pcov = curve_fit(sigmoid,stim_amps,spike_probs,
                          param_guess,maxfev=maxfev)
    sigmoid_fit = sigmoid(stim_amps,*popt)

    return sigmoid_fit,popt

def get_stimamp_sigmoidfit(prob,popt):
    '''
    Args:
        prob: probability of spiking (obviously > 0 and < 1). Don't choose
            exactly 0 or 1 (scalar).
        popt: array of optimal parameters from curve fitting procedure.

    Returns:
        stim_amp: stimulation amplitude (positive, uA) corresponding to the 
                  user defined probability of spiking.
    Raises:
        ValueError:
            1. If probability is not a real probability.
            2. If optimal parameters doesn't have a shape of (2,)

    Function to return the estimated stimulation ampltitude from the fitted 
    sigmoid, given a specific probability of spiking.
    '''

    # Check inputs, can't use exactly 0 or 1.
    if prob <= 0 or prob >= 1:
        raise ValueError("Must pass a valid probability (between 0 and 1).")
        
    if np.shape(popt) != (2,):
        raise ValueError("Optimal paramaters array must contain two elements"\
                         " for sigmoidal fit.")
        
    # Solve for x in the sigmoid function given the parameters.
    return -np.log(1 / prob - 1) / popt[0] + popt[1]
