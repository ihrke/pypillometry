"""
preproc.py
==========

preprocessing functions (blinks, smoothing, missing data...)
"""
import numpy as np
import scipy.optimize
import pylab as plt

from .convenience import *


def smooth_window(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    adapted from SciPy Cookbook: `<https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html>`_.
    
    Parameters
    ----------    
    x: the input signal 
    window_len: the dimension of the smoothing window; should be an odd integer
    window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Returns
    -------
    np.array: the smoothed signal        
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window should be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y[(window_len-1):(-window_len+1)]



#<https://figshare.com/articles/A_simple_way_to_reconstruct_pupil_size_during_eye_blinks/688001>

def detect_blinks(sy, min_duration, blink_val):
    """
    Detect blinks as consecutive sequence of `blink_val` (f.eks., 0 or NaN) of at least
    `min_duration` successive values in the signal `sy`.
    Detected blinks are put a matrix `blinks` (nblinks x 2) where start and end
    are stored as indexes.
    
    Parameters
    ----------
    sy: np.array (float)
        signal
    min_duration: int
        minimum number of consecutive samples for a sequence of missing numbers to be treated as blink
    blink_val: 
        "missing value" code
    
    Returns
    -------
    np.array (nblinks x 2) containing the indices of the start/end of the blinks
    """
    x=np.r_[0, np.diff((sy==blink_val).astype(np.int))]
    starts=np.where(x==1)[0]
    ends=np.where(x==-1)[0]-1
    if ends.size!=starts.size: 
        ## is the first start earlier than the first end?
        if starts[0]>ends[0]:
            ends=ends[1:] # drop first end
        else:
            starts=starts[:-1] # drop last start
    blinks=[ [start,end] for start,end in zip(starts,ends) if end-start>=min_duration]
    return np.array(blinks)
    