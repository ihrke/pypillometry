import numpy as np
import scipy.signal as signal
import scipy
import math

import scipy.interpolate
from scipy.interpolate import interp1d, splrep, splev

from .pupil import *


# filter-functions for baseline-regressor
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def downsample(y,R):
    """
    Simple downsampling scheme using mean within the downsampling window.
    
    Parameters:
    -----------
    
    y: np.array
        signal to downsample
        
    R: int
        decimate-factor
    """
    pad_size = int(math.ceil(float(y.size)/R)*R - y.size)
    y_padded = np.append(y, np.zeros(pad_size)*np.NaN)
    y2=scipy.nanmean(y_padded.reshape(-1,R), axis=1)
    return y2

def baseline_envelope(tx,sy,event_onsets, fs=1000, lp=2, prominence_thr=80, interp_method="cubic"):
    """
    Extract baseline based on the lower envelope of the (filtered) signal.

    Steps: 
    
    - filter away noise
    - detect high-prominence throughs in the signal 
    - calculate lower envelope based on these peaks
    
    Parameters:
    -----------
    
    tx : np.ndarray
        time-vector in seconds
        
    sy : np.ndarray
        raw pupil signal
        
    event_onsets : list
        onsets of events (stimuli/responses) in seconds
        
    fs : float
        sampling rate in Hz
    
    lp : float
        low-pass filter cutoff for removing random noise
        
    prominence_thr : float in [0,100]
        percentile of the prominence distribution (of the peaks) to 
        use for determining prominent peaks (see `scipy.stats.peak_prominences()`)
        
    interp_method : string, one of ["linear", "cubic", "spline"]
        "linear" - linear interpolation between the high-prominence peaks
        "cubic"  - cubic interpolation through all high-prominence peaks
        "spline" - a smoothing spline that is guaranteed to go through all
                   high-prominence peaks and smoothes through all the other
                   (lower-prominence) peaks
    """
    syc=butter_lowpass_filter(sy, fs=fs, order=2, cutoff=lp)
    peaks_ix=signal.find_peaks(-syc)[0]
    prominences=signal.peak_prominences(-syc, peaks_ix)[0]
    res=signal.peak_widths(-syc, peaks_ix)
    width_locs=(-res[1],res[2]/fs,res[3]/fs)
    widths=res[0]
    peaks=tx[peaks_ix]
    widths=widths/fs # in seconds
    prominence_cutoff=np.percentile(prominences,prominence_thr)
    real_peaks=peaks[prominences>prominence_cutoff]
    real_peaks_ix=peaks_ix[prominences>prominence_cutoff]
    
    if interp_method in ["linear","cubic"]:
        ## interpolate only most prominent peaks
        xinterp=np.concatenate( ([tx.min()],real_peaks,[tx.max()]) )
        yinterp=np.concatenate( ([syc[0]], syc[real_peaks_ix], [syc[-1]]) )
        f=interp1d(xinterp,yinterp, kind=interp_method)
    elif interp_method=="spline":
        ## use all peaks for interpolation and use "real" peaks as inner knots
        xinterp=np.concatenate( ([tx.min()],peaks,[tx.max()]) )
        yinterp=np.concatenate( ([syc[0]], syc[peaks_ix], [syc[-1]]) )
        f=scipy.interpolate.LSQUnivariateSpline(xinterp,yinterp,real_peaks)
    else:
        raise ValueError("interp_method must be one of 'linear','cubic','spline'")
    x0=f(tx)
    
    return x0


def baseline_pupil_model(tx,sy,event_onsets, fs=1000, lp1=2, lp2=0.2):
    """
    Extract baseline based on filtering after removing stim-locked activity.
    
    Steps:
    
    - filter away noise
    - regress out event-locked activity from the filtered signal using NNLS
    - remove modeled signal from filtered data
    - run another lowpass-filter to get rid of spurious signals
    
    Parameters:
    -----------
    
    tx : np.ndarray
        time-vector in seconds
        
    sy : np.ndarray
        raw pupil signal
        
    event_onsets : list
        onsets of events (stimuli/responses) in seconds
        
    fs : float
        sampling rate in Hz
    
    lp1 : float
        low-pass filter cutoff for removing random noise
        
    lp2 : float
        low-pass filter cutoff for removing spurious peaks from the baseline-signal        
    """
    syc=butter_lowpass_filter(sy, fs=fs, order=2, cutoff=lp1)
    
    # calculate indices for event-onsets
    event_onsets_ix=np.argmin(np.abs(np.tile(event_onsets, (sy.size,1)).T-tx), axis=1)

    # set up a single regressor
    x1=np.zeros(sy.size, dtype=np.float)
    x1[event_onsets_ix]=1
    kernel=pupil_kernel(4, fs=fs)
    x1=np.convolve(x1, kernel, mode="full")[0:x1.size]

    # solve with non-negative least-squares
    X=np.stack( (x1, np.ones(x1.size)))
    coef=scipy.optimize.nnls(X.T, syc)[0]
    pred=coef[1]+coef[0]*x1
    resid=syc-pred+coef[1]

    resid_lp=butter_lowpass_filter(resid, fs=fs, order=2, cutoff=lp2)
    x0 = resid_lp
    
    return x0
