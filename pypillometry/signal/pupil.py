"""
pupil.py
========

functions related to pupillary responses.
"""
import numpy as np
import scipy.optimize
import pylab as plt
from loguru import logger
from ..convenience import *
from scipy.ndimage import uniform_filter1d


def lowpass_filter_iterative(signal, cutoff, fs, order=2, max_iter=5):
    """
    Lowpass filter that handles NaN values via iterative interpolation.
    
    NaN values are initially filled with the signal mean, then iteratively
    replaced with filtered values until convergence. This avoids sharp
    artifacts at gap boundaries that would occur with simple interpolation.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (can contain NaN values)
    cutoff : float
        Lowpass filter cutoff frequency in Hz
    fs : float
        Sampling rate in Hz
    order : int, optional
        Butterworth filter order (default: 2)
    max_iter : int, optional
        Maximum number of iterations (default: 5)
        
    Returns
    -------
    filtered : np.ndarray
        Lowpass-filtered signal with NaN gaps smoothly filled
        
    Notes
    -----
    The iterative approach works by:
    1. Fill NaN positions with the signal mean
    2. Apply lowpass filter to the filled signal
    3. Replace NaN positions with the filtered values
    4. Repeat steps 2-3 until max_iter reached
    
    This converges to a smooth solution that respects the observed data
    while filling gaps with values consistent with the surrounding signal.
    
    Examples
    --------
    >>> signal = np.array([1, 2, np.nan, np.nan, 5, 6])
    >>> filtered = lowpass_filter_iterative(signal, cutoff=2.0, fs=100)
    """
    valid = ~np.isnan(signal)
    n_valid = np.sum(valid)
    
    if n_valid == 0:
        # All NaN - return as is
        return signal.copy()
    
    # Import here to avoid circular import (baseline.py imports from pupil.py)
    from .baseline import butter_lowpass_filter
    
    if n_valid == len(signal):
        # No NaN - just apply regular filter
        return butter_lowpass_filter(signal, cutoff, fs, order)
    
    # Initialize: fill NaN with signal mean
    filled = signal.copy()
    filled[~valid] = np.nanmean(signal)
    
    # Iterative filtering
    for _ in range(max_iter):
        # Filter the filled signal
        smoothed = butter_lowpass_filter(filled, cutoff, fs, order)
        # Replace only the NaN positions with filtered values
        filled[~valid] = smoothed[~valid]
    
    # Final filter pass
    result = butter_lowpass_filter(filled, cutoff, fs, order)
    return result


def pupil_signal_quality(signal, fs, mask=None, lowpass_cutoff=4.0, 
                         window_size_ms=500, metric="snr"):
    """
    Compute local signal quality metric for pupil signal.
    
    Estimates signal quality by separating the signal into a slow "true signal"
    component (via lowpass filtering) and a fast "noise" component (the residual).
    Quality is then computed in sliding windows.
    
    Parameters
    ----------
    signal : np.ndarray or np.ma.MaskedArray
        Pupil signal. Can be either:
        - A regular numpy array, in which case `mask` must be provided
        - A numpy masked array, in which case its mask is used
    fs : float
        Sampling rate in Hz
    mask : np.ndarray, optional
        Boolean mask where True indicates invalid/masked samples.
        Required if `signal` is a regular numpy array.
        Ignored if `signal` is a masked array.
    lowpass_cutoff : float, optional
        Cutoff frequency for separating signal from noise (default: 4.0 Hz).
        Pupil responses are typically < 4 Hz, so higher frequencies are noise.
    window_size_ms : float, optional
        Window size for local quality estimation in milliseconds (default: 500 ms)
    metric : str, optional
        Quality metric to return:
        - "snr": signal-to-noise ratio (signal_power / noise_power)
        - "snr_db": SNR in decibels, 10 * log10(snr)
          (0 dB = equal, +10 dB = 10x better, -10 dB = 10x worse)
        - "noise_power": raw noise power (signal units squared)
        - "noise_cv": coefficient of variation (noise_std / signal),
          interpretable as fractional noise (0.05 = 5% noise)
          
    Returns
    -------
    quality : np.ndarray
        Local quality metric array (same length as input).
        Values are 0 at masked locations.
        
    Raises
    ------
    ValueError
        If metric is not one of the supported options.
        If signal is a regular array and mask is not provided.
        If there are NaN values in the unmasked (valid) portion of the signal.
        
    Notes
    -----
    The signal-noise separation assumes pupil dynamics are slow (< lowpass_cutoff Hz).
    The "signal" estimate is the lowpass-filtered version, and "noise" is the
    high-frequency residual (original - lowpass).
    
    Local power is computed as the mean of squared values in a sliding window.
    SNR = signal_power / noise_power indicates how much true signal there is
    relative to measurement noise.
    
    Examples
    --------
    >>> # With explicit mask
    >>> mask = pupil == 0  # e.g., blinks marked as 0
    >>> snr = pupil_signal_quality(pupil, fs=500, mask=mask, metric="snr")
    >>> 
    >>> # With masked array
    >>> pupil_ma = np.ma.array(pupil, mask=(pupil == 0))
    >>> snr = pupil_signal_quality(pupil_ma, fs=500, metric="snr")
    >>> 
    >>> # Get noise as coefficient of variation (percentage)
    >>> noise_cv = pupil_signal_quality(pupil, fs=500, mask=mask, metric="noise_cv")
    >>> print(f"Average noise level: {np.mean(noise_cv[~mask])*100:.1f}%")
    """
    valid_metrics = ["snr", "snr_db", "noise_power", "noise_cv"]
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")
    
    # Handle masked array vs regular array + mask
    if isinstance(signal, np.ma.MaskedArray):
        data = signal.data.copy()
        invalid_mask = signal.mask
        # Handle scalar mask (all False or all True)
        if isinstance(invalid_mask, np.bool_):
            invalid_mask = np.full(data.shape, invalid_mask, dtype=bool)
    else:
        if mask is None:
            raise ValueError("mask must be provided when signal is a regular numpy array")
        data = signal.copy()
        invalid_mask = np.asarray(mask, dtype=bool)
    
    # Check for NaN in valid (unmasked) data
    valid_data = data[~invalid_mask]
    if np.any(np.isnan(valid_data)):
        raise ValueError("NaN values found in unmasked (valid) portion of signal. "
                        "All invalid samples should be marked in the mask.")
    
    # Convert window size to samples
    window_samples = int(window_size_ms / 1000 * fs)
    if window_samples < 3:
        window_samples = 3
    
    # Prepare signal for filtering: set masked values to NaN for iterative filter
    signal_with_nan = data.copy()
    signal_with_nan[invalid_mask] = np.nan
    
    # Get signal estimate via iterative lowpass filter
    signal_smooth = lowpass_filter_iterative(signal_with_nan, lowpass_cutoff, fs, order=2)
    
    # Compute noise (residual) only for valid samples
    # For masked samples, use 0 (they won't contribute to the result anyway)
    noise = np.zeros_like(data)
    noise[~invalid_mask] = data[~invalid_mask] - signal_smooth[~invalid_mask]
    
    # Compute local power using sliding window
    # uniform_filter1d computes local mean, so we apply it to squared values
    signal_power = uniform_filter1d(signal_smooth**2, window_samples, mode='nearest')
    noise_power = uniform_filter1d(noise**2, window_samples, mode='nearest')
    
    # Small epsilon to avoid division by zero
    eps = 1e-10
    
    # Compute requested metric
    if metric == "snr":
        result = signal_power / (noise_power + eps)
    elif metric == "snr_db":
        snr = signal_power / (noise_power + eps)
        result = 10 * np.log10(snr + eps)
    elif metric == "noise_power":
        result = noise_power
    elif metric == "noise_cv":
        # CV = noise_std / signal_mean = sqrt(noise_power) / abs(signal_smooth)
        noise_std = np.sqrt(noise_power)
        result = noise_std / (np.abs(signal_smooth) + eps)
    
    # Set output to 0 at masked locations
    result[invalid_mask] = 0
    
    return result


def pupil_kernel_t(t,npar,tmax):
    """
    According to Hoeks and Levelt (1993, 
    https://link.springer.com/content/pdf/10.3758%2FBF03204445.pdf), 
    the PRF can be described by the so-called Erlang gamma function
    $$h_{HL}(t) = t^n e^{-nt/t_{max}}$$
    which we normalize to
    $$h(t)=\\frac{1}{h_{max}} h_{HL}(t)$$
    where $$h_{max} = \max_t{\\left(h_{HL}(t)\\right)} = e^{-n}t_{max}^{n}$$
    which yields a maximum value of 1 for this function. 
    The function $h(t)$ is implemented in :py:func:`pupil_kernel()`.
    
    This version of the function evaluates the PRF at inputs `t`.
    
    Parameters
    -----------
    t: float/np.array
        in ms
    npar: float
        n in the equation above
    tmax: float
        t_{max} in the equation above

    Returns
    --------
    h: np.array
        PRF evaluated at `t`
    """
    t=np.array(t)
    npar=float(npar)
    tmax=float(tmax)
    hmax=np.exp(-npar)*tmax**npar ## theoretical maximum
    h = t**(npar) * np.exp(-npar*t / tmax)   #Erlang gamma function Hoek & Levelt (1993)
    h=h/hmax
    return h

def pupil_get_max_duration(npar,tmax,thr=1e-8,stepsize=1.):
    """
    Get the time when the PRF with parameters $n$ and $t_{max}$ is decayed to
    `thr`. This gives an indication of how long the `duration` parameter
    in :py:func:`pupil_kernel()` should be chosen.
    
    Parameters
    -----------
    npar,tmax: float
        PRF parameters, see :py:func:`pupil_kernel()`
    thr: float
        desired value to which the PRF is decayed (the lower `thr`, the longer the duration)
    stepsize: float
        precision of the maximum duration in ms
        
    Returns
    --------
    tdur: float
        first value of t so that PRF(t)<`thr`
    """
    # start looking from `tmax` (which is time of the peak)
    tdur=tmax
    while pupil_kernel_t(tdur,npar,tmax)>thr:
        tdur=tdur+stepsize # in steps of `stepsize` ms
    return tdur

def pupil_kernel(duration=4000, fs=1000, npar=10.1, tmax=930.0):
    """
    According to Hoeks and Levelt (1993, 
    https://link.springer.com/content/pdf/10.3758%2FBF03204445.pdf), 
    the PRF can be described by the so-called Erlang gamma function
    $$h_{HL}(t) = t^n e^{-nt/t_{max}}$$
    which we normalize to
    $$h(t)=\\frac{1}{h_{max}} h_{HL}(t)$$
    where $$h_{max} = \max_t{\\left(h_{HL}(t)\\right)} = e^{-n}t_{max}^{n}$$
    which yields a maximum value of 1 for this function. 
    The function $h(t)$ is implemented in `pp.pupil_kernel()`.
    
    Parameters
    -----------
    
    duration: float
        in ms; maximum of the time window for which to calculate the PRF [0,duration]
    fs: float
        sampling rate for resolving the PRF
    npar: float
        n in the equation above
    tmax: float
        t_{max} in the equation above
        
    Returns
    --------
    
    h: np.array
        sampled version of h(t) over the interval [0,`duration`] with sampling rate `fs`
    """
    n=int(duration/1000.*fs)
    t = np.linspace(0,duration, n, dtype = float)
    h=pupil_kernel_t(t,npar,tmax)
    #h = t**(npar) * np.exp(-npar*t / tmax)   #Erlang gamma function Hoek & Levelt (1993)
    #hmax=np.exp(-npar)*tmax**npar ## theoretical maximum
    return h#/h.max() # rescale to height=1

def plot_prf(npar=10.1,tmax=930,max_duration="estimate",fs=500,**kwargs):
    """
    Plot profile of the pupil-response function (PRF) with 
    parameters `npar` and `tmax`.
    """
    if max_duration=="estimate":
        max_duration=pupil_get_max_duration(npar,tmax)
    n=int(fs*(max_duration/1000.))
    tx=np.linspace(0,max_duration,n)
    prf=pupil_kernel_t(tx,npar,tmax)
    plt.plot(tx,prf,**kwargs)
    plt.xlabel("time [ms]")
    plt.ylabel("AU")
    

def pupil_build_design_matrix(tx,event_onsets,fs,npar,tmax,max_duration="estimate"):
    """
    Construct design matrix (nevents x n).
    Each column has a single pupil-kernel with parameters `npar`, `tmax` starting at 
    each `event_onset`.
    
    Parameters
    ----------
    
    tx: np.array
        in ms
    event_onsets: np.array
        timing of the events
    fs: float
        sampling rate (Hz)
    npar: float
        n in the equation of :py:func:`pypillometry.pupil.pupil_kernel()`
    tmax: float
        t_{max} in :py:func:`pypillometry.pupil.pupil_kernel()`
    max_duration: float or "estimate"
        either maximum duration in milliseconds or string "estimate" which
        causes the function to determine an appropriate duration based on the
        kernel parameters `npar` and `tmax`
        
    Returns
    -------
    x1: np.array (nevents x n) 
        design matrix
    """
    if max_duration=="estimate":
        max_duration=pupil_get_max_duration(npar,tmax)

        
    h=pupil_kernel(duration=max_duration, fs=fs, npar=npar, tmax=tmax) ## pupil kernel

    # event-onsets for each event
    x1 = np.zeros((event_onsets.size, tx.size), dtype=float) # onsets

    # event-onsets as indices of the txd array
    evon_ix=np.argmin(np.abs(np.tile(event_onsets, (tx.size,1)).T-tx), axis=1)

    for i in range(evon_ix.size):
        slic_add=h.size if (evon_ix[i]+h.size)<x1.shape[1] else x1.shape[1]-evon_ix[i]            
        x1[i,evon_ix[i]:evon_ix[i]+slic_add]=h[0:slic_add]
                
      
    ## old, vectorized version (I thought it would be faster but it is, in fact, a lot slower :-(
    # # prepare stimulus and response-regressors
    # h=pupil_kernel(duration=max_duration, fs=fs, npar=npar, tmax=tmax) ## pupil kernel
    # 
    # # event-onsets for each event
    # x1 = np.zeros((event_onsets.size, tx.size), dtype=float) # onsets
    # 
    # # event-onsets as indices of the txd array
    # evon_ix=np.argmin(np.abs(np.tile(event_onsets, (tx.size,1)).T-tx), axis=1)
    # 
    # X=np.meshgrid(np.arange(x1.shape[1]), np.arange(x1.shape[0]))[0]
    # evon_ix_M1=np.tile(evon_ix, (x1.shape[1],1)).T
    # evon_ix_M2=np.tile(evon_ix+h.size, (x1.shape[1],1)).T
    # 
    # x1[ np.arange(event_onsets.size), evon_ix ]=1
    # x1[ np.logical_and(X>=evon_ix_M1, X<evon_ix_M2) ]=np.tile(h, evon_ix.size)
    return x1

def pupil_response(tx, sy, event_onsets, fs, npar="free", tmax="free", verbose=10, 
                   bounds={"npar":(1,20), "tmax":(100,2000)}, display_progress=True):
    """
    Estimate pupil-response based on event-onsets.
    
    tx : np.ndarray
        time-vector in milliseconds        
    sy : np.ndarray
        (baseline-corrected) pupil signal
    event_onsets : list
        onsets of events (stimuli/responses) in milliseconds        
    fs : float
        sampling rate in Hz
    npar: float
        npar-parameter for the canonical response-function or "free";
        in case of "free", the function optimizes for this parameter
    tmax: float
        tmax-parameter for the canonical response-function or "free";
        in case of "free", the function optimizes for this parameter
    bounds: dict
        in case that one or both parameters are estimated, give the lower
        and upper bounds for the parameters
    """
    if npar=="free" and tmax=="free":
        logger.info("optimizing both npar and tmax, might take a while...")
        def objective(x, event_onsets, tx,sy,fs):
            npar_t,tmax_t=x
            npar,tmax=x
            #npar=trans_logistic_vec(npar_t, a=bounds["npar"][0], b=bounds["npar"][1], inverse=True)
            #tmax=trans_logistic_vec(tmax_t, a=bounds["tmax"][0], b=bounds["tmax"][1], inverse=True)
            maxdur=pupil_get_max_duration(npar,tmax)
            logger.debug("npar,tmax,maxdur=(%.2f,%.2f,%i)"%(npar,tmax,maxdur))
            x1=pupil_build_design_matrix(tx, event_onsets, fs, npar, tmax, maxdur)
            coef=scipy.optimize.nnls(x1.T, sy)[0]    
            pred=np.dot(x1.T, coef)  ## predicted signal
            resid=sy-pred         ## residuals            
            return np.sum(resid**2)
        
        #npar_start_trans=trans_logistic_vec(10,a=bounds["npar"][0], b=bounds["npar"][1],inverse=False)
        #tmax_start_trans=trans_logistic_vec(900,a=bounds["tmax"][0], b=bounds["tmax"][1],inverse=False)
        #r=scipy.optimize.minimize(objective, (npar_start_trans, tmax_start_trans), 
        #                          args=(event_onsets,tx,sy,fs), 
        #                          method="Nelder-Mead")    
        r=scipy.optimize.minimize(objective, (10,900), #(npar_start_trans, tmax_start_trans), 
                                  args=(event_onsets,tx,sy,fs), bounds=[bounds["npar"],bounds["tmax"]],
                                 options={"disp":display_progress})
                                  #method="Nelder-Mead")    
        
        #npar=trans_logistic_vec(r.x[0], a=bounds["npar"][0], b=bounds["npar"][1], inverse=False)
        #tmax=trans_logistic_vec(r.x[1], a=bounds["tmax"][0], b=bounds["tmax"][1], inverse=False)
        npar,tmax=r.x[0],r.x[1]
    elif npar=="free":
        logger.info("optimizing npar only, might take a while...")
        def objective(x, tmax, event_onsets, tx,sy,fs):
            npar=x
            maxdur=pupil_get_max_duration(npar,tmax)
            logger.debug("npar,maxdur=(%.2f,%i)"%(npar,maxdur))
            x1=pupil_build_design_matrix(tx, event_onsets, fs, npar, tmax, maxdur)            
            coef=scipy.optimize.nnls(x1.T, sy)[0]    
            pred=np.dot(x1.T, coef)  ## predicted signal
            resid=sy-pred         ## residuals            
            return np.sum(resid**2)
        r=scipy.optimize.minimize_scalar(objective, bounds=bounds["npar"],
                                  args=(tmax,event_onsets,tx,sy,fs), 
                                  method="bounded", options={"disp":display_progress,"xatol":.1})        
        npar=r.x
    elif tmax=="free":
        logger.info("optimizing tmax only, might take a while...")
        def objective(x, npar, event_onsets, tx,sy,fs):
            tmax=x
            maxdur=pupil_get_max_duration(npar,tmax)   
            logger.debug("tmax,maxdur=(%.2f,%i)"%(tmax,maxdur)) 
            x1=pupil_build_design_matrix(tx, event_onsets, fs, npar, tmax, maxdur)
            coef=scipy.optimize.nnls(x1.T, sy)[0]    
            pred=np.dot(x1.T, coef)  ## predicted signal
            resid=sy-pred         ## residuals            
            return np.sum(resid**2)
        r=scipy.optimize.minimize_scalar(objective, bounds=bounds["tmax"],
                                  args=(npar,event_onsets,tx,sy,fs), 
                                  method="bounded",options={"disp":display_progress,"xatol":1})        
        tmax=r.x
  
    maxdur=pupil_get_max_duration(npar,tmax)
    x1=pupil_build_design_matrix(tx, event_onsets, fs, npar, tmax, maxdur)
    coef=scipy.optimize.nnls(x1.T, sy)[0]    
    pred=np.dot(x1.T, coef)  ## predicted signal
    
    return pred, coef, npar, tmax, x1

        
     
def pupilresponse_nnls(tx, sy, event_onsets, fs, npar=10.1, tmax=930):
    """
    Estimate single-event pupil responses based on canonical PRF (`pupil_kernel()`)
    using non-negative least-squares (NNLS).
        
    Parameters
    -----------
    
    tx : np.ndarray
        time-vector in milliseconds
        
    sy : np.ndarray
        (baseline-corrected) pupil signal
        
    event_onsets : list
        onsets of events (stimuli/responses) in seconds
        
    fs : float
        sampling rate in Hz
        
    npar,tmax: float
        parameters for :py:func:`pypillometry.pupil.pupil_kernel()`
        
    Returns
    --------
    
    (coef,pred,resid): tuple
        coef: purely-positive regression coefficients
        pred: predicted signal
        resid: residuals (sy-pred)
    """
    x1=pupil_build_design_matrix(tx, event_onsets, fs, npar, tmax, "estimate")
    
    ## we use a non-negative least squares solver to force the PRF-coefficients to be positive
    coef=scipy.optimize.nnls(x1.T, sy)[0]    
    pred=np.dot(x1.T, coef)  ## predicted signal
    resid=sy-pred         ## residual

    return coef,pred,resid
    

    