"""
pupil.py
========

functions related to pupillary responses.
"""
import numpy as np
import scipy.optimize

def pupil_kernel(duration=4, fs=1000, npar=10.1, tmax=930.0):
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
        maximum of the time window for which to calculate the PRF [0,duration]
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
    n=int(duration*fs)
    t = np.linspace(0,duration, n, dtype = np.float)*1000 # in ms
    h = t**(npar) * np.exp(-npar*t / tmax)   #Erlang gamma function Hoek & Levelt (1993)
    #hmax=np.exp(-npar)*tmax**npar ## theoretical maximum
    return h/h.max() # rescale to height=1

def pupilresponse_nnls(tx, sy, event_onsets, fs, npar=10.1, tmax=930):
    """
    Estimate single-event pupil responses based on canonical PRF (`pupil_kernel()`)
    using non-negative least-squares (NNLS).
        
    Parameters
    -----------
    
    tx : np.ndarray
        time-vector in seconds
        
    sy : np.ndarray
        probably baseline-corrected pupil signal
        
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
    # prepare stimulus and response-regressors
    h=pupil_kernel(fs=fs, npar=npar, tmax=tmax) ## pupil kernel

    # event-onsets for each event
    x1 = np.zeros((event_onsets.size, sy.size), dtype=np.float) # onsets

    # event-onsets as indices of the txd array
    evon_ix=np.argmin(np.abs(np.tile(event_onsets, (sy.size,1)).T-tx), axis=1)
    x1[ np.arange(event_onsets.size), evon_ix ]=1

    # convolve with PRF to get single-trial regressors
    for i in range(event_onsets.size):
        x1[i,]=np.convolve(x1[i,], h, mode="full")[0:x1[i,].size]
    
    ## we use a non-negative least squares solver to force the PRF-coefficients to be positive
    coef=scipy.optimize.nnls(x1.T, sy)[0]    
    pred=np.dot(x1.T, coef)  ## predicted signal
    resid=sy-pred         ## residual

    return coef,pred,resid
    