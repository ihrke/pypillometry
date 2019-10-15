import numpy as np
import scipy.optimize

def pupil_kernel(duration=4, fs=1000, npar=10.1, tmax=930.0):
    n=int(duration*fs)
    t = np.linspace(0,duration, n, dtype = np.float)*1000 # in ms
    h = t**(npar) * np.exp(-npar*t / tmax)   #Erlang gamma function Hoek & Levelt (1993)
    return h/h.max() # rescale to height=1

def pupilresponse_nnls(tx, sy, event_onsets, fs, npar=10.1, tmax=930):
    """
    Estimate single-event pupil responses based on canonical PRF (`pupil_kernel()`)
    using non-negative least-squares (NNLS).
        
    Parameters:
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
        parameters for `pupil_kernel()`
        
    Returns:
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
    