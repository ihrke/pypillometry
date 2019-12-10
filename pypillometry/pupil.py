"""
pupil.py
========

functions related to pupillary responses.
"""
import numpy as np
import scipy.optimize
from .convenience import *

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
    t = np.linspace(0,duration, n, dtype = np.float)
    h=pupil_kernel_t(t,npar,tmax)
    #h = t**(npar) * np.exp(-npar*t / tmax)   #Erlang gamma function Hoek & Levelt (1993)
    #hmax=np.exp(-npar)*tmax**npar ## theoretical maximum
    return h#/h.max() # rescale to height=1


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
    
    # prepare stimulus and response-regressors
    h=pupil_kernel(duration=max_duration, fs=fs, npar=npar, tmax=tmax) ## pupil kernel

    # event-onsets for each event
    x1 = np.zeros((event_onsets.size, tx.size), dtype=np.float) # onsets

    # event-onsets as indices of the txd array
    evon_ix=np.argmin(np.abs(np.tile(event_onsets, (tx.size,1)).T-tx), axis=1)

    X=np.meshgrid(np.arange(x1.shape[1]), np.arange(x1.shape[0]))[0]
    evon_ix_M1=np.tile(evon_ix, (x1.shape[1],1)).T
    evon_ix_M2=np.tile(evon_ix+h.size, (x1.shape[1],1)).T

    x1[ np.arange(event_onsets.size), evon_ix ]=1
    x1[ np.logical_and(X>=evon_ix_M1, X<evon_ix_M2) ]=np.tile(h, evon_ix.size)
    return x1

def pupil_response(tx, sy, event_onsets, fs, npar="free", tmax="free", verbose=10):
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
    """
    def vprint(v, s):
        if v<=verbose:
            print(s,end="")

    if npar=="free" and tmax=="free":
        print("MSG: optimizing both npar and tmax, might take a while...")
        def objective(x, event_onsets, tx,sy,fs):
            vprint(50,".")
            npar_t,tmax_t=x
            npar=trans_logistic_vec(npar_t, a=1, b=20, inverse=True)
            tmax=trans_logistic_vec(tmax_t, a=100, b=2000, inverse=True)
            maxdur=pupil_get_max_duration(npar,tmax)
            vprint(100, "\nnpar,tmax,maxdur=(%.2f,%.2f,%i)"%(npar,tmax,maxdur))            
            x1=pupil_build_design_matrix(tx, event_onsets, fs, npar, tmax, maxdur)
            coef=scipy.optimize.nnls(x1.T, sy)[0]    
            pred=np.dot(x1.T, coef)  ## predicted signal
            resid=sy-pred         ## residuals            
            return np.sum(resid**2)
        
        npar_start_trans=trans_logistic_vec(10,a=1,b=20,inverse=False)
        tmax_start_trans=trans_logistic_vec(900,a=100,b=2000,inverse=False)
        r=scipy.optimize.minimize(objective, (npar_start_trans, tmax_start_trans), 
                                  args=(event_onsets,tx,sy,fs), 
                                  method="Nelder-Mead")        
        npar=trans_logistic_vec(r.x[0], a=1, b=20, inverse=False)
        tmax=trans_logistic_vec(r.x[1], a=100, b=2000, inverse=False)
    elif npar=="free":
        print("MSG: optimizing npar only, might take a while...")
        def objective(x, tmax, event_onsets, tx,sy,fs):
            vprint(50,".")
            npar=x
            vprint(100, "\nnpar=(%.2f)"%(npar))            
            x1=pupil_build_design_matrix(tx, event_onsets, fs, npar, tmax, 6000)
            coef=scipy.optimize.nnls(x1.T, sy)[0]    
            pred=np.dot(x1.T, coef)  ## predicted signal
            resid=sy-pred         ## residuals            
            return np.sum(resid**2)
        r=scipy.optimize.minimize_scalar(objective, bounds=(0,20),
                                  args=(tmax,event_onsets,tx,sy,fs), 
                                  method="bounded")        
        npar=r.x
    elif tmax=="free":
        print("MSG: optimizing tmax only, might take a while...")
        def objective(x, npar, event_onsets, tx,sy,fs):
            vprint(50,".")            
            tmax=x
            vprint(100, "\ntmax=(%.2f)"%(tmax))            
            x1=pupil_build_design_matrix(tx, event_onsets, fs, npar, tmax, 6000)
            coef=scipy.optimize.nnls(x1.T, sy)[0]    
            pred=np.dot(x1.T, coef)  ## predicted signal
            resid=sy-pred         ## residuals            
            return np.sum(resid**2)
        r=scipy.optimize.minimize_scalar(objective, bounds=(400,1500),
                                  args=(npar,event_onsets,tx,sy,fs), 
                                  method="bounded")        
        tmax=r.x
    
    x1=pupil_build_design_matrix(tx, event_onsets, fs, npar, tmax, 6000)
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
    