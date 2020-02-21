"""
baseline.py
===========

Functions for estimating tonic pupillary fluctuations (baseline).
"""
import numpy as np
import scipy.signal as signal
import scipy
import math

import scipy.interpolate
from scipy.interpolate import interp1d, splrep, splev

from .pupil import *
from .convenience import *

stan_code_baseline_model_asym_laplac="""
// Stan model for pupil-baseline estimation
//
functions{
    // asymmetric laplace function with the mu, sigma, tau parametrization
    real skew_double_exponential_lpdf(real y, real mu, real sigma, real tau) {
      return log(tau) + log1m(tau)
        - log(sigma)
        - 2 * ((y < mu) ? (1 - tau) * (mu - y) : tau * (y - mu)) / sigma;
    }
    
    // zero-centered asymmetric laplace function with the mu, lambda, kappa parametrization
    real skew_double_exponential2_lpdf(real y, real lam, real kappa) {
      return log(lam) - log(kappa+1/kappa)
        + ((y<0) ? (lam/kappa) : (-lam*kappa))*(y);
    }
}
data{ 
  int<lower=1> n; // number of timepoints in the signal
  vector[n] sy;   // the pupil signal
    
  int<lower=1> ncol; // number of basis functions (columns in B)
  matrix[n,ncol] B;  // spline basis functions
    
  int<lower=1> npeaks;          // number of lower peaks in the signal
  int<lower=1> peakix[npeaks];  // index of the lower peaks in sy
  vector<lower=0>[npeaks] lam_prominences; // lambda-converted prominence values
    
  real<lower=0> lam_sig;    // lambda for the signal where there is no peak
  real<lower=0,upper=1> pa; // proportion of allowed distribution below 0
}

transformed data{
    vector[n] lam;       // lambda at each timepoint
    real<lower=0> kappa; // calculated kappa from pa
    kappa=sqrt(pa)/sqrt(1-pa);
    
    lam=rep_vector(lam_sig, n); 
    for(i in 1:npeaks){
        lam[peakix[i]]=lam_prominences[i];
    }
}
parameters {
    vector[ncol] coef; // coefficients for the basis-functions
}

transformed parameters{
    
}

model {
    {
    vector[n] d;
        
    coef ~ normal(0,5);
    d=sy-(B*coef); // center at estimated baseline
    for( i in 1:n ){
        d[i] ~ skew_double_exponential2(lam[i], kappa);
    }
    }
}
"""


def bspline(txd, knots, spline_degree=3):
    """
    Re-implementation from https://mc-stan.org/users/documentation/case-studies/splines_in_stan.html.
    Similar behaviour as R's bs() function from the splines-library.
    
    Parameters
    -----------
    
    txd: np.array
        time-vector
    knots: np.array
        location of the knots
    spline_degree: int
        degree of the spline
        
    Returns
    --------
    
    B: np.array
        matrix of basis functions
    """
    n=txd.shape[0]
    num_knots=knots.shape[0]

    def build_b_spline(t, ext_knots, ind, order):
        n=t.shape[0]
        b_spline=np.zeros(n)
        w1=np.zeros(n)
        w2=np.zeros(n)
        if order==1:
            b_spline=np.zeros(n)
            b_spline[np.logical_and(t>=ext_knots[ind], t<ext_knots[ind+1]) ]=1
        else:
            if ext_knots[ind] != ext_knots[ind+order-1]:
                w1=(t-np.array([ext_knots[ind]]*n))/(ext_knots[ind+order-1]-ext_knots[ind])
            if ext_knots[ind+1]!=ext_knots[ind+order]:
                w2=1-(t-np.array([ext_knots[ind+1]]*n))/(ext_knots[ind+order]-ext_knots[ind+1])
            b_spline=w1*build_b_spline(t,ext_knots,ind,order-1)+w2*build_b_spline(t,ext_knots,ind+1,order-1)
        return b_spline

    num_basis = num_knots + spline_degree - 1; # total number of B-splines
    B=np.zeros( (num_basis, n) )
    ext_knots=np.concatenate( ([knots[0]]*spline_degree, knots, 
                              [knots[num_knots-1]]*spline_degree) )
    for i in range(num_basis):
        B[i,:] = build_b_spline(txd, ext_knots, i, spline_degree+1)
    B[num_basis-1,n-1]=1
    return B.T

def butter_lowpass(cutoff, fs, order=5):
    """
    Get lowpass-filter coefficients for Butterworth-filter.
    
    Parameters
    -----------
    
    cutoff: float
        lowpass-filter cutoff
    fs: float
        sampling rate
    order: int
        filter order
        
    Returns
    -------
    
    (b,a): tuple (float,float)
        filter coefficients
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Lowpass-filter signal using a Butterworth-filter.
    
    Parameters
    -----------
    
    data: np.array
        data to lowpass-filter
    cutoff: float
        lowpass-filter cutoff
    fs: float
        sampling rate
    order: int
        filter order
        
    Returns
    -------
    
    y: np.array
        filtered data
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def downsample(y,R):
    """
    Simple downsampling scheme using mean within the downsampling window.
    
    Parameters
    -----------
    
    y: np.array
        signal to downsample
        
    R: int
        decimate-factor
        
    Returns
    -------
    
    y: np.array
        downsampled data
    """
    pad_size = int(math.ceil(float(y.size)/R)*R - y.size)
    y_padded = np.append(y, np.zeros(pad_size)*np.NaN)
    y2=scipy.nanmean(y_padded.reshape(-1,R), axis=1)
    return y2


    
def baseline_envelope_iter_bspline(tx,sy,event_onsets,fs, fsd=10, lp=2, 
                                   lam_sig=1, lam_min=1, lam_max=100,
                                   verbose=0):
    """
    Extract baseline based (re-)estimating the lower envelope using B-splines.
    See notebook `baseline_interp.ipynb` for details.
    The signal is downsampled (to `fsd` Hz) for speed.
    
    Parameters
    -----------
    
    tx : np.ndarray
        time-vector in seconds
        
    sy : np.ndarray
        raw pupil signal
        
    event_onsets : list
        onsets of events (stimuli/responses) in milliseconds
        
    fs : float
        sampling rate in Hz
    
    fsd : float
        downsampled sampling rate (if too slow, decrease)
       
    lp : float
        lowpass-filter cutoff (Hz)
    
    lam_sig: float
        parameter steering how much the baseline is shaped by the non-peaks of the signal
    
    lam_min,lam_max: float
        parameters mapping how much low- and high-prominence peaks influence the baseline
        
    verbose: int [0, 100]
        how much information to print (0 nothing, 100 everything)
    
    Returns
    -------
    
    (txd,syd,base2,base1) : tuple
        txd: downsampled time-array
        syd: downsampled and lowpass-filtered pupil signal
        base1: is the estimated base after the first round
        base2: is the final baseline estimate
        
    """
    def vprint(v, s):
        if v<=verbose:
            print(">",s)

    dsfac=int(fs/fsd) # calculate downsampling factor
    vprint(100, "Downsampling factor is %i"%dsfac)
    
    # downsampling
    syc=butter_lowpass_filter(sy, lp, fs, order=2)
    syd=downsample(syc, dsfac)

    # z-scale for easier model-fitting
    symean,sysd=np.mean(syd),np.std(syd)
    syd=(syd-symean)/sysd
    txd=downsample(tx, dsfac)
    vprint(100, "Downsampling done")

    # peak finding and spline-building
    peaks_ix=signal.find_peaks(-syd)[0]
    prominences=signal.peak_prominences(-syd, peaks_ix)[0]
    peaks=txd[peaks_ix]
    vprint(100, "Peak-detection done, %i peaks detected"%peaks.shape[0])
    knots=np.concatenate( ([txd.min()], peaks, [txd.max()]) ) ## peaks as knots
    B=bspline(txd, knots, 3)
    vprint(100, "B-spline matrix built, dims=%s"%str(B.shape))
    

    # convert
    def prominence_to_lambda(w, lam_min=1, lam_max=100):
        w2=lam_min+((w-np.min(w))/(np.max(w-np.min(w))))*(lam_max-lam_min)
        return w2
    
    w=prominence_to_lambda(prominences, lam_min=lam_min, lam_max=lam_max)
    
    # load or compile model
    vprint(10, "Compiling Stan model")

    sm = StanModel_cache(stan_code_baseline_model_asym_laplac)
    
    ## put the data for the model together
    data={
        'n':syd.shape[0],
        'sy':syd,
        'ncol':B.shape[1],
        'B':B,
        'npeaks':peaks_ix.shape[0],
        'peakix':peaks_ix,
        'lam_sig':lam_sig,
        'pa':0.05,
        'lam_prominences':w
    }
    
    ## variational optimization
    vprint(10, "Optimizing Stan model")
    opt=sm.vb(data=data)
    vbc=opt["mean_pars"]
    meansigvb=np.dot(B, vbc)
    vprint(10, "Done optimizing Stan model")
    
    ## PRF model
    # new "signal"
    syd2=syd-meansigvb

    vprint(10, "Estimating PRF model (NNLS)")    
    #coef,pred,resid=pupilresponse_nnls(txd,syd2,event_onsets,fs=fsd)
    pred, coef, _, _, _=pupil_response(txd, syd2, event_onsets, fsd, npar=10, tmax=917)
    resid=syd-pred
    vprint(10, "Done Estimating PRF model (NNLS)")
    
    ### 2nd iteration
    ## get new peaks
    syd3=syd-pred
    peaks2_ix=signal.find_peaks(-syd3)[0]
    prominences2=signal.peak_prominences(-syd3, peaks2_ix)[0]
    peaks2=txd[peaks2_ix]
    vprint(100, "2nd Peak-detection done, %i peaks detected"%peaks2.shape[0])

    knots2=np.concatenate( ([txd.min()], peaks2, [txd.max()]) ) ## peaks as knots
    B2=bspline(txd, knots2, 3)
    vprint(100, "2nd B-spline matrix built, dims=%s"%str(B2.shape))

    w2=prominence_to_lambda(prominences2, lam_min=lam_min, lam_max=lam_max)

    data2={
        'n':syd3.shape[0],
        'sy':syd3,
        'ncol':B2.shape[1],
        'B':B2,
        'npeaks':peaks2_ix.shape[0],
        'peakix':peaks2_ix,
        'lam_sig':lam_sig,
        'pa':0.05,
        'lam_prominences':w2
    }
    
    ##  variational optimization
    vprint(10, "2nd Optimizing Stan model")
    opt=sm.vb(data=data2)
    vbc2=opt["mean_pars"]
    meansigvb2=np.dot(B2, vbc2)  
    vprint(10, "Done 2nd Optimizing Stan model")
    
    return txd,(syd*sysd)+symean,(meansigvb2*sysd)+symean, (meansigvb*sysd)+symean

def baseline_envelope(tx,sy,event_onsets, fs=1000, lp=2, prominence_thr=80, interp_method="cubic"):
    """
    Extract baseline based on the lower envelope of the (filtered) signal.

    Steps: 
    
    - filter away noise
    - detect high-prominence throughs in the signal 
    - calculate lower envelope based on these peaks
    
    Parameters
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
                   
    Returns
    --------
    
    base: np.array
        baseline estimate
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
    
    Parameters
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
        
    Returns
    --------
    
    base: np.array
        baseline estimate        
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
