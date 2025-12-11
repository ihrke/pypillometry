"""
preproc.py
==========

preprocessing functions (blinks, smoothing, missing data...)
21.10.2025 (Josephine) Added fix to detect_blinks_velocity() line 123
"""
import numpy as np
import scipy.optimize
import pylab as plt
from loguru import logger

from ..convenience import *


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


def backward_smooth(x, window_len=11):
    """
    Smooth signal using only past and current samples (backward-looking).
    
    This is useful for detecting signal changes (like blink onsets) right before
    missing data, where you don't want future NaN values to affect the estimate.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    window_len : int
        Size of the smoothing window (uses current sample and window_len-1 past samples)
        
    Returns
    -------
    np.ndarray
        Smoothed signal (same length as input)
        
    Notes
    -----
    Uses uniform (moving average) smoothing. The window includes the current sample
    and looks backward, so the smoothed value at index i depends only on x[i-window_len+1:i+1].
    
    See Also
    --------
    forward_smooth : Smooth using only current and future samples
    smooth_window : Symmetric smoothing (uses both past and future)
    """
    from scipy.ndimage import uniform_filter1d
    
    if x.size < window_len:
        return x.copy()
    
    # origin > 0 shifts window backward (uses past samples)
    # origin = window_len//2 makes it fully backward-looking
    return uniform_filter1d(x.astype(float), window_len, mode='nearest', origin=window_len//2)


def forward_smooth(x, window_len=11):
    """
    Smooth signal using only current and future samples (forward-looking).
    
    This is useful for detecting signal changes (like blink offsets) right after
    missing data, where you don't want past NaN values to affect the estimate.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    window_len : int
        Size of the smoothing window (uses current sample and window_len-1 future samples)
        
    Returns
    -------
    np.ndarray
        Smoothed signal (same length as input)
        
    Notes
    -----
    Uses uniform (moving average) smoothing. The window includes the current sample
    and looks forward, so the smoothed value at index i depends only on x[i:i+window_len].
    
    See Also
    --------
    backward_smooth : Smooth using only past and current samples
    smooth_window : Symmetric smoothing (uses both past and future)
    """
    from scipy.ndimage import uniform_filter1d
    
    if x.size < window_len:
        return x.copy()
    
    # origin < 0 shifts window forward (uses future samples)
    # origin = -window_len//2 makes it fully forward-looking
    return uniform_filter1d(x.astype(float), window_len, mode='nearest', origin=-window_len//2)


def helper_merge_blinks(b1,b2):
    if b1.size==0:
        return b2
    elif b2.size==0:
        return b1
    on=np.sort(np.concatenate( (b1[:,0], b2[:,0]) ))
    off=np.sort(np.concatenate( (b1[:,1], b2[:,1]) ))
    b=np.vstack((on,off)).T

    newb=[]
    on,off=b[0,:]
    for i in range(1,b.shape[0]):
        if b[i,0]<=off:
            # absorb onset from next 
            off=max(off,b[i,1])
        else:
            newb.append([on,off])
            on,off=b[i,:]
    off=b[-1,1]
    newb.append([on,off])
    return np.array(newb)

def detect_blinks_velocity(sy, smooth_winsize, vel_onset, vel_offset, min_onset_len=5, min_offset_len=5):
    """
    Detect blinks as everything between a fast downward and a fast upward-trending PD-changes.
    
    Uses asymmetric (one-sided) smoothing to handle NaN values properly:
    - Backward smoothing for onset detection (so future NaN doesn't affect the estimate)
    - Forward smoothing for offset detection (so past NaN doesn't affect the estimate)
    
    This works similarly to :py:func:`blink_onsets_mahot()`.
    
    Parameters
    ----------
    sy: np.array
        pupil data (can contain NaN for missing values)
    smooth_winsize: int (odd)
        size of the smoothing window in sampling points
    vel_onset: float
        velocity-threshold to detect the onset of the blink (negative value)
    vel_offset: float
        velocity-threshold to detect the offset of the blink (positive value)
    min_onset_len: int
        minimum number of consecutive samples that cross the threshold to detect onset
    min_offset_len: int
        minimum number of consecutive samples that cross the threshold to detect offset
        
    Returns
    -------
    np.array (nblinks x 2)
        Array of blink onset/offset index pairs
    """
    n = len(sy)
    
    # Track NaN/zero locations (invalid data)
    invalid_mask = np.isnan(sy) | (sy == 0)
    
    # Fill invalid values with linear interpolation for smoothing
    # (the asymmetric smoothing will prevent NaN from affecting edges)
    sy_filled = sy.copy()
    valid_indices = np.where(~invalid_mask)[0]
        
    # Simple linear interpolation for NaN regions
    invalid_indices = np.where(invalid_mask)[0]
    if len(invalid_indices) > 0:
        sy_filled[invalid_indices] = np.interp(
            invalid_indices, valid_indices, sy[valid_indices]
        )
    
    # Generate velocity profiles using asymmetric smoothing
    # Backward smooth for onset detection (only uses past samples)
    sym_backward = backward_smooth(sy_filled, smooth_winsize)
    vel_backward = np.r_[0, np.diff(sym_backward)]
    
    # Forward smooth for offset detection (only uses future samples)  
    sym_forward = forward_smooth(sy_filled, smooth_winsize)
    vel_forward = np.r_[np.diff(sym_forward), 0]
    
    # Don't detect inside invalid regions
    vel_backward[invalid_mask] = 0
    vel_forward[invalid_mask] = 0
    
    logger.debug(f"Generated asymmetric velocity profiles, length {n}")

    # Find onset candidates using backward velocity (fast drop before blink)
    onsets = np.where(vel_backward <= vel_onset)[0]
    logger.debug(f"Found {len(onsets)} potential onset points")
    
    if len(onsets) == 0:
        logger.debug("No onsets found - returning empty array")
        return np.array([])
        
    onsets_ixx = np.r_[np.diff(onsets), 10] > 1
    onsets_len = np.diff(np.r_[0, np.where(onsets_ixx)[0] + 1])
    onsets = onsets[:-1][onsets_ixx[:-1]]
    onsets_len = onsets_len[:len(onsets)]
    onsets = onsets[onsets_len > min_onset_len]
    logger.debug(f"After filtering, {len(onsets)} onsets remain")

    # Find offset candidates using forward velocity (fast rise after blink)
    offsets = np.where(vel_forward >= vel_offset)[0]
    logger.debug(f"Found {len(offsets)} potential offset points")
    
    if len(offsets) == 0:
        logger.debug("No offsets found - returning empty array")
        return np.array([])
        
    offsets_ixx = np.r_[10, np.diff(offsets)] > 1
    offsets_len = np.diff(np.r_[np.where(offsets_ixx)[0], offsets.size])
    offsets = offsets[offsets_ixx]
    offsets = offsets[offsets_len > min_offset_len]
    logger.debug(f"After filtering, {len(offsets)} offsets remain")
    
    if len(onsets) == 0 or len(offsets) == 0:
        logger.debug("No valid onset/offset pairs - returning empty array")
        return np.array([])
    
    # Find corresponding on- and off-sets
    blinks = []
    on = onsets[0]
    while on is not None:
        offs = offsets[offsets > on]
        off = offs[0] if offs.size > 0 else n
        blinks.append([on, off])
        ons = onsets[onsets > off]
        on = ons[0] if ons.size > 0 else None
    logger.debug(f"Found {len(blinks)} blink pairs")
        
    # If on/off-sets fall in an invalid region, grow until first valid sample
    blinks2 = []
    for (on, off) in blinks:
        while on > 0 and invalid_mask[on]:
            on -= 1
        while off < n - 1 and invalid_mask[off]:
            off += 1
        blinks2.append([on, off])
    
    result = np.array(blinks2)
    logger.debug(f"Returning {len(result)} blinks after invalid-region adjustment")
    return result

def detect_blinks_zero(sy, min_duration, blink_val=0):
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
    x=np.r_[0, np.diff((sy==blink_val).astype(int))]
    starts=np.where(x==1)[0]
    ends=np.where(x==-1)[0]-1
    
    logger.debug(f"Found {len(starts)} potential blink starts and {len(ends)} potential blink ends")
    
    if len(starts) == 0 or len(ends) == 0:
        logger.debug("No valid blinks found - returning empty array")
        return np.array([])
        
    if sy[0]==blink_val: ## first value missing?
        starts=np.r_[0,starts]    
        logger.debug("First value is missing - adding 0 to starts")
        
    if ends.size!=starts.size: 
        ## is the first start earlier than the first end?
        if starts[0]>ends[0]:
            ends=ends[1:] # drop first end
            logger.debug("First start after first end - dropping first end")
        else:
            starts=starts[:-1] # drop last start
            logger.debug("Last start without end - dropping last start")
            
    if len(ends) > 0 and ends[-1]==x.size:
        ends[-1]-=1
        logger.debug("Last end at signal boundary - adjusting index")
        
    blinks=[ [start,end] for start,end in zip(starts,ends) if end-start>=min_duration]
    logger.debug(f"Found {len(blinks)} blinks after duration filtering")
    
    return np.array(blinks)
    
def blink_onsets_mahot(sy, blinks, smooth_winsize, vel_onset, vel_offset, margin, blinkwindow):
    """
    Method for finding the on- and offset for each blink (excluding transient).
    See https://figshare.com/articles/A_simple_way_to_reconstruct_pupil_size_during_eye_blinks/688001.
    
    Parameters
    ----------
    sy: np.array
        pupil data
    blinks: np.array (nblinks x 2) 
        blink onset/offset matrix (contiguous zeros)
    smooth_winsize: int (odd)
        size of the Hanning-window in sampling points
    vel_onset: float
        velocity-threshold to detect the onset of the blink
    vel_offset: float
        velocity-threshold to detect the offset of the blink
    margin: tuple (int,int)
        margin that is subtracted/added to onset and offset (in sampling points)
    blinkwindow: int
        how much time before and after each blink to include (in sampling points)        
    """
    # generate smoothed signal and velocity-profile
    sym=smooth_window(sy, smooth_winsize, "hanning")
    vel=np.r_[0,np.diff(sym)] 
    blinkwindow_ix=blinkwindow
    n=sym.size
    
    newblinks=[]
    for ix,(start,end) in enumerate(blinks):                
        winstart,winend=max(0,start-blinkwindow_ix), min(end+blinkwindow_ix, n)
        slic=slice(winstart, winend) #start-blinkwindow_ix, end+blinkwindow_ix)
        winlength=vel[slic].size

        onsets=np.where(vel[slic]<=vel_onset)[0]
        offsets=np.where(vel[slic]>=vel_offset)[0]
        if onsets.size==0 or offsets.size==0:
            continue

        ## onsets are in "local" indices of the windows, start-end of blink global
        startl,endl=blinkwindow_ix if winstart>0 else start,end-start+blinkwindow_ix

        # find vel-crossing next to start of blink and move back to start of that crossing
        onset_ix=np.argmin(np.abs((onsets-startl<=0)*(onsets-startl)))
        while(onsets[onset_ix-1]+1==onsets[onset_ix]):
            onset_ix-=1
        onset=onsets[onset_ix]
        onset=max(0, onset-margin[0]) # avoid overflow to the left

        # find start of "reversal period" and move forward until it drops back
        offset_ix=np.argmin(np.abs(((offsets-endl<0)*np.iinfo(int).max)+(offsets-endl)))
        while(offset_ix<(len(offsets)-1) and offsets[offset_ix+1]-1==offsets[offset_ix]):
            offset_ix+=1        
        offset=offsets[offset_ix]
        offset=min(winlength-1, offset+margin[1]) # avoid overflow to the right
        newblinks.append( [onset+winstart,offset+winstart] )
    
    return np.array(newblinks)    
