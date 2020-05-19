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


def detect_blinks_velocity(sy, smooth_winsize, vel_onset, vel_offset, min_onset_len=5, min_offset_len=5):
    """
    Detect blinks as everything between a fast downward and a fast upward-trending PD-changes.
    
    This works similarly to :py:func:`blink_onsets_mahot()`.
    
    Parameters
    ----------
    sy: np.array
        pupil data
    smooth_winsize: int (odd)
        size of the Hanning-window in sampling points
    vel_onset: float
        velocity-threshold to detect the onset of the blink
    vel_offset: float
        velocity-threshold to detect the offset of the blink
    min_onset_len: int
        minimum number of consecutive samples that cross the threshold to detect onset
    min_offset_len: int
        minimum number of consecutive samples that cross the threshold to detect offset
    """
    # generate smoothed signal and velocity-profile
    sym=smooth_window(sy, smooth_winsize, "hanning")
    vel=np.r_[0,np.diff(sym)] 
    n=sym.size

    # find first negative vel-crossing 
    onsets=np.where(vel<=vel_onset)[0]
    onsets_ixx=np.r_[np.diff(onsets),10]>1
    onsets_len=np.diff(np.r_[0,np.where(onsets_ixx)[0]])
    onsets=onsets[onsets_ixx]
    onsets=onsets[onsets_len>min_onset_len]

    ## offset finding
    offsets=np.where(vel>=vel_offset)[0]
    offsets_ixx=np.r_[10,np.diff(offsets)]>1
    offsets_len=np.diff(np.r_[np.where(offsets_ixx)[0],offsets.size])
    offsets=offsets[offsets_ixx]
    offsets=offsets[offsets_len>min_offset_len]
    
    
    ## find corresponding on- and off-sets
    blinks=[]
    on=onsets[0]
    while on is not None:
        offs=offsets[offsets>on]
        off=offs[0] if offs.size>0 else n
        blinks.append([on,off])
        ons=onsets[onsets>off]
        on=ons[0] if ons.size>0 else None
        
    ## if on- off-sets fall in a zero-region, grow until first non-zero sample
    blinks2=[]
    for (on,off) in blinks:
        while(on>0 and sy[on]==0):
            on-=1
        while(off<n-1 and sy[off]==0):
            off+=1
        blinks2.append([on,off])
    return np.array(blinks2)

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
    x=np.r_[0, np.diff((sy==blink_val).astype(np.int))]
    starts=np.where(x==1)[0]
    ends=np.where(x==-1)[0]-1
    if ends.size!=starts.size: 
        ## is the first start earlier than the first end?
        if starts[0]>ends[0]:
            ends=ends[1:] # drop first end
        else:
            starts=starts[:-1] # drop last start
    if ends[-1]==x.size:
        ends[-1]-=1
    blinks=[ [start,end] for start,end in zip(starts,ends) if end-start>=min_duration]
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
        offset_ix=np.argmin(np.abs(((offsets-endl<0)*np.iinfo(np.int).max)+(offsets-endl)))
        while(offset_ix<(len(offsets)-1) and offsets[offset_ix+1]-1==offsets[offset_ix]):
            offset_ix+=1        
        offset=offsets[offset_ix]
        offset=min(winlength-1, offset+margin[1]) # avoid overflow to the right
        newblinks.append( [onset+winstart,offset+winstart] )
    
    return np.array(newblinks)    