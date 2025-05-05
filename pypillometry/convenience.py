"""
convenience.py
==============

Some convenience functions.
"""


import numpy as np
import pandas as pd
from typing import Union, Dict
from contextlib import contextmanager
import os

@contextmanager
def change_dir(path: Union[str]):
    """Temporarily change the current working directory.
    
    Parameters
    ----------
    path : str
        Directory to change to
        
    Examples
    --------
    >>> with change_dir("/path/to/dir"):
    ...     # do something in that directory
    ...     pass
    >>> # back in original directory
    """
    old_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_dir)

def mask_to_intervals(mask):
    """
    Convert a boolean mask to a list of intervals.
    
    Parameters
    ----------
    mask : np.ndarray
        boolean mask
    
    Returns
    -------
    list of tuples
        list of intervals
    """
    if not isinstance(mask, np.ndarray):
        mask=np.array(mask)
    if mask.ndim>1:
        raise ValueError("mask must be 1D")
    if mask.dtype not in (bool,int):
        raise ValueError("mask must be int or boolean")
    if mask.size==0:
        return []
    m=np.concatenate(([0], mask, [0]))
    idxs = np.flatnonzero(m[1:] != m[:-1])
    ivs = list(zip(idxs[::2], idxs[1::2]))
    ivs = [(max(start,0), min(end,mask.size-1)) 
           for start, end in ivs]
    return ivs



def nprange(ar):
    return (ar.min(),ar.max())

import pickle
from hashlib import md5

def zscale(y):
    return (y-np.nanmean(y))/np.nanstd(y)


def p_asym_laplac(y, mu, sigma, tau):
    """
    Asymmetric laplace distribution https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution;
    Parametrization as here: https://github.com/stan-dev/stan/issues/2312
    
    tau in [0,1]
    """
    I=np.array(y<=mu, dtype=int)
    return (2*tau*(1-tau))/sigma*np.exp(-2/sigma * ( (1-tau)*I*(mu-y) + tau*(1-I)*(y-mu) ) )

def p_asym_laplac_kappa(y, mu, lam, kappa):
    """
    Asymmetric laplace distribution https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution;
    Wikipedia parametrization.
    
    kappa in [0, infty] where 1 means symmetry
    """
    I=np.array(y<=mu, dtype=int)
    return (lam)/(kappa+1./kappa)*np.exp( ( (lam/kappa)*I*(y-mu) - lam*kappa*(1-I)*(y-mu) ) )


def trans_logistic_vec(x, a, b, inverse=False):
    """
    vectorized version of trans_logistic()
    
    goes from [a,b] to [-inf,+inf] and back;
    inverse=False: [a,b] -> [-inf, +inf]
    inverse=True:  [-inf,+inf] -> [a,b]
    if a or b is +/-infty, a logarithmic/exponential transform is used
    """
    eps=1e-15
    if inverse==False:
        # variables from [a,inf]
        x=np.where( (a>-np.inf) & (b==np.inf), np.log(np.maximum(x-a, eps)), x)
        # variables from [-inf, b]
        x=np.where( (a==-np.inf) & (b<np.inf), np.log(np.maximum(b-x, eps)), x)
        # variables from [a, b]
        x=np.where( (a>-np.inf) & (b<np.inf), -np.log( (b-a)/(x-a)-1 ), x)
    elif inverse==True:
        # variables from [-inf,inf] -> [a,inf]
        x=np.where( (a>-np.inf) & (b==np.inf), np.exp(x)+a, x)
        # variables from [-inf, inf] -> [-inf, b]
        x=np.where( (a==-np.inf) & (b<np.inf), b-np.exp(x), x)
        # variables from [-inf,inf] -> [a, b]
        x=np.where( (a>-np.inf) & (b<np.inf), (1./(1.+np.exp(-x)))*(b-a)+a, x)
    
    return x

def plot_pupil_ipy(tx, sy, event_onsets=None, overlays=None, overlay_labels=None, 
                   blinks=None, interpolated=None,
                   figsize=(16,8), xlab="ms", nsteps=100):
    """
    Plotting with interactive adjustment of plotting window.
    To use this, do

    $ pip install ipywidgets
    $ jupyter nbextension enable --py widgetsnbextension
    $ jupyter labextension install @jupyter-widgets/jupyterlab-manager

    Parameters
    ----------
    
    tx : np.ndarray
        time-vector in seconds    
    sy : np.ndarray
        raw pupil signal        
    event_onsets : list
        onsets of events (stimuli/responses) in seconds
    overlays: tuple of np.array
        signals to overlay over the plot, given as tuple of arrays of same length as `tx`
    overlay_labels: tuple of strings
        labels for the overlays to be displayed in the legend
    figsize: tuple of int
        dimensions for the plot
    xlab: str
        label for x-axis
    nsteps: int
        number of steps for slider
    """
    import pylab as plt
    from ipywidgets import interact, interactive, fixed, interact_manual, Layout
    import ipywidgets as widgets

    def draw_plot(plotxrange):
        xmin,xmax=plotxrange
        ixmin=np.argmin(np.abs(tx-xmin))
        ixmax=np.argmin(np.abs(tx-xmax))
        plt.figure(figsize=figsize)

        plt.plot(tx[ixmin:ixmax],sy[ixmin:ixmax], label="signal")
        if overlays is not None:
            if type(overlays) is np.ndarray:
                plt.plot(tx[ixmin:ixmax],overlays[ixmin:ixmax],label=overlay_labels)
            else:
                for i,overlay in enumerate(overlays):
                    lab=overlay_labels[i] if overlay_labels is not None else None
                    plt.plot(tx[ixmin:ixmax],overlay[ixmin:ixmax], label=lab)
        for istart,iend in interpolated:
            plt.gca().axvspan(tx[istart],tx[iend],color="green", alpha=0.1)
        for istart,iend in blinks:
            plt.gca().axvspan(tx[istart],tx[iend],color="red", alpha=0.1)

        plt.vlines(event_onsets, *plt.ylim(), color="grey", alpha=0.5)
        plt.xlim(xmin,xmax)
        plt.xlabel(xlab)
        if overlay_labels is not None:
            plt.legend()


    wid_range=widgets.FloatRangeSlider(
        value=[tx.min(), tx.max()],
        min=tx.min(),
        max=tx.max(),
        step=(tx.max()-tx.min())/nsteps,
        description=' ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=Layout(width='100%', height='80px')
    )

    interact(draw_plot, plotxrange=wid_range)
    

def sizeof_fmt(num, suffix='B'):
    """
    Convert number of bytes in `num` into human-readable string representation.
    Taken from https://stackoverflow.com/a/1094933
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

class ByteSize(int):
    """Class to represent size in bytes with human-readable formatting."""
    def __new__(cls, bytes: Union[int, Dict[str, int]]):
        if isinstance(bytes, dict):
            total_bytes = bytes['memory'] + bytes['disk']
            obj = super().__new__(cls, total_bytes)
            obj.cached_bytes = bytes['disk']
        else:
            obj = super().__new__(cls, bytes)
            obj.cached_bytes = 0
        return obj
            
    def __str__(self):
        if self.cached_bytes > 0:
            return f"{sizeof_fmt(int(self) - self.cached_bytes)} (+{sizeof_fmt(self.cached_bytes)} cached)"
        else:
            return sizeof_fmt(int(self))
        
    def is_cached(self):
        return self.cached_bytes > 0
            
    def __repr__(self):
        return self.__str__()

def test_selector(obj, selfun, **kwargs):
    """
    Function to help test the output of a selector function.

    Parameters
    -----------
    obj: GenericEyeData object
        object to test the selector function on
    selfun: function
        function that takes a single event label as input and returns True/False
        to indicate whether the event should be included or not
    kwargs: dict
        arguments to pass onto the selector function

    Examples
    --------
    # here, events are coded as F_1, S_1, R_1, F_2, S_2, R_2, ...
    # to indicate fixation cross, stimulus, response for each trial
    # we build selector functions to extract all fixation crosses etc
    def fix_selector(evlab):
        return evlab.startswith("F")
    def stim_selector(evlab):
        return evlab.startswith("S")
    obj.test_selector(fix_selector)

    # we can do use kwargs to pass on arguments to the selector function
    # for example, we can restrict to only a single trial
    def fix_selector(evlab, trial=None):
        return evlag.startswith("F") and evlab.endswith("_{}".format(trial))
    obj.test_selector(fix_selector, trial=1)
    """
    if not callable(selfun):
        raise ValueError("selfun must be a function")
    return obj.event_labels[np.array([selfun(evlab, **kwargs) for evlab in self.event_labels])]
