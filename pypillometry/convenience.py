"""
convenience.py
==============

Some convenience functions.
"""


import numpy as np
import pandas as pd
from typing import Union, Dict
from contextlib import contextmanager
import os, sys
from tqdm import tqdm

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


@contextmanager
def suppress_all_output():
    """Suppress all output including C library output."""
    # Save original file descriptors
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    
    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    
    try:
        # Redirect file descriptors
        os.dup2(devnull_fd, 1)  # stdout
        os.dup2(devnull_fd, 2)  # stderr
        sys.stdout.flush()
        sys.stderr.flush()
        yield
    finally:
        # Restore original file descriptors
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        
        # Close file descriptors
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)
        os.close(devnull_fd)

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

def is_url(url):
    """Check if a string is a URL.
    
    Parameters
    ----------
    url : str
        URL to check
        
    Returns
    -------
    bool
        True if the string is a URL, False otherwise
    """
    from urllib.parse import urlparse
    if not isinstance(url, str):
        url = str(url)
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    return True



## Decorator to check for optional dependencies
import functools
from typing import Callable, Any

def requires_package(package_name: str, install_hint: str = None):
    """
    Decorator to mark functions that require optional dependencies.
    
    Parameters
    ----------
    package_name : str
        Name of the required package
    install_hint : str, optional
        Custom installation instruction
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                __import__(package_name)
            except ImportError:
                hint = install_hint or f"pip install {package_name}"
                raise ImportError(
                    f"Function '{func.__name__}' requires the '{package_name}' package. "
                    f"Install it with: {hint}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Unit handling utilities
UNIT_ALIASES = {
    # milliseconds
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    # seconds
    "s": "sec",
    "sec": "sec",
    "secs": "sec",
    "second": "sec",
    "seconds": "sec",
    # minutes
    "m": "min",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    # hours
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    # indices (samples)
    "index": None,
    "indices": None,
    "idx": None,
    "sample": None,
    "samples": None,
}

CANONICAL_UNITS = {"ms", "sec", "min", "h"}


def normalize_unit(unit: Union[str, None]) -> Union[str, None]:
    """
    Normalize a time unit string to its canonical form.
    
    Accepts various aliases (e.g., "seconds", "s", "secs") and returns the
    canonical unit name ("sec"). Returns None if input is None.

    Parameters
    ----------
    unit : str or None
        Unit string to normalize (e.g., "seconds", "ms", "h"). If None, returns None.
    
    Returns
    -------
    str or None
        Canonical unit name ("ms", "sec", "min", "h") or None if input is None
    
    Raises
    ------
    ValueError
        If unit string is not recognized
    
    Examples
    --------
    >>> normalize_unit("seconds")
    'sec'
    >>> normalize_unit("ms")
    'ms'
    >>> normalize_unit("hours")
    'h'
    >>> normalize_unit(None)
    None
    >>> normalize_unit("invalid")
    ValueError: Unknown unit 'invalid'. Valid units are: ms, sec, min, h
    """
    # Handle None case
    if unit is None:
        return None
    
    # Normalize: lowercase and strip whitespace
    unit_normalized = str(unit).lower().strip()
    
    # Look up in aliases
    if unit_normalized in UNIT_ALIASES:
        return UNIT_ALIASES[unit_normalized]
    
    # Not found
    valid_units_str = ", ".join(sorted(CANONICAL_UNITS))
    raise ValueError(f"Unknown unit '{unit}'. Valid units are: {valid_units_str}")


