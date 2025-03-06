"""
convenience.py
==============

Some convenience functions.
"""


import numpy as np
import pandas as pd

def get_example_data():
    ## loading the raw samples from the asc file
    from .eyedata import EyeData

    from importlib.resources import files
    fname_samples = files('pypillometry.data').joinpath('002_rlmw_samples_short.asc')
    fname_events = files('pypillometry.data').joinpath('002_rlmw_events_short.asc')
    df=pd.read_table(fname_samples, index_col=False,
                    names=["time", "left_x", "left_y", "left_p",
                            "right_x", "right_y", "right_p"])

    ## Eyelink tracker puts "   ." when no data is available for x/y coordinates
    left_x=df.left_x.values
    left_x[left_x=="   ."] = np.nan
    left_x = left_x.astype(float)

    left_y=df.left_y.values
    left_y[left_y=="   ."] = np.nan
    left_y = left_y.astype(float)

    right_x=df.right_x.values
    right_x[right_x=="   ."] = np.nan
    right_x = right_x.astype(float)

    right_y=df.right_y.values
    right_y[right_y=="   ."] = np.nan
    right_y = right_y.astype(float)

    ## Loading the events from the events file
    # read the whole file into variable `events` (list with one entry per line)
    with open(fname_events) as f:
        events=f.readlines()

    # keep only lines starting with "MSG"
    events=[ev for ev in events if ev.startswith("MSG")]
    experiment_start_index=np.where(["experiment_start" in ev for ev in events])[0][0]
    events=events[experiment_start_index+1:]
    df_ev=pd.DataFrame([ev.split() for ev in events])
    df_ev=df_ev[[1,2]]
    df_ev.columns=["time", "event"]

    # Creating EyeData object that contains both X-Y coordinates
    # and pupil data
    d = EyeData(time=df.time, name="test short",
                screen_resolution=(1280,1024), physical_screen_size=(33.75,27),
                screen_eye_distance=60,
                left_x=left_x, left_y=left_y, left_pupil=df.left_p,
                right_x=right_x, right_y=right_y, right_pupil=df.right_p,
                event_onsets=df_ev.time, event_labels=df_ev.event,
                keep_orig=True)\
                .reset_time()
    d.set_experiment_info(screen_eye_distance=60, 
                        screen_resolution=(1280,1024), 
                        physical_screen_size=(30, 20))
    return d
    

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
