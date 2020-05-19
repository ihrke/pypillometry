"""
erpd.py
============

Event-related pupil dilation.

"""
from .io import *

import pylab as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy import interpolate
import scipy


from typing import Sequence, Union, List, TypeVar, Optional, Tuple, Callable
PupilArray=Union[np.ndarray, List[float]]
import collections.abc


class ERPD:
    """
    Class representing a event-related pupillary dilation (ERPD) for one subject.
    """
    def __init__(self, name, tx, erpd, missing, baselines):
        self.name=name
        self.baselines=baselines
        self.tx=tx
        self.erpd=erpd
        self.missing=missing

    def summary(self) -> dict:
        """Return a summary of the :class:`.PupilData`-object."""
        summary=dict(
            name=self.name,
            nevents=self.erpd.shape[0],
            n=self.erpd.shape[1],
            window=(self.tx.min(), self.tx.max())
        )
        return summary

    def write_file(self, fname:str):
        """
        Save to file (using :mod:`pickle`).
        
        Parameters
        ----------
        
        fname: str
            filename
        """
        pd_write_pickle(self, fname)
       
    @classmethod
    def from_file(cls, fname:str):
        """
        Reads a :class:`.ERPD` object from a pickle-file.
        Use as ``pypillometry.ERPD.from_file("yourfile.pd")``.
        
        Parameters
        ----------
        
        fname: str
            filename
        """
        r=pd_read_pickle(fname)
        return r
    
    def __repr__(self) -> str:
        """Return a string-representation of the dataset."""
        pars=self.summary()
        del pars["name"]
        s="ERPD({name}):\n".format(name=self.name)
        flen=max([len(k) for k in pars.keys()])
        for k,v in pars.items():
            s+=(" {k:<"+str(flen)+"}: {v}\n").format(k=k,v=v)
        return s
                
    def plot(self, overlays=None, meanfct=np.mean, varfct=scipy.stats.sem, plot_missing: bool=True): 
        """
        Plot mean and error-ribbons using `varct`.
        
        Parameters
        ----------
        
        overlays: single or sequence of :class:`.ERPDSingleSubject`-objects 
            the overlays will be added to the same plot
        
        meanfct: callable
            mean-function to apply to the single-trial ERPDs for plotting
        varfct: callable or None
            function to calculate error-bands (e.g., :func:`numpy.std` for standard-deviation 
            or :func:`scipy.stats.sem` for standard-error)
            if None, no error bands are plotted
            
        plot_missing: bool
            plot percentage interpolated/missing data per time-point?
        """
        merpd=meanfct(self.erpd, axis=0)
        sderpd=varfct(self.erpd, axis=0) if callable(varfct) else None
        percmiss=np.mean(self.missing, axis=0)*100.
        ax1=plt.gca()        
        if sderpd is not None:
            ax1.fill_between(self.tx, merpd-sderpd, merpd+sderpd, color="grey", alpha=0.3)
        ax1.plot(self.tx, merpd, label=self.name)        
        ax1.axvline(x=0, color="red")        
        ax1.set_ylabel("mean PD")
        ax1.set_xlabel("time (ms)")
        ax1.set_title(self.name)
        if plot_missing:
            ax2=ax1.twinx()
            ax2.plot(self.tx, percmiss, alpha=0.3)
            ax2.set_ylim(0,100)
            ax2.set_ylabel("% missing")
            
        if overlays is not None:
            if not isinstance(overlays, collections.abc.Sequence):
                overlays=[overlays]
            for ov in overlays:
                merpd=meanfct(ov.erpd, axis=0)
                sderpd=varfct(ov.erpd, axis=0) if callable(varfct) else None
                percmiss=np.mean(ov.missing, axis=0)*100.
                if sderpd is not None:
                    ax1.fill_between(self.tx, merpd-sderpd, merpd+sderpd, color="grey", alpha=0.3)
                ax1.plot(self.tx, merpd, label=ov.name)        
                if plot_missing:
                    ax2.plot(ov.tx, percmiss, alpha=0.3)
        ax1.legend()
        

def plot_erpds(erpds):
    """
    Plot a list of ERPD objects.
    """
    erpds[0].plot(erpds[1:len(erpds)])
    

def group_erpd(datasets: List, erpd_name: str, event_select, 
               baseline_win: Optional[Tuple[float,float]]=None, 
               time_win: Tuple[float,float]=(-500, 2000),
               subj_meanfct=np.mean):
    """
    Calculate group-level ERPDs by applying `subj_meanfct` to each subj-level ERPD.
    

    Parameters
    ----------
    datasets: list of PupilData objects
        one PupilData object for each subject that should go into the group-level ERPD.
        
    erpd_name: str
        identifier for the result (e.g., "cue-locked" or "conflict-trials")

    baseline_win: tuple (float,float) or None
        if None, no baseline-correction is applied
        if tuple, the mean value in the window in milliseconds (relative to `time_win`) is 
            subtracted from the single-trial ERPDs (baseline-correction)

    event_select: str or function
        variable describing which events to select and align to
        - if str: use all events whose label contains the string
        - if function: apply function to all labels, use those where the function returns True

    time_win: Tuple[float, float]
        time before and after event to include (in ms)
        
    subj_meanfct: fct
        function to summarize each individual ERPD
        
    Returns
    -------

    an :py:class:`ERPD` object for the group
    """        
    erpds=[d.get_erpd(erpd_name, event_select, baseline_win, time_win) for d in datasets]
    merpd=np.array([subj_meanfct(e.erpd, axis=0) for e in erpds])
    mmissing=np.array([subj_meanfct(e.missing, axis=0) for e in erpds])
    tx=erpds[0].tx
    erpd=ERPD(erpd_name, tx, merpd, mmissing, None)    
    return erpd
