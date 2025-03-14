"""
erpd.py
============

Event-related pupil dilation.

"""
from types import NoneType
from .io import *
from .eyedata import EyeDataDict
from loguru import logger
import pylab as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import numpy.ma as ma
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy import interpolate
import scipy



from typing import Iterable, Sequence, Union, List, TypeVar, Optional, Tuple, Callable
PupilArray=Union[np.ndarray, List[float]]
import collections.abc


class ERPD:
    """
    Class representing a event-related pupillary dilation (ERPD) for one subject.

    Parameters
    ----------
    name: str
        name of the ERPD (e.g., "cue-locked" or "conflict-trials")
    tx: np.ndarray
        time-axis (in ms)
    erpd: EyeDataDict
        dictionary containing the ERPD (2D-array with shape (nevents, ntimepoints))
    """
    def __init__(self, name: str, tx: np.array, erpd: EyeDataDict):
        self.name=name
        self.tx=tx
        self.erpd=erpd

    @property
    def n(self):
        return self.erpd.shape[0]

    @property
    def nevents(self):
        return self.erpd.shape[1]

    def summary(self) -> dict:
        """Return a summary of the :class:`.PupilData`-object."""
        summary=dict(
            name=self.name,
            eyes=self.erpd.get_available_eyes(),
            variables=self.erpd.get_available_variables(),
            nevents=self.nevents, 
            n=self.n, 
            window=(self.tx.min(), self.tx.max()),
            glimpse=repr(self.erpd)
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
                
    def baseline_correct(self, 
                         baseline_win: Tuple[float,float]|float|NoneType=0,
                         eyes: list|str=[]):
        """
        Apply baseline-correction to the ERPD.
        
        Parameters
        ----------
        
        eyes: list of str
            list of eyes to apply baseline-correction to
            if empty, all eyes are baseline-corrected
        
        baseline_win: tuple (float,float) or float (default 0) or None
            if None, no baseline-correction is applied
            if tuple, the mean value in the window in milliseconds (relative to `time_win`) is 
            subtracted from the single-trial ERPDs (baseline-correction)
            if float, the value where the time-array is closest to this value is used as baseline
        """
        if baseline_win is None:
            logger.warning("No baseline-correction applied")
            return           
        if isinstance(eyes, str):
            eyes=[eyes]        
        if isinstance(eyes, Iterable) and len(eyes)==0:
            eyes=self.erpd.get_available_eyes()
        
        for eye in eyes:
            if not isinstance(baseline_win, Iterable):
                blwin_ix=np.argmin(np.abs(self.tx-baseline_win))
            else:
                blwin_ix=tuple(( np.argmin(np.abs(bw-self.tx)) for bw in baseline_win ))
            logger.info("Baseline-correction for eye {eye} using window {blwin_ix}".format(
                eye=eye, blwin_ix=blwin_ix))
            for i in range(self.nevents):
                if isinstance(blwin_ix, Iterable) and len(blwin_ix)==2:
                    baseline = np.mean(self.erpd[eye,"erpd"][blwin_ix[0]:blwin_ix[1],i])
                else:
                    baseline = self.erpd[eye,"erpd"][blwin_ix,i]
                self.erpd[eye,"erpd"][:,i] -= baseline

        
    
    def plot(self, 
             eyes: str|list=[],
             overlays=None, meanfct=np.mean, varfct=scipy.stats.sem, 
             plot_missing: bool=True,
             title: str=None): 
        """
        Plot mean and error-ribbons using `varct`.
        
        Parameters
        ----------
        
        eyes: str or list of str
            list of eyes to plot
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
        title: str
            title of the plot or None (in which case the name of the ERPD is used as title)
        """
        if isinstance(eyes, str):
            eyes=[eyes]
        if len(eyes)==0:
            eyes=self.erpd.get_available_eyes()
        
        ax1=plt.gca()
        if plot_missing:
            ax2=ax1.twinx()
        for eye in eyes:
            erpd = ma.masked_array(self.erpd[eye,"erpd"], mask=self.erpd.mask[eye+"_erpd"])
            merpd=meanfct(erpd, axis=1)
            sderpd=varfct(erpd, axis=1) if callable(varfct) else None
            percmiss=np.mean(self.erpd.mask[eye+"_erpd"], axis=1)*100.
            if sderpd is not None:
                ax1.fill_between(self.tx, merpd-sderpd, merpd+sderpd, color="grey", alpha=0.3)
            ax1.plot(self.tx, merpd, label=eye)        
            ax1.axvline(x=0, color="red")        
            ax1.set_ylabel("mean PD")
            ax1.set_xlabel("time (ms)")
            if title is not None:
                ax1.set_title(title)
            else:
                ax1.set_title(self.name)
            if plot_missing:
                ax2.plot(self.tx, percmiss, alpha=0.3)
                ax2.set_ylim(0,100)
                ax2.set_ylabel("% missing")
        ax1.legend()
        

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
