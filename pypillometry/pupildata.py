"""
pupildata.py
============

Main object-oriented entry point
"""

from .convenience import *
from .baseline import *
from .fakedata import *
from .preproc import *
from .io import *
from .erpd import *

import pylab as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy import interpolate
import scipy
from random import choice
import pickle

import collections.abc

import copy
import math

#from pytypes import typechecked
from typing import Sequence, Union, List, TypeVar, Optional, Tuple, Callable
PupilArray=Union[np.ndarray, List[float]]


_inplace=False ## default for whether or not inplace-operations should be used


import inspect
import functools

## decoratory to keep a history of actions performed on dataset
# can only be used with functions that return "self" 
def keephistory(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        obj=func(*args,**kwargs)
        funcname=func.__name__
        argstr=",".join(["%s"%(v) for v in args[1:]])
        kwargstr=",".join(["%s=%s"%(k,v) for k,v in kwargs.items()])
        allargs=argstr
        if len(allargs)>0 and len(kwargstr)>0:
            allargs+=","+kwargstr
        elif len(kwargstr)>0:
            allargs+=kwargstr
        fstr="{func}({allargs})".format(func=funcname, allargs=allargs)
        #fstr="{func}({argstr},{kwargstr})".format(func=funcname,argstr=argstr,kwargstr=kwargstr)
        obj.add_to_history({"funcstring":fstr, "funcname":funcname, "args":args[1:], "kwargs":kwargs})
        return obj
    return wrapper
        
             

#@typechecked
class PupilData:
    """
    Class representing pupillometric data. 
    """
    
    def add_to_history(self, event):
        """Add event to history"""
        try:
            self.history.append(event)
        except:
            self.history=[event]
            
    def print_history(self):
        """
        Pretty-print the history of the current dataset (which manipulations where done on it).
        """
        print("* "+self.name)
        try:
            for i,ev in enumerate(self.history):
                print(" "*(i)+"└ " + ev["funcstring"])
        except:
            print("no history")
   
    def apply_history(self, obj):
        """
        Apply history of operations done on `self` to `obj`.
        
        Parameters:
        -----------
        
        obj: :class:`.PupilData`
            object of class :class:`.PupilData` to which the operations are to be transferred
            
        Returns:
        --------
        
        copy of the :class:`.PupilData`-object to which the operations in `self` were applied
        """
        for ev in self.history:
            obj=getattr(obj, ev["funcname"])(*ev["args"], **ev["kwargs"])
        return obj

    def __len__(self) -> int:
        """Return number of sampling points in the pupil data."""
        return self.sy.size
    
    def nevents(self) -> int:
        """Return number of events in pupillometric data."""
        return self.event_onsets.size

    def nblinks(self) -> int:
        """
        Return number of detected blinks. Should be run after `detect_blinks()`.
        """
        return self.blinks.shape[0]
    
    def get_duration(self, units="min"):
        fac=self._unit_fac(units)
        return (len(self)/self.fs*1000)*fac
    
    def _random_id(self, n:int=8) -> str:
        """
        Create a random ID string that is easy to recognise.
        Based on <http://code.activestate.com/recipes/526619-friendly-readable-id-strings/>.
        """
        v = 'aeiou'
        c = 'bdfghklmnprstvw'

        return ''.join([choice(v if i%2 else c) for i in range(n)])
    
    def __init__(self,
                 pupil: PupilArray, 
                 sampling_rate: Optional[float]=None,
                 time: Optional[PupilArray]=None,
                 event_onsets: Optional[PupilArray]=None,
                 event_labels: Optional[PupilArray]=None,
                 name: Optional[str]=None,
                 keep_orig: bool=True,
                 fill_time_discontinuities: bool=True):
        """
        Parameters
        ----------
        
        name: 
            name of the dataset or `None` (in which case a random string is selected)
        time: 
            timing array or `None`, in which case the time-array goes from [0,maxT]
            using `sampling_rate` (in ms)
        pupil:
            pupillary data at times `time` assumed to be in ms
        event_onsets: 
            time-onsets of any events that are to be modelled in the pupil
        event_labels:
            for each event in `event_onsets`, provide a label
        sampling_rate: float
            sampling-rate of the pupillary signal in Hz
        keep_orig: bool
            keep a copy of the original dataset? If `True`, a copy of the :class:`.PupilData` object
            as initiated in the constructor is stored in member `PupilData.original`
        fill_time_discontinuities: bool
            sometimes, when the eyetracker loses signal, no entry in the EDF is made; 
            when this option is True, such entries will be made and the signal set to 0 there
        """
        self.sy=np.array(pupil, dtype=np.float)
        if sampling_rate is None and time is None:
            raise ValueError("you have to specify either sampling_rate or time-vector (or both)")
        
        if time is None:
            maxT=len(self)/sampling_rate*1000.
            self.tx=np.linspace(0,maxT, num=len(self))
        else:
            self.tx=np.array(time, dtype=np.float)
        
        if sampling_rate is None:
            self.fs=np.round(1000./np.median(np.diff(self.tx)))
        else:
            self.fs=sampling_rate
            
        if fill_time_discontinuities:
            ## find gaps in the time-vector
            tx=self.tx
            sy=self.sy
            stepsize=np.median(np.diff(tx))
            n=tx.size
            gaps_end_ix=np.where(np.r_[stepsize,np.diff(tx)]>2*stepsize)[0]
            ngaps=gaps_end_ix.size
            if ngaps!=0:
                ## at least one gap here
                print("> Filling in %i gaps"%ngaps)
                gaps_start_ix=gaps_end_ix-1
                print( ((tx[gaps_end_ix]-tx[gaps_start_ix])/1000), "seconds" )
                
                ntx=[tx[0:gaps_start_ix[0]]] # initial
                nsy=[sy[0:gaps_start_ix[0]]]
                for i in range(ngaps):
                    start,end=gaps_start_ix[i], gaps_end_ix[i]
                    # fill in the gap
                    ntx.append( np.linspace(tx[start],tx[end], int((tx[end]-tx[start])/stepsize), endpoint=False) )
                    nsy.append( np.zeros(ntx[-1].size) )

                    # append valid signal
                    if i==ngaps-1:
                        nstart=n
                    else:
                        nstart=gaps_start_ix[i+1]
                    ntx.append( tx[end:nstart] )
                    nsy.append( sy[end:nstart] )

                ntx=np.concatenate(ntx)
                nsy=np.concatenate(nsy) 
                self.tx=ntx
                self.sy=nsy
            
        if event_onsets is None:
            self.event_onsets=np.array([], dtype=np.float)
        else:
            self.event_onsets=np.array(event_onsets, dtype=np.float)
        
        # check whether onsets are in range
        for onset in self.event_onsets:
            if onset<self.tx.min() or onset>self.tx.max():
                raise ValueError("some event-onsets outside data range according to time-vector")
            
            
        if event_labels is None:
            self.event_labels=np.zeros_like(self.event_onsets)
        else:
            if self.event_onsets.size!=np.array(event_labels).size:
                raise ValueError("event_labels must have same length as event_onsets")
            self.event_labels=np.array(event_labels)
          
        if self.tx.size != self.sy.size:
            raise ValueError("time and pupil-array must have same length, found {} vs {}".format(
                self.tx.size,self.sy.size))
        
        if name is None:
            self.name = self._random_id()
        else:
            self.name=name
        
        ## initialize baseline signal
        self.scale_params={"mean":0, "sd":1}
        self.baseline=np.zeros(len(self))
        self.baseline_estimated=False
        
        ## initialize response-signal
        self.response=np.zeros(len(self))
        self.response_pars=None
        self.response_estimated=False
        
        ## initialize blinks
        self.blinks=np.empty((0,2), dtype=np.int)
        self.blink_mask=np.zeros(len(self), dtype=np.int)
        
        ## interpolated mask
        self.interpolated_mask=np.zeros(len(self), dtype=np.int)
        self.missing=np.zeros(len(self), dtype=np.int)
        self.missing[self.sy==0]=1
        
        self.original=None
        if keep_orig: 
            self.original=self.copy()
            
        ## start with empty history    
        self.history=[]
       
    @keephistory
    def drop_original(self, inplace=_inplace):
        """
        Drop original dataset from record (e.g., to save space).
        """
        obj=self if inplace else self.copy()
        obj.original=None
        return obj
    
    @keephistory
    def reset_time(self, t0: float=0, inplace=_inplace):
        """
        Resets time so that the time-array starts at time zero (t0).
        Resets onsets etc.
        
        Parameters
        ----------
        t0: float
            time at which the :class:`.PupilData`'s time-vector starts
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes
        """
        tmin=self.tx.min()
        obj=self if inplace else self.copy()            
        obj.tx=(self.tx-tmin)+t0
        obj.event_onsets=(self.event_onsets-tmin)+t0
        return obj
        
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
        Reads a :class:`.PupilData` object from a pickle-file.
        Use as ``pypillometry.PupilData.from_file("yourfile.pd")``.
        
        Parameters
        ----------
        
        fname: str
            filename
        """
        r=pd_read_pickle(fname)
        return r
        
    def _unit_fac(self, units):
        if units=="sec":
            fac=1./1000.
        elif units=="min":
            fac=1./1000./60.
        elif units=="h":
            fac=1./1000./60./60.
        else:
            fac=1.
        return fac

    @keephistory
    def sub_slice(self, start: float=-np.inf, end: float=np.inf, units: str="sec"):
        """
        Return a new `PupilData` object that is a shortened version
        of the current one (contains all data between `start` and
        `end` in units given by `units` (one of "ms", "sec", "min", "h").

        Parameters
        ----------
        
        start: float
            start for new dataset
        end: float
            end of new dataset
        units: str
            time units in which `start` and `end` are provided
        """
        slic=self.copy()
        fac=self._unit_fac(units)
        tx=self.tx*fac
        keepix=np.where(np.logical_and(tx>=start, tx<=end))
        for k, v in slic.__dict__.items():
            if isinstance(v,np.ndarray) and v.size==self.sy.size:
                slic.__dict__[k]=slic.__dict__[k][keepix]
        evon=slic.event_onsets*slic._unit_fac(units)
        keepev=np.logical_and(evon>=start, evon<=end)
        slic.event_onsets=slic.event_onsets[keepev]
        slic.event_labels=slic.event_labels[keepev]
        ## just remove all detected blinks (need to rerun `detect_blinks()`)
        slic.blinks=np.empty((0,2), dtype=np.int)
        slic.blink_mask=np.zeros(len(slic), dtype=np.int)
        return slic

    def summary(self) -> dict:
        """Return a summary of the :class:`.PupilData`-object."""
        summary=dict(
            name=self.name,
            n=len(self),
            nmiss=np.sum(self.missing),#np.sum(np.isnan(self.sy))+np.sum(self.sy==0),
            perc_miss=np.sum(self.missing)/len(self)*100.,#(np.sum(np.isnan(self.sy))+np.sum(self.sy==0))/len(self)*100.,
            nevents=self.nevents(), 
            nblinks=self.nblinks(),
            ninterpolated=self.interpolated_mask.sum(),
            blinks_per_min=self.nblinks()/(len(self)/self.fs/60.),
            fs=self.fs, 
            duration_minutes=self.get_duration("min"),
            start_min=self.tx.min()/1000./60.,
            end_min=self.tx.max()/1000./60.,
            baseline_estimated=self.baseline_estimated,
            response_estimated=self.response_estimated)        
        return summary
    
    def size_bytes(self):
        """
        Return size of current dataset in bytes.
        """
        nbytes=len(pickle.dumps(self, -1))
        return nbytes
    
    def __repr__(self) -> str:
        """Return a string-representation of the dataset."""
        pars=self.summary()
        del pars["name"]
        s="PupilData({name}, {size}):\n".format(name=self.name, size=sizeof_fmt(self.size_bytes()))
        flen=max([len(k) for k in pars.keys()])
        for k,v in pars.items():
            s+=(" {k:<"+str(flen)+"}: {v}\n").format(k=k,v=v)
        s+=" History:\n *\n"
        try:
            for i,ev in enumerate(self.history):
                s+=" "*(i+1)+"└ " + ev["funcstring"] +"\n"
        except:
            s+=" └no history\n"
        return s
    
    @keephistory    
    def unscale(self, mean: Optional[float]=None, sd: Optional[float]=None, inplace=_inplace):
        """
        Scale back to original values using either values provided as arguments
        or the values stored in `scale_params`.
        
        Parameters
        ----------
        mean: float
            mean to add from signal
        sd: float
            sd to scale with        
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes
        
        """
        if mean is None:
            mean=self.scale_params["mean"]
        if sd is None:
            sd=self.scale_params["sd"]
        
        obj=self if inplace else self.copy()
        obj.scale_params={"mean":0, "sd":1}
        obj.sy=(self.sy*sd)+mean
        obj.baseline=(self.baseline*sd)+mean
        obj.response=(self.response*sd)
        return obj
    
    @keephistory
    def scale(self, mean: Optional[float]=None, sd: Optional[float]=None, inplace=_inplace):
        """
        Scale the pupillary signal by subtracting `mean` and dividing by `sd`.
        If these variables are not provided, use the signal's mean and std.
        
        Parameters
        ----------
        
        mean: float
            mean to subtract from signal
        sd: float
            sd to scale with
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes        
        
        Note
        ----
        Scaling-parameters are being saved in the `scale_params` argument. 
        """
        if mean is None:
            mean=np.nanmean(self.sy)
        if sd is None:
            sd=np.nanstd(self.sy)

        obj=self if inplace else self.copy()            
        obj.scale_params={"mean":mean, "sd":sd}
        obj.sy=(self.sy-mean)/sd
        obj.baseline=(self.baseline-mean)/sd
        obj.response=(self.response)/sd
        return obj
    
    @keephistory
    def lowpass_filter(self, cutoff: float, order: int=2, inplace=_inplace):
        """
        Lowpass-filter signal using a Butterworth-filter, 
        see :py:func:`pypillometry.baseline.butter_lowpass_filter()`.
    
        Parameters
        -----------

        cutoff: float
            lowpass-filter cutoff
        order: int
            filter order
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                    
        """
        obj=self if inplace else self.copy()                    
        obj.sy=butter_lowpass_filter(self.sy, cutoff, self.fs, order)
        return obj

    @keephistory
    def smooth_window(self, window: str="hanning", winsize: float=11, inplace=_inplace):
        """
        Apply smoothing of the signal using a moving window. See :func:`.smooth_window()`.
        
        Parameters
        ----------
        window: str
            (the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'); 
             flat window will produce a moving average smoothing.
        winsize: float
            the length of the window in ms 
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                            
        """
        winsize_ix=int(np.ceil(winsize/1000.*self.fs)) 
        obj=self if inplace else self.copy()                            
        obj.sy=smooth_window(self.sy, winsize_ix, window )
        return obj
    
    @keephistory
    def downsample(self, fsd: float, dsfac: bool=False, inplace=_inplace):
        """
        Simple downsampling scheme using mean within the downsampling window.
        See :py:func:`pypillometry.baseline.downsample()`.

        Parameters
        -----------

        fsd: 
            new sampling-rate or decimate-factor
        dsfac:
            if False, `fsd` is the new sampling rate;
            if True, `fsd` is the decimate factor
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                        
        """
        if dsfac:
            dsfac=fsd
            fsd=float(self.fs/dsfac)
        else:
            dsfac=int(self.fs/fsd) # calculate downsampling factor
            
        obj=self if inplace else self.copy()
        
        ## downsample all arrays that have the original sy-length
        # (this is so that the function is general for subclasses, as well)
        nd=self.sy.size
        for k, v in obj.__dict__.items():
            if isinstance(v,np.ndarray) and v.size==nd:
                obj.__dict__[k]=downsample(self.__dict__[k], dsfac)
            
        obj.fs=fsd
        return obj

    def copy(self, new_name: Optional[str]=None):
        """
        Make and return a deep-copy of the pupil data.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v))
        if new_name is None:
            result.name=self.name+"_"+self._random_id(n=2)
        else:
            result.name=new_name
        return result        

    def _plot(self, plot_range, overlays, overlay_labels, units, interactive, highlight_blinks, highlight_interpolated):
        fac=self._unit_fac(units)
        if units=="sec":
            xlab="seconds"
        elif units=="min":
            xlab="minutes"
        elif units=="h":
            xlab="hours"
        else:
            xlab="ms"
        tx=self.tx*fac
        evon=self.event_onsets*fac
        
        start,end=plot_range
        if start==-np.infty:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.infty:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))
        
        tx=tx[startix:endix]
        
        ixx=np.logical_and(evon>=start, evon<end)
        evlab=self.event_labels[ixx]
        evon=evon[ixx]
        overlays=(ov[startix:endix] for ov in overlays)
        
        if interactive:
            blinks=np.empty((0,2), dtype=np.int)
            interpolated=np.empty((0,2), dtype=np.int)
            if highlight_blinks:
                blinks=[]
                for sblink,eblink in self.blinks:
                    if eblink<startix or sblink>endix:
                        continue
                    else:
                        sblink=max(0,sblink-startix)
                        eblink=min(endix,eblink-startix)
                    blinks.append([sblink,eblink])
                blinks=np.array(blinks)
            if highlight_interpolated:
                a=np.diff(np.r_[0, self.interpolated_mask[startix:endix], 0])[:-1]
                istarts=np.where(a>0)[0]
                iends=np.where(a<0)[0]
                interpolated=[]
                for istart,iend in zip(istarts,iends):
                    interpolated.append([istart,iend])
            plot_pupil_ipy(tx, self.sy[startix:endix], evon,
                           overlays=overlays, overlay_labels=overlay_labels,
                           blinks=blinks, interpolated=interpolated,
                          xlab=xlab)
        else:
            plt.plot(tx, self.sy[startix:endix], label="signal")
            for i,ov in enumerate(overlays):
                plt.plot(tx, ov, label=overlay_labels[i])
            plt.vlines(evon, *plt.ylim(), color="grey", alpha=0.5)
            ll,ul=plt.ylim()
            for ev,lab in zip(evon,evlab):
                plt.text(ev, ll+(ul-ll)/2., "%s"%lab, fontsize=8, rotation=90)
            if highlight_interpolated:
                a=np.diff(np.r_[0, self.interpolated_mask[startix:endix], 0])[:-1]
                istarts=np.where(a>0)[0]
                iends=np.where(a<0)[0]
                for istart,iend in zip(istarts,iends):
                    plt.gca().axvspan(tx[istart],tx[iend],color="green", alpha=0.1)
            if highlight_blinks:
                for sblink,eblink in self.blinks:
                    if eblink<startix or sblink>endix:
                        continue
                    else:
                        sblink=min(tx.size-1, max(0,sblink-startix))
                        eblink=min(endix-startix-1,eblink-startix)
                    
                    plt.gca().axvspan(tx[sblink],tx[eblink],color="red", alpha=0.2)
                
                
            plt.legend()
            plt.xlabel(xlab)        
    
    def plot(self, plot_range: Tuple[float,float]=(-np.infty, +np.infty),
             interactive: bool=False, 
             baseline: bool=True, 
             response: bool=False,
             model: bool=True,
             highlight_blinks: bool=True,
             highlight_interpolated: bool=True,
             units: str="sec"
            ) -> None:
        """
        Make a plot of the pupil data using `matplotlib` or :py:func:`pypillometry.convenience.plot_pupil_ipy()`
        if `interactive=True`.

        Parameters
        ----------
        plot_range: tuple (start,end)
            plot from start to end (in units of `units`)
        baseline: bool
            plot baseline if estimated
        response: bool
            plot response if estimated
        model: bool
            plot full model if baseline and response have been estimated
        interactive: bool
            if True, plot with sliders to adjust range
        units: str
            one of "sec"=seconds, "ms"=millisec, "min"=minutes, "h"=hours
        """

        overlays=tuple()
        overlay_labels=tuple()
        if baseline and self.baseline_estimated:
            overlays+=(self.baseline,)                
            overlay_labels+=("baseline",)
        if response and self.response_estimated:
            overlays+=(self.response,)
            overlay_labels+=("response",)             
        if model and self.baseline_estimated and self.response_estimated:
            overlays+=(self.baseline+self.response,)
            overlay_labels+=("model",)
        self._plot(plot_range, overlays, overlay_labels, units, interactive, highlight_blinks, highlight_interpolated)

    def plot_segments(self, overlay=None, pdffile: Optional[str]=None, interv: float=1, figsize=(15,5), ylim=None, **kwargs):
        """
        Plot the whole dataset chunked up into segments (usually to a PDF file).

        Parameters
        ----------

        pdffile: str or None
            file name to store the PDF; if None, no PDF is written 
        interv: float
            duration of each of the segments to be plotted (in minutes)
        figsize: Tuple[int,int]
            dimensions of the figures
        kwargs: 
            arguments passed to :func:`.PupilData.plot()`

        Returns
        -------

        figs: list of :class:`matplotlib.Figure` objects
        """

        # start and end in minutes
        smins,emins=self.tx.min()/1000./60., self.tx.max()/1000./60.
        segments=[]
        cstart=smins
        cend=smins
        while cend<emins:
            cend=min(emins, cstart+interv)
            segments.append( (cstart,cend) )
            cstart=cend

        figs=[]
        _backend=mpl.get_backend()
        mpl.use("pdf")
        plt.ioff() ## avoid showing plots when saving to PDF 

        for start,end in segments:
            plt.figure(figsize=figsize)
            self.plot( (start,end), units="min", **kwargs)
            if overlay is not None:
                overlay.plot( (start, end), units="min", **kwargs)  
            if ylim is not None:
                plt.ylim(*ylim)
            figs.append(plt.gcf())


        if isinstance(pdffile, str):
            print("> Writing PDF file '%s'"%pdffile)
            with PdfPages(pdffile) as pdf:
                for fig in figs:
                    pdf.savefig(fig)         

        ## switch back to original backend and interactive mode                        
        mpl.use(_backend) 
        plt.ion()

        return figs        
    
    @keephistory
    def estimate_baseline(self, method: str="envelope_iter_bspline_2", inplace=_inplace, **kwargs):
        """
        Apply one of the baseline-estimation methods.
        
        Parameters
        ----------
        
        method: 
            "envelope_iter_bspline_1": :py:func:`pypillometry.baseline.baseline_envelope_iter_bspline()` 
                                        with one iteration
            "envelope_iter_bspline_2": :py:func:`pypillometry.baseline.baseline_envelope_iter_bspline()` 
                                        with two iterations
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                        
            
        kwargs:
            named arguments passed to the low-level function in :py:mod:`pypillometry.baseline`.
            
        Note
        -----
        the results of the estimation is stored in member `baseline`
        
        """
        obj=self if inplace else self.copy()
        if method=="envelope_iter_bspline_2":
            txd,syd,base2,base1=baseline_envelope_iter_bspline(self.tx, self.sy,self.event_onsets,self.fs,**kwargs)
            f=interpolate.interp1d(txd, base2, kind="cubic", bounds_error=False, fill_value="extrapolate")
            obj.baseline=f(self.tx)
        elif method=="envelope_iter_bspline_1": 
            txd,syd,base2,base1=baseline_envelope_iter_bspline(self.tx, self.sy,self.event_onsets,self.fs,**kwargs)
            f=interpolate.interp1d(txd, base1, kind="cubic", bounds_error=False, fill_value="extrapolate")
            obj.baseline=f(self.tx)            
        else:
            raise ValueError("Undefined method for baseline estimation: %s"%method)         
        obj.baseline_estimated=True
        return obj

    def stat_per_event(self, interval: Tuple[float,float], event_select=None, statfct: Callable=np.mean, return_missing: Optional[str]=None):
        """
        Return result of applying a statistical function to pupillometric data in a
        given interval relative to event-onsets. For example, extract mean 
        pupil-size in interval before trial onset.

        Parameters
        -----------
        event_select: str or function
            variable describing which events to select and align to
            - if str: use all events whose label contains the string
            - if function: apply function to all labels, use those where the function returns True
        
        interval : tuple (min,max)
            time-window in ms relative to event-onset (0 is event-onset)

        statfct : function
            function mapping np.array to a single number

        return_missing: None, "nmiss", "prop"
            if None, only an array with the stats per event is return
            if "nmiss", returns a tuple (stat, nmiss) where `nmiss` is the number of missing vales in the timewin
            if "prop", return a tuple (stat, prop_miss) where `prop_miss` is the proportion missing vales in the timewin
    
        Returns
        --------

        result: np.array
            number of event-onsets long result array
        """
        if callable(event_select):
            event_ix=np.array([bool(event_select(evlab)) for evlab in self.event_labels])
        elif isinstance(event_select, str):
            event_ix=np.array([event_select in evlab for evlab in self.event_labels])
        elif event_select is None:
            event_ix=np.arange(self.nevents())
        else:
            raise ValueError("event_select must be string or function")
        
        stat =stat_event_interval(self.tx, self.sy, self.event_onsets[event_ix], interval, statfct)
        if return_missing=="nmiss":
            nmiss=stat_event_interval(self.tx, np.logical_or(self.missing, self.interpolated_mask), 
                                      self.event_onsets[event_ix], interval, np.sum)
            ret=(stat,nmiss)
        elif return_missing=="prop":
            prop_miss=stat_event_interval(self.tx, np.logical_or(self.missing, self.interpolated_mask), 
                                          self.event_onsets[event_ix], interval, np.mean)
            ret=(stat,prop_miss)            
        else: 
            ret=stat
        return ret
    
    @keephistory
    def estimate_response(self, npar: Union[str,float]="free", tmax: Union[str,float]="free", 
                          verbose: int=50,
                          bounds: dict={"npar":(1,20), "tmax":(100,2000)},
                          inplace=_inplace):
        """
        Estimate pupil-response based on event-onsets, see
        :py:func:`pypillometry.pupil.pupil_response()`.
        

        npar: float
            npar-parameter for the canonical response-function or "free";
            in case of "free", the function optimizes for this parameter
        tmax: float
            tmax-parameter for the canonical response-function or "free";
            in case of "free", the function optimizes for this parameter
        bounds: dict
            in case that one or both parameters are estimated, give the lower
            and upper bounds for the parameters        
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                        
        
        Note
        ----
        the results of the estimation is stored in members `response`, `response_x` (design matrix) 
        and `response_pars`

        """
        if not self.baseline_estimated:
            print("WARNING: no baseline estimated yet, using zero as baseline")
        
        pred, coef, npar_est, tmax_est, x1=pupil_response(self.tx, self.sy-self.baseline, 
                                                          self.event_onsets, self.fs, 
                                                          npar=npar, tmax=tmax, verbose=verbose,
                                                         bounds=bounds)
        obj=self if inplace else self.copy()
        obj.response_pars={"npar":npar_est,
                            "npar_free":True if npar=="free" else False,
                            "tmax":tmax_est,
                            "tmax_free":True if tmax=="free" else False,
                            "coef":coef,
                            "bounds":bounds
                           }
        
        obj.response=pred
        obj.response_x=x1
        obj.response_estimated=True
        return obj
    

    @keephistory
    def blinks_detect(self, min_duration:float=20, blink_val:float=0,
                      winsize: float=11, vel_onset: float=-5, vel_offset: float=5, 
                      min_onset_len: int=5, min_offset_len: int=5,
                      strategies: List[str]=["zero","velocity"],
                      units="ms", inplace=_inplace):
        """
        Detect blinks in the pupillary signal using several strategies.
        First, blinks are detected as consecutive sequence of `blink_val` 
        (f.eks., 0 or NaN). Second, blinks are defined as everything between
        two crossings of the velocity profile (from negative to positive).
        
        Detected blinks are put into member `blinks` (matrix 2 x nblinks where start and end
        are stored as indexes) and member `blink_mask` which codes for each sampling point
        whether there is a blink (1) or not (0).

        Finally, detected blinks have to be at least `min_duration` duration (in `units`).
        
        Parameters
        ----------
        min_duration: float
            minimum duration for a sequence of missing numbers to be treated as blink
        blink_val: float
            "missing value" code
        winsize:
            window-size for smoothing for velocity profile (in units)
        vel_onset:
            negative velocity that needs to be crossed; arbitrary units that depend on
            sampling rate etc
        vel_offset:
            positive velocity that needs to be exceeded; arbitrary units that depend on
            sampling rate etc
        min_onset_len: int
            minimum number of consecutive samples that crossed threshold in the velocity
            profile to detect as onset (to avoid noise-induced changes)
        min_offset_len: int
            minimum number of consecutive samples that crossed threshold in the velocity
            profile to detect as offset (to avoid noise-induced changes)            
        strategies: list of strategies to use
            so far, use a list containing any combination of "zero" and "velocity"
        units: str
            one of "ms", "sec", "min", "h"
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                                    
        """
        fac=self._unit_fac(units)
        winsize_ms=winsize*fac
        winsize_ix=int(winsize_ms/1000.*self.fs)
        if winsize_ix % 2==0:
            winsize += 1
        min_duration_ms=min_duration*fac
        min_duration_ix=int(min_duration_ms/1000.*self.fs)        

        obj=self if inplace else self.copy()
        
        # check for unknown strategies
        for strat in strategies:
            if strat not in ["zero", "velocity"]:
                print("WARN: strategy '%s' unknown"%strat)
        
        ## detect blinks with the different strategies
        if "velocity" in strategies:
            blinks_vel=detect_blinks_velocity(self.sy, winsize_ix, vel_onset, vel_offset, min_onset_len, min_offset_len)
        else: 
            blinks_vel=np.array([])
            
        if "zero" in strategies:
            blinks_zero=detect_blinks_zero(self.sy, 1, blink_val)
        else:
            blinks_zero=np.array([])

        ## merge the two blinks
        blinks=helper_merge_blinks(blinks_vel, blinks_zero)
        obj.blinks=np.array([[on,off] for (on,off) in blinks if off-on>=min_duration_ix])
        
        obj.blink_mask=np.zeros(self.sy.size, dtype=np.int)
        
        for start,end in obj.blinks:
            obj.blink_mask[start:end]=1
        return obj    
    
    def blinks_plot(self, pdf_file: Optional[str]=None, nrow: int=5, ncol: int=3, 
                    figsize: Tuple[int,int]=(10,10), 
                    pre_blink: float=500, post_blink: float=500, units: str="ms", 
                    plot_index: bool=True):
        """
        Plot the detected blinks into separate figures each with nrow x ncol subplots. 

        Parameters
        ----------
        pdf_file: str or None
            if the name of a file is given, the figures are saved into a multi-page PDF file
        ncol: int
            number of columns for the blink-plots
        pre_blink: float
            extend plot a certain time before each blink (in ms)
        post_blink: float
            extend plot a certain time after each blink (in ms)
        units: str
            units in which the signal is plotted
        plot_index: bool
            plot a number with the blinks' index (e.g., for identifying abnormal blinks)

        Returns
        -------

        list of plt.Figure objects each with nrow*ncol subplots
        in Jupyter Notebook, those are displayed inline one after the other
        """
        fac=self._unit_fac(units)
        pre_blink_ix=int((pre_blink/1000.)*self.fs)
        post_blink_ix=int((post_blink/1000.)*self.fs)

        nblinks=self.blinks.shape[0]
        nsubplots=nrow*ncol # number of subplots per figure
        nfig=int(np.ceil(nblinks/nsubplots))

        figs=[]
        if isinstance(pdf_file,str):
            _backend=mpl.get_backend()
            mpl.use("pdf")
            plt.ioff() ## avoid showing plots when saving to PDF 
        
        iblink=0
        for i in range(nfig):
            fig=plt.figure(figsize=figsize)
            axs = fig.subplots(nrow, ncol).flatten()

            for ix,(start,end) in enumerate(self.blinks[(i*nsubplots):(i+1)*nsubplots]):
                iblink+=1
                slic=slice(start-pre_blink_ix,end+post_blink_ix)
                ax=axs[ix]
                ax.plot(self.tx[slic]*fac,self.sy[slic])

                ## highlight interpolated data
                a=np.diff(np.r_[0,self.interpolated_mask[slic],0])[:-1]
                istarts=start-pre_blink_ix+np.where(a>0)[0]
                iends=start-pre_blink_ix+np.where(a<0)[0]
                for istart,iend in zip(istarts,iends):
                    ax.axvspan(self.tx[istart]*fac,self.tx[iend]*fac,color="green", alpha=0.1)

                ## highlight blink
                ax.axvspan(self.tx[start]*fac,self.tx[end]*fac,color="red", alpha=0.2)

                if plot_index: 
                    ax.text(0.5, 0.5, '%i'%(iblink), fontsize=12, horizontalalignment='center',     
                            verticalalignment='center', transform=ax.transAxes)
            figs.append(fig)

        if pdf_file is not None:
            print("> Saving file '%s'"%pdf_file)
            with PdfPages(pdf_file) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
            ## switch back to original backend and interactive mode                
            mpl.use(_backend) 
            plt.ion()
            
        return figs    

    @keephistory
    def blinks_merge(self, distance: float=100, remove_signal: bool=False, inplace=_inplace):
        """
        Merge together blinks that are close together. 
        Some subjects blink repeatedly and standard detection/interpolation can result in weird results.
        This function simply treats repeated blinks as one long blink.

        Parameters
        ----------

        distance: float
            merge together blinks that are closer together than `distance` in ms
        remove_signal: bool
            if True, set all signal values during the "new blinks" to zero so 
            that :func:`.detect_blinks()` will pick them up; interpolation will work
            either way
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                                    
        """
        distance_ix=distance/self.fs*1000.

        newblinks=[] 
        i=1
        cblink=self.blinks[0,:] ## start with first blink
        while(i<self.nblinks()):
            if (self.blinks[i,0]-cblink[1])<=distance_ix:
                # merge
                cblink[1]=self.blinks[i,1]
            else:
                newblinks.append(cblink)
                cblink=self.blinks[i,:]
            i+=1            
        newblinks.append(cblink)
        newblinks=np.array(newblinks)       

        obj=self if inplace else self.copy()
        obj.blinks=newblinks

        ## set signal to zero within the new blinks
        if remove_signal:
            for start,end in obj.blinks:
                obj.sy[start:end]=0

        return obj    
    
    @keephistory
    def blinks_interpolate(self, winsize: float=11, 
                           vel_onset: float=-5, vel_offset: float=5, 
                           margin: Tuple[float,float]=(10,30), 
                           interp_type: str="cubic", inplace=_inplace):
        """
        Interpolation of missing data "in one go".
        Detection of blinks happens using Mahot (2013), see :func:`.blink_onsets_mahot()`.
        
        Parameters
        ----------
        winsize: float
            size of the Hanning-window in ms
        vel_onset: float
            velocity-threshold to detect the onset of the blink
        vel_offset: float
            velocity-threshold to detect the offset of the blink
        margin: Tuple[float,float]
            margin that is subtracted/added to onset and offset (in ms)
        interp_type: str
            type of interpolation accepted by :func:`scipy.interpolate.interp1d()`   
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                                    
        """
        # parameters in sampling units (from ms)
        winsize_ix=int(np.ceil(winsize/1000.*self.fs)) 
        margin_ix=tuple(int(np.ceil(m/1000.*self.fs)) for m in margin)
        if winsize_ix % 2==0: ## ensure smoothing window is odd
            winsize_ix+=1 

        # generate smoothed signal and velocity-profile
        sym=smooth_window(self.sy, winsize_ix, "hanning")
        vel=np.r_[0,np.diff(sym)] 

        blink_onsets=blink_onsets_mahot(self.sy, self.blinks, winsize_ix, vel_onset, vel_offset,
                                        margin_ix, int(np.ceil(500/1000*self.fs)))
        obj=self if inplace else self.copy()
        obj.interpolated_mask=np.zeros(self.sy.size)
        for on,off in blink_onsets:
            obj.interpolated_mask[on:off]=1
        f=scipy.interpolate.interp1d(self.tx[obj.interpolated_mask==0], sym[obj.interpolated_mask==0], 
                                     kind=interp_type, bounds_error=False, fill_value=0)
        syr=f(self.tx)
        obj.sy=syr
        
        
        return obj
    
    @keephistory
    def blinks_interp_mahot(self, winsize: float=11, 
                           vel_onset: float=-5, vel_offset: float=5, 
                           margin: Tuple[float,float]=(10,30), 
                           blinkwindow: float=500,
                           interp_type: str="cubic",
                           plot: Optional[str]=None, 
                           plot_dim: Tuple[int,int]=(5,3),
                           plot_figsize: Tuple[int,int]=(10,8),
                           inplace=_inplace):
        """
        Implements the blink-interpolation method by Mahot (2013).
        
        Mahot, 2013:
        https://figshare.com/articles/A_simple_way_to_reconstruct_pupil_size_during_eye_blinks/688001.

        This procedure relies heavily on eye-balling (reconstructing visually convincing signal),
        so a "plot" option is provided that will plot many diagnostics (see paper linked above) that
        can help to set good parameter values for `winsize`, `vel_onset`, `vel_offset` and `margin`.

        Parameters
        ----------
        winsize: float
            size of the Hanning-window in ms
        vel_onset: float
            velocity-threshold to detect the onset of the blink
        vel_offset: float
            velocity-threshold to detect the offset of the blink
        margin: Tuple[float,float]
            margin that is subtracted/added to onset and offset (in ms)
        blinkwindow: float
            how much time before and after each blink to include (in ms)
        interp_type: str
            type of interpolation accepted by :func:`scipy.interpolate.interp1d()`
        plot: True, str or None
            if a string, the plot is going to be saved to a multipage PDF file; 
            if None, no plotting is done
            if True, plot is not saved but produced
        plot_dim: tuple nrow x ncol 
            number of subplots
        plot_figsize: tuple (width, height)
            dimensions for each figure
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                                    
        """
        # parameters in sampling units (from ms)
        winsize_ix=int(np.ceil(winsize/1000.*self.fs)) 
        margin_ix=tuple(int(np.ceil(m/1000.*self.fs)) for m in margin)
        blinkwindow_ix=int(blinkwindow/1000.*self.fs)
        if winsize_ix % 2==0: ## ensure smoothing window is odd
            winsize_ix+=1 

        # generate smoothed signal and velocity-profile
        sym=smooth_window(self.sy, winsize_ix, "hanning")
        vel=np.r_[0,np.diff(sym)] 
        syr=self.sy.copy() ## reconstructed signal

        nrow,ncol=plot_dim
        nsubplots=nrow*ncol    
        nfig=int(np.ceil(self.nblinks()/nsubplots))
        figs=[]
        if isinstance(plot,str):
            _backend=mpl.get_backend()
            mpl.use("pdf")
            plt.ioff() ## avoid showing plots when saving to PDF 

        blink_onsets=blink_onsets_mahot(self.sy, self.blinks, winsize_ix, vel_onset, vel_offset,
                                           margin_ix, blinkwindow_ix)
          
        obj=self if inplace else self.copy()    
        # loop through blinks
        for ix,(onset,offset) in enumerate(blink_onsets):                
            if plot is not None:            
                if ix % nsubplots==0:
                    fig,axs=plt.subplots(nrow,ncol,figsize=plot_figsize)
                    axs=axs.flatten()
                    figs.append(fig)

            # calc the 4 time points
            t2,t3=onset,offset
            t1=max(0,t2-t3+t2)
            t4=min(t3-t2+t3, len(self)-1)
            if t1==t2:
                t2+=1
            if t3==t4:
                t3-=1
            
            txpts=[self.tx[pt] for pt in [t1,t2,t3,t4]]
            sypts=[self.sy[pt] for pt in [t1,t2,t3,t4]]
            intfct=interp1d(txpts,sypts, kind=interp_type)
            islic=slice(t2, t3)
            syr[islic]=intfct(self.tx[islic])

            ## record the interpolated datapoints
            obj.interpolated_mask[islic]=1

            slic=slice(max(0,onset-blinkwindow_ix), min(offset+blinkwindow_ix, len(self)))
            
            ## plotting for diagnostics
            #--------------------------
            if plot is not None:            
                #fig,ax1=plt.subplots()
                ax1=axs[ix % nsubplots]
                ax1.plot(self.tx[slic]/1000., self.sy[slic], color="blue", label="raw")
                ax1.plot(self.tx[slic]/1000., sym[slic], color="green", label="smoothed")
                ax1.plot(self.tx[slic]/1000., syr[slic], color="red", label="interpolated")
                ax2=ax1.twinx()
                ax2.plot(self.tx[slic]/1000., vel[slic], color="orange", label="velocity")

                for pt in (t1,t2,t3,t4):
                    ax1.plot(self.tx[pt]/1000., sym[pt], "o", color="red")
                ax1.text(0.5, 0.5, '%i'%(ix+1), fontsize=12, horizontalalignment='center',     
                    verticalalignment='center', transform=ax1.transAxes)
                if ix % nsubplots==0:
                    handles1, labels1 = ax1.get_legend_handles_labels()
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    handles=handles1+handles2
                    labels=labels1+labels2
                    fig.legend(handles, labels, loc='upper right')            
        if isinstance(plot, str):
            print("> Writing PDF file '%s'"%plot)
            with PdfPages(plot) as pdf:
                for fig in figs:
                    pdf.savefig(fig)         
            ## switch back to original backend and interactive mode                
            mpl.use(_backend) 
            plt.ion()
        elif plot is not None:
            for fig in figs:
                pass
                #fig.show()

        # replace signal with the reconstructed one
        obj.sy=syr

        return obj
    
    def get_erpd(self, erpd_name: str, event_select, 
                 baseline_win: Optional[Tuple[float,float]]=None, 
                 time_win: Tuple[float,float]=(-500, 2000)):
        """
        Extract event-related pupil dilation (ERPD).
        No attempt is being made to exclude overlaps of the time-windows.

        Parameters
        ----------
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


        """
        if callable(event_select):
            event_ix=np.array([bool(event_select(evlab)) for evlab in self.event_labels])
        elif isinstance(event_select, str):
            event_ix=np.array([event_select in evlab for evlab in self.event_labels])
        else:
            raise ValueError("event_select must be string or function")


        nev=event_ix.sum()
        time_win_ix=tuple(( int(np.ceil(tw/1000.*self.fs)) for tw in time_win ))
        duration_ix=time_win_ix[1]-time_win_ix[0]
        txw=np.linspace(time_win[0], time_win[1], num=duration_ix)

        ## resulting matrix and missing (interpolated/blinks/...) indicator for each datapoint
        erpd=np.zeros((nev,duration_ix))
        missing=np.ones((nev,duration_ix))

        # event-onsets as indices of the tx array
        evon=self.event_onsets[event_ix]
        # vectorized version (seems to be worse than naive one)
        #evon_ix=np.argmin(np.abs(np.tile(evon, (self.tx.size,1)).T-self.tx), axis=1)
        # naive version
        evon_ix=np.array([np.argmin(np.abs(ev-self.tx)) for ev in evon])

        for i,ev in enumerate(evon_ix):
            on,off=ev+time_win_ix[0], ev+time_win_ix[1]
            onl,offl=0,duration_ix # "local" window indices
            if on<0: ## pad with zeros in case timewindow starts before data
                onl=np.abs(on)
                on=0
            if off>=self.tx.size:
                offl=offl-(off-self.tx.size)
                off=self.tx.size

            erpd[i,onl:offl]=self.sy[on:off]
            missing[i,onl:offl]=np.logical_or(self.interpolated_mask[on:off], self.missing[on:off])

        baselines=[None for _ in range(nev)]
        if baseline_win is not None:
            if baseline_win[0]<time_win[0] or baseline_win[0]>time_win[1] or baseline_win[1]<time_win[0] or baseline_win[1]>time_win[1]:
                print("WARNING: baseline-window misspecified %s vs. %s; NOT doing baseline correction"%(baseline_win, time_win))
            else:
                blwin_ix=tuple(( np.argmin(np.abs(bw-txw)) for bw in baseline_win ))

                for i in range(nev):
                    baselines[i]=np.mean(erpd[i,blwin_ix[0]:blwin_ix[1]])
                    erpd[i,:]-=baselines[i]

        return ERPD(erpd_name, txw, erpd, missing, baselines)
    

    
#@typechecked   
class FakePupilData(PupilData):
    """
    Simulated pupil data for validation purposes.
    """
    def __init__(self,
                 pupil: PupilArray, 
                 sampling_rate: Optional[float]=None,
                 time: Optional[PupilArray]=None,
                 event_onsets: Optional[PupilArray]=None,
                 event_labels: Optional[PupilArray]=None,
                 name: Optional[str]=None,
                 sim_params: dict={},
                 real_baseline: Optional[PupilArray]=None,
                 real_response_coef: Optional[PupilArray]=None):
        """
        Constructor for artifical pupil data.
        """
        super().__init__(pupil,sampling_rate,time,event_onsets,event_labels,name)
        self.name="fake_"+self.name
        self.sim_params=sim_params
        self.sim_baseline=real_baseline
        
        ## OBS: not the real model but a simplification (npar/tmax may be different per event)
        x1=pupil_build_design_matrix(self.tx, self.event_onsets, self.fs, 
                                     sim_params["prf_npar"][0], sim_params["prf_tmax"][0], 6000)
        amp=np.mean(real_baseline)*sim_params["evoked_response_perc"]
        real_response=amp*np.dot(x1.T, real_response_coef)  ## predicted signal
        
        self.sim_response=real_response
        self.sim_response_coef=real_response_coef

    @keephistory    
    def unscale(self, mean: Optional[float]=None, sd: Optional[float]=None, inplace=_inplace):
        """
        Scale back to original values using either values provided as arguments
        or the values stored in `scale_params`.
        
        Parameters
        ----------
        mean: mean to add from signal
        sd: sd to scale with        
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                                
        """
        mmean,ssd=self.scale_params["mean"],self.scale_params["sd"]
        obj=super().unscale(mean,sd,inplace)
        obj.sim_baseline=(self.sim_baseline*ssd)+mmean
        obj.sim_response=(self.sim_response*ssd)
        return obj
    
    @keephistory
    def scale(self, mean: Optional[float]=None, sd: Optional[float]=None, inplace=_inplace) -> None:
        """
        Scale the pupillary signal by subtracting `mean` and dividing by `sd`.
        If these variables are not provided, use the signal's mean and std.
        
        Parameters
        ----------
        
        mean: mean to subtract from signal
        sd: sd to scale with
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                        
        
        Note
        ----
        Scaling-parameters are being saved in the `scale_params` argument. 
        """
        obj=super().scale(mean,sd)
        mean,sd=obj.scale_params["mean"],obj.scale_params["sd"]
        obj.sim_baseline=(self.sim_baseline-mean)/sd
        obj.sim_response=(self.sim_response)/sd
        return obj

    @keephistory
    def sub_slice(self, start: float=-np.inf, end: float=np.inf, units: str="sec"):
        """
        Return a new `PupilData` object that is a shortened version
        of the current one (contains all data between `start` and
        `end` in units given by `units` (one of "ms", "sec", "min", "h").
        """
        slic=super().sub_slice(start,end,units)
        evon=self.event_onsets*self._unit_fac(units)
        keepev=np.logical_and(evon>=start, evon<=end)
        slic.sim_response_coef=slic.sim_response_coef[keepev]
        return slic
            
    
    def plot(self,
             plot_range: Tuple[float,float]=(-np.infty, +np.infty),             
             interactive: bool=False, 
             baseline: bool=True, 
             response: bool=False,
             model: bool=True,
             simulated: bool=True,
             units: str="sec"
            ) -> None:
        """
        Make a plot of the pupil data using `matplotlib` or :py:func:`pypillometry.convenience.plot_pupil_ipy()`
        if `interactive=True`.

        Parameters
        ----------
        plot_range: tuple (start,end): plot from start to end (in units of `units`)
        baseline: plot baseline if estimated
        response: plot response if estimated
        model: plot full model if baseline and response have been estimated
        simulated: plot also the "ground-truth" baseline and response (i.e., the simulated one)?
        interactive: if True, plot with sliders to adjust range
        units: one of "sec"=seconds, "ms"=millisec, "min"=minutes, "h"=hours
        """
        overlays=tuple()
        overlay_labels=tuple()
        if baseline and self.baseline_estimated:
            overlays+=(self.baseline,)
            overlay_labels+=("baseline",)
        if baseline and simulated:
            overlays+=(self.sim_baseline,)
            overlay_labels+=("sim_baseline",)
        if response and self.response_estimated:
            overlays+=(self.response,)
            overlay_labels+=("response",)
        if response and simulated:
            overlays+=(self.sim_response,)
            overlay_labels+=("sim_response",)
        if model and self.baseline_estimated and self.response_estimated:
            overlays+=(self.baseline+self.response,)
            overlay_labels+=("model",)
        if model and simulated:
            overlays+=(self.sim_baseline+self.sim_response,)
            overlay_labels+=("real model",)
        self._plot(plot_range, overlays, overlay_labels, units, interactive, False, False)

        
        
def plotpd_ia(*args: PupilData, figsize: Tuple[int]=(16,8), baseline: bool=True, events: Optional[int]=0):
    """
    Interactive plotting for multiple `PupilData` objects.
    
    Parameters
    ----------
    args: `PupilData` datasets to plot
    figsize: dimensions of the plot
    baseline: plot baselines, too?
    events: plot event-markers? if None, no events are plotted, otherwise `events` 
            is the index of the `PupilData` object to take the events from
    """

    import pylab as plt
    from ipywidgets import interact, interactive, fixed, interact_manual, Layout
    import ipywidgets as widgets

    def draw_plot(plotxrange):
        xmin,xmax=plotxrange
        plt.figure(figsize=figsize)

        for i,pd in enumerate(args):
            ixmin=np.argmin(np.abs(pd.tx-xmin))
            ixmax=np.argmin(np.abs(pd.tx-xmax))


            plt.plot(pd.tx[ixmin:ixmax],pd.sy[ixmin:ixmax],label=pd.name)
            if baseline and pd.baseline_estimated:
                plt.plot(pd.tx[ixmin:ixmax], pd.baseline[ixmin:ixmax], label="BL: "+pd.name)
            
        if not events is None: 
            plt.vlines(args[events].event_onsets, *plt.ylim(), color="grey", alpha=0.5)
        plt.xlim(xmin,xmax)
        plt.legend()

    xmin=np.min([pd.tx.min() for pd in args])
    xmax=np.max([pd.tx.max() for pd in args])
    wid_range=widgets.FloatRangeSlider(
        value=[xmin,xmax],
        min=xmin,
        max=xmax,
        step=1,
        description=' ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=Layout(width='100%', height='80px')
    )

    interact(draw_plot, plotxrange=wid_range)
    
    

    
def plotpd(*args: PupilData, subplots: bool=False, baseline: bool=False):
    """
    Plotting for `PupilData` objects.
    
    Parameters
    ----------
    
    subplots: plot the different `PupilData`-objects in the same plot or subplots
    """
    if len(args)<3:
        ncol=len(args)
        nrow=1
    else:
        ncol=3
        nrow=np.ceil(len(args)/3.0)
    for i,pd in enumerate(args):
        if subplots:
            plt.subplot(nrow,ncol,i+1)
            plt.title(pd.name)
        plt.plot(pd.tx/1000./60., pd.sy, label=pd.name)
        if baseline and pd.baseline_estimated:
            plt.plot(pd.tx/1000./60., pd.baseline, label="BL: "+pd.name)
        if i==0:
            plt.xlabel("time (min)")
            plt.ylabel("PD")
    if not subplots:
        plt.legend()
        
    
def create_fake_pupildata(**kwargs):
    """
    Return a :py:class:`pyillometry.pupildata.FakePupilData` object by buildling it with
    :py:func:`pypillometry.fakedata.get_dataset()`.
    
    Parameters
    -----------
    
    ntrials:int
        number of trials
    isi: float
        inter-stimulus interval in seconds
    rtdist: tuple (float,float)
        mean and std of a (truncated at zero) normal distribution to generate response times
    pad: float
        padding before the first and after the last event in seconds        
    fs: float
        sampling rate in Hz
    baseline_lowpass: float
        cutoff for the lowpass-filter that defines the baseline
        (highest allowed frequency in the baseline fluctuations)        
    evoked_response_perc: float
        amplitude of the pupil-response as proportion of the baseline     
    response_fluct_sd: float
        How much do the amplitudes of the individual events fluctuate?
        This is determined by drawing each individual pupil-response to 
        a single event from a (positive) normal distribution with mean as determined
        by `evoked_response_perc` and sd `response_fluct_sd` (in units of 
        `evoked_response_perc`).
    prf_npar: tuple (float,float)
        (mean,std) of the npar parameter from :py:func:`pypillometry.pupil.pupil_kernel()`. 
        If the std is exactly zero, then the mean is used for all pupil-responses.
        If the std is positive, npar is taken i.i.d. from ~ normal(mean,std) for each event.
    prf_tmax: tuple (float,float)
        (mean,std) of the tmax parameter from :py:func:`pypillometry.pupil.pupil_kernel()`. 
        If the std is exactly zero, then the mean is used for all pupil-responses.
        If the std is positive, tmax is taken i.i.d. from ~ normal(mean,std) for each event.
    prop_spurious_events: float
        Add random events to the pupil signal. `prop_spurious_events` is expressed
        as proportion of the number of real events. 
    noise_amp: float
        Amplitude of random gaussian noise that sits on top of the simulated signal.
        Expressed in units of mean baseline pupil diameter.
    """
    sim_params={
        "ntrials":100,
        "isi":1000.0,
        "rtdist":(1000.0,500.0),
        "pad":5000.0,
        "fs":1000.0,
        "baseline_lowpass":0.1,
        "evoked_response_perc":0.001,
        "response_fluct_sd":1,
        "prf_npar":(10.35,0),
        "prf_tmax":(917.0,0),
        "prop_spurious_events":0.1,
        "noise_amp":0.0001
    }
    sim_params.update(kwargs)
    #print(sim_params)
    tx,sy,baseline,event_onsets,response_coef=get_dataset(**sim_params)
    event_labels=["event" for _ in range(event_onsets.size)]
    ds=FakePupilData(sy,sim_params["fs"],tx, event_onsets,event_labels=event_labels,
                     sim_params=sim_params, 
                     real_baseline=baseline, real_response_coef=response_coef)
    return ds
    
    