"""
pupildata.py
============

Main object-oriented entry point
"""

from .convenience import *
from .baseline import *
from .fakedata import *

import pylab as plt
import numpy as np
import scipy.signal as signal
from scipy import interpolate
import scipy
from random import choice

import copy
import math

#from pytypes import typechecked
from typing import Sequence, Union, List, TypeVar, Optional, Tuple, Callable
PupilArray=Union[np.ndarray, List[float]]

#@typechecked
class PupilData:
    """
    Class representing pupillometric data. 
    """
    
    def __len__(self) -> int:
        """Return number of sampling points in the pupil data."""
        return self.sy.size
    
    def nevents(self) -> int:
        """Return number of events in pupillometric data."""
        return self.event_onsets.size
    
    
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
                 name: Optional[str]=None):
        """
        Parameters
        ----------
        
        name: 
            name of the dataset or `None` (in which case a random string is selected)
        time: 
            timing array or `None`, in which case the time-array goes from [0,maxT]
            using `sampling_rate`
        pupil:
            pupillary data at times `time` assumed to be in ms
        event_onsets:
            time-onsets of any events that are to be modelled in the pupil
        sampling_rate:
            sampling-rate of the pupillary signal in Hz
        """
        self.sy=np.array(pupil)
        if sampling_rate is None and time is None:
            raise ValueError("you have to specify either sampling_rate or time-vector (or both)")
        
        if time is None:
            maxT=len(self)/sampling_rate*1000.
            self.tx=np.linspace(0,maxT, num=len(self))
        else:
            self.tx=time
        
        if sampling_rate is None:
            self.fs=np.round(1000./np.median(np.diff(self.tx)))
        else:
            self.fs=sampling_rate
            
        if event_onsets is None:
            self.event_onsets=np.array([])
        else:
            self.event_onsets=np.array(event_onsets)
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
    
    def sub_slice(self, start: float=-np.inf, end: float=np.inf, units: str="sec"):
        """
        Return a new `PupilData` object that is a shortened version
        of the current one (contains all data between `start` and
        `end` in units given by `units` (one of "ms", "sec", "min", "h").
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
        return slic
        
    def __repr__(self) -> str:
        """Return a string-representation of the dataset."""
        pars=dict(
            n=len(self),
            nmiss=np.sum(np.isnan(self.sy)),
            nevents=self.nevents(), 
            fs=self.fs, 
            duration_minutes=len(self)/self.fs/60,
            baseline_estimated=self.baseline_estimated,
            response_estimated=self.response_estimated)
        s="PupilData({name}):\n".format(name=self.name)
        flen=max([len(k) for k in pars.keys()])
        for k,v in pars.items():
            s+=(" {k:<"+str(flen)+"}: {v}\n").format(k=k,v=v)
        return s
        
    def unscale(self, mean: Optional[float]=None, sd: Optional[float]=None):
        """
        Scale back to original values using either values provided as arguments
        or the values stored in `scale_params`.
        
        Parameters
        ----------
        mean: mean to add from signal
        sd: sd to scale with        
        """
        if mean is None:
            mean=self.scale_params["mean"]
        if sd is None:
            sd=self.scale_params["sd"]
        
        self.scale_params={"mean":0, "sd":1}
        self.sy=(self.sy*sd)+mean
        self.baseline=(self.baseline*sd)+mean
        self.response=(self.response*sd)
        return self
        
    def scale(self, mean: Optional[float]=None, sd: Optional[float]=None):
        """
        Scale the pupillary signal by subtracting `mean` and dividing by `sd`.
        If these variables are not provided, use the signal's mean and std.
        
        Parameters
        ----------
        
        mean: mean to subtract from signal
        sd: sd to scale with
        
        Note
        ----
        Scaling-parameters are being saved in the `scale_params` argument. 
        """
        if mean is None:
            mean=np.nanmean(self.sy)
        if sd is None:
            sd=np.nanstd(self.sy)
        
        self.scale_params={"mean":mean, "sd":sd}
        self.sy=(self.sy-mean)/sd
        self.baseline=(self.baseline-mean)/sd
        self.response=(self.response)/sd
        return self
        
    def lowpass_filter(self, cutoff: float, order: int=2):
        """
        Lowpass-filter signal using a Butterworth-filter, 
        see :py:func:`pypillometry.baseline.butter_lowpass_filter()`.
    
        Parameters
        -----------

        cutoff: float
            lowpass-filter cutoff
        order: int
            filter order
        """
        self.sy=butter_lowpass_filter(self.sy, cutoff, self.fs, order)
        return self

        
    def downsample(self, fsd: float, dsfac: bool=False):
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
        """
        if dsfac:
            dsfac=fsd
            fsd=float(self.fs/dsfac)
        else:
            dsfac=int(self.fs/fsd) # calculate downsampling factor
        
        ## downsample all arrays that have the original sy-length
        # (this is so that the function is general for subclasses, as well)
        nd=self.sy.size
        for k, v in self.__dict__.items():
            if isinstance(v,np.ndarray) and v.size==nd:
                self.__dict__[k]=downsample(self.__dict__[k], dsfac)
            
        #self.tx=downsample(self.tx, dsfac)
        #self.sy=downsample(self.sy, dsfac)
        #self.baseline=downsample(self.baseline, dsfac)
        self.fs=fsd
        return self

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

    def _plot(self, plot_range, overlays, overlay_labels, units, interactive):
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
        evon=np.array([ev for ev in evon if ev>=start and ev<end])
        overlays=(ov[startix:endix] for ov in overlays)
        
        if interactive:
            plot_pupil_ipy(tx, self.sy[startix:endix], evon,
                           overlays=overlays, overlay_labels=overlay_labels,
                          xlab=xlab)
        else:
            plt.plot(tx, self.sy[startix:endix])
            for i,ov in enumerate(overlays):
                plt.plot(tx, ov, label=overlay_labels[i])
            plt.vlines(evon, *plt.ylim(), color="grey", alpha=0.5)            
            plt.legend()
            plt.xlabel(xlab)        
    
    def plot(self, plot_range: Tuple[float,float]=(-np.infty, +np.infty),
             interactive: bool=False, 
             baseline: bool=True, 
             response: bool=False,
             model: bool=True,
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
        interactive: if True, plot with sliders to adjust range
        units: one of "sec"=seconds, "ms"=millisec, "min"=minutes, "h"=hours
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
        self._plot(plot_range, overlays, overlay_labels, units, interactive)

            
    def estimate_baseline(self, method: str="envelope_iter_bspline_2", **kwargs):
        """
        Apply one of the baseline-estimation methods.
        
        Parameters
        ----------
        
        method: 
            "envelope_iter_bspline_1": :py:func:`pypillometry.baseline.baseline_envelope_iter_bspline()` 
                                        with one iteration
            "envelope_iter_bspline_2": :py:func:`pypillometry.baseline.baseline_envelope_iter_bspline()` 
                                        with two iterations
            
        kwargs:
            named arguments passed to the low-level function in :py:mod:`pypillometry.baseline`.
            
        Note
        -----
        the results of the estimation is stored in member `baseline`
        
        """
        if method=="envelope_iter_bspline_2":
            txd,syd,base2,base1=baseline_envelope_iter_bspline(self.tx, self.sy,self.event_onsets,self.fs,**kwargs)
            f=interpolate.interp1d(txd, base2, kind="cubic", bounds_error=False, fill_value="extrapolate")
            self.baseline=f(self.tx)
        elif method=="envelope_iter_bspline_1": 
            txd,syd,base2,base1=baseline_envelope_iter_bspline(self.tx, self.sy,self.event_onsets,self.fs,**kwargs)
            f=interpolate.interp1d(txd, base1, kind="cubic", bounds_error=False, fill_value="extrapolate")
            self.baseline=f(self.tx)            
        else:
            raise ValueError("Undefined method for baseline estimation: %s"%method)         
        self.baseline_estimated=True
        return self

    def stat_per_event(self, interval: Tuple[float,float], statfct: Callable=np.mean):
        """
        Return result of applying a statistical function to pupillometric data in a
        given interval relative to event-onsets. For example, extract mean 
        pupil-size in interval before trial onset.

        Parameters
        -----------
        interval : tuple (min,max)
            time-window in ms relative to event-onset (0 is event-onset)

        statfct : function
            function mapping np.array to a single number

        Returns
        --------

        result: np.array
            number of event-onsets long result array
        """
        return stat_event_interval(self.tx, self.sy, self.event_onsets, interval, statfct)
        
    def estimate_response(self, npar: Union[str,float]="free", tmax: Union[str,float]="free", 
                          verbose: int=50,
                          bounds: dict={"npar":(1,20), "tmax":(100,2000)}):
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
        self.response_pars={"npar":npar_est,
                            "npar_free":True if npar=="free" else False,
                            "tmax":tmax_est,
                            "tmax_free":True if tmax=="free" else False,
                            "coef":coef,
                            "bounds":bounds
                           }
        
        self.response=pred
        self.response_x=x1
        self.response_estimated=True
        return self
        
    
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
                 name: Optional[str]=None,
                 sim_params: dict={},
                 real_baseline: Optional[PupilArray]=None,
                 real_response_coef: Optional[PupilArray]=None):
        """
        """
        super().__init__(pupil,sampling_rate,time,event_onsets,name)
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

        
    def unscale(self, mean: Optional[float]=None, sd: Optional[float]=None):
        """
        Scale back to original values using either values provided as arguments
        or the values stored in `scale_params`.
        
        Parameters
        ----------
        mean: mean to add from signal
        sd: sd to scale with        
        """
        mmean,ssd=self.scale_params["mean"],self.scale_params["sd"]
        super().unscale(mean,sd)
        self.sim_baseline=(self.sim_baseline*ssd)+mmean
        self.sim_response=(self.sim_response*ssd)
        return self
        
    def scale(self, mean: Optional[float]=None, sd: Optional[float]=None) -> None:
        """
        Scale the pupillary signal by subtracting `mean` and dividing by `sd`.
        If these variables are not provided, use the signal's mean and std.
        
        Parameters
        ----------
        
        mean: mean to subtract from signal
        sd: sd to scale with
        
        Note
        ----
        Scaling-parameters are being saved in the `scale_params` argument. 
        """
        super().scale(mean,sd)
        mean,sd=self.scale_params["mean"],self.scale_params["sd"]
        self.sim_baseline=(self.sim_baseline-mean)/sd
        self.sim_response=(self.sim_response)/sd
        return self

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
        self._plot(plot_range, overlays, overlay_labels, units, interactive)

        
        
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
    ds=FakePupilData(sy,sim_params["fs"],tx, event_onsets,sim_params=sim_params, 
                     real_baseline=baseline, real_response_coef=response_coef)
    return ds
    
    