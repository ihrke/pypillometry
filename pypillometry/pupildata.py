"""
pupildata.py
============

Main object-oriented entry point
"""

from .convenience import *
from .baseline import *

import numpy as np
import scipy.signal as signal
from scipy import interpolate
import scipy
from random import choice

import copy
import math

#from pytypes import typechecked
from typing import Sequence, Union, List, TypeVar, Optional
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
        
        
    def __repr__(self) -> str:
        """Return a string-representation of the dataset."""
        pars=dict(
            n=len(self),
            nmiss=np.sum(np.isnan(self.sy)),
            nevents=self.nevents(), 
            fs=self.fs, 
            duration=len(self)/self.fs/60,
            baseline_estimated=self.baseline_estimated)
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
        if mean is None:
            mean=np.nanmean(self.sy)
        if sd is None:
            sd=np.nanstd(self.sy)
        
        self.scale_params={"mean":mean, "sd":sd}
        self.sy=(self.sy-mean)/sd
        
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
        self.tx=downsample(self.tx, dsfac)
        self.sy=downsample(self.sy, dsfac)
        self.fs=fsd

    def copy(self, new_name: Optional[str]=None) -> PupilData:
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
        
    def plot(self, interactive: bool=False, baseline: bool=True) -> None:
        """
        Make a plot of the pupil data using `matplotlib` or :py:func:`pypillometry.convenience.plot_pupil_ipy()`
        if `interactive=True`.

        Parameters
        ----------
        
        interactive: if True, plot with sliders to adjust range
        """
        if interactive:
            if baseline:
                overlays=(self.baseline,)
            else:
                overlays=tuple()
            plot_pupil_ipy(self.tx, self.sy, self.event_onsets,overlays=overlays)
        else:
            plt.plot(self.tx, self.sy)
            if baseline:
                plt.plot(self.tx, self.baseline)
            
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
            f=interpolate.interp1d(txd, base2, kind="cubic")
            self.baseline=f(self.tx)
        elif method=="envelope_iter_bspline_1": 
            txd,syd,base2,base1=baseline_envelope_iter_bspline(self.tx, self.sy,self.event_onsets,self.fs,**kwargs)
            f=interpolate.interp1d(txd, base1, kind="cubic")
            self.baseline=f(self.tx)            
        else:
            raise ValueError("Undefined method for baseline estimation: %s"%method)         
        self.baseline_estimated=True

    def estimate_response(self):
        """
        
        
        Note
        ----
        the results of the estimation is stored in members `response` and `response_pars`

        """
        if not self.baseline_estimated:
            print("WARNING: no baseline estimated yet, using zero as baseline")
        
        pred, coef, x1=pupil_response(self.tx, self.sy-self.baseline, 
                                      self.event_onsets, self.fs, 
                                      npar="free", tmax="free")
        
        self.response_estimated=True
        
    
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
        self.sim_response_coef=real_response_coef
        
        
    
    
def plotpd(*args: PupilData, subplots: bool=False, baseline: bool=True):
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
        if baseline:
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
        "noise_amp":0.0002
    }
    sim_params.update(kwargs)
    print(sim_params)
    tx,sy,baseline,event_onsets,response_coef=pp.get_dataset(**sim_params)
    ds=FakePupilData(sy,sim_params["fs"],tx, event_onsets,sim_params=sim_params, 
                     real_baseline=baseline, real_response_coef=response_coef)
    return ds
    
    