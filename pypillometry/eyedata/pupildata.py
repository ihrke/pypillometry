"""
pupildata.py
============

Class representing pupillometric data.
"""

import itertools
from typing import Optional, Tuple
from .eyedatadict import EyeDataDict
from ..erpd import ERPD
from .generic import GenericEyeData, keephistory
#from .. import convenience
from ..signal import baseline
from ..signal import preproc
from ..signal import pupil
#from .. import io
from ..plot import PupilPlotter
from ..intervals import IntervalStats, get_interval_stats

import json
from loguru import logger
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate
import scipy

class PupilData(GenericEyeData):
    """
    Class representing pupillometric data. 

    The class is a subclass of :class:`.GenericEyedata` and inherits all its methods.

    If eye-tracking data is available in addition to pupillometry, use the :class:`.EyeData` class.

    Parameters
    ----------
    time: 
        timing array or `None`, in which case the time-array goes from [0,maxT]
        using `sampling_rate` (in ms)
    left_pupil:
        data from left eye (at least one of the eyes must be provided)
    right_pupil:
        data from right eye (at least one of the eyes must be provided)
    sampling_rate: float
        sampling-rate of the signal in Hz; if None, 
    name: 
        name of the dataset or `None` (in which case a random string is selected)
    event_onsets: 
        time-onsets of any events in the data (in ms, matched in `time` vector)
    event_labels:
        for each event in `event_onsets`, provide a label
    keep_orig: bool
        keep a copy of the original dataset? If `True`, a copy of the object
        as initiated in the constructor is stored in member `original`
    fill_time_discontinuities: bool
        sometimes, when the eyetracker loses signal, no entry in the EDF is made; 
        when this option is True, such entries will be made and the signal set to 0 there
        (or do it later using `fill_time_discontinuities()`)
    inplace: bool
        if True, the object is modified in place; if False, a new object is returned
        this object-level property can be overwritten by the method-level `inplace` argument
        default is "False"
    use_cache: bool
        Whether to use cached storage for data arrays. Default is False.
    cache_dir: str
        Directory to store cache files. If None, creates a temporary directory.
    max_memory_mb: float
        Maximum memory usage in MB when using cache. Default is 100MB.
    """    
    def __init__(self,
                 time: np.ndarray = None,
                 left_pupil: np.ndarray=None,
                 right_pupil: np.ndarray=None, 
                 event_onsets: np.ndarray = None,
                 event_labels: np.ndarray = None,
                 sampling_rate: float = None,
                 name: str = None,
                 fill_time_discontinuities: bool = True,
                 keep_orig: bool = False,
                 notes: str = None,
                 inplace: bool = False,
                 use_cache: bool = False,
                 cache_dir: Optional[str] = None,
                 max_memory_mb: float = 100):
        """Constructor for PupilData object.
        """

        logger.info("Creating PupilData object")
        if (left_pupil is None and right_pupil is None):
            raise ValueError("At least one of the eyes, left_pupil or right_pupil, must be provided")
        self.data=EyeDataDict(left_pupil=left_pupil, right_pupil=right_pupil)

        self._init_common(time, sampling_rate, 
                          event_onsets, event_labels, 
                          name, fill_time_discontinuities, 
                          notes, inplace,
                          use_cache=use_cache,
                          cache_dir=cache_dir,
                          max_memory_mb=max_memory_mb)

        # store original
        self.original=None
        if keep_orig: 
            self.original=self.copy()

    @property
    def plot(self):
        return PupilPlotter(self)


    def summary(self):
        """
        Return a summary of the dataset as a dictionary.

        Returns
        -------
        dict
            dictionary containing description of dataset
        """

        summary=dict(
            name=self.name, 
            n=len(self),
            sampling_rate=self.fs,
            eyes=self.eyes,
            data=list(self.data.keys()),
            nevents=self.nevents(), 
            nblinks=self.nblinks(), 
            blinks=self.blink_stats(),
            duration_minutes=self.get_duration("min"),
            start_min=self.tx.min()/1000./60.,
            end_min=self.tx.max()/1000./60.,
            params=self._strfy_params(),
            glimpse=repr(self.data)
        )
        
        return summary            

    @keephistory
    def pupil_lowpass_filter(self,  cutoff: float, order: int=2, eyes=[], inplace=None):
        """
        Lowpass-filter pupil signal using a Butterworth-filter, 
        see :func:`baseline.butter_lowpass_filter()`.
    
        Parameters
        -----------
        cutoff: float
            lowpass-filter cutoff
        order: int
            filter order
        eyes: list
            list of eyes to filter; if empty, all available eyes are filtered
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes           
            if `None`, use the object-level setting         
        """
        obj = self._get_inplace(inplace)
        eyes,_=self._get_eye_var(eyes,[])

        for eye in eyes:
            obj.data[eye,"pupil"]=baseline.butter_lowpass_filter(obj.data[eye,"pupil"], cutoff, obj.fs, order)
        return obj

    @keephistory
    def pupil_smooth_window(self, eyes=[], window: str="hanning", winsize: float=11, inplace=None):
        """
        Apply smoothing of the signal using a moving window. See :func:`baseline.smooth_window()`.
        
        Parameters
        ----------
        eyes: list
            list of eyes to smooth; if empty, all available eyes are smoothed
        window: str
            (the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'); 
             flat window will produce a moving average smoothing.
        winsize: float
            the length of the window in ms 
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                            
        """
        obj = self._get_inplace(inplace)
        eyes,_=self._get_eye_var(eyes,[])

        # convert winsize to index based on sampling rate
        winsize_ix=int(np.ceil(winsize/1000.*self.fs)) 

        # process requested eyes
        for eye in eyes:
            obj.data[eye,"pupil"]=preproc.smooth_window(obj.data[eye,"pupil"], winsize_ix, window )

        return obj


    @keephistory
    def pupil_blinks_detect(self, eyes=[], min_duration:float=20, blink_val:float=0,
                      winsize: float=11, vel_onset: float=-5, vel_offset: float=5, 
                      min_onset_len: int=5, min_offset_len: int=5,
                      strategies: list=["zero","velocity"],
                      units="ms", inplace=None):
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
        eyes: list
            list of eyes to process; if empty, all available eyes are processed
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
        obj = self._get_inplace(inplace)
        eyes,_=self._get_eye_var(eyes,[])

        fac=self._unit_fac(units)
        winsize_ms=winsize*fac
        winsize_ix=int(winsize_ms/1000.*self.fs)
        if winsize_ix % 2==0:
            winsize += 1
        min_duration_ms=min_duration*fac
        min_duration_ix=int(min_duration_ms/1000.*self.fs)        

        
        # check for unknown strategies
        for strat in strategies:
            if strat not in ["zero", "velocity"]:
                logger.warning("Strategy '%s' unknown"%strat)
        
        for eye in eyes:
            ## detect blinks with the different strategies
            if "velocity" in strategies:
                blinks_vel=preproc.detect_blinks_velocity(self.data[eye,"pupil"], winsize_ix, vel_onset, vel_offset, min_onset_len, min_offset_len)
            else: 
                blinks_vel=np.array([])
                
            if "zero" in strategies:
                blinks_zero=preproc.detect_blinks_zero(self.data[eye,"pupil"], 1, blink_val)
            else:
                blinks_zero=np.array([])

            ## merge the two blinks
            blinks=preproc.helper_merge_blinks(blinks_vel, blinks_zero)
            obj.set_blinks(eye, "pupil", np.array([[on,off] for (on,off) in blinks if off-on>=min_duration_ix]))
            
        return obj    
    
    @keephistory
    def pupil_blinks_interpolate(self, eyes: str|list=[],
                                 store_as: str="pupil", 
                                 method="mahot", winsize: float=11, 
                                 vel_onset: float=-5, vel_offset: float=5, 
                                 margin: Tuple[float,float]=(10,30), 
                                 blinkwindow: float=500,
                                 interp_type: str="cubic",
                                 inplace=None):
        """Interpolating blinks in the pupillary signal.

        Implements the blink-interpolation method by Mahot (2013).
        
        Mahot, 2013:
        https://figshare.com/articles/A_simple_way_to_reconstruct_pupil_size_during_eye_blinks/688001.

        This procedure relies heavily on eye-balling (reconstructing visually convincing signal),
        so a "plot" option is provided that will plot many diagnostics (see paper linked above) that
        can help to set good parameter values for `winsize`, `vel_onset`, `vel_offset` and `margin`.

        Parameters
        ----------
        eyes: str or list
            str or list of eyes to process; if empty, all available eyes are processed
        store_as: str
            how to store the interpolated data; either "pupil" (default) which replaces
            the original pupil data or a string that will be used as the new variable
            name in the data (e.g., "pupilinterp")
        method: str
            method to use; so far, only "mahot" is implemented
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
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes              
        """
        obj = self._get_inplace(inplace)
        eyes,_=self._get_eye_var(eyes,[])

        if not isinstance(store_as, str):
            raise ValueError("store_as must be a string")

        # parameters in sampling units (from ms)
        winsize_ix=int(np.ceil(winsize/1000.*self.fs)) 
        margin_ix=tuple(int(np.ceil(m/1000.*self.fs)) for m in margin)
        blinkwindow_ix=int(blinkwindow/1000.*self.fs)
        if winsize_ix % 2==0: ## ensure smoothing window is odd
            winsize_ix+=1 

        for eye in eyes:
            syr=obj.data[eye,"pupil"].copy() ## interpolated signal
            mask=obj.data.mask[eye+"_pupil"].copy() # copy of mask
            bls = self.get_blinks(eye, "pupil")
            blink_onsets=preproc.blink_onsets_mahot(obj.data[eye,"pupil"], bls, 
                                                    winsize_ix, 
                                                    vel_onset, vel_offset,
                                                    margin_ix, blinkwindow_ix)
          
            # loop through blinks
            for ix,(onset,offset) in enumerate(blink_onsets):                
                # calc the 4 time points
                t2,t3=onset,offset
                t1=max(0,t2-t3+t2)
                t4=min(t3-t2+t3, len(self)-1)
                if t1==t2:
                    t2+=1
                if t3==t4:
                    t3-=1
                
                txpts=[obj.tx[pt] for pt in [t1,t2,t3,t4]]
                sypts=[obj.data[eye,"pupil"][pt] for pt in [t1,t2,t3,t4]]
                intfct=interp1d(txpts,sypts, kind=interp_type)
                islic=slice(t2, t3)
                syr[islic]=intfct(obj.tx[islic])

                # store interpolated data
                obj.data[eye,store_as]=syr
                # record the interpolated datapoints
                obj.data.mask[eye+"_"+store_as]=mask
                obj.data.mask[eye+"_"+store_as][islic]=1                

        return obj
            


    @keephistory
    def pupil_estimate_baseline(self, eyes=[], variable="pupil",
                                method: str="envelope_iter_bspline_2", inplace=None, **kwargs):
        """
        Apply one of the baseline-estimation methods.
        
        Parameters
        ----------
        
        eyes: list or str
            str or list of eyes to process; if empty, all available eyes are processed
        variable: str
            default is to use the "pupil" but it could be used to process, e.g., 
            interpolated pupil data stored in a different variables, e.g., "pupilinterp"
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
            
        Returns
        -------
        PupilData
            object with baseline estimated (stored in data["eye_baseline"])        
        """
        obj = self._get_inplace(inplace)
        eyes,_=self._get_eye_var(eyes,[])
        if not isinstance(variable, str):
            logger.warning("variable must be a string; using default 'pupil'")
            variable="pupil"

        for eye in eyes:
            logger.info("Estimating baseline for eye %s"%eye)

            if method=="envelope_iter_bspline_2":
                txd,syd,base2,base1=baseline.baseline_envelope_iter_bspline(self.tx, self.data[eye,variable],
                                                                            self.event_onsets,self.fs,**kwargs)
                f=interpolate.interp1d(txd, base2, kind="cubic", bounds_error=False, fill_value="extrapolate")
                obj.data[eye,"baseline"]=f(self.tx)
            elif method=="envelope_iter_bspline_1": 
                txd,syd,base2,base1=baseline.baseline_envelope_iter_bspline(self.tx, self.data[eye,variable],
                                                                            self.event_onsets,self.fs,**kwargs)
                f=interpolate.interp1d(txd, base1, kind="cubic", bounds_error=False, fill_value="extrapolate")
                obj.data[eye,"baseline"]=f(self.tx)
            else:
                raise ValueError("Undefined method for baseline estimation: %s"%method)         
            
        return obj
        
    @keephistory
    def pupil_estimate_response(self, 
                          eyes=[], 
                          npar: str|float="free", tmax: str|float="free", 
                          verbose: int=50,
                          bounds: dict={"npar":(1,20), "tmax":(100,2000)},
                          inplace=None):
        """
        Estimate pupil-response based on event-onsets, see
        :py:func:`pypillometry.pupil.pupil_response()`.
        

        Parameters
        ----------
        eyes: list or str
            str or list of eyes to process; if empty, all available eyes are processed
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
        obj = self._get_inplace(inplace)
        eyes,_=self._get_eye_var(eyes,[])

        obj.params["response"]=dict()
        for eye in eyes:
            logger.info("Estimating response for eye %s"%eye)
            if not eye+"_baseline" in obj.data.keys():
                logger.warning("Eye %s: no baseline estimated yet, using zero as baseline"%eye)
                base=np.zeros(len(obj.tx))
            else:
                base=obj.data[eye,"baseline"]
        
            syd = obj.data[eye,"pupil"]-base
            pred, coef, npar_est, tmax_est, x1=pupil.pupil_response(obj.tx, syd, 
                                                            obj.event_onsets, obj.fs, 
                                                            npar=npar, tmax=tmax, verbose=verbose,
                                                            bounds=bounds)
            
            obj.data[eye+"_response"]=pred

            obj.params["response"][eye]={"npar":npar_est,
                                "npar_free":True if npar=="free" else False,
                                "tmax":tmax_est,
                                "tmax_free":True if tmax=="free" else False,
                                "coef":coef,
                                "bounds":bounds
                            }
        
        return obj
    

    def get_erpd(self, erpd_name: str, event_select, 
                 eyes: list=[], variable: str="pupil",
                 baseline_win: Optional[Tuple[float,float]]=None, 
                 interval: Tuple[float,float]=(-500, 2000), 
                 **kwargs):
        """
        Extract event-related pupil dilation (ERPD).
        No attempt is being made to exclude overlaps of the time-windows.

        Parameters
        ----------
        erpd_name: str
            identifier for the result (e.g., "cue-locked" or "conflict-trials")
        eyes: list or str
            str or list of eyes to process; if empty, all available eyes are processed
        variable: str
            default is to use the "pupil" but it could be used to process, e.g.,
            interpolated pupil data stored in a different variables, e.g., "pupilinterp"
        baseline_win: tuple (float,float) or None
            if None, no baseline-correction is applied
            if tuple, the mean value in the window in milliseconds (relative to `time_win`) is 
                subtracted from the single-trial ERPDs (baseline-correction)
        event_select: str or function
            variable describing which events to select and align to
            - if str: use all events whose label contains the string
            - if function: apply function to all labels, use those where the function returns True
            see :class:`GenericEyeData.get_intervals()` for details
        interval: Tuple[float, float]
            time before and after event to include (in ms)
        kwargs:
            additional arguments passed to the `event_select` function
        """
        eyes,_=self._get_eye_var(eyes,[])
        if not isinstance(variable, str):
            logger.warning("variable must be a string; using default 'pupil'")
            variable="pupil"

        # convert interval into sampling units
        interval_ix=tuple(( int(np.ceil(tw/1000.*self.fs)) for tw in interval ))
        duration_ix=interval_ix[1]-interval_ix[0]
        txw=np.linspace(interval[0], interval[1], num=duration_ix)

        # units=None means, we get indices into self.tx back from self.get_intervals()
        # use inter in sampling units with units=None to find closest points in tx
        intervals = self.get_intervals(event_select, interval_ix, units=None, **kwargs)
        nintv=len(intervals)


        data = EyeDataDict()

        for eye in eyes:
            # matrix/mask for the ERPDs for each eye
            erpd=np.zeros((duration_ix,nintv))
            mask=np.ones((duration_ix,nintv))

            for i,(on,off) in enumerate(intervals):
                onl,offl=0,duration_ix # "local" window indices
            
                if on<0: ## pad with zeros in case timewindow starts before data
                    onl=np.abs(on)
                    on=0
                if off>=self.tx.size-1:
                    offl=(off-on)
                    off=self.tx.size
                #print(on,off,onl,offl)
                erpd[onl:offl,i]=self.data[eye,variable][on:off]
                mask[onl:offl,i]=self.data.mask[eye+"_"+variable][on:off]

            data[eye,"erpd"]=erpd
            data.mask[eye+"_erpd"]=mask
        
        rerpd = ERPD(erpd_name, txw, data)
        if baseline_win is not None:
            blwin=np.array(baseline_win)
            if np.any(blwin<interval[0]) or np.any(blwin>=interval[1]):
                logger.warning("Baseline window misspecified %s vs. %s; "
                               "NOT doing baseline correction"%(baseline_win, interval))
            else:
                rerpd.baseline_correct(baseline_win=baseline_win) 

        return rerpd