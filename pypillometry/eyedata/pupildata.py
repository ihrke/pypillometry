"""
pupildata.py
============

Class representing pupillometric data.
"""

import itertools
from .eyedatadict import EyeDataDict
from .generic import GenericEyeData, keephistory, IntervalStats
#from .. import convenience
from ..signal import baseline
from ..signal import preproc
from ..signal import pupil
#from .. import io
from ..plot import PupilPlotter


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
                 inplace: bool = False):
        """Constructor for PupilData object.
        """

        logger.info("Creating PupilData object")
        if (left_pupil is None and right_pupil is None):
            raise ValueError("At least one of the eyes, left_pupil or right_pupil, must be provided")
        self.data=EyeDataDict(left_pupil=left_pupil, right_pupil=right_pupil)

        self._init_common(time, sampling_rate, 
                          event_onsets, event_labels, 
                          name, fill_time_discontinuities, inplace)

        # store original
        self.original=None
        if keep_orig: 
            self.original=self.copy()

    @property
    def plot(self):
        return PupilPlotter(self)


    def blink_stats(self, eyes=[], units: str="ms"):
        """
        Return statistics on blinks.

        Parameters
        ----------
        eyes: list
            list of eyes to process; if empty, all available eyes are processed
        units: str
            one of "ms", "sec", "min", "h"
        """
        eyes,_=self._get_eye_var(eyes,[])
        fac=self._unit_fac(units)

        stats=dict()
        for eye in eyes:
            blinks=self.get_blinks(eye, "pupil")
            blink_durations=[(off-on)/self.fs*1000*fac for on,off in blinks]
            stats[eye]=IntervalStats(
                n=len(blink_durations),
                mean=np.mean(blink_durations),
                median=np.median(blink_durations),
                min=np.min(blink_durations),
                max=np.max(blink_durations),
                sd=np.std(blink_durations)
            )
        return stats

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
            nmiss=np.sum(self.missing),
            perc_miss=np.sum(self.missing)/len(self)*100.,
            duration_minutes=self.get_duration("min"),
            start_min=self.tx.min()/1000./60.,
            end_min=self.tx.max()/1000./60.,
            ninterpolated={eye:self.data[eye+"_interpolated"].sum() for eye in self.eyes if eye+"_interpolated" in self.data},
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
    def pupil_estimate_baseline(self, eyes=[],
                                method: str="envelope_iter_bspline_2", inplace=None, **kwargs):
        """
        Apply one of the baseline-estimation methods.
        
        Parameters
        ----------
        
        eyes: list or str
            str or list of eyes to process; if empty, all available eyes are processed
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

        for eye in eyes:
            logger.info("Estimating baseline for eye %s"%eye)

            if method=="envelope_iter_bspline_2":
                txd,syd,base2,base1=baseline.baseline_envelope_iter_bspline(self.tx, self.data[eye,"pupil"],
                                                                            self.event_onsets,self.fs,**kwargs)
                f=interpolate.interp1d(txd, base2, kind="cubic", bounds_error=False, fill_value="extrapolate")
                obj.data[eye,"baseline"]=f(self.tx)
            elif method=="envelope_iter_bspline_1": 
                txd,syd,base2,base1=baseline.baseline_envelope_iter_bspline(self.tx, self.data[eye,"pupil"],
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
    
