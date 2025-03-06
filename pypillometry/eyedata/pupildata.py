"""
pupildata.py
============

Class representing pupillometric data.
"""

import itertools
from .eyedatadict import EyeDataDict
from .generic import GenericEyeData, keephistory
from ..parameters import Parameters
#from .. import convenience
from ..signal import baseline
#from .. import io
from ..plot import PupilPlotter


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

        ## set plotter 
        self.plot=PupilPlotter(self)

        ## default parameters for scaling signal
        self.scale_params=Parameters()

        ## initialize baseline signal
        self.baseline_estimated=False
        
        ## initialize response-signal
        self.response_params=Parameters()
        self.response_estimated=False
        
        ## initialize blinks
        self.blinks={eye:np.empty((0,2), dtype=int) for eye in self.eyes}
        
        ## masks for blinks and interpolated segments of the data
        for eye in self.eyes:
            self.data[eye+"_blinkmask"]=np.zeros(len(self), dtype=int)
            self.data[eye+"_interpolated"]=np.zeros(len(self), dtype=int)

        # store original
        self.original=None
        if keep_orig: 
            self.original=self.copy()
       
    def nblinks(self, eyes=[]) -> int:
        """
        Return number of detected blinks. Should be run after `detect_blinks()`.

        Parameters
        ----------
        eyes: list
            list of eyes to consider; if empty, all eyes are considered

        Returns
        -------
        int
            number of detected blinks
        """
        if len(eyes)==0:
            eyes=self.eyes
        return {eye:self.blinks[eye].shape[0] for eye in eyes}

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
            blinks_per_min={k:v/(len(self)/self.fs/60.) for k,v in self.nblinks().items()},
            nmiss=np.sum(self.missing),
            perc_miss=np.sum(self.missing)/len(self)*100.,
            duration_minutes=self.get_duration("min"),
            start_min=self.tx.min()/1000./60.,
            end_min=self.tx.max()/1000./60.,
            ninterpolated={eye:self.data[eye+"_interpolated"].sum() for eye in self.eyes},
            baseline_estimated=self.baseline_estimated,
            response_estimated=self.response_estimated,
            scale_params=self.scale_params,
            response_params=self.response_params,
            glimpse=repr(self.data)
        )
        
        return summary            

    @keephistory
    def pupil_lowpass_filter(self,  cutoff: float, order: int=2, eyes=[], inplace=None):
        """
        Lowpass-filter pupil signal using a Butterworth-filter, 
        see :py:func:`pypillometry.baseline.butter_lowpass_filter()`.
    
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
        if inplace is None:
            inplace=self.inplace
        obj=self if inplace else self.copy()
        if len(eyes)==0:
            eyes=obj.eyes
        for eye in eyes:
            obj.data[eye,"pupil"]=baseline.butter_lowpass_filter(obj.data[eye,"pupil"], cutoff, obj.fs, order)
        return obj
