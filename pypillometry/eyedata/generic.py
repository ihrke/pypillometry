"""
eyedata_generic.py
==================

Generic Eyedata class for use with the pypillometry package.
All other eyedata classes should inherit from this class.
"""

from collections.abc import Iterable
from .. import io
from ..convenience import sizeof_fmt, ByteSize
from .eyedatadict import CachedEyeDataDict, EyeDataDict
from ..signal import baseline
from ..intervals import stat_event_interval, get_interval_stats, merge_intervals

import numpy as np
import itertools
from loguru import logger
#from pytypes import typechecked
from typing import Sequence, Union, List, TypeVar, Optional, Tuple, Callable, Dict, Any
import functools
from random import choice
import copy
import pickle
import inspect
import h5py
import tempfile
import os
import sys

## abstract base class to enforce implementation of some functions for all classes
from abc import ABC, abstractmethod 


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
        

class GenericEyeData(ABC):
    """
    Generic class for eyedata. 
    Defines the basic structure of an eyedata object and 
    implements some basic functions.
    """
    name: str  ## name of dataset
    fs: float  ## sampling rate
    data: EyeDataDict ## dictionary with data (contains ndarrays)
    tx: np.ndarray ## time vector
    event_onsets: np.ndarray ## vector with event onsets in time units
    inplace: bool ## whether to make changes in-place
    notes: str ## optional notes about the dataset

    @abstractmethod
    def __init__():
        """Constructor"""
        pass

    def _init_common(self, time: np.ndarray,
                    sampling_rate: float,
                    event_onsets: np.ndarray,
                    event_labels: np.ndarray,
                    name: str,
                    fill_time_discontinuities: bool,
                    inplace: bool,
                    notes: str = None,
                    use_cache: bool = False,
                    cache_dir: Optional[str] = None,
                    max_memory_mb: float = 100):
        """
        Common code for the child-classes of GenericEyeData.
        Assumes that self.data is already set and filled.

        Private method.
        """
        logger.debug("Initializing common data")
        if self.data is None:
            raise ValueError("data must be available before calling _init_common()")
        
        if time is None and sampling_rate is None:
            raise ValueError("Either `time` or `sampling_rate` must be provided")

        ## name
        if name is None:
            self.name = self._random_id()
        else:
            self.name=name

        ## set time array and sampling rate
        if time is None:
            maxT=len(self.data)/sampling_rate*1000.
            self.tx=np.linspace(0,maxT, num=len(self.data))
        else:
            self.tx=np.array(time, dtype=float)

        if sampling_rate is None:
            self.fs=np.round(1000./np.median(np.diff(self.tx)))
        else:
            self.fs=sampling_rate
            
        self.set_event_onsets(event_onsets, event_labels)

        self._init_blinks()

        ## empty parameter dict
        self.params = dict()

        ## start with empty history    
        self.history=[]            

        ## init whether or not to do operations in place
        self.inplace=inplace 

        ## set notes
        self.notes = notes

        ## Initialize caching if requested
        if use_cache:
            # Create cache directory if it doesn't exist
            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)
                logger.info(f"Created cache directory at {cache_dir}")
            
            # Convert to cached version
            old_data = self.data
            self.data = CachedEyeDataDict(cache_dir=cache_dir, max_memory_mb=max_memory_mb)
            for k, v in old_data.items():
                self.data[k] = v
                if k in old_data.mask:
                    self.data.set_mask(k, old_data.mask[k])

        ## fill in time discontinuities
        if fill_time_discontinuities:
            self.fill_time_discontinuities()

    def _unit_fac(self, units):
        """for converting units"""
        if units=="sec":
            fac=1./1000.
        elif units=="min":
            fac=1./1000./60.
        elif units=="h":
            fac=1./1000./60./60.
        else:
            fac=1.
        return fac

    def _strfy_params(self):
        """
        Return a shortened string representation of the parameters.
        """
        ks = self.params.keys()
        r="{"+",".join(["%s: {...}"%k for k in ks])
        r+="}"
        return r

    def __len__(self):
        """Return number of sampling points"""
        return len(self.tx)

    def nevents(self) -> int:
        """Return number of events in data."""
        return self.event_onsets.size

    def _get_eye_var(self, eyes,variables):
        """Private helper function"""        
        if isinstance(eyes, str):
            eyes=[eyes]
        if len(eyes)==0:
            eyes=self.eyes

        if isinstance(variables, str):
            variables=[variables]
        if len(variables)==0:
            variables=self.variables

        funcname = inspect.stack()[1].function
        logger.debug("%s(): eyes=%s, vars=%s" % (funcname, repr(eyes), repr(variables)))
        return eyes, variables

    def _get_inplace(self, inplace):
        """Private helper function"""
        if inplace is None:
            return self
        else:
            return self if inplace else self.copy()


    @property
    def blinks(self):
        """Return blinks (intervals) for all eyes.

        Returns
        -------
        dict
            dictionary with blinks for each eye
        """
        return self._blinks
    
    def set_blinks(self, eye:str, variable:str, blinks):
        """
        Set blinks for a given eye and variable.

        Parameters
        ----------
        eye: str
            a single eye to set the blinks for
        variable: str
            a single variable to set the blinks for
        blinks: ndarray or None
            ndarrays of blinks (nblinks x 2); 
        """
        if not ( isinstance(blinks, np.ndarray) or blinks is None ):
            raise ValueError("Blinks must be a numpy.ndarray or None")
        
        self._blinks[eye+"_"+variable]=blinks
        # update mask in EyeDataDict
        if blinks is not None:
            for bstart,bend in blinks:
                self.data.mask[eye+"_"+variable][bstart:bend]=1

    def get_blinks(self, eye:str, variable:str):
        """Get blinks for a given eye and variable.

        Parameters
        ----------
        eye : str
            a single eye to get blinks for
        variable : str
            a single variable to get blinks for
        """
        key = eye+"_"+variable
        if key in self._blinks.keys():
            return self._blinks[key]
        else: 
            return None

    def get_blinks_merged(self, eyes=[], variables=[]) -> np.ndarray:
        """Get blinks merged over given eyes and variables.

        Parameters
        ----------
        eyes : list or str
            list of eyes to consider; if empty, all eyes are considered
        variables : list or str
            list of variables to consider; if empty, all variables are considered
        """
        eyes,variables=self._get_eye_var(eyes,variables)

        blinks=[]
        for eye,var in itertools.product(eyes,variables):
            cblinks=self.get_blinks(eye,var)
            if cblinks is None: 
                continue
            elif isinstance(cblinks, np.ndarray):
                blinks += cblinks.tolist()
            elif isinstance(blinks, Iterable):
                blinks += cblinks
        mblinks = merge_intervals(blinks)
        return mblinks

        #bmask=np.any([self.data.mask[eye+"_"+variable] 
        #        for eye,variable in itertools.product(eyes,variables)], 
        #       axis=0)
        #
        #a=np.diff(np.r_[0, bmask, 0])[:-1]
        #bstarts=np.where(a>0)[0]
        #bends=np.where(a<0)[0]
#
        ##z = np.concatenate(([0], bmask, [0]))
        ##start = np.flatnonzero(~z[:-1] & z[1:])   
        ##end = np.flatnonzero(z[:-1] & ~z[1:])
        #blinks = np.column_stack((bstarts, bends))  


    def _init_blinks(self, eyes=[], variables=[]):
        """Initialize blink-arrays for selected eyes and variables.

        All by default

        Parameters
        ----------
        eyes: str or list
            list of eyes to initialize blinks for; if empty, initialize for all
        variables: str or list
            list of variables to initialize blinks for; if empty, initialize for all
        """        
        eyes,variables=self._get_eye_var(eyes,variables)

        ## initialize blinks
        self._blinks={}
        for eye,variable in itertools.product(eyes, variables):
            self.set_blinks(eye,variable, None)        

    def nblinks(self, eyes=[], variables=[]) -> int:
        """
        Return number of detected blinks. Should be run after `detect_blinks()`.

        By default, all eyes and variables are considered.

        Parameters
        ----------
        eyes: list
            list of eyes to consider; if empty, all eyes are considered

        Returns
        -------
        int
            number of detected blinks
        """
        eyes,variables=self._get_eye_var(eyes,variables)
        
        return {eye+"_"+var:self.get_blinks(eye,var).shape[0] 
                for eye,var in itertools.product(eyes,variables)
                if self.get_blinks(eye,var) is not None}


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
            blinks=self.get_blinks(eye,"pupil")
            if blinks is not None:
                blinks=blinks/self.fs*1000*fac
                stats[eye]=get_interval_stats(blinks)            
            else:
                stats[eye]=None
        return stats


    @property
    def eyes(self) -> List[str]:
        """
        Return a list of available eyes in the dataset.
        """
        return self.data.get_available_eyes()
    
    @property
    def variables(self) -> List[str]:
        """
        Return a list of available variables in the dataset.
        """
        return self.data.get_available_variables()

    def get_duration(self, units="min"):
        """Return duration of the dataset in units specified.

        Parameters
        ----------
        units : str, optional
            unit one of "min", "sec", "h", by default "min"

        Returns
        -------
        float
            duration of dataset in specified units
        """        
        fac=self._unit_fac(units)
        return (len(self)/self.fs*1000)*fac

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
        
        Parameters
        ----------
        obj: :class:`GenericEyedata`
            object of class :class:`GenericEyedata` to which the operations are to be transferred
            
        Returns
        -------
        copy of the :class:`GenericEyedata`-object to which the operations in `self` were applied
        """
        for ev in self.history:
            obj=getattr(obj, ev["funcname"])(*ev["args"], **ev["kwargs"])
        return obj

    def _random_id(self, n:int=8) -> str:
        """
        Create a random ID string that is easy to recognise.
        Based on <http://code.activestate.com/recipes/526619-friendly-readable-id-strings/>.
        """
        v = 'aeiou'
        c = 'bdfghklmnprstvw'

        return ''.join([choice(v if i%2 else c) for i in range(n)])


    @keephistory
    def drop_original(self, inplace=None):
        """
        Drop original dataset from record (e.g., to save space).

        Parameters
        ----------
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes
            if `None`, use the setting of the object (specified in constructor)
        """
        obj = self._get_inplace(inplace)
        obj.original=None
        return obj
    

    @keephistory
    def reset_time(self, t0: float=0, inplace=None):
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
            if `None`, use the setting of the object (specified in constructor)
        """
        obj = self._get_inplace(inplace)
        tmin=self.tx.min()
        obj.tx=(self.tx-tmin)+t0
        obj.event_onsets=(self.event_onsets-tmin)+t0
        return obj

    @keephistory
    def sub_slice(self, 
                start: float=-np.inf, 
                end: float=np.inf, 
                units: str=None, inplace=None):
        """
        Return a new `EyeData` object that is a shortened version
        of the current one (contains all data between `start` and
        `end` in units given by `units` (one of "ms", "sec", "min", "h").
        If units is `None`, use the units in the time vector.

        Parameters
        ----------
        
        start: float
            start for new dataset
        end: float
            end of new dataset
        units: str
            time units in which `start` and `end` are provided.
            (one of "ms", "sec", "min", "h").
            If units is `None`, use the units in the time vector.
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes
            if `None`, use the setting of the object (specified in constructor)
        """
        obj = self._get_inplace(inplace)

        if units is not None: 
            fac=self._unit_fac(units)
            tx = self.tx*fac
            evon=obj.event_onsets*fac
        else: 
            tx = self.tx.copy()
            evon=obj.event_onsets.copy()
        keepix=np.where(np.logical_and(tx>=start, tx<=end))

        ndata={}
        for k,v in obj.data.items():
            ndata[k]=v[keepix]
        obj.data=EyeDataDict(ndata)
        obj.tx=obj.tx[keepix]

        
        keepev=np.logical_and(evon>=start, evon<=end)
        obj.event_onsets=obj.event_onsets[keepev]
        obj.event_labels=obj.event_labels[keepev]
        
        return obj
    
    def write_file(self, fname:str):
        """
        Save to file (using :mod:`pickle`).
        
        Parameters
        ----------
        
        fname: str
            filename
        """
        io.write_pickle(self, fname)
       
    @classmethod
    def from_file(cls, fname:str):
        """
        Reads a :class:`.GenericEyedata` object from a pickle-file.
        Use as ``pypillometry.PupilData.from_file("yourfile.pd")``.
        
        Parameters
        ----------
        
        fname: str
            filename
        """
        r=io.read_pickle(fname)
        return r

    @abstractmethod
    def summary(self) -> dict:
        """Return a summary of the :class:`.GenericEyedata`-object."""
        pass

    def __repr__(self) -> str:
        """Return a string-representation of the dataset."""
        pars=self.summary()
        del pars["name"]
        size = self.get_size()
        s="{cname}({name}, {size}):\n".format(cname=self.__class__.__name__,
                                              name=self.name, 
                                              size=str(size))
        flen=max([len(k) for k in pars.keys()])
        for k,v in pars.items():
            s+=(" {k:<"+str(flen)+"}: {v}\n").format(k=k,v=v)
        if self.notes:
            s+=" Notes:\n"
            for line in self.notes.split('\n'):
                s+="  " + line + "\n"
        s+=" History:\n *\n"
        try:
            for i,ev in enumerate(self.history):
                s+=" "*(i+1)+"└ " + ev["funcstring"] +"\n"
        except:
            s+=" └no history\n"
        return s
    
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

    def set_event_onsets(self, event_onsets: np.ndarray, event_labels: np.ndarray):
        """
        Set onsets of events in the data

        Parameters
        ----------
        onsets: np.ndarray
            array of onsets (in ms)
        labels: np.ndarray
            array of labels (strings)
        """
        if event_onsets is None:
            self.event_onsets=np.array([], dtype=float)
        else:
            self.event_onsets=np.array(event_onsets, dtype=float)
            
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

    @keephistory
    def fill_time_discontinuities(self, yval=0, print_info=True):
        """
        find gaps in the time-vector and fill them in
        (add zeros to the signal)
        """
        tx=self.tx
        stepsize=np.median(np.diff(tx))
        n=len(self)
        gaps_end_ix=np.where(np.r_[stepsize,np.diff(tx)]>2*stepsize)[0]
        ngaps=gaps_end_ix.size
        if ngaps!=0:
            ## at least one gap here
            if print_info:
                print("> Filling in %i gaps"%ngaps)
            gaps_start_ix=gaps_end_ix-1
            print( ((tx[gaps_end_ix]-tx[gaps_start_ix])/1000), "seconds" )
            
            ## build new time-vector
            ntx=[tx[0:gaps_start_ix[0]]] # initial
            for i in range(ngaps):
                start,end=gaps_start_ix[i], gaps_end_ix[i]
                # fill in the gap
                ntx.append( np.linspace(tx[start],tx[end], int((tx[end]-tx[start])/stepsize), endpoint=False) )

                # append valid signal
                if i==ngaps-1:
                    nstart=n
                else:
                    nstart=gaps_start_ix[i+1]
                ntx.append( tx[end:nstart] )
            ntx=np.concatenate(ntx)

            ## fill in missing data
            newd = {}
            for k,v in self.data.items():
                nv = np.zeros_like(ntx)
                nv=[v[0:gaps_start_ix[0]]]
                for i in range(ngaps):
                    start,end=gaps_start_ix[i], gaps_end_ix[i]
                    nv.append( yval*np.ones_like(ntx[start:end], dtype=float) )
                # append valid signal
                if i==ngaps-1:
                    nstart=n
                else:
                    nstart=gaps_start_ix[i+1]                    
                nv.append( v[end:nstart] )
                newd[k]=np.concatenate(nv)
            
            self.data=EyeDataDict(newd)
            self.tx=ntx
        return self
    
    def get_intervals(self, 
                    event_select,
                    interval: tuple=(-200,200),
                    units: str="ms", **kwargs):
        """
        Return a list of intervals relative to event-onsets. For example, extract
        the interval before and after a stimulus has been presented.
        It is possible to select based on the event label, e.g. only select
        events matching "stimulus" or "response". It is also 
        possible to select the intervals situated between two different events,
        e.g., trial start and response by specifying `event_select` as a tuple
        with two entriess.

        Parameters
        -----------
        event_select: str, tuple or function
            variable describing which events to select and align to
            - if str: use all events whose label contains the string
            - if function: apply function to all labels, use those where the function returns True
            - if tuple: use all events between the two events specified in the tuple. 
                Both selectors must result in identical number of events.

        interval : tuple (min,max)
            time-window relative to event-onset (0 is event-onset); i.e., negative
            numbers are before the event, positive numbers after. Units are defined
            by the `units` parameter

        units : str
            units of the interval (one of "ms", "sec", "min"); units=None means
            that the interval in sampling units (i.e., indices into the time-array)

        kwargs : dict
            passed onto the event_select function
            
        Returns
        --------

        result: 
            interval on- and offsets for each match, units determined by `units`
        """
        fac=self._unit_fac(units)

        if not isinstance(interval, Iterable):
            raise ValueError("interval must be iterable")
        if interval[1]<=interval[0]:
            raise ValueError("interval must be (min,max) with min<max, got {}".format(interval))
        if isinstance(event_select, tuple):
            # two events and interval in between
            if len(event_select)!=2:
                raise ValueError("event_select must be tuple with two entries")
            event_ix=[None,None]
            for i,evsel in enumerate(event_select):
                if callable(evsel):
                    event_ix[i]=np.array([bool(evsel(evlab, **kwargs)) 
                                          for evlab in self.event_labels])
                elif isinstance(evsel, str):
                    event_ix[i]=np.array([evsel in evlab for evlab in self.event_labels])
                else:
                    raise ValueError("event_select must be string or function")
                if np.sum(event_ix[i])==0:
                    raise ValueError("no events found matching event_select")
            if np.sum(event_ix[0])!=np.sum(event_ix[1]):
                raise ValueError("event_select must result in same number of events for both "
                "selectors, got {} and {}".format(np.sum(event_ix[0]), np.sum(event_ix[1])))
            
            if units is None:
                eix1=np.argmin(np.abs(self.tx-self.event_onsets[event_ix[0]]))
                eix2=np.argmin(np.abs(self.tx-self.event_onsets[event_ix[1]]))
                sti=min(eix1+interval[0], 0)
                ste=max(eix2+interval[1], len(self.tx))
            else:
                sti=(self.event_onsets[event_ix[0]]*fac)+interval[0]
                ste=(self.event_onsets[event_ix[1]]*fac)+interval[1]
        else:
            # one event with padding interval
            if callable(event_select):
                event_ix=np.array([bool(event_select(evlab, **kwargs)) 
                                   for evlab in self.event_labels])
            elif isinstance(event_select, str):
                event_ix=np.array([event_select in evlab for evlab in self.event_labels])
            else:
                raise ValueError("event_select must be string or function")
            if np.sum(event_ix)==0:
                raise ValueError("no events found matching event_select")
            if units is None:
                eix=np.array([np.argmin(np.abs(self.tx-ev)) 
                              for ev in self.event_onsets[event_ix]])
                sti=np.maximum(eix+interval[0], 0)
                ste=np.minimum(eix+interval[1], len(self.tx))
            else:
                sti=(self.event_onsets[event_ix]*fac)+interval[0]
                ste=(self.event_onsets[event_ix]*fac)+interval[1]
        
        intervals = [(s,e) for s,e in zip(sti,ste)]

        return intervals

    @keephistory
    def scale(self, variables=[], mean: Union[float,dict,None]=None, 
              sd: Union[float,dict,None]=None, eyes=[], inplace=None):
        """
        Scale the signal by subtracting `mean` and dividing by `sd`.
        If these variables are not provided, use the signal's mean and std.
        Specify whether to scale `x`, `y`, `pupil` or other variables.
        
        Parameters
        ----------
        
        variables: str or list
            variables to scale; can be "pupil", "x","y", "baseline", "response" or any other variable
            that is available in the dataset; either a string or a list of strings;
            available variables can be checked with `obj.variables`; empty list means all variables
        mean: None, float or dict 
            mean to subtract from signal; if `None`, use the signal's mean
            if dict, provide mean for each eye and variable configuration
        sd: None, float or dict
            sd to scale with; if `None`, use the signal's std
            if dict, provide sd for each eye and variable configuration
        eyes: str or list
            list of eyes to consider; if empty, all available eyes are considered
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes    
            if `None`, use the object's default setting    
        
        Note
        ----
        Scaling-parameters are being saved in the `params["scale"]` argument. 
        """
        obj = self._get_inplace(inplace)
        eyes,variables=self._get_eye_var(eyes,variables)

        if mean is None:
            mean={eye:{var:np.nanmean(obj.data[eye,var]) for var in variables} for eye in eyes}
        elif isinstance(mean, float):
            mean={eye:{var:mean for var in variables} for eye in eyes}            
        elif not isinstance(mean, dict):
            raise ValueError("mean must be None, float or dict")

        if sd is None:
            sd={eye:{var:np.nanstd(obj.data[eye,var]) for var in variables} for eye in eyes}
        elif isinstance(sd, float):
            sd={eye:{var:sd for var in variables} for eye in eyes}
        elif not isinstance(sd, dict):
            raise ValueError("sd must be None, float or dict")
        logger.debug("Mean: %s" % mean)
        logger.debug("SD: %s" % sd)

        # check that all eye/var combinations are present in mean/sd
        for eye in eyes:
            if eye not in mean or eye not in sd:
                raise ValueError("mean and sd must be provided for each eye")
            for var in variables:
                if var not in mean[eye] or var not in sd[eye]:
                    raise ValueError("mean and sd must be provided for each variable")

        # store scaling parameters
        obj.params["scale"]={
            "mean": mean,
            "sd": sd
        }

        for var in variables:
            for eye in eyes:
                if eye not in obj.eyes:
                    raise ValueError("No data for eye %s available" % eye)
                if var not in obj.variables:
                    raise ValueError("No data for variable %s available" % var)
                obj.data[eye, var]=(obj.data[eye, var]-mean[eye][var])/sd[eye][var]
        return obj
        
    @keephistory    
    def unscale(self, variables=[], mean=None, sd=None, eyes=[], inplace=None):
        """
        Scale back to original values using either values provided as arguments
        or the values stored in `scale_params`.
        
        Parameters
        ----------
        variables: str or list
            variables to scale; can be "pupil", "x","y", "baseline", "response" or any other variable
            that is available in the dataset; either a string or a list of strings;
            available variables can be checked with `obj.variables`
        eyes: str or list
            list of eyes to consider; if empty, all available eyes are considered
        mean: None, float or dict
            mean to subtract from signal; if `None`, use the signal's mean
            if dict, provide mean for each eye and variable configuration
        sd: None, float or Parameters
            sd to scale with; if `None`, use the signal's std
            if dict, provide sd for each eye and variable configuration
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes    
            if `None`, use the object's default setting            
        """
        obj = self._get_inplace(inplace)
        eyes,variables=self._get_eye_var(eyes,variables)

        # if no parameters are provided, use the stored ones (normal)
        if mean is None:
            mean=obj.params["scale"]["mean"]
        if sd is None:
            sd=obj.params["scale"]["sd"]
        
        # check whether unscaling parameters are provided
        for eye in eyes:
            if eye not in mean or eye not in sd:
                raise ValueError("mean and sd must be provided for each eye")
            for var in variables:
                if var not in mean[eye] or var not in sd[eye]:
                    raise ValueError("mean and sd must be provided for each variable")

        for eye in eyes:
            for var in variables:                
                if var not in obj.variables:
                    raise ValueError("No data for variable %s available" % var)
                obj.data[eye, var]=(obj.data[eye, var]*sd[eye][var])+mean[eye][var]
        return obj    

    
    @keephistory
    def downsample(self, fsd: float, dsfac: bool=False, inplace=None):
        """
        Simple downsampling scheme using mean within the downsampling window.

        All data fields are downsampled simultaneously.
        See :func:`baseline.downsample()`.

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
        obj = self._get_inplace(inplace)

        if dsfac:
            dsfac=fsd
            fsd=float(obj.fs/dsfac)
        else:
            dsfac=int(obj.fs/fsd) # calculate downsampling factor
            
        # downsample all of the variables/eyes in the data
        ndata={}
        for k,v in obj.data.items():
            ndata[k]=baseline.downsample(v, dsfac)
        obj.data=EyeDataDict(ndata)

        # also downsample the time-vector
        obj.tx=baseline.downsample(obj.tx, dsfac)

        # set new sampling rate            
        obj.fs=fsd
        return obj

    @keephistory
    def merge_eyes(self, eyes=[], variables=[], method="mean", keep_eyes=True, inplace=None):
        """Merge data from both eyes into a single variable.

        Parameters
        ----------
        eyes : list, optional
            list of eyes to merge, by default [] meaning all eyes
        variables : list, optional
            list of variables to merge, by default [] meaning all variables
        method : str, optional
            which method to use for merging (one of "mean", "regress"), by default "mean"
        keep_eyes : bool, optional
            keep original eye data or drop?, by default True
        inplace : make change inplace, optional
            if None, use default value in class, by default None
        """        
        obj = self._get_inplace(inplace)
        eyes,variables=self._get_eye_var(eyes,variables)

        if method=="mean":            
            for var in variables:
                meanval = np.zeros(len(obj.tx))
                for eye in eyes:
                    if var not in obj.variables:
                        raise ValueError("No data for variable %s available" % var)
                    if eye not in obj.eyes:
                        raise ValueError("No data for eye %s available" % eye)
                    meanval += obj.data[eye,var]
                meanval /= len(eyes)
                obj.data["mean",var]=meanval
                if not keep_eyes:
                    for eye in eyes:
                        logger.debug("Dropping eye %s for variable %s" % (eye, var))
                        del obj.data[eye+"_"+var]
                        obj.set_blinks(eye,var,None)
        else:
            raise ValueError("Method %s not implemented" % method)

        return obj
        
    @keephistory
    def blinks_merge(self, eyes=[], variables=[], distance: float=100, inplace=None):
        """Merge together blinks that are close together. 

        Some subjects blink repeatedly and standard detection/interpolation can result in weird results.
        This function simply treats repeated blinks as one long blink.

        Parameters
        ----------

        eyes: str or list
            list of eyes to consider; if empty, consider all
        variables: str or list
            list of variables to consider; if empty, consider all
        distance: float
            merge together blinks that are closer together than `distance` in ms
        inplace: bool
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                                    
        """
        obj = self._get_inplace(inplace)
        eyes,variables=self._get_eye_var(eyes,variables)

        distance_ix=distance/self.fs*1000.

        for eye,var in itertools.product(eyes, variables):
            blinks = obj.get_blinks(eye, var)
            if blinks is None:
                continue

            newblinks=[] 
            i=1
            cblink=blinks[0,:] ## start with first blink
            while(i<blinks.shape[0]):
                if (blinks[i,0]-cblink[1])<=distance_ix:
                    # merge
                    cblink[1]=blinks[i,1]
                else:
                    newblinks.append(cblink)
                    cblink=blinks[i,:]
                i+=1            
            newblinks.append(cblink)
            newblinks=np.array(newblinks)       

            obj.set_blinks(eye,var,newblinks)

        return obj    
            
    def stat_per_event(self, 
                       interval: Tuple[float,float], event_select=None,                        
                       eyes: str|list=[], variables: str|list=[],                       
                       statfct: Callable=np.mean, units: str="ms",
                       **kwargs):
        """
        Return result of applying a statistical function to data in a
        given interval relative to event-onsets. For example, extract mean 
        pupil-size in interval before trial onset.

        Parameters
        -----------

        eyes: str or list
            list of eyes to consider; if empty, consider all
        variables: str or list
            list of variables to consider; if empty, consider all
        event_select: str or function
            variable describing which events to select and align to
            see :class:`GenericEyeData.get_intervals()` for details
        interval : tuple (min,max)
            time-window in ms relative to event-onset (0 is event-onset)
        statfct : function
            function mapping np.array to a single number
        kwargs : dict
            passed onto the event_select function
    
        Returns
        --------

        result: np.array or dict
            number of event-onsets long result array; in case of multiple eyes/variables, a dict is returned
        """
        eyes,variables=self._get_eye_var(eyes,variables)
        units = self._unit_fac(units)
        intervs=self.get_intervals(event_select, interval, units=units, **kwargs)

        stat={}
        for eye,var in itertools.product(eyes, variables):
            stat[eye+"_"+var]=stat_event_interval(self.tx, self.data[eye,var], intervs, statfct)

        return stat

    def set_cache_options(self, use_cache: bool = None, cache_dir: Optional[str] = None, max_memory_mb: float = None):
        """Update cache settings.
        
        Parameters
        ----------
        use_cache : bool, optional
            Whether to use cached storage.
        cache_dir : str, optional
            Directory to store cache files.
        max_memory_mb : float, optional
            Maximum memory usage in MB.
        """
        if use_cache is not None:
            self.use_cache = use_cache
        if cache_dir is not None:
            self.cache_dir = cache_dir
        if max_memory_mb is not None:
            self.max_memory_mb = max_memory_mb
            
        if self.use_cache and isinstance(self.data, EyeDataDict):
            # Convert to cached version
            old_data = self.data
            self.data = CachedEyeDataDict(cache_dir=self.cache_dir, max_memory_mb=self.max_memory_mb)
            for k, v in old_data.items():
                self.data[k] = v
                if k in old_data.mask:
                    self.data.set_mask(k, old_data.mask[k])
        elif not self.use_cache and isinstance(self.data, CachedEyeDataDict):
            # Convert to in-memory version
            old_data = self.data
            self.data = EyeDataDict()
            for k, v in old_data.items():
                self.data[k] = v
                if k in old_data.mask:
                    self.data.set_mask(k, old_data.mask[k])
            old_data.clear_cache()
            
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if using cached storage."""
        if isinstance(self.data, CachedEyeDataDict):
            return self.data.get_cache_stats()
        return None
        
    def clear_cache(self):
        """Clear memory cache if using cached storage."""
        if isinstance(self.data, CachedEyeDataDict):
            self.data.clear_cache()
            
    def set_max_memory(self, max_memory_mb: float):
        """Set maximum memory usage in MB if using cached storage."""
        if isinstance(self.data, CachedEyeDataDict):
            self.data.set_max_memory(max_memory_mb)
            self.max_memory_mb = max_memory_mb

    def get_size(self) -> ByteSize:
        """Return the size of the object in bytes.
        
        Returns
        -------
        ByteSize
            Total size of the object, including data and all attributes.
        """
        # Get size of the data dictionary
        data_size = self.data.get_size()
        
        # Calculate size of other attributes
        other_size = 0
        
        # Time vector and event arrays
        other_size += self.tx.nbytes
        other_size += self.event_onsets.nbytes if self.event_onsets is not None else 0
        other_size += self.event_labels.nbytes if self.event_labels is not None else 0
        
        # String attributes
        other_size += sys.getsizeof(self.name) if self.name is not None else 0
        other_size += sys.getsizeof(self.notes) if self.notes is not None else 0
        
        # Numeric attributes
        other_size += sys.getsizeof(self.fs)
        other_size += sys.getsizeof(self.inplace)
        
        # Dictionary attributes
        other_size += sys.getsizeof(self.params)
        for k, v in self.params.items():
            other_size += sys.getsizeof(k)
            other_size += sys.getsizeof(v)
            
        # History list
        other_size += sys.getsizeof(self.history)
        for item in self.history:
            other_size += sys.getsizeof(item)
            if isinstance(item, dict):
                for k, v in item.items():
                    other_size += sys.getsizeof(k)
                    other_size += sys.getsizeof(v)
        
        if data_size.is_cached():
            # For cached objects, add other_size to memory usage
            return ByteSize({
                'memory': data_size-data_size.cached_bytes + other_size,
                'disk': data_size.cached_bytes
            })
        else:
            # For non-cached objects, return total size
            return ByteSize(data_size + other_size)

    @keephistory
    def merge_masks(self, inplace=None):
        """Merge masks of all variables into a single joint mask.
        
        This function creates a joint mask by taking the logical OR of all individual masks
        using `get_mask()`, then applies this joint mask to all variables.
        
        Parameters
        ----------
        inplace : bool, optional
            If True, make change in-place and return the object.
            If False, make and return copy before making changes.
            If None, use the object's default setting.
            
        Returns
        -------
        GenericEyeData
            The object with merged masks.
        """
        obj = self._get_inplace(inplace)
        
        # Get joint mask across all variables
        joint_mask = obj.data.get_mask()
        
        # Apply joint mask to all variables
        for key in obj.data.keys():
            obj.data.set_mask(key, joint_mask)
            
        return obj