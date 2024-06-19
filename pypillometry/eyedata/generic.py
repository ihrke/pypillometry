"""
eyedata_generic.py
==================

Generic Eyedata class for use with the pypillometry package.
All other eyedata classes should inherit from this class.
"""

import numpy as np
from .. import io
from ..convenience import sizeof_fmt
from .eyedatadict import EyeDataDict

#from pytypes import typechecked
from typing import Sequence, Union, List, TypeVar, Optional, Tuple, Callable
import functools
from random import choice
import copy
import pickle

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
    missing: np.ndarray ## missing data vector (1=missing, 0=not missing)
    event_onsets: np.ndarray ## vector with event onsets in time units
    inplace: bool ## whether to make changes in-place

    @abstractmethod
    def __init__():
        """Constructor"""
        pass

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

    def __len__(self):
        """Return number of sampling points"""
        return len(self.tx)

    def nevents(self) -> int:
        """Return number of events in data."""
        return self.event_onsets.size


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
        
        Parameters:
        -----------
        
        obj: :class:`.GenericEyedata`
            object of class :class:`.GenericEyedata` to which the operations are to be transferred
            
        Returns:
        --------
        
        copy of the :class:`.GenericEyedata`-object to which the operations in `self` were applied
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
        obj=self if inplace else self.copy()
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
        tmin=self.tx.min()
        obj=self if inplace else self.copy()            
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
        if inplace is None:
            inplace=self.inplace
        obj=self if inplace else self.copy()
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
    
    def size_bytes(self):
        """
        Return size of current dataset in bytes.
        """
        nbytes=len(pickle.dumps(self, -1))
        return nbytes

    def write_file(self, fname:str):
        """
        Save to file (using :mod:`pickle`).
        
        Parameters
        ----------
        
        fname: str
            filename
        """
        io.eyedata_write_pickle(self, fname)
       
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
        r=io.eyedata_read_pickle(fname)
        return r

    @abstractmethod
    def summary(self) -> dict:
        """Return a summary of the :class:`.GenericEyedata`-object."""
        pass

    def __repr__(self) -> str:
        """Return a string-representation of the dataset."""
        pars=self.summary()
        del pars["name"]
        s="{cname}({name}, {size}):\n".format(cname=self.__class__.__name__,
                                              name=self.name, 
                                              size=sizeof_fmt(self.size_bytes()))
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
                    padding: tuple=(-200,200),
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

        padding : tuple (min,max)
            time-window relative to event-onset (0 is event-onset); i.e., negative
            numbers are before the event, positive numbers after. Units are defined
            by the `units` parameter

        units : str
            units of the padding (one of "ms", "sec", "min"); units=None means
            that the padding is in time units taken from the data

        kwargs : dict
            passed onto the event_select function
            
        Returns
        --------

        result: dict with 3 np.arrays (label, start, end)
            interval on- and offsets for each match, units determined by `units`
        """
        if units is None:
            fac=1.0
        else:
            fac=self._unit_fac(units)

        if padding[1]<=padding[0]:
            raise ValueError("padding must be (min,max) with min<max, got {}".format(padding))
        if isinstance(event_select, tuple):
            # two events and interval in between
            if len(event_select)!=2:
                raise ValueError("event_select must be tuple with two entries")
            event_ix=[None,None]
            for i,evsel in enumerate(event_select):
                if callable(evsel):
                    event_ix[i]=np.array([bool(evsel(evlab, **kwargs)) for evlab in self.event_labels])
                elif isinstance(evsel, str):
                    event_ix[i]=np.array([evsel in evlab for evlab in self.event_labels])
                else:
                    raise ValueError("event_select must be string or function")
                if np.sum(event_ix[i])==0:
                    raise ValueError("no events found matching event_select")
            if np.sum(event_ix[0])!=np.sum(event_ix[1]):
                raise ValueError("event_select must result in same number of events for both selectors, got {} and {}".format(np.sum(event_ix[0]), np.sum(event_ix[1])))
            sti=(self.event_onsets[event_ix[0]]*fac)+padding[0]
            ste=(self.event_onsets[event_ix[1]]*fac)+padding[1]
        else:
            # one event with padding interval
            if callable(event_select):
                event_ix=np.array([bool(event_select(evlab, **kwargs)) for evlab in self.event_labels])
            elif isinstance(event_select, str):
                event_ix=np.array([event_select in evlab for evlab in self.event_labels])
            else:
                raise ValueError("event_select must be string or function")
            if np.sum(event_ix)==0:
                raise ValueError("no events found matching event_select")
            sti=(self.event_onsets[event_ix]*fac)+padding[0]
            ste=(self.event_onsets[event_ix]*fac)+padding[1]
        
        intervals = [(s,e) for s,e in zip(sti,ste)]

        return intervals
