"""
eyedata_generic.py
==================

Generic Eyedata class for use with the pypillometry package.
All other eyedata classes should inherit from this class.
"""

import numpy as np
from pypillometry import _inplace
from .io import *
from .convenience import sizeof_fmt

#from pytypes import typechecked
from typing import Sequence, Union, List, TypeVar, Optional, Tuple, Callable
PupilArray=Union[np.ndarray, List[float]]
import functools
from random import choice

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
        

from collections.abc import MutableMapping
class EyeDataDict(MutableMapping):
    """
    A dictionary that contains 1-dimensional ndarrays of equal length
    and with the same datatype (float).
    Drops empty entries (None or length-0 ndarrays).
    """

    def __init__(self, *args, **kwargs):
        self.data = dict()
        self.length=0
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if value is None or len(value)==0:
            return
        value=np.array(value)
        if not isinstance(value, np.ndarray):
            raise ValueError("Value must be numpy.ndarray")
        if len(value.shape)>1:
            raise ValueError("Array must be 1-dimensional")
        if self.length==0:
            self.length=value.shape[0]
        if value.shape[0]!=self.length:
            raise ValueError("Array must have same length as existing arrays")
        self.data[key] = value.astype(float)

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return self.length
    def __repr__(self) -> str:
        r="EyeDataDict(vars=%i,n=%i): \n"%(len(self.data), self.length)
        for k,v in self.data.items():
            r+="  %s (%s): "%(k,v.dtype)
            r+=", ".join(v[0:(min(5,self.length))].astype(str))
            if self.length>5:
                r+="..."
            r+="\n"
        return r

class GenericEyedata(ABC):
    """
    Generic class for eyedata. 
    Defines the basic structure of an eyedata object and 
    implements some basic functions.
    """
    name: str  ## name of dataset
    fs: float  ## sampling rate
    data: dict ## dictionary with data (contains ndarrays)
    tx: np.ndarray ## time vector
    missing: np.ndarray ## missing data vector (1=missing, 0=not missing)
    event_onsets: np.ndarray ## vector with event onsets in time units

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
        return len(self.tx)

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
