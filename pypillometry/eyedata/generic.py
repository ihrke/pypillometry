"""
eyedata_generic.py
==================

Generic Eyedata class for use with the pypillometry package.
All other eyedata classes should inherit from this class.
"""

from collections.abc import Iterable
from .. import io
from ..convenience import sizeof_fmt, ByteSize, _round_values, requires_package, is_url, suppress_all_output, normalize_unit, _warn_if_ignores_mask
from ..units import UNSET, _has_units, _time_values_and_units, parse_time
from ..io import download
from .eyedatadict import EyeDataDict
from .experimental_setup import ExperimentalSetup
from ..signal import baseline
from ..intervals import stat_event_interval, Intervals
from ..logging import logging_get_level
from .. import config
from .. import state

import numpy as np
import itertools
import numbers
import warnings
from loguru import logger
#from pytypes import typechecked
from typing import Sequence, Union, List, TypeVar, Optional, Tuple, Callable, Dict, Any
import functools
from random import choice
from importlib import resources
from io import BytesIO
import base64
import copy
import pickle
import inspect
import h5py
import tempfile
import os
import sys

## abstract base class to enforce implementation of some functions for all classes
from abc import ABC, abstractmethod 


def _history_arg(v):
    """Short text for an argument in the history (no reference to the object)."""
    if isinstance(v, str):
        return repr(v)
    if v is None or isinstance(v, (bool, int, float, np.number)):
        return str(v)
    if isinstance(v, np.ndarray):
        return f"array{v.shape}"
    if callable(v):
        return getattr(v, "__name__", type(v).__name__)
    r = repr(v) if isinstance(v, (list, tuple, dict)) else str(v)
    if len(r) <= 40 and "\n" not in r:
        return r  # short enough, e.g. [1, 2], (1280, 1024), 30 cm
    try:
        return f"{type(v).__name__}({len(v)})"  # e.g. Intervals(13)
    except TypeError:
        return type(v).__name__


# number of decorated methods currently running; only the outermost call
# (the one made by the user) is recorded
_history_depth = 0


def keephistory(func):
    """
    Decorator that records a method call in the history of the returned object.

    After the decorated method has run, a short text version of the call is
    added (via ``add_to_history()``) to the object it returned (or to `self`
    if it returned nothing, e.g. for setters), provided that object is an
    eyedata object (i.e., has an ``add_to_history`` method).
    Calls made from inside another decorated method are not recorded, and
    large arguments are only summarized (e.g., ``Intervals(13)``). The history
    is a log of the operations; show it with
    :meth:`~pypillometry.GenericEyeData.print_history`.

    Parameters
    ----------
    func : callable
        method to wrap; its first argument must be the object (`self`)

    Returns
    -------
    callable
        wrapped method that returns the same value as `func`
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        global _history_depth
        _history_depth += 1
        try:
            obj=func(*args,**kwargs)
        finally:
            _history_depth -= 1
        # record in the returned object, or in `self` for in-place setters
        # that return nothing
        target = obj if obj is not None else args[0]
        if _history_depth == 0 and hasattr(target, 'add_to_history'):
            allargs = [_history_arg(v) for v in args[1:]]
            allargs += [f"{k}={_history_arg(v)}" for k, v in kwargs.items()]
            fstr = f"{func.__name__}({','.join(allargs)})"
            target.add_to_history({"funcstring": fstr, "funcname": func.__name__})
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
    inplace: Optional[bool] ## whether to make changes in-place (None: package default)
    info: dict ## optional info about the dataset
    params: dict ## parameters 

    @abstractmethod
    def __init__():
        """Constructor"""
        pass

    def _init_common(self, time: np.ndarray,
                    sampling_rate: float,
                    event_onsets: np.ndarray,
                    event_labels: np.ndarray,
                    name: str|None,
                    fill_time_discontinuities: bool,
                    inplace: Optional[bool],
                    info: dict = None):
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

        ## set info
        self.info = info if info is not None else {}

        ## fill in time discontinuities (part of construction, so bypass
        ## @keephistory and keep it out of the history)
        if fill_time_discontinuities:
            type(self).fill_time_discontinuities.__wrapped__(self)

    def _unit_fac(self, units):
        """Convert time units to factor for converting from milliseconds.
        
        Parameters
        ----------
        units : str or None
            Time unit (accepts aliases like "seconds", "s", "hrs", etc.)
            
        Returns
        -------
        float
            Conversion factor from milliseconds to target unit
        """
        # Normalize unit (handles aliases and None)
        units = normalize_unit(units)
        
        if units == "sec":
            fac = 1./1000.
        elif units == "min":
            fac = 1./1000./60.
        elif units == "h":
            fac = 1./1000./60./60.
        else:  # None or "ms"
            fac = 1.
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

    def __getitem__(self, key):
        """
        Provide dict-style access to data arrays, returning masked arrays.
        Supports special keys for retrieving time in different units.
        """
        normalized_key = key

        # Handle tuple keys (e.g., ("left", "x") or ("time", "sec"))
        if isinstance(key, tuple):
            if len(key) == 0:
                raise KeyError("Empty key tuple")
            if key[0] == "time":
                unit = key[1] if len(key) > 1 else None
                return self._get_time_array(unit)
            normalized_key = "_".join(key)

        if isinstance(normalized_key, str):
            if normalized_key.startswith("time"):
                prefix, _, suffix = normalized_key.partition("_")
                if prefix == "time":
                    return self._get_time_array(suffix or None)

        # Delegate to EyeDataDict and wrap using the stored mask
        data_array = self.data[normalized_key]
        try:
            mask = self.data.mask[normalized_key]
        except KeyError:
            # Fallback if mask not directly accessible (e.g., cached dict),
            # letting EyeDataDict resolve the key.
            mask = self.data.get_mask(normalized_key)

        return np.ma.masked_array(data_array, mask=mask, fill_value=np.nan)

    @keephistory
    def __setitem__(self, key, value):
        """
        Set data arrays using dict-style access.
        
        Accepts numpy arrays (mask will be all zeros) or masked arrays
        (mask and data stored separately in EyeDataDict).
        
        Special handling for "time" keys:
        - data["time"] = array: Updates tx directly (assumes milliseconds)
        - data["time_sec"] = array: Converts from seconds to milliseconds
        - data["time_min"] = array: Converts from minutes to milliseconds
        - data["time_h"] = array: Converts from hours to milliseconds
        - data["time", "sec"] = array: Same as "time_sec"
        
        Parameters
        ----------
        key : str or tuple
            Dictionary-style key (e.g., "left_x" or ("left", "x"))
        value : np.ndarray or np.ma.MaskedArray
            Array to store. If masked array, mask is extracted and stored separately.
        
        Examples
        --------
        >>> data["left_x"] = np.array([1, 2, 3])
        >>> data["left", "pupil"] = np.ma.masked_array([10, 20, 30], mask=[0, 1, 0])
        >>> data["time_sec"] = np.array([0, 0.1, 0.2])  # Converts to ms
        """
        normalized_key = key
        
        # Handle tuple keys (e.g., ("left", "x") or ("time", "sec"))
        if isinstance(key, tuple):
            if len(key) == 0:
                raise KeyError("Empty key tuple")
            if key[0] == "time":
                unit = key[1] if len(key) > 1 else None
                self._set_time_array(value, unit)
                return
            normalized_key = "_".join(key)
        
        # Handle string keys starting with "time"
        if isinstance(normalized_key, str):
            if normalized_key.startswith("time"):
                prefix, _, suffix = normalized_key.partition("_")
                if prefix == "time":
                    self._set_time_array(value, suffix or None)
                    return
        
        # For non-time keys, set data in EyeDataDict
        # Check for masked array BEFORE converting with asarray
        if isinstance(value, np.ma.MaskedArray):
            # Extract data and mask from masked array
            self.data.data[normalized_key] = value.data.astype(float)
            self.data.mask[normalized_key] = value.mask.astype(int)
        else:
            # Regular array - let EyeDataDict handle it (creates zero mask)
            value = np.asarray(value)
            self.data[normalized_key] = value

    def _set_time_array(self, value, unit: Optional[str]):
        """
        Set the time vector from the provided array, converting to milliseconds if needed.
        
        Parameters
        ----------
        value : np.ndarray
            Time array in the specified units
        unit : str or None
            Unit of the input array (e.g., "sec", "min", "h", "ms").
            None or "" or "ms" means the array is already in milliseconds.
        """
        value = np.asarray(value)
        
        if unit is None or unit == "":
            # Already in milliseconds
            self.tx = value.astype(float)
            return
        
        # Normalize unit (handles all aliases)
        unit = normalize_unit(unit)
        
        if unit == "ms" or unit is None:
            # Already in milliseconds
            self.tx = value.astype(float)
        else:
            # Convert to milliseconds
            # _unit_fac gives factor to convert FROM ms to target unit
            # So to convert TO ms, we divide by the factor
            factor = self._unit_fac(unit)
            self.tx = (value / factor).astype(float)

    @keephistory
    def __delitem__(self, key):
        """
        Delete data arrays using dict-style access.
        
        Parameters
        ----------
        key : str or tuple
            Dictionary-style key (e.g., "left_x" or ("left", "x"))
        
        Examples
        --------
        >>> del data["left_x"]
        >>> del data["left", "pupil"]
        
        Raises
        ------
        KeyError
            If the key does not exist in the data
        """
        normalized_key = key
        
        # Handle tuple keys (e.g., ("left", "x"))
        if isinstance(key, tuple):
            if len(key) == 0:
                raise KeyError("Empty key tuple")
            if key[0] == "time":
                raise ValueError("Cannot delete time array using __delitem__")
            normalized_key = "_".join(key)
                
        # Delete from EyeDataDict (removes both data and mask)
        del self.data[normalized_key]

    def _get_time_array(self, unit: Optional[str]):
        """
        Return the time vector in the requested units.

        Parameters
        ----------
        unit : str or None
            Unit suffix (e.g., "sec", "min", "h", "ms"). Accepts aliases.
            None or "" or "ms" returns raw time in milliseconds.
        """
        if unit is None or unit == "":
            return self.tx

        # Normalize unit (handles all aliases)
        unit = normalize_unit(unit)

        if unit == "ms" or unit is None:
            return self.tx

        # Use _unit_fac which already handles all valid units
        factor = self._unit_fac(unit)
        return self.tx * factor

    def __getattr__(self, name):
        """
        Delegate unknown attribute access to the plot object.
        
        This allows calling plot methods directly on the data object:
        
        Examples
        --------
        >>> data.plot_timeseries()  # instead of data.plot.plot_timeseries()
        >>> data.plot_intervals(intervals)
        >>> data.pupil_plot(plot_range=(0, 1000))
        
        The original syntax still works:
        >>> data.plot.plot_timeseries()  # this still works too
        """
        # Avoid infinite recursion - only delegate if 'plot' property exists
        if name != 'plot':
            try:
                # Use object.__getattribute__ to avoid recursion
                plot_obj = object.__getattribute__(self, 'plot')
                if hasattr(plot_obj, name):
                    return getattr(plot_obj, name)
            except AttributeError:
                pass
        
        # If not found in plot object, raise the normal AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

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
        """Private helper function.
        
        Returns self or a copy depending on inplace setting. `inplace=None`
        falls back to the object-level setting (constructor), and if that is
        None too, to the package-level default (:func:`set_default_inplace`).
        When copying, preserves the original name to avoid name changes
        during internal processing.
        """
        if inplace is None:
            inplace = getattr(self, "inplace", None)  # object-level setting
        if inplace is None:
            inplace = config.get_default_inplace()    # package-level setting
        return self if inplace else self.copy(new_name=self.name)


    @property
    def blinks(self):
        """Return blinks (intervals) for all eyes.

        Returns
        -------
        dict
            dictionary with blinks for each eye
        """
        return self._blinks
    
    @keephistory
    def set_blinks(self, intervals, eyes=[], variables=[], apply_mask=True):
        """
        Set blinks for specified eyes and variables (in-place).

        Parameters
        ----------
        intervals : Intervals, dict or None
            Either:

            - a single :class:`~pypillometry.Intervals` object: stored for all
              specified eyes/variables;
            - a dict of Intervals (or `None`) with keys like "left_pupil",
              "right_x", etc.; when a dict is provided, `eyes` and
              `variables` are ignored;
            - `None`: remove the stored blinks for the specified
              eyes/variables (the mask is not changed).
        eyes : str or list of str, optional
            eye(s) to set blinks for (e.g., ``["left", "right"]``);
            if empty, all available eyes. Ignored if `intervals` is a dict.
        variables : str or list of str, optional
            variable(s) to set blinks for (e.g., ``["pupil", "x", "y"]``);
            if empty, all available variables. Ignored if `intervals` is a dict.
        apply_mask : bool, optional
            If True, also apply the intervals as mask to the data
            (existing masks are kept). If False, only store intervals.
            Default is True.
        """
        # Check if intervals is a dict
        if isinstance(intervals, dict):
            # Store each Intervals object to its corresponding eye_variable
            for key, interval_obj in intervals.items():
                if interval_obj is not None and not isinstance(interval_obj, Intervals):
                    raise TypeError(f"Dict values must be Intervals objects or None, got {type(interval_obj)} for key '{key}'")
                
                self._blinks[key] = interval_obj
            
            # Apply mask if requested (filter out None values)
            if apply_mask:
                intervals_to_mask = {k: v for k, v in intervals.items() if v is not None}
                if intervals_to_mask:
                    self.mask_intervals(intervals_to_mask, inplace=True)
        else:
            # Single Intervals object - apply to specified eyes/variables
            if intervals is not None and not isinstance(intervals, Intervals):
                raise TypeError(f"intervals must be an Intervals object, dict, or None, got {type(intervals)}")
            
            eyes, variables = self._get_eye_var(eyes, variables)
            
            # Store for all combinations of eyes and variables
            for eye in eyes:
                for var in variables:
                    key = f"{eye}_{var}"
                    self._blinks[key] = intervals
            
            # Apply mask if requested and intervals exist
            if apply_mask and intervals is not None:
                self.mask_intervals(intervals, eyes=eyes, variables=variables, inplace=True)

    def get_blinks(self, eyes: str|list = [], variables: str|list = [], 
                   units: str|None = None):
        """
        Get blinks as Intervals object or dict of Intervals.

        If no blinks are stored for an eye/variable, an empty Intervals
        object is returned for it.

        Parameters
        ----------
        eyes : str or list of str, optional
            Eye(s) to get blinks for; if empty, all available eyes.
            If multiple, returns dict.
        variables : str or list of str, optional
            Variable(s) to get blinks for; if empty, all available variables.
            If multiple, returns dict.
        units : str or None, optional
            Units for intervals: "ms", "sec", "min", "h", or None for indices
            
        Returns
        -------
        Intervals or dict
            If single eye and variable: returns Intervals object.
            If multiple eyes/variables: returns dict with keys like "left_pupil".
            
        Examples
        --------
        >>> blinks = data.get_blinks('left', 'pupil')
        >>> blinks_ms = data.get_blinks('left', 'pupil', units="ms")
        >>> blinks_dict = data.get_blinks(['left', 'right'], 'pupil')
        >>> blinks_idx = blinks_ms.to_units("indices")
        """
        eyes, variables = self._get_eye_var(eyes, variables)
        
        # Normalize units (handles aliases like "seconds", "hrs", etc.)
        units = normalize_unit(units)
        
        # Check if we need to return dict for multiple eyes/variables
        return_dict = len(eyes) > 1 or len(variables) > 1
        
        if return_dict:
            # Return dict of Intervals
            result = {}
            for e, v in itertools.product(eyes, variables):
                key = e + "_" + v
                if key in self._blinks and self._blinks[key] is not None:
                    intervals_obj = self._blinks[key]
                    # Ensure sampling_rate is set
                    if intervals_obj.sampling_rate is None:
                        intervals_obj.sampling_rate = self.fs
                else:
                    # No blinks stored - create empty Intervals
                    intervals_obj = Intervals(
                        intervals=[],
                        units=None,
                        label=f"{e}_{v}_blinks",
                        data_time_range=(0, len(self.tx)),
                        sampling_rate=self.fs,
                        time_offset=self.tx[0]
                    )
                
                # Convert to requested units if needed
                if units is not None:
                    intervals_obj = intervals_obj.to_units(units)
                
                result[key] = intervals_obj
            
            return result
        else:
            # Single eye/variable - return Intervals directly
            key = eyes[0] + "_" + variables[0]
            if key in self._blinks and self._blinks[key] is not None:
                result = self._blinks[key]
                # Ensure sampling_rate is set
                if result.sampling_rate is None:
                    result.sampling_rate = self.fs
            else:
                # No blinks stored - return empty Intervals
                result = Intervals(
                    intervals=[],
                    units=None,
                    label=f"{eyes[0]}_{variables[0]}_blinks",
                    data_time_range=(0, len(self.tx)),
                    sampling_rate=self.fs,
                    time_offset=self.tx[0]
                )
            
            # Convert to requested units if needed
            if units is not None:
                result = result.to_units(units)
            
            return result  

    def get_interpolated_blinks(self, eyes: str|list = [], variables: str|list = [], 
                                units: str|None = None):
        """
        Get interpolated blinks as Intervals object or dict of Intervals.
        
        Interpolated blinks are the blink regions that were successfully interpolated
        by `pupil_blinks_interpolate()`. These may differ from detected blinks because:

        - Blink boundaries are refined using velocity thresholds
        - Some blinks may be skipped if they're too short to interpolate

        If no interpolated blinks are stored for an eye/variable, an empty
        Intervals object is returned for it.

        Parameters
        ----------
        eyes : str or list of str, optional
            Eye(s) to get interpolated blinks for; if empty, all available eyes.
            If multiple, returns dict.
        variables : str or list of str, optional
            Variable(s) to get interpolated blinks for; if empty, all available
            variables. If multiple, returns dict.
        units : str or None, optional
            Units for intervals: "ms", "sec", "min", "h", or None for indices
            
        Returns
        -------
        Intervals or dict
            If single eye and variable: returns Intervals object.
            If multiple eyes/variables: returns dict with keys like "left_pupil".
            
        See Also
        --------
        get_blinks : Get detected (not interpolated) blinks
        pupil_blinks_interpolate : Interpolate blinks
        """
        eyes, variables = self._get_eye_var(eyes, variables)
        
        # Multiple eyes/variables - return dict
        if len(eyes) > 1 or len(variables) > 1:
            result = {}
            for e, v in itertools.product(eyes, variables):
                key = e + "_" + v
                if key in self._interpolated_blinks and self._interpolated_blinks[key] is not None:
                    intervals_obj = self._interpolated_blinks[key]
                    # Ensure sampling_rate is set
                    if intervals_obj.sampling_rate is None:
                        intervals_obj.sampling_rate = self.fs
                    # Convert to requested units if needed
                    if units is not None:
                        intervals_obj = intervals_obj.to_units(units)
                    result[key] = intervals_obj
                else:
                    # No interpolated blinks stored - return empty Intervals
                    result[key] = Intervals(
                        intervals=[],
                        units=None,
                        label=f"{key}_interpolated_blinks",
                        data_time_range=(0, len(self.tx)),
                        sampling_rate=self.fs,
                        time_offset=self.tx[0]
                    )
                    if units is not None:
                        result[key] = result[key].to_units(units)
            return result
        else:
            # Single eye/variable - return Intervals directly
            key = eyes[0] + "_" + variables[0]
            if key in self._interpolated_blinks and self._interpolated_blinks[key] is not None:
                result = self._interpolated_blinks[key]
                # Ensure sampling_rate is set
                if result.sampling_rate is None:
                    result.sampling_rate = self.fs
            else:
                # No interpolated blinks stored - return empty Intervals
                result = Intervals(
                    intervals=[],
                    units=None,
                    label=f"{eyes[0]}_{variables[0]}_interpolated_blinks",
                    data_time_range=(0, len(self.tx)),
                    sampling_rate=self.fs,
                    time_offset=self.tx[0]
                )
            
            # Convert to requested units if needed
            if units is not None:
                result = result.to_units(units)
            
            return result

    def mask_as_intervals(self, eyes: str|list = [], variables: str|list = [], 
                          units: str|None = None, label: str|None = None):
        """
        Convert masks to Intervals objects.

        Each contiguous run of masked samples becomes one interval.

        Parameters
        ----------
        eyes : str or list of str, optional
            Eye(s) to get masks for; if empty, all available eyes.
            If multiple, returns dict.
        variables : str or list of str, optional
            Variable(s) to get masks for; if empty, all available variables.
            If multiple, returns dict (combinations not present in the data
            are left out).
        units : str or None, optional
            Units for intervals: "ms", "sec", "min", "h", or None for indices.
            Default is None (indices).
        label : str, optional
            Descriptive label for the intervals. Default is None.
            
        Returns
        -------
        Intervals or dict
            If single eye and variable: returns Intervals object.
            If multiple eyes/variables: returns dict with keys like "left_pupil".
            
        Examples
        --------
        >>> mask_intervals = data.mask_as_intervals('left', 'pupil')
        >>> mask_ms = data.mask_as_intervals('left', 'pupil', units='ms')
        >>> mask_dict = data.mask_as_intervals(['left', 'right'], 'pupil')
        """
        eyes, variables = self._get_eye_var(eyes, variables)
        
        # Normalize units (handles aliases like "seconds", "hrs", etc.)
        units = normalize_unit(units)
        
        # Check if we need to return dict for multiple eyes/variables
        return_dict = len(eyes) > 1 or len(variables) > 1
        
        if return_dict:
            # Return dict of Intervals
            result = {}
            for e, v in itertools.product(eyes, variables):
                key = e + "_" + v
                if key not in self.data.mask:
                    continue
                    
                mask = self.data.mask[key]
                intervals_list = self._mask_to_intervals_list(mask)
                
                intervals_obj = Intervals(
                    intervals_list,
                    units=None,
                    label=label or f"{key}_mask",
                    data_time_range=(0, len(self.tx)),
                    sampling_rate=self.fs,
                    time_offset=self.tx[0]
                )
                
                # Convert to requested units if needed
                if units is not None:
                    intervals_obj = intervals_obj.to_units(units)
                
                result[key] = intervals_obj
            
            return result
        else:
            # Single eye/variable - return Intervals directly
            key = eyes[0] + "_" + variables[0]
            if key not in self.data.mask:
                empty_intervals = Intervals([], units=None, label=label or f"{key}_mask",
                               data_time_range=(0, len(self.tx)), sampling_rate=self.fs,
                               time_offset=self.tx[0])
                if units is not None:
                    return empty_intervals.to_units(units)
                return empty_intervals
                
            mask = self.data.mask[key]
            intervals_list = self._mask_to_intervals_list(mask)
            
            result = Intervals(
                intervals_list,
                units=None,
                label=label or f"{key}_mask",
                data_time_range=(0, len(self.tx)),
                sampling_rate=self.fs,
                time_offset=self.tx[0]
            )
            
            # Convert to requested units if needed
            if units is not None:
                result = result.to_units(units)
            
            return result
    
    def _mask_to_intervals_list(self, mask):
        """Helper method to convert mask array to list of interval tuples.
        
        Returns intervals with exclusive end (Python convention): [start, end)
        This means the interval (1, 4) covers indices 1, 2, 3 (not 4).
        
        Uses Intervals.from_mask() internally for the conversion.
        """
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask.ndim > 1:
            raise ValueError("mask must be 1D")
        if mask.size == 0 or not np.any(mask):
            return []
        
        # Use Intervals.from_mask() for the conversion
        intervals_obj = Intervals.from_mask(mask)
        return intervals_obj.intervals

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
        self._interpolated_blinks={}
        for eye,variable in itertools.product(eyes, variables):
            self._blinks[f"{eye}_{variable}"] = None
            self._interpolated_blinks[f"{eye}_{variable}"] = None        

    def nblinks(self, eyes=[], variables=[]) -> Dict[str, int]:
        """
        Return number of detected blinks. Should be run after `pupil_blinks_detect()`.

        By default, all eyes and variables are considered.

        Parameters
        ----------
        eyes : str or list of str, optional
            eye(s) to consider; if empty, all available eyes
        variables : str or list of str, optional
            variable(s) to consider; if empty, all available variables

        Returns
        -------
        dict
            number of detected blinks per eye/variable, with keys like
            "left_pupil"; combinations without blinks are left out
        """
        eyes,variables=self._get_eye_var(eyes,variables)
        
        return {eye+"_"+var:len(self.get_blinks(eye,var)) 
                for eye,var in itertools.product(eyes,variables)
                if len(self.get_blinks(eye,var)) > 0}


    def blink_stats(self, eyes: list = [], units: str = "ms") -> dict:
        """
        Return statistics on blink durations.

        Uses the blinks stored for the "pupil" variable of each eye
        (see :meth:`~pypillometry.Intervals.stats`).

        Parameters
        ----------
        eyes : str or list of str, optional
            eye(s) to consider; if empty, all available eyes
        units : str, optional
            Units for statistics: "ms", "sec", "min", or "h". Default "ms".

        Returns
        -------
        dict
            Dictionary mapping eye name to stats dict (number, mean, sd,
            min, max and total duration of blinks), or None if no blinks
        """
        eyes, _ = self._get_eye_var(eyes, [])
        
        stats = dict()
        for eye in eyes:
            blinks = self.get_blinks(eye, "pupil", units=units)
            if len(blinks) > 0:
                stats[eye] = blinks.stats()
            else:
                stats[eye] = None
        return stats


    def quality_report(self, thresholds: Optional[dict] = None, eyes: str|list = []):
        """
        Report the quality of the data.

        Measures that need no preprocessing are always computed (on the
        stored values, where missing samples are NaN):

        - pupil (if available): missing data, dropouts (periods of missing
          data) and high-frequency noise
        - gaze (if available): missing data, off-screen samples (needs the
          screen resolution) and precision (in degrees if screen size and
          eye distance are set in the experimental setup, else in pixels)
        - both eyes: correlation of the pupils and disparity of the gaze

        Further measures are added if available: blinks (after
        :meth:`~pypillometry.PupilData.pupil_blinks_detect`), masked and
        interpolated data (after preprocessing) and calibration/validation
        errors (e.g., when read from an Eyelink file).

        Where it makes sense, each measure is labeled as "good", "ok",
        "poor" or "bad". The overall label is the worst of all labels.

        Parameters
        ----------
        thresholds : dict, optional
            thresholds for the labels, overriding the defaults in
            :data:`pypillometry.quality.DEFAULT_QUALITY_THRESHOLDS`.
            Keys are ``"<section>.<metric>"`` (e.g., ``"pupil.missing_pct"``),
            values are three bounds ``(good, ok, poor)``. Increasing bounds
            mean that lower values are better, decreasing bounds that higher
            values are better. `None` turns the label off.
        eyes : str or list of str, optional
            eye(s) to report; if empty, all available eyes

        Returns
        -------
        :class:`~pypillometry.QualityReport`
            the report; print it for a text version, display it in a
            notebook for an HTML version or use
            :meth:`~pypillometry.QualityReport.to_dict` to get a dict

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_rlmw_002_short()  # doctest: +SKIP
        >>> print(d.quality_report())  # doctest: +SKIP
        >>> d.quality_report(thresholds={"pupil.missing_pct": (2, 10, 20)})  # doctest: +SKIP
        """
        from ..quality import _compute_quality_report
        return _compute_quality_report(self, thresholds=thresholds, eyes=eyes)

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
        """Add event to history.

        Usually called by the ``keephistory`` decorator.

        Parameters
        ----------
        event : dict
            description of the operation with keys "funcstring" (printable
            call) and "funcname" (method name)
        """
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

        Sets the `original` attribute (stored when the object was created
        with ``keep_orig=True``) to `None`.

        Parameters
        ----------
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the object without the original data (`self` or a copy,
            depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        obj.original=None
        return obj
    

    @keephistory
    def reset_time(self, t0: float|None=None, inplace=None):
        """
        Resets time so that time `t0` becomes time zero.

        `t0` is subtracted from the time-vector and from the event onsets.

        Parameters
        ----------
        t0 : float or None, optional
            time (in ms) that becomes the new time zero; if None, use the first
            time point (so that the time-vector starts at zero)
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the object with shifted time (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        if t0 is None:
            t0 = self.tx[0]
        obj.tx=self.tx-t0
        obj.event_onsets=self.event_onsets-t0
        return obj

    @keephistory
    def sub_slice(self,
                start: float=-np.inf, 
                end: float=np.inf, 
                units: str=None, inplace=None):
        """
        Return a shortened version of the current object.

        The result contains all data (and events) between `start` and `end`
        (inclusive), e.g. ``d.sub_slice(3*u.min, 5*u.min)`` or
        ``d.sub_slice("3 min", "5 min")``.
        `start` and `end` refer to the values of the time-vector (not to
        the time since the start of the recording). Masks are kept; stored
        blinks (in sample indices) are shifted and cut to the new range.

        Parameters
        ----------
        start : float, str or Quantity, optional
            start for new dataset; default: beginning of the data
        end : float, str or Quantity, optional
            end of new dataset; default: end of the data
        units : str or None, optional
            time units of `start` and `end` if they are plain numbers
            (one of "ms", "sec", "min", "h"). If units is `None`, plain
            numbers are ms (the units of the time vector).
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the shortened object (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)

        default = normalize_unit(units) or "ms"
        start = parse_time(start, default=default)
        end = parse_time(end, default=default)
        tx = self.tx.copy()
        evon=obj.event_onsets.copy()
        keepix=np.where(np.logical_and(tx>=start, tx<=end))

        keepix=keepix[0]
        ndata=EyeDataDict()
        for k,v in obj.data.items():
            ndata.set_with_mask(k, v[keepix], mask=obj.data.mask[k][keepix])
        obj.data=ndata
        obj.tx=obj.tx[keepix]

        # shift stored blinks in sample indices and cut them to the new range
        # (blinks in time units stay valid)
        first = int(keepix[0]) if keepix.size>0 else 0
        n = len(obj.tx)
        for store in (obj._blinks, obj._interpolated_blinks):
            for key, iv in store.items():
                if iv is None or iv.units is not None:
                    continue
                shifted = [(max(int(on)-first, 0), min(int(off)-first, n)) for on, off in iv.intervals]
                store[key] = Intervals([(on, off) for on, off in shifted if off > on], None,
                                       label=iv.label, data_time_range=(0, n), sampling_rate=obj.fs,
                                       time_offset=obj.tx[0] if n > 0 else 0.0)

        keepev=np.logical_and(evon>=start, evon<=end)
        obj.event_onsets=obj.event_onsets[keepev]
        obj.event_labels=obj.event_labels[keepev]
        
        return obj
    
    def write_file(self, fname:str, compress:bool=True):
        """
        Save to file (using :func:`~pypillometry.io.write_pickle`).

        The format follows from the extension: HDF5 for ``.h5`` and
        ``.hdf5``, pickle otherwise (e.g., ``.pypil``).

        Parameters
        ----------
        fname: str
            filename
        compress: bool
            compress the file (default True; pickle: zstd, HDF5: gzip)
        """
        io.write_pickle(self, fname, compress=compress)


    @classmethod
    def from_file(cls, fname:str, **kwargs):
        """
        Reads a :class:`~pypillometry.GenericEyeData` object from a pickle
        or HDF5 file (see :meth:`write_file`).
        Use as ``pypillometry.PupilData.from_file("yourfile.pypil")``.

        The file is read with :func:`pypillometry.io.read_pickle`, so
        `fname` can also be an HTTP(S) or SSH URL. Files written by older
        versions of pypillometry are upgraded to the current version.

        Parameters
        ----------
        fname : str
            filename (or URL)
        **kwargs
            passed to :func:`pypillometry.io.read_pickle` (e.g.,
            `cache_file`, `cache_dir`, `cache_layout`, `autodelete`,
            `session`)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the object stored in the file (its class is the one that was
            stored, independent of the class this method is called on)
        """
        r=io.read_pickle(fname, **kwargs)
        return r

    @classmethod
    def from_eyelink(
        cls, fname:str, 
        return_edf_obj:bool=False, 
        remove_eyelink_triggers:bool=True, **kwargs):
        """
        Reads an eyedata object from an Eyelink file.

        All "messages" are stored as event_labels (can be filtered later) and
        it tries to be smart to detect different information from the EDF file.
        Spatial calibration/validation data is automatically parsed and stored
        in the `calibration` attribute as SpatialCalibration objects.
        Eyelink codes missing pupil samples (e.g., during blinks) as 0; these
        are converted to NaN (and therefore masked).

        However, the EDF file may contain other information of interest, e.g.,
        eyelink-detected saccades, etc.
        If you need more control, use the "eyelinkio" package directly and
        create the object manually.

        Parameters
        ----------
        fname : str
            filename of the EDF file, HTTP(S) URL or SSH URL
            (``ssh://server/folder/file.edf``, see
            :func:`pypillometry.io.fetch_url`)
        return_edf_obj : bool, optional
            if True, return a tuple with the eyedata object and the object
            returned by the "eyelinkio" package;
            if False (default), return only the eyedata object
        remove_eyelink_triggers : bool, optional
            if True (default), remove all Eyelink triggers from the event_labels
            (related to eye-tracker settings, calibration, validation etc)
        **kwargs
            additional arguments passed to :func:`pypillometry.io.read_eyelink`
            (e.g., `cache_file`, `cache_dir`, `cache_layout`, `autodelete`,
            `session`)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData` or tuple
            object of the class this method is called on (e.g.,
            :class:`~pypillometry.EyeData`), or tuple with that object and the
            object returned by the "eyelinkio" package
        
        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.EyeData.from_eyelink('data.edf')  # doctest: +SKIP
        >>> # Access calibration data
        >>> print(d.calibration['left'])  # doctest: +SKIP
        <SpatialCalibration: left eye, HV9, n=9, error=0.45±0.23° (max=0.79°)>
        """
        edf = io.read_eyelink(fname, **kwargs)
        
        # convert data from EDF to EyeDataDict
        avail_data_fields = edf["info"]["sample_fields"]
        d = {}
        
        # Check for binocular data (separate left/right fields)
        left_fields = ['xpos_left', 'ypos_left', 'ps_left']
        right_fields = ['xpos_right', 'ypos_right', 'ps_right']
        
        if all(field in avail_data_fields for field in left_fields):
            d['left_x'] = edf["samples"][avail_data_fields.index("xpos_left")]
            d['left_y'] = edf["samples"][avail_data_fields.index("ypos_left")]
            d['left_pupil'] = edf["samples"][avail_data_fields.index("ps_left")]
        
        if all(field in avail_data_fields for field in right_fields):
            d['right_x'] = edf["samples"][avail_data_fields.index("xpos_right")]
            d['right_y'] = edf["samples"][avail_data_fields.index("ypos_right")]
            d['right_pupil'] = edf["samples"][avail_data_fields.index("ps_right")]
        
        # Check for monocular data (no left/right suffix)
        monocular_fields = ['xpos', 'ypos', 'ps']
        if all(field in avail_data_fields for field in monocular_fields):
            # Determine which eye was tracked (usually stored in edf["info"]["eye"])
            # Map to 'left' by default, but check if eye info is available
            tracked_eye = edf["info"].get("eye", "left").lower()
            if tracked_eye not in ['left', 'right']:
                tracked_eye = 'left'  # Default to left if unclear
            
            d[f'{tracked_eye}_x'] = edf["samples"][avail_data_fields.index("xpos")]
            d[f'{tracked_eye}_y'] = edf["samples"][avail_data_fields.index("ypos")]
            d[f'{tracked_eye}_pupil'] = edf["samples"][avail_data_fields.index("ps")]

        # Eyelink codes missing pupil samples (e.g., blinks) as 0
        for key in d:
            if key.endswith("_pupil"):
                d[key] = np.where(d[key] == 0, np.nan, d[key])

        # get events (Eyelink stores in seconds, convert to ms)
        evon = edf["discrete"]["messages"]["stime"]*1000
        evlab = edf["discrete"]["messages"]["msg"].astype(str)
        
        # store the info from the EDF file
        info = {}
        info["eyelink_info"] = edf["info"]
        
        # get time vector (Eyelink stores in seconds, convert to ms)
        sfreq = edf["info"]["sfreq"]
        tx = edf["times"]*1000

        # Parse calibration data (use validations, which are post-calibration checks)
        calibration = None
        if 'calibrations' in edf["info"] and len(edf["info"]["calibrations"]) > 0:
            from .spatial_calibration import SpatialCalibration
            calibration = {}
            screen_res = edf["info"].get("screen_coords", None)
            
            for val_data in edf["info"]["calibrations"]:
                eye = val_data.get('eye', 'unknown').lower()
                # Use most recent validation for each eye (overwrite if multiple)
                calibration[eye] = SpatialCalibration.from_eyelink(
                    val_data, screen_res
                )
        
        # build the object with experimental setup
        screen_coords = edf["info"]["screen_coords"]
        setup = ExperimentalSetup(screen_resolution=tuple(screen_coords)) if screen_coords is not None else None
        
        kwargs = dict(**d, calibration=calibration, experimental_setup=setup)
        # only pass what the constructor accepts (e.g., PupilData has no gaze)
        accepted = inspect.signature(cls.__init__).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        obj = cls(time=tx, **kwargs, event_onsets=evon, event_labels=evlab,
            sampling_rate=sfreq, info=info, name=edf["info"]["filename"])

        # filter out irrelevant triggers
        def eyelink_filter_func(lab):
            ban_strings = ["!CAL", "VALIDATE", "RECCFG", "GAZE_COORDS", "ELCL_WINDOW_SIZES", 
                        "CAMERA_LENS_FOCAL_LENGTH", "ELCL_PROC", "!MODE", "ELCLCFG", "THRESHOLDS",
                        "ELCL_PCR_PARAM"]
            isbanned = [bstr in lab for bstr in ban_strings]
            return np.logical_not(np.any(isbanned))

        if remove_eyelink_triggers:
            logger.info("Filtering out Eyelink triggers")
            ev = obj.get_events()
            evs = ev.filter(eyelink_filter_func)
            obj = obj.set_events(evs)

        if return_edf_obj:
            return obj, edf
        else:
            return obj

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
        flen=max(len(k) for k in list(pars.keys())+["glimpse"])
        for k,v in pars.items():
            s+=(" {k:<"+str(flen)+"}: {v}\n").format(k=k,v=_round_values(v))
        s+=(" {k:<"+str(flen)+"}: {v}\n").format(k="glimpse", v=repr(self.data))
        if self.info:
            s+=f" Info: {len(self.info)} keys: {', '.join(list(self.info.keys())[:5])}" + ("..." if len(self.info) > 5 else "") + "\n"
        s+=" History:\n *\n"
        try:
            for i,ev in enumerate(self.history):
                s+=" "*(i+1)+"└ " + ev["funcstring"] +"\n"
        except:
            s+=" └no history\n"
        return s
    
    @requires_package("jinja2")
    def _repr_html_(self):
        """Rich HTML representation for notebooks using Jinja templates."""
        from jinja2 import Environment

        summary = self.summary()
        blink_stats_raw = summary.get("blinks", {}) or {}
        history_items = [ev["funcstring"] for ev in getattr(self, "history", [])[-5:]]

        size_str = sizeof_fmt(int(self.get_size()))

        overview = [
            ("Samples", f"{summary['n']:,}"),
            ("Sampling rate", f"{summary['sampling_rate']} Hz"),
            (
                "Duration",
                f"{summary['duration_minutes']:.2f} min "
                f"(start {summary['start_min']:.2f}, end {summary['end_min']:.2f})",
            ),
            ("Eyes", ", ".join(summary["eyes"])),
            ("Variables", ", ".join(summary["data"])),
            ("Events", summary["nevents"]),
            ("Size", size_str),
        ]

        if blink_stats_raw:
            counts = ", ".join(
                f"{eye}: {stats['n']}" for eye, stats in blink_stats_raw.items() if stats
            )
            overview.insert(-1, ("Blinks", counts))

        blink_stats = []
        for eye, stats in blink_stats_raw.items():
            if not stats:
                continue
            blink_stats.append(
                {
                    "eye": eye,
                    "count": stats["n"],
                    "mean_sd": f"{stats['mean']:.1f} ± {stats['sd']:.1f}",
                    "range": f"[{stats['min']:.1f}, {stats['max']:.1f}]",
                }
            )

        sparklines = []
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            for key in sorted(self.data.keys()):
                eye, var = key if isinstance(key, tuple) else (None, key)
                label = f"{eye} {var}".strip() if eye else var
                data = self[key]  # Use accessor to get masked array
                if data is None or len(data) == 0:
                    continue
                # Average over ~200 bins, ignoring NaN/masked samples, so
                # sparse data still shows up (bins without data stay NaN)
                values = np.ma.filled(np.ma.asarray(data, dtype=float), np.nan)
                valid = ~np.isnan(values)
                nbins = min(200, len(values))
                starts = np.linspace(0, len(values), nbins + 1).astype(int)[:-1]
                sums = np.add.reduceat(np.where(valid, values, 0.0), starts)
                counts = np.add.reduceat(valid.astype(int), starts)
                with np.errstate(invalid="ignore", divide="ignore"):
                    sample = np.where(counts > 0, sums / counts, np.nan)
                fig = Figure(figsize=(1.8, 0.6), dpi=100)
                ax = fig.subplots()
                ax.plot(sample, color="#1f77b4", linewidth=1)
                ax.axis("off")
                canvas = FigureCanvasAgg(fig)
                buf = BytesIO()
                canvas.print_png(buf)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("ascii")
                sparklines.append(
                    {
                        "label": label,
                        "data": f"data:image/png;base64,{img_b64}",
                    }
                )
        except Exception as exc:  # pragma: no cover - optional feature
            logger.debug(f"Could not render sparklines: {exc}")
            sparklines = []

        context = {
            "class_name": self.__class__.__name__,
            "name": summary["name"],
            "overview": overview,
            "glimpse": repr(self.data),
            "blink_stats": blink_stats,
            "history": history_items,
            "sparklines": sparklines,
        }

        template_source = resources.read_text("pypillometry.templates", "eyedata_repr.html")
        # autoescape, so that names, labels etc. with "<" or "&" are shown as text
        template = Environment(autoescape=True).from_string(template_source)
        return template.render(**context)

    def _to_state(self) -> dict:
        """
        State of the object (see :mod:`pypillometry.state`).

        Subclasses with further attributes extend it (and :meth:`_set_state`).
        """
        def ivs(blinks):
            return {k: None if v is None else v._to_state() for k, v in blinks.items()}
        return {
            "name": self.name,
            "fs": self.fs,
            "tx": self.tx,
            **self.data._to_state(),  # data and mask
            "event_onsets": self.event_onsets,
            "event_labels": self.event_labels,
            "blinks": ivs(self._blinks),
            "interpolated_blinks": ivs(self._interpolated_blinks),
            "params": self.params,
            "history": self.history,
            "inplace": self.inplace,
            "info": self.info,
            "original": None if self.original is None else state.to_state(self.original),
        }

    def _set_state(self, st: dict):
        """Set the attributes from a state of the current version (from :meth:`_to_state`)."""
        def ivs(blinks):
            return {k: None if v is None else Intervals._from_state(v)
                    for k, v in blinks.items()}
        self.data = EyeDataDict._from_state(st)
        self.name = st["name"]
        self.tx = st["tx"]
        self.fs = st["fs"]
        self.event_onsets = st["event_onsets"]
        self.event_labels = st["event_labels"]
        self.history = list(st["history"])
        self._blinks = ivs(st["blinks"])
        self._interpolated_blinks = ivs(st["interpolated_blinks"])
        self.params = st["params"]
        self.inplace = st["inplace"]
        self.info = st["info"]
        self.original = None if st["original"] is None else state.from_state(st["original"])

    def __getstate__(self):
        """State for pickling (see :func:`pypillometry.state.to_state`)."""
        return state.to_state(self)

    def __setstate__(self, st):
        """Restore from a pickled state, upgrading older versions first."""
        self._set_state(state.migrate(st, type(self).__name__))

    def copy(self, new_name: Optional[str]=None):
        """
        Make and return a deep-copy of the eyedata object.

        Parameters
        ----------
        new_name : str, optional
            New name for the copy. If None, "_copy" is appended to the original name.

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            deep copy of the object (same class as the original)
        """
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v))
        if new_name is None:
            result.name = self.name + "_copy"
        else:
            result.name = new_name
        return result

    @keephistory
    def drop_data(self, eyes: list = [], variables: list = []):
        """
        Drop specified eyes and/or variables from the data and return a new object.
        
        This method never works in place - it always returns a new object.
        All data of the given eyes and all data of the given variables are
        dropped (together with their stored blinks).
        The returned object's class is determined by what data remains:

        - If only pupil variables remain: returns PupilData
        - If only x/y variables remain: returns GazeData
        - If both pupil and x/y remain: returns EyeData

        Parameters
        ----------
        eyes : str or list of str, optional
            Eyes to drop (e.g., ['left'] or ['right'] or ['left', 'right']);
            if empty, no eye is dropped
        variables : str or list of str, optional
            Variables to drop (e.g., ['pupil'] or ['x', 'y']);
            if empty, no variable is dropped
            
        Returns
        -------
        GenericEyeData subclass
            A new object with the specified data removed, cast to the appropriate class.
            
        Raises
        ------
        ValueError
            If dropping would result in no data remaining.
            
        Examples
        --------
        >>> # Drop left eye data
        >>> right_only = data.drop_data(eyes=['left'])
        >>> 
        >>> # Drop pupil data, keep only gaze
        >>> gaze_only = data.drop_data(variables=['pupil'])
        >>> 
        >>> # Drop right eye and pupil data
        >>> left_gaze = data.drop_data(eyes=['right'], variables=['pupil'])
        """
        # Normalize inputs to lists
        if isinstance(eyes, str):
            eyes = [eyes]
        if isinstance(variables, str):
            variables = [variables]
            
        # Make a deep copy first
        result = self.copy(new_name=self.name)
        
        # Get current keys
        keys_to_remove = []
        for key in list(result.data.data.keys()):
            parts = key.split("_", 1)
            if len(parts) != 2:
                continue
            eye, var = parts
            
            # Check if this key should be removed
            if eye in eyes or var in variables:
                keys_to_remove.append(key)
        
        # Remove the keys from data and mask
        for key in keys_to_remove:
            if key in result.data.data:
                del result.data.data[key]
            if key in result.data.mask:
                del result.data.mask[key]
        
        # Check what remains
        remaining_vars = result.data.get_available_variables()
        remaining_eyes = result.data.get_available_eyes()
        
        if len(remaining_vars) == 0 or len(remaining_eyes) == 0:
            raise ValueError("Dropping the specified data would leave no data remaining")
        
        # Also clean up related stored data (blinks, interpolated_blinks, etc.)
        # Remove blinks for dropped eyes/variables
        blinks_to_remove = []
        for key in result._blinks.keys():
            parts = key.split("_", 1)
            if len(parts) == 2:
                eye, var = parts
                if eye in eyes or var in variables:
                    blinks_to_remove.append(key)
        for key in blinks_to_remove:
            del result._blinks[key]
            
        # Remove interpolated blinks for dropped eyes/variables
        interp_to_remove = []
        for key in result._interpolated_blinks.keys():
            parts = key.split("_", 1)
            if len(parts) == 2:
                eye, var = parts
                if eye in eyes or var in variables:
                    interp_to_remove.append(key)
        for key in interp_to_remove:
            del result._interpolated_blinks[key]
        
        return result._cast_to_data_class()

    def _cast_to_data_class(self):
        """
        Return the object cast to the class that matches its data.

        Objects with pupil and gaze data become EyeData, objects with only
        pupil data PupilData and objects with only gaze data GazeData. Objects
        with neither keep their class. The attributes are shared with `self`
        (not copied).
        """
        # Import here to avoid circular imports
        from . import PupilData, GazeData, EyeData

        variables = self.data.get_available_variables()
        has_pupil = 'pupil' in variables
        has_gaze = 'x' in variables or 'y' in variables
        if has_pupil and has_gaze:
            target_class = EyeData
        elif has_pupil:
            target_class = PupilData
        elif has_gaze:
            target_class = GazeData
        else:
            target_class = self.__class__

        if target_class is self.__class__:
            return self
        result = target_class.__new__(target_class)
        result.__dict__.update(self.__dict__)
        return result

    @keephistory
    def set_event_onsets(self, event_onsets: np.ndarray, event_labels: np.ndarray):
        """
        Set onsets of events in the data (in-place).

        Parameters
        ----------
        event_onsets : np.ndarray or None
            array of onsets (in ms); `None` removes all events
        event_labels : np.ndarray or None
            array of labels (strings), same length as `event_onsets`;
            if `None`, all labels are set to 0

        Raises
        ------
        ValueError
            if any onset is outside the range of the time-vector or if
            `event_labels` and `event_onsets` differ in length

        See Also
        --------
        set_events : set events from an :class:`~pypillometry.Events` object
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

    @property
    def events(self):
        """Events (onsets in ms) as an :class:`~pypillometry.Events` object.

        Shortcut for :meth:`get_events` with default arguments.
        """
        return self.get_events()

    def get_events(self, units: str = "ms"):
        """
        Get events as an Events object.
        
        Returns the event onsets and labels stored in the dataset as an
        Events object, which provides convenient filtering and display capabilities.
        
        Parameters
        ----------
        units : str, optional
            Units for the event onsets: "ms", "sec", "min", or "h".
            Default is "ms" (the internal storage format).
            
        Returns
        -------
        Events
            Events object containing the event onsets and labels
            
        Examples
        --------
        >>> events = data.get_events()
        >>> len(events)
        42
        
        >>> # Get events in seconds
        >>> events_sec = data.get_events(units="sec")
        
        >>> # Filter and work with events
        >>> stim_events = data.get_events().filter("stim")
        
        See Also
        --------
        set_events : Replace events from an Events object
        get_intervals : Extract intervals around events
        """
        from ..events import Events
        
        # Normalize units (handles aliases)
        units = normalize_unit(units) if units is not None else "ms"
        
        # Get data time range in ms (internal format)
        data_time_range = (self.tx[0], self.tx[-1])
        
        # Create Events object (always in ms first, since that's internal format)
        events = Events(
            onsets=self.event_onsets.copy(),
            labels=self.event_labels.copy(),
            units="ms",
            data_time_range=data_time_range
        )
        
        # Convert to requested units if different from ms
        if units != "ms":
            events = events.to_units(units)
        
        return events
    
    @keephistory
    def set_events(self, events, inplace=None):
        """
        Replace event onsets and labels from an Events object.
        
        This method updates the internal event_onsets and event_labels arrays
        from an Events object. The Events object's onsets are automatically
        converted to milliseconds (the internal storage format) if needed.
        
        Parameters
        ----------
        events : Events
            Events object containing the new events to set
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the object with updated events (`self` or a copy, depending on `inplace`)
            
        Examples
        --------
        >>> # Filter events and set them back
        >>> events = data.get_events()
        >>> stim_events = events.filter("stim")
        >>> data.set_events(stim_events)
        
        >>> # Modify events and update
        >>> events = data.get_events()
        >>> # ... filter or modify events ...
        >>> data.set_events(events)
        
        See Also
        --------
        get_events : Get events as an Events object
        set_event_onsets : Lower-level method to set events from arrays
        """
        from ..events import Events
        
        obj = self._get_inplace(inplace)
        
        if not isinstance(events, Events):
            raise TypeError(f"events must be an Events object, got {type(events)}")
        
        # Normalize units (handles aliases) and convert to ms if needed
        # (internal storage is always in ms)
        event_units = normalize_unit(events.units)
        if event_units != "ms" and event_units is not None:
            events = events.to_units("ms")
        
        # Use the existing set_event_onsets method
        obj.set_event_onsets(events.onsets, events.labels)
        
        return obj

    @keephistory
    def fill_time_discontinuities(self, yval=np.nan, print_info=True):
        """
        Find gaps in the time-vector and fill them in (in-place).

        A gap is a step between consecutive time points that is larger than
        twice the median step. New time points are inserted into each gap
        at the median step size and all data variables are set to `yval`
        at these new time points; they are masked as missing. Existing masks
        are kept and stored blinks (in sample indices) are moved to the new
        sample positions. This is called by the constructor when
        ``fill_time_discontinuities=True``.

        Parameters
        ----------
        yval : float, optional
            value used for the data in the filled-in gaps; default NaN,
            so the filled-in samples are masked as missing data
        print_info : bool, optional
            if True (default), log the number of gaps that are filled in

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the object itself (always modified in-place)
        """
        tx=self.tx
        stepsize=np.median(np.diff(tx))
        n=len(self)
        gaps_end_ix=np.where(np.r_[stepsize,np.diff(tx)]>2*stepsize)[0]
        ngaps=gaps_end_ix.size
        if ngaps!=0:
            ## at least one gap here
            if print_info:
                logger.info("Filling in %i gaps"%ngaps)
            gaps_start_ix=gaps_end_ix-1
            logger.info( str(((tx[gaps_end_ix]-tx[gaps_start_ix])/1000))+ " seconds" )
            
            ## build new time-vector and data: keep all valid segments
            ## and insert `yval` at regularly spaced time points in each gap
            bounds=np.r_[0, gaps_end_ix, n]
            ntx=[]
            nv={k:[] for k in self.data.keys()}
            nm={k:[] for k in self.data.keys()}
            for i in range(ngaps+1):
                seg=slice(bounds[i], bounds[i+1])
                ntx.append( tx[seg] )
                for k,v in self.data.items():
                    nv[k].append( v[seg] )
                    nm[k].append( self.data.mask[k][seg] )
                if i<ngaps:
                    start,end=gaps_start_ix[i], gaps_end_ix[i]
                    nfill=int((tx[end]-tx[start])/stepsize)-1
                    filltx=tx[start]+stepsize*np.arange(1, nfill+1)
                    ntx.append( filltx )
                    for k in nv:
                        nv[k].append( yval*np.ones_like(filltx, dtype=float) )
                        nm[k].append( np.ones(len(filltx), dtype=int) )  # filled = missing
            # new position of each original sample (ntx alternates kept
            # segments and filled gaps)
            lens=[len(t) for t in ntx]
            offsets=np.cumsum([0]+lens[:-1])
            newpos=np.concatenate([offsets[2*i]+np.arange(lens[2*i]) for i in range(ngaps+1)])
            ntx=np.concatenate(ntx)

            ndata=EyeDataDict()
            for k in nv:
                ndata.set_with_mask(k, np.concatenate(nv[k]), mask=np.concatenate(nm[k]))
            self.data=ndata
            self.tx=ntx

            # move stored blinks in sample indices to the new sample positions
            # (blinks in time units stay valid)
            for store in (self._blinks, self._interpolated_blinks):
                for key, iv in store.items():
                    if iv is None or iv.units is not None:
                        continue
                    moved = [(int(newpos[on]), int(newpos[off-1])+1) for on, off in iv.intervals]
                    store[key] = Intervals(moved, None, label=iv.label,
                                           data_time_range=(0, len(ntx)), sampling_rate=self.fs,
                                           time_offset=ntx[0])
        return self
    
    def get_intervals(self, 
                    event_select,
                    interval: tuple=(-200,200),
                    units: str|None=UNSET, label: str|None=None, **kwargs):
        """
        Return an Intervals object containing intervals relative to event-onsets.
        For example, extract the interval before and after a stimulus has been presented.
        It is possible to select based on the event label, e.g. only select
        events matching "stimulus" or "response". It is also 
        possible to select the intervals situated between two different events,
        e.g., trial start and response by specifying `event_select` as a tuple
        with two entries.

        Parameters
        ----------
        event_select : str, tuple, function, or Events
            variable describing which events to select and align to

            - if str: use all events whose label contains the string
            - if function: apply function to all labels, use those where the
              function returns True
            - if tuple: use the intervals between the events selected by the
              first and the second entry (each a str or function); `interval[0]`
              is added to the first and `interval[1]` to the second event
              (so ``interval=(0, 0)`` gives exactly the span between them).
              Both selectors must result in identical number of events;
              events are paired by order (n-th first with n-th second event)
              and a ValueError is raised if a second event comes before
              its first event.
            - if Events: use the events from the
              :class:`~pypillometry.Events` object directly
        interval : tuple (min,max) or Quantity, optional
            time-window relative to event-onset (0 is event-onset); i.e., negative
            numbers are before the event, positive numbers after, e.g.
            ``(-200, 800) * u.ms`` or ``("-0.2 sec", "0.8 sec")``. Plain numbers
            are in units given by the `units` parameter. Default ``(-200, 200)``.
        units : str or None, optional
            units of the returned intervals (one of "ms", "sec", "min", "h", or
            None for sampling units, i.e., indices into the time-array); plain
            numbers in `interval` are in these units, an `interval` with units
            is converted to them. If not given, the units of `interval` are used
            (sampling units if `interval` has no units).
        label : str or None, optional
            optional label for the Intervals object. If None, a label is automatically
            generated from the event_select parameter
        **kwargs
            passed onto the event_select function

        Returns
        -------
        :class:`~pypillometry.Intervals`
            Intervals object containing interval on- and offsets for each match,
            with associated metadata (event labels, indices, units)
        """
        from ..intervals import Intervals
        from ..events import Events
        
        if not isinstance(interval, Iterable) or isinstance(interval, str):
            raise ValueError("interval must be iterable")
        if _has_units(interval):
            interval, units = _time_values_and_units(interval, units, self.fs)
            if units is None:
                interval = np.round(interval).astype(int)
            interval = tuple(interval.tolist())
        elif units is UNSET:
            units = None
        fac=self._unit_fac(units)

        # between two events, interval[0] and interval[1] shift different
        # events, so e.g. (0,0) is valid; the resulting intervals are checked below
        if not isinstance(event_select, tuple) and interval[1]<=interval[0]:
            raise ValueError("interval must be (min,max) with min<max, got {}".format(interval))
        
        # Handle Events object as event_select
        if isinstance(event_select, Events):
            # Convert Events to ms if needed (internal format)
            if event_select.units != "ms":
                events_ms = event_select.to_units("ms")
            else:
                events_ms = event_select
            
            # Use events directly - create intervals around them
            if units is None:
                # Work in indices
                eix = np.array([np.argmin(np.abs(self.tx - ev)) 
                               for ev in events_ms.onsets])
                sti = np.maximum(eix + interval[0], 0)
                ste = np.minimum(eix + interval[1], len(self.tx))
                event_onsets_list = eix.tolist()
            else:
                # Work in specified units
                sti = (events_ms.onsets * fac) + interval[0]
                ste = (events_ms.onsets * fac) + interval[1]
                event_onsets_list = (events_ms.onsets * fac).tolist()
            
            intervals_list = [(s, e) for s, e in zip(sti, ste)]
            
            # Create label from Events if not provided
            if label is None:
                label = f"events_{len(events_ms)}"
            
            # Get data time range
            if units is None:
                data_time_range = (0, len(self.tx))
                time_offset = self.tx[0]
            else:
                data_time_range = (self.tx[0] * fac, self.tx[-1] * fac)
                time_offset = 0.0
            
            return Intervals(
                intervals=intervals_list,
                units=units,
                label=label,
                event_labels=events_ms.labels.tolist(),
                event_indices=None,  # We don't have indices from Events object
                data_time_range=data_time_range,
                event_onsets=event_onsets_list,
                sampling_rate=self.fs,
                time_offset=time_offset,
                interval_window=interval
            )
        
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
            # pairs are matched by order: the n-th second event must not
            # come before the n-th first event (catches missing/extra events)
            on1=self.event_onsets[event_ix[0]]
            on2=self.event_onsets[event_ix[1]]
            bad=np.where(on2<on1)[0]
            if len(bad)>0:
                raise ValueError("second event before first event in pair(s) {} "
                    "(onsets {} and {}); events are paired by order, check for "
                    "missing or extra events".format(bad.tolist(), on1[bad].tolist(),
                                                     on2[bad].tolist()))

            if units is None:
                eix1=np.array([np.argmin(np.abs(self.tx-ev))
                               for ev in self.event_onsets[event_ix[0]]])
                eix2=np.array([np.argmin(np.abs(self.tx-ev))
                               for ev in self.event_onsets[event_ix[1]]])
                sti=np.maximum(eix1+interval[0], 0)
                ste=np.minimum(eix2+interval[1], len(self.tx))
            else:
                sti=(self.event_onsets[event_ix[0]]*fac)+interval[0]
                ste=(self.event_onsets[event_ix[1]]*fac)+interval[1]
            bad=np.where(ste<=sti)[0]
            if len(bad)>0:
                raise ValueError("interval {} gives empty intervals for pair(s) {}"
                                 .format(interval, bad.tolist()))

            # Create label for between-events intervals if not provided
            if label is None:
                if isinstance(event_select[0], str) and isinstance(event_select[1], str):
                    label = f"{event_select[0]}_to_{event_select[1]}"
                else:
                    label = "between_events"
            
            # Get labels and indices for selected events
            selected_labels = [f"{self.event_labels[i]}_to_{self.event_labels[j]}" 
                             for i, j in zip(np.where(event_ix[0])[0], np.where(event_ix[1])[0])]
            selected_indices = np.where(event_ix[0])[0]
            
            # Get event onsets (use first event of the pair as reference)
            if units is None:
                # In index units
                event_onsets_list = [np.argmin(np.abs(self.tx-self.event_onsets[i])) 
                                    for i in np.where(event_ix[0])[0]]
            else:
                # In specified units
                event_onsets_list = [self.event_onsets[i] * fac 
                                    for i in np.where(event_ix[0])[0]]
            
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
            
            # Create label from event_select if not provided
            if label is None:
                if isinstance(event_select, str):
                    label = event_select
                else:
                    label = "custom_events"
            
            # Get labels and indices for selected events
            selected_indices = np.where(event_ix)[0]
            selected_labels = [self.event_labels[i] for i in selected_indices]
            
            # Get event onsets in appropriate units
            if units is None:
                # In index units
                event_onsets_list = eix.tolist()
            else:
                # In specified units
                event_onsets_list = (self.event_onsets[event_ix] * fac).tolist()
        
        intervals_list = [(s,e) for s,e in zip(sti,ste)]
        
        # Get data time range for plotting (in same units as intervals)
        if units is None:
            # Units are indices
            data_time_range = (0, len(self.tx))
            time_offset = self.tx[0]
        else:
            # Convert tx range to specified units
            data_time_range = (self.tx[0] * fac, self.tx[-1] * fac)
            time_offset = 0.0
        
        return Intervals(
            intervals=intervals_list,
            units=units,
            label=label,
            event_labels=selected_labels,
            event_indices=selected_indices,
            data_time_range=data_time_range,
            event_onsets=event_onsets_list,
            sampling_rate=self.fs,
            time_offset=time_offset,
            interval_window=interval
        )

    def get_segments(self, intervals, variable: str, name: str = None):
        """
        Extract segments from data using intervals.
        
        Creates a SegmentedEyeData object containing equal-length segments
        of a single variable extracted at the specified intervals.
        
        Parameters
        ----------
        intervals : Intervals
            Intervals object defining the segments to extract (from get_intervals())
        variable : str
            Variable to extract in "eye_variable" format (e.g., "left_pupil", "right_x")
        name : str, optional
            Name for the segmented dataset. If None, uses intervals.label or "segments"
            
        Returns
        -------
        SegmentedEyeData
            Object containing the extracted segments with mask information
            
        Examples
        --------
        >>> # Extract pupil segments around stimulus events
        >>> intervals = data.get_intervals("stimulus", interval=(-200, 800), units="ms")
        >>> segments = data.get_segments(intervals, "left_pupil")
        >>> segments.plot()
        
        >>> # Apply baseline correction
        >>> segments_bc = segments.baseline_correct(window=(-200, 0))
        
        See Also
        --------
        get_intervals : Create Intervals object from events
        SegmentedEyeData : Class holding segmented data
        """
        from ..segmented import SegmentedEyeData
        return SegmentedEyeData.from_eyedata(self, intervals, variable, name=name)

    @keephistory
    def scale(self, variables=[], mean: Union[float,dict,None]=None, 
              sd: Union[float,dict,None]=None, eyes=[], inplace=None):
        """
        Scale the signal by subtracting `mean` and dividing by `sd`.
        If these variables are not provided, use the signal's mean and std.
        Specify whether to scale `x`, `y`, `pupil` or other variables.
        
        Parameters
        ----------
        variables : str or list of str, optional
            variables to scale; can be "pupil", "x","y", "baseline", "response" or any other variable
            that is available in the dataset; either a string or a list of strings;
            available variables can be checked with `obj.variables`; empty list means all variables
        mean : None, float or dict, optional
            mean to subtract from signal; if `None`, use the signal's mean
            (of the non-masked samples);
            if float, use the same value for all eyes and variables;
            if dict, provide mean for each eye and variable as
            ``{eye: {variable: value}}``
        sd : None, float or dict, optional
            sd to scale with; if `None`, use the signal's std (of the
            non-masked samples);
            if float, use the same value for all eyes and variables;
            if dict, provide sd for each eye and variable as
            ``{eye: {variable: value}}``
        eyes : str or list of str, optional
            eye(s) to consider; if empty, all available eyes
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the scaled object (`self` or a copy, depending on `inplace`)

        Notes
        -----
        Scaling-parameters are being saved in the `params["scale"]` attribute
        and are used by :meth:`unscale`.
        """
        obj = self._get_inplace(inplace)
        eyes,variables=self._get_eye_var(eyes,variables)

        if mean is None:
            # use only non-masked samples
            mean={eye:{var:np.nanmean(obj[eye,var].compressed()) for var in variables} for eye in eyes}
        elif isinstance(mean, numbers.Real):
            mean={eye:{var:mean for var in variables} for eye in eyes}            
        elif not isinstance(mean, dict):
            raise ValueError("mean must be None, a number or dict")

        if sd is None:
            sd={eye:{var:np.nanstd(obj[eye,var].compressed()) for var in variables} for eye in eyes}
        elif isinstance(sd, numbers.Real):
            sd={eye:{var:sd for var in variables} for eye in eyes}
        elif not isinstance(sd, dict):
            raise ValueError("sd must be None, a number or dict")
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
                # scale the masked samples too (masked-array arithmetic
                # would leave them unchanged), keep the mask
                x = obj[eye, var]
                obj[eye, var] = np.ma.array((x.data - mean[eye][var]) / sd[eye][var],
                                            mask=np.ma.getmaskarray(x))
        return obj
        
    @keephistory    
    def unscale(self, variables=[], mean=None, sd=None, eyes=[], inplace=None):
        """
        Scale back to original values using either values provided as arguments
        or the values stored in `params["scale"]` by :meth:`scale`.

        The signal is multiplied by `sd` and `mean` is added.

        Parameters
        ----------
        variables : str or list of str, optional
            variables to unscale; can be "pupil", "x","y", "baseline", "response" or any other variable
            that is available in the dataset; either a string or a list of strings;
            available variables can be checked with `obj.variables`; empty list means all variables
        mean : None, float or dict, optional
            mean to add to the signal; a number (same for all eyes and
            variables) or ``{eye: {variable: value}}``;
            if `None`, use the mean stored by :meth:`scale`
        sd : None, float or dict, optional
            sd to multiply the signal with; a number (same for all eyes and
            variables) or ``{eye: {variable: value}}``;
            if `None`, use the sd stored by :meth:`scale`
        eyes : str or list of str, optional
            eye(s) to consider; if empty, all available eyes
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the unscaled object (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        eyes,variables=self._get_eye_var(eyes,variables)

        # if no parameters are provided, use the stored ones (normal)
        if mean is None:
            mean=obj.params["scale"]["mean"]
        elif isinstance(mean, numbers.Real):
            mean={eye:{var:mean for var in variables} for eye in eyes}
        elif not isinstance(mean, dict):
            raise ValueError("mean must be None, a number or dict")
        if sd is None:
            sd=obj.params["scale"]["sd"]
        elif isinstance(sd, numbers.Real):
            sd={eye:{var:sd for var in variables} for eye in eyes}
        elif not isinstance(sd, dict):
            raise ValueError("sd must be None, a number or dict")
        
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
                # unscale the masked samples too, keep the mask
                x = obj[eye, var]
                obj[eye, var] = np.ma.array(x.data * sd[eye][var] + mean[eye][var],
                                            mask=np.ma.getmaskarray(x))
        return obj    

    
    @keephistory
    def downsample(self, fsd: float, dsfac: bool=False, inplace=None):
        """
        Simple downsampling scheme using mean within the downsampling window.

        All data fields and the time-vector are downsampled simultaneously.
        See :func:`pypillometry.signal.baseline.downsample`. Masked samples are
        left out of the window means; a new sample is masked only if all
        samples in its window were masked. Stored blinks (in sample indices)
        are rescaled to the new sampling rate.

        Parameters
        ----------
        fsd : float
            new sampling-rate (in Hz) or decimate-factor
        dsfac : bool, optional
            if False (default), `fsd` is the new sampling rate (the decimate
            factor is ``int(fs/fsd)``);
            if True, `fsd` is the decimate factor
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the downsampled object (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)

        if dsfac:
            dsfac=fsd
            fsd=float(obj.fs/dsfac)
        else:
            dsfac=int(obj.fs/fsd) # calculate downsampling factor
            
        dsfac=int(dsfac)
        # downsample all of the variables/eyes in the data; masked samples are
        # left out of the window means, and a new sample is masked (NaN) only
        # if all samples in its window were masked
        ndata=EyeDataDict()
        for k,v in obj.data.items():
            v=np.where(obj.data.mask[k].astype(bool), np.nan, v)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
                y=baseline.downsample(v, dsfac)
            ndata.set_with_mask(k, y, mask=np.isnan(y).astype(int))
        obj.data=ndata

        # also downsample the time-vector
        obj.tx=baseline.downsample(obj.tx, dsfac)

        # set new sampling rate
        obj.fs=fsd

        # rescale stored blinks in sample indices so that they still cover the
        # blinks: round starts down and ends up (blinks in time units stay valid)
        n=len(obj.tx)
        for store in (obj._blinks, obj._interpolated_blinks):
            for key, iv in store.items():
                if iv is None or iv.units is not None:
                    continue
                rescaled = [(int(on)//dsfac, min(int(np.ceil(off/dsfac)), n)) for on, off in iv.intervals]
                store[key] = Intervals(rescaled, None, label=iv.label, data_time_range=(0, n),
                                       sampling_rate=obj.fs, time_offset=obj.tx[0])
        return obj

    @keephistory
    def merge_eyes(self, eyes=[], variables=[], method="mean", keep_eyes=True, inplace=None):
        """Merge data from both eyes into a single variable.

        The merged data is stored under the eye name "mean" (e.g.,
        ``obj["mean", "pupil"]``). Each sample is the mean over the eyes
        in which it is not masked; it is masked in the merged data only if
        it is masked in all merged eyes (it then holds the mean of the
        stored values, e.g., of interpolated blinks).

        Parameters
        ----------
        eyes : str or list of str, optional
            eye(s) to merge; if empty, all available eyes
        variables : str or list of str, optional
            variable(s) to merge; if empty, all available variables
        method : str, optional
            which method to use for merging; currently only "mean" is
            implemented, by default "mean"
        keep_eyes : bool, optional
            keep original eye data (True, default) or drop it together
            with its stored blinks (False)
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the object with merged data (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        eyes,variables=self._get_eye_var(eyes,variables)

        if method=="mean":            
            for var in variables:
                # Collect masked arrays from all eyes
                eye_data = []
                for eye in eyes:
                    if var not in obj.variables:
                        raise ValueError("No data for variable %s available" % var)
                    if eye not in obj.eyes:
                        raise ValueError("No data for eye %s available" % eye)
                    eye_data.append(obj[eye, var])
                
                # Mean over the unmasked eyes; masked only where all eyes
                # are masked
                stacked = np.ma.stack(eye_data, axis=0)
                meanval = np.ma.mean(stacked, axis=0)
                allmasked = np.ma.getmaskarray(meanval)
                # where all eyes are masked, keep the mean of the stored
                # values (e.g., interpolated blinks), still masked
                raw = np.ma.masked_invalid(np.stack([obj.data[eye, var] for eye in eyes]))
                rawmean = np.ma.mean(raw, axis=0).filled(np.nan)
                values = np.where(allmasked, rawmean, np.ma.getdata(meanval))
                obj["mean", var] = np.ma.array(values, mask=allmasked)
                
                if not keep_eyes:
                    for eye in eyes:
                        logger.debug("Dropping eye %s for variable %s" % (eye, var))
                        del obj.data[eye+"_"+var]
                        obj.set_blinks(None, eyes=[eye], variables=[var])
        else:
            raise ValueError("Method %s not implemented" % method)

        return obj
        
    @keephistory
    def blinks_merge_close(
        self,
        eyes: list = [],
        variables: list = [],
        distance: float = 100,
        units: str = "ms",
        apply_mask: bool = False,
        inplace: bool | None = None,
    ):
        """
        Merge blinks that are close together.
        
        Parameters
        ----------
        eyes : str or list of str, optional
            eye(s) to process; if empty, all available eyes
        variables : str or list of str, optional
            variable(s) to process; if empty, all available variables
        distance : float, str or Quantity, optional
            Merge blinks closer than this distance, e.g. ``100 * u.ms`` or
            ``"0.1 sec"``; plain numbers are in units given by ``units``.
            Default 100.
        units : str, optional
            Units for ``distance`` if it is a plain number. Can be "ms",
            "sec", "min", or "h". Default "ms".
        apply_mask : bool, default=False
            If True, store the merged blinks, apply them as masks to the data
            and return the object.
            If False, return merged blinks as dict of Intervals objects without
            storing them or applying masks (same default as
            :meth:`~pypillometry.PupilData.pupil_blinks_detect`).
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor).
            Only relevant when apply_mask=True.
            
        Returns
        -------
        GenericEyeData or dict of Intervals
            If apply_mask=True: returns modified object for chaining.
            If apply_mask=False: returns dict of Intervals objects (one per eye/variable combination).
        """
        obj = self._get_inplace(inplace)
        eyes, variables = obj._get_eye_var(eyes, variables)

        distance_ix = parse_time(distance, to="samples", default=normalize_unit(units) or "ms",
                                 fs=obj.fs)

        logger.debug(f"Distance '{distance}' in index units: {distance_ix}")

        # Dictionary to store merged intervals
        merged_dict = {}

        for eye, var in itertools.product(eyes, variables):
            blinks = obj.get_blinks(eye, var, units=None)
            if len(blinks) == 0:
                continue
            logger.debug(f"Processing blinks for eye {eye} and variable {var}: {len(blinks)} blinks")
            blinks_array = np.array(blinks).astype(int)
            
            newblinks = []
            i = 1
            cblink = blinks_array[0, :].copy()
            while i < blinks_array.shape[0]:
                if (blinks_array[i, 0] - cblink[1]) <= distance_ix:
                    cblink[1] = blinks_array[i, 1]
                else:
                    newblinks.append(cblink)
                    cblink = blinks_array[i, :].copy()
                i += 1
            newblinks.append(cblink)
            
            # Store merged blinks as Intervals
            merged_intervals = Intervals(
                intervals=[(int(s), int(e)) for s, e in newblinks],
                units=None,
                label=f"{eye}_{var}_blinks",
                data_time_range=(0, len(obj.tx)),
                sampling_rate=obj.fs,
                time_offset=obj.tx[0]
            )
            
            key = f"{eye}_{var}"
            merged_dict[key] = merged_intervals
            
            # Apply mask if requested
            if apply_mask:
                obj.set_blinks(merged_intervals, eyes=[eye], variables=[var], apply_mask=True)

        # Return appropriate result
        if apply_mask:
            return obj
        else:
            return merged_dict    
            
    def stat_per_event(self, 
                       intervals=None,
                       interval: Tuple[float,float]=None, event_select=None,                        
                       eyes: str|list=[], variables: str|list=[],                       
                       statfct: Callable=np.mean, units: str="ms",
                       **kwargs):
        """
        Return result of applying a statistical function to data in a
        given interval relative to event-onsets. For example, extract mean 
        pupil-size in interval before trial onset.

        Parameters
        ----------
        intervals : Intervals, optional
            Intervals object to use, in any time units or in sample indices
            (`units=None`, the default of :meth:`get_intervals`). If
            provided, interval and event_select are ignored.
        interval : tuple (min,max) or Quantity, optional
            time-window relative to event-onset (0 is event-onset), e.g.
            ``(-200, 0) * u.ms``; plain numbers are in units given by `units`.
            Required if intervals is not provided.
        event_select : str or function, optional
            variable describing which events to select and align to
            see :meth:`get_intervals` for details.
            Required if intervals is not provided.
        eyes : str or list of str, optional
            eye(s) to consider; if empty, all available eyes
        variables : str or list of str, optional
            variable(s) to consider; if empty, all available variables
        statfct : function, optional
            function mapping a masked array to a single number; it must
            ignore masked samples (e.g., `np.mean` (default) or
            `np.ma.median`, but not `np.median`; a warning is logged
            otherwise). Intervals where all samples are masked give NaN.
        units : str, optional
            units for plain numbers in `interval` (if used); default "ms"
        **kwargs
            passed onto the event_select function

        Returns
        -------
        dict
            one array per eye/variable (keys like "left_pupil"), with one
            value per interval/event
        """
        from ..intervals import Intervals, stat_event_interval
        
        eyes,variables=self._get_eye_var(eyes,variables)
        
        # Accept either Intervals object or old-style parameters
        if intervals is not None:
            if not isinstance(intervals, Intervals):
                raise TypeError("intervals must be an Intervals object")
            intervs = intervals
        else:
            if event_select is None or interval is None:
                raise ValueError("Must provide either 'intervals' or both 'event_select' and 'interval'")
            intervs = self.get_intervals(event_select, interval, units=units, **kwargs)

        # stat_event_interval matches the bounds against the time vector it
        # gets: use sample numbers for index-based Intervals, else tx in ms
        if intervs.units is None:
            tx = np.arange(len(self.tx))
            intervals_list = intervs.intervals
        else:
            tx = self.tx
            intervals_list = intervs.to_units("ms").intervals

        _warn_if_ignores_mask(statfct, "statfct")
        stat={}
        for eye,var in itertools.product(eyes, variables):
            # masked array, so that masked samples are ignored
            stat[eye+"_"+var]=stat_event_interval(tx, self[eye,var], intervals_list, statfct)

        return stat

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
        other_size += sys.getsizeof(self.info) if self.info is not None else 0
        
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
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the object with merged masks (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        
        # Get joint mask across all variables
        joint_mask = obj.data.get_mask()
        
        # Apply joint mask to all variables
        for key in obj.data.keys():
            obj.data.set_mask(key, joint_mask)
            
        return obj

    @keephistory
    def mask_intervals(self, intervals, eyes=[], variables=[], inplace=None):
        """Apply Intervals as masks to selected data.
        
        Takes detected intervals (e.g., blinks, artifacts) and applies them
        as masks to specified eyes and variables. Existing masks are kept
        (the new mask is added to them).

        Parameters
        ----------
        intervals : Intervals or dict
            Either:

            - A single Intervals object: applied to all specified eyes/variables
            - A dict of Intervals: keys like "left_pupil", "right_x", etc.
              When dict is provided, `eyes` and `variables` are ignored.
        eyes : str or list of str, optional
            eye(s) to apply mask to (e.g., ``["left", "right"]``);
            if empty, all available eyes. Ignored if `intervals` is a dict.
        variables : str or list of str, optional
            variable(s) to apply mask to (e.g., ``["pupil", "x", "y"]``);
            if empty, all available variables. Ignored if `intervals` is a dict.
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            The object with masks applied (`self` or a copy, depending on `inplace`).
            
        Examples
        --------
        Apply blinks from a single Intervals object to left eye pupil data:
        
        >>> blinks = data.pupil_blinks_detect(apply_mask=False)
        >>> data.mask_intervals(blinks['left_pupil'], eyes=['left'], variables=['pupil'])
        
        Apply a dict of intervals (one per eye/variable):
        
        >>> blinks = data.pupil_blinks_detect(apply_mask=False)
        >>> data.mask_intervals(blinks)  # eyes and variables ignored
        
        Chain with other operations:
        
        >>> data.mask_intervals(blinks).pupil_lowpass_filter(cutoff=5)
        
        See Also
        --------
        pupil_blinks_detect : Detect blinks with optional masking
        get_blinks : Retrieve stored blink intervals
        """
        obj = self._get_inplace(inplace)

        # Check if intervals is a dict
        if isinstance(intervals, dict):
            # Apply each Intervals object to its corresponding eye_variable
            for key, interval_obj in intervals.items():
                if not isinstance(interval_obj, Intervals):
                    raise TypeError(f"Dict values must be Intervals objects, got {type(interval_obj)} for key '{key}'")
                
                # Ensure sampling_rate is set for unit conversion
                if interval_obj.sampling_rate is None:
                    interval_obj.sampling_rate = obj.fs
                
                # Convert to index-based intervals if needed
                intervals_array = np.array(interval_obj.to_units("indices")).astype(int)
                
                # Apply mask for this specific eye_variable
                if key not in obj.data.data:
                    logger.warning(f"Key '{key}' not found in data, skipping")
                    continue
                    
                for start, end in intervals_array:
                    start_idx = int(max(0, min(len(obj.tx) - 1, start)))
                    end_idx = int(max(0, min(len(obj.tx), end)))
                    obj.data.mask[key][start_idx:end_idx] = 1
        else:
            # Single Intervals object - apply to specified eyes/variables
            if not isinstance(intervals, Intervals):
                raise TypeError(f"intervals must be an Intervals object or dict, got {type(intervals)}")
            
            # Ensure sampling_rate is set for unit conversion
            if intervals.sampling_rate is None:
                intervals.sampling_rate = obj.fs
            
            eyes, variables = obj._get_eye_var(eyes, variables)
            
            # Convert to index-based intervals if needed
            intervals_array = np.array(intervals.to_units("indices")).astype(int)
            
            # Apply to all combinations of eyes and variables
            for eye in eyes:
                for var in variables:
                    key = f"{eye}_{var}"
                    if key not in obj.data.data:
                        logger.warning(f"Key '{key}' not found in data, skipping")
                        continue
                    
                    for start, end in intervals_array:
                        start_idx = int(max(0, min(len(obj.tx) - 1, start)))
                        end_idx = int(max(0, min(len(obj.tx), end)))
                        obj.data.mask[key][start_idx:end_idx] = 1
        
        return obj

    @keephistory
    def interpolate_intervals(self, intervals, eyes=[], variables=[], 
                             method="linear", margin=(-20, 20),
                             store_as_suffix=None, inplace=None):
        """
        Interpolate data within specified intervals.
        
        This function replaces data within the specified intervals with interpolated
        values based on boundary points outside each interval. The
        interpolated samples are also marked as masked. Intervals without
        valid (non-NaN) boundary points on both sides are skipped with a
        warning.

        Parameters
        ----------
        intervals : Intervals or dict
            Either:

            - A single Intervals object: applied to all specified eyes/variables
            - A dict of Intervals: keys like "left_pupil", "right_x", etc.
              When dict is provided, `eyes` and `variables` are ignored.
        eyes : str or list of str, optional
            eye(s) to interpolate; if empty, all available eyes.
            Ignored if `intervals` is a dict.
        variables : str or list of str, optional
            variable(s) to interpolate; if empty, all available variables.
            Ignored if `intervals` is a dict.
        method : str, default "linear"
            Interpolation method. Options: "linear", "nearest", "zero", 
            "slinear", "quadratic", "cubic", "previous", "next".
            Passed to scipy.interpolate.interp1d.
        margin : tuple, default (-20, 20)
            Time window (in ms) before and after the interval to collect boundary
            points for interpolation, relative to the start and end of each
            interval: points are taken from ``start + margin[0]`` to the start
            and from the end to ``end + margin[1]``, so `margin[0]` must be
            <= 0 and `margin[1]` >= 0. Masked or NaN samples are not used.
            For higher-order methods like "cubic" or "quadratic", more
            boundary points from this margin are used to fit the
            interpolation curve.
        store_as_suffix : str or None, default None
            If None, replace the original data.
            If a string, store interpolated data in a new variable with this suffix
            (e.g., "_interp" creates "left_pupil_interp").
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            The object with interpolated data (`self` or a copy, depending on `inplace`).
            
        Examples
        --------
        Interpolate blinks for left eye pupil data:
        
        >>> blinks = data.get_blinks(eyes='left', variables='pupil')
        >>> data.interpolate_intervals(blinks, eyes=['left'], variables=['pupil'])
        
        Interpolate using a dict of intervals:
        
        >>> blinks = data.get_blinks()  # returns dict
        >>> data.interpolate_intervals(blinks)
        
        Store interpolated data in a new variable:
        
        >>> data.interpolate_intervals(blinks, store_as_suffix="_interp")
        >>> # Access via data['left', 'pupil_interp']
        
        Use cubic interpolation with wider margin:
        
        >>> data.interpolate_intervals(blinks, method="cubic", margin=(-50, 50))
        
        See Also
        --------
        mask_intervals : Apply intervals as masks without interpolation
        get_blinks : Retrieve stored blink intervals
        """
        from scipy.interpolate import interp1d
        
        obj = self._get_inplace(inplace)
        
        if margin[0] > 0 or margin[1] < 0:
            raise ValueError(f"margin must be (<=0, >=0) in ms, got {margin}")

        # Convert margin from ms to samples
        margin_left_ix = int(-margin[0] / 1000.0 * obj.fs)
        margin_right_ix = int(margin[1] / 1000.0 * obj.fs)
        
        def interpolate_single_key(key, interval_obj):
            """Local helper to interpolate a single data key."""
            # Ensure sampling_rate is set
            if interval_obj.sampling_rate is None:
                interval_obj.sampling_rate = obj.fs
            
            # Convert to index-based intervals
            intervals_array = np.array(interval_obj.to_units("indices")).astype(int)
            
            # Get the data and mask
            data = obj.data[key].copy()
            mask = obj.data.mask[key].copy()
            
            # Interpolate each interval
            for start, end in intervals_array:
                start_idx = int(max(0, start))
                end_idx = int(min(len(obj.tx), end))
                
                if end_idx <= start_idx:
                    continue
                
                # Collect boundary points from margin windows
                # Left margin: from (start_idx - margin_left_ix) to (start_idx - 1)
                left_margin_start = max(0, start_idx - margin_left_ix)
                left_margin_end = start_idx  # exclusive
                
                # Right margin: from end_idx to (end_idx + margin_right_ix - 1)
                right_margin_start = end_idx
                right_margin_end = min(len(obj.tx), end_idx + margin_right_ix)
                
                # Get valid (non-NaN, non-masked) points from left margin;
                # `mask` also covers intervals interpolated before
                left_indices = []
                left_times = []
                left_values = []
                for idx in range(left_margin_start, left_margin_end):
                    if not np.isnan(data[idx]) and not mask[idx]:
                        left_indices.append(idx)
                        left_times.append(obj.tx[idx])
                        left_values.append(data[idx])
                
                # Get valid (non-NaN, non-masked) points from right margin
                right_indices = []
                right_times = []
                right_values = []
                for idx in range(right_margin_start, right_margin_end):
                    if not np.isnan(data[idx]) and not mask[idx]:
                        right_indices.append(idx)
                        right_times.append(obj.tx[idx])
                        right_values.append(data[idx])
                
                # Need at least one point on each side
                if len(left_values) == 0 or len(right_values) == 0:
                    logger.warning(f"Cannot interpolate interval [{start_idx}:{end_idx}] - "
                                  f"insufficient valid boundary points")
                    continue
                
                # Combine boundary points
                boundary_times = np.array(left_times + right_times)
                boundary_values = np.array(left_values + right_values)
                
                # Determine if we have enough points for the requested method
                n_points = len(boundary_times)
                min_points = {"cubic": 4, "quadratic": 3}.get(method, 2)
                
                use_method = method
                if n_points < min_points:
                    logger.warning(f"Method '{method}' requires {min_points}+ points but only {n_points} "
                                  f"available for interval [{start_idx}:{end_idx}]. "
                                  f"Falling back to linear interpolation.")
                    use_method = "linear"
                
                # Interpolate
                if use_method == "linear" and n_points == 2:
                    # Use numpy's interp for simple 2-point linear
                    data[start_idx:end_idx] = np.interp(
                        obj.tx[start_idx:end_idx],
                        boundary_times,
                        boundary_values
                    )
                else:
                    f = interp1d(boundary_times, boundary_values, 
                                kind=use_method, fill_value="extrapolate")
                    data[start_idx:end_idx] = f(obj.tx[start_idx:end_idx])
                
                # Update mask for interpolated region
                mask[start_idx:end_idx] = 1
            
            # Store result
            target_key = key if store_as_suffix is None else f"{key}{store_as_suffix}"
            obj.data.set_with_mask(target_key, data, mask=mask)
        
        # Handle dict vs single Intervals
        if isinstance(intervals, dict):
            for key, interval_obj in intervals.items():
                if not isinstance(interval_obj, Intervals):
                    raise TypeError(f"Dict values must be Intervals objects, got {type(interval_obj)}")
                
                if key not in obj.data.data:
                    logger.warning(f"Key '{key}' not found in data, skipping")
                    continue
                
                interpolate_single_key(key, interval_obj)
        else:
            if not isinstance(intervals, Intervals):
                raise TypeError(f"intervals must be Intervals or dict, got {type(intervals)}")
            
            eyes, variables = obj._get_eye_var(eyes, variables)
            
            for eye in eyes:
                for var in variables:
                    key = f"{eye}_{var}"
                    if key not in obj.data.data:
                        logger.warning(f"Key '{key}' not found in data, skipping")
                        continue
                    
                    interpolate_single_key(key, intervals)
        
        return obj