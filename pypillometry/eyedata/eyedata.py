"""
eyedata.py
==========

The class :class:`EyeData`: pupil and/or gaze data of one or two eyes.
The helpers it uses (and that the namespaces ``d.pupil``/``d.gaze`` use as
well) are in :mod:`pypillometry.eyedata.generic`.
"""

import base64
import copy
import inspect
import numbers
import sys
import warnings
from collections.abc import Iterable
from importlib import resources
from io import BytesIO
from random import choice
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from .. import config, io, state
from ..convenience import (sizeof_fmt, ByteSize, _round_values, _plain_values, requires_package,
                           normalize_unit, _warn_if_ignores_mask)
from ..epochs import Epochs
from ..events import Events, _pair_with_next, _warn_if_needs_sync
from ..intervals import Intervals, _as_intervals
from ..plot import PlotNamespace
from ..signal import baseline
from ..units import (UNSET, _has_units, _time_values_and_units, parse_time, _ms_to_samples,
                     _samples_to_ms)
from .eyedatadict import EyeDataDict, REC_LAYERS, _join_key
from .experimental_setup import ExperimentalSetup
from .gaze import GazeNamespace
from .generic import (_check_per_pair, _check_timebase, _decimate_dict,
                      _decimate_factor, _history_arg, _insert_dict, _keep_dict, _merge_usage,
                      _per_pair, _some, _store_merged, keephistory)
from .namespace import accessor
from .pipes import step, Step, Pipeline, _missing_call
from .pupil import PupilNamespace
from .spatial_calibration import SpatialCalibration


class EyeData(state.VersionedState):
    """
    Class for eye-tracking and pupillometric data: pupil size and/or gaze
    position (x, y) of one or two eyes.

    The object can hold any of these channels (e.g., only pupil data, only
    gaze data, or both). The methods for pupil data are in the namespace
    ``d.pupil`` (:class:`~pypillometry.eyedata.pupil.PupilNamespace`, e.g.
    ``d.pupil.blinks_detect()``), those for gaze data in ``d.gaze``
    (:class:`~pypillometry.eyedata.gaze.GazeNamespace`) and the plots in
    ``d.plot`` (:class:`~pypillometry.plot.PlotNamespace`, e.g.
    ``d.plot.pupil_plot()``); they raise a ValueError if the object has no
    such data. The methods for any channel (e.g., :meth:`crop`) are methods
    of the object. All transforming methods can also be used as pipeline steps, e.g.
    ``d | pp.pupil.blinks_detect() | pp.crop(0, "10 min")`` (see
    :class:`~pypillometry.Pipeline`).

    Parameters
    ----------
    time : np.ndarray or None
        timing array (in ms) or `None`, in which case the time-array goes from [0,maxT]
        using `sampling_rate`
    left_x, left_y, left_pupil : np.ndarray or None
        data from the left eye: gaze position (x and y, both or neither)
        and pupil size; at least one channel of one eye must be provided
    right_x, right_y, right_pupil : np.ndarray or None
        data from the right eye (as for the left eye)
    event_onsets : np.ndarray or None
        time-onsets of any events in the data (in ms, matched in `time` vector)
    event_labels : np.ndarray or None
        for each event in `event_onsets`, provide a label
    events : Events or None
        the events as an :class:`~pypillometry.Events` object (e.g., with
        metadata); instead of `event_onsets` and `event_labels`
    sampling_rate : float or None
        sampling-rate of the signal in Hz; if `None`, it is estimated from
        the median difference in `time` (either `time` or `sampling_rate`
        must be provided)
    experimental_setup : ExperimentalSetup, optional
        Geometric model of the experimental setup including screen geometry,
        eye position, camera position, and screen orientation.
    name : str or None
        name of the dataset or `None` (in which case a random string is selected)
    calibration : dict, optional
        Dictionary of SpatialCalibration objects, one per eye (keys: 'left', 'right')
    fill_time_discontinuities : bool
        sometimes, when the eyetracker loses signal, no entry in the EDF is made; 
        when this option is True, such entries will be made and the signal set to 0 there
        (or do it later using `fill_time_discontinuities()`)
    keep_orig : bool
        keep a copy of the original dataset? If `True`, a copy of the object
        as initiated in the constructor is stored in member `original`
    info : dict or None
        optional dictionary with additional information about the dataset
        (stored in member `info`)
    inplace : bool or None
        default for methods called with `inplace=None`: if True, they modify the
        object in place; if False, they return a modified copy; if None (default),
        use the package-level setting (see :func:`~pypillometry.set_default_inplace`,
        initially True). The method-level `inplace` argument overrides this.

    Examples
    --------
    >>> import numpy as np
    >>> import pypillometry as pp
    >>> rng = np.random.default_rng(1)
    >>> d = pp.EyeData(time=np.arange(0, 1000, 1),
    ...                left_x=rng.normal(size=1000),
    ...                left_y=rng.normal(size=1000),
    ...                left_pupil=rng.normal(5, 0.1, size=1000))
    >>> d.eyes, d.variables, len(d)
    (['left'], ['x', 'y', 'pupil'], 1000)
    >>> print(d)  # summary of the data  # doctest: +SKIP
    """
    pupil = accessor(PupilNamespace)
    gaze = accessor(GazeNamespace)
    plot = accessor(PlotNamespace)

    name: str  ## name of dataset
    fs: float  ## sampling rate
    data: EyeDataDict ## dictionary with data (contains ndarrays)
    tx: np.ndarray ## time vector
    _events: Events ## events (onsets in ms), see `events`
    inplace: Optional[bool] ## whether to make changes in-place (None: package default)
    info: dict ## optional info about the dataset
    params: dict ## parameters 
    experimental_setup: Optional[ExperimentalSetup] ## geometry of the setup (or None)
    calibration: Optional[dict] ## SpatialCalibration per eye (or None)

    # state (see pypillometry.state); subclasses (FakeEyeData) add their own fields
    _STATE_FIELDS = (
        "name", "fs", "tx", ("data", "EyeDataDict"), ("events", "Events", "_events"),
        "params", "history", "inplace", "info", ("original", "versioned"),
        ("experimental_setup", "ExperimentalSetup"),
        ("calibration", "dict:SpatialCalibration"))
    # EyeDataDicts with one value per sample; the operations that change
    # the samples (_keep_samples, _decimate, _insert_samples) apply to all
    _SAMPLE_ALIGNED = ("data",)

    def _set_state(self, st: dict):
        # PupilData states of development versions of format 3 (before
        # 2.2) have no experimental setup and calibration; this can be
        # removed before 2.2 is released
        super()._set_state({"experimental_setup": None, "calibration": None, **st})

    def __init__(self, 
                    time: np.ndarray = None,
                    left_x: np.ndarray = None,
                    left_y: np.ndarray = None,
                    left_pupil: np.ndarray = None,
                    right_x: np.ndarray = None,
                    right_y: np.ndarray = None,
                    right_pupil: np.ndarray = None,
                    event_onsets: np.ndarray = None,
                    event_labels: np.ndarray = None,
                    events: Optional[Events] = None,
                    sampling_rate: float = None,
                    experimental_setup: Optional[ExperimentalSetup] = None,
                    name: str = None,
                    calibration: Optional[Dict[str, SpatialCalibration]] = None,
                    fill_time_discontinuities: bool = True,
                    keep_orig: bool = False, 
                    info: dict = None,
                    inplace: Optional[bool] = None):
        """Constructor for the EyeData class.
        """
        logger.debug("Creating EyeData object")
        if all(v is None for v in (left_x, left_y, left_pupil, right_x, right_y, right_pupil)):
            raise ValueError("At least one channel (e.g., left_pupil, or left_x and "
                             "left_y) must be provided")
        for eye, x, y in (("left", left_x, left_y), ("right", right_x, right_y)):
            if (x is None) != (y is None):
                raise ValueError(f"{eye}_x and {eye}_y must be provided together")
        self.data=EyeDataDict(left_x=left_x, left_y=left_y, left_pupil=left_pupil,
                                right_x=right_x, right_y=right_y, right_pupil=right_pupil)
        
        
        if events is not None:
            if event_onsets is not None or event_labels is not None:
                raise ValueError("give either events or event_onsets/event_labels, not both")
        self._init_common(time, sampling_rate, 
                          event_onsets, event_labels, 
                          name, fill_time_discontinuities, 
                          info=info, inplace=inplace,
                          experimental_setup=experimental_setup,
                          calibration=calibration, events=events)

        self.original=None
        if keep_orig: 
            self.original=self.copy()



    def _init_common(self, time: np.ndarray,
                    sampling_rate: float,
                    event_onsets: np.ndarray,
                    event_labels: np.ndarray,
                    name: str|None,
                    fill_time_discontinuities: bool,
                    inplace: Optional[bool],
                    info: dict = None,
                    experimental_setup: Optional[ExperimentalSetup] = None,
                    calibration: Optional[dict] = None,
                    events: Optional[Events] = None):
        """
        Common code for the child-classes of EyeData.
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
            tx=np.arange(len(self.data))*1000./sampling_rate
        else:
            tx=np.array(time, dtype=float)

        if sampling_rate is None:
            sampling_rate=np.round(1000./np.median(np.diff(tx)))
        self._set_timebase(tx, sampling_rate)

        if events is not None:
            self._events = self._check_events(events)
        else:
            if event_onsets is None:
                event_onsets = []
            if event_labels is None:
                event_labels = np.full(len(event_onsets), "")
            self._events = self._check_events(Events(event_onsets, event_labels, units="ms"))

        ## empty parameter dict
        self.params = dict()

        ## start with empty history    
        self.history=[]            

        ## init whether or not to do operations in place
        self.inplace=inplace 

        ## set info
        self.info = info if info is not None else {}

        ## experimental setup (geometry) and spatial calibration data
        self.experimental_setup = experimental_setup
        self.calibration = calibration

        ## fill in time discontinuities (part of construction, so bypass
        ## @keephistory and keep it out of the history)
        if fill_time_discontinuities:
            inspect.unwrap(type(self).fill_time_discontinuities)(self, inplace=True)

    def _set_timebase(self, tx, fs=None):
        """
        Set the time vector `tx` (ms) and the sampling rate `fs` (Hz) (in place).

        This is the only place where `tx` and `fs` are changed. `fs` must
        fit the time vector (see :func:`_check_timebase`).

        Parameters
        ----------
        tx : array_like
            new time vector in ms
        fs : float or None, optional
            new sampling rate in Hz; None: keep the current one

        Raises
        ------
        ValueError
            if `fs` does not match the time vector
        """
        tx = np.asarray(tx, dtype=float)
        if fs is None:
            fs = self.fs
        _check_timebase(tx, fs)
        self.tx = tx
        self.fs = fs

    def _samples(self, duration, units="ms", rounding="down"):
        """Number of samples in `duration` (e.g., a window size or margin).

        Plain numbers are in `units`; `duration` can also carry units (e.g.,
        ``50 * u.ms``) or be a tuple. `rounding` is "down" (towards zero),
        "up" or "nearest". Returns an int (or a tuple of ints).
        """
        n = parse_time(duration, to="samples", default=units, fs=self.fs)
        n = np.round(n, 9)  # rounding errors of the unit conversion
        n = {"down": np.trunc, "up": np.ceil, "nearest": np.round}[rounding](n)
        return tuple(int(v) for v in n) if np.ndim(n) else int(n)

    def _unit_fac(self, units):
        """Convert time units to factor for converting from milliseconds.
        
        Parameters
        ----------
        units : str
            Time unit (accepts aliases like "seconds", "s", "hrs", etc.)
            
        Returns
        -------
        float
            Conversion factor from milliseconds to target unit
        """
        units = normalize_unit(units)
        if units == "samples":
            raise ValueError("samples are not a time unit")
        
        if units == "sec":
            fac = 1./1000.
        elif units == "min":
            fac = 1./1000./60.
        elif units == "h":
            fac = 1./1000./60./60.
        else:  # "ms"
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

    def __or__(self, other):
        """
        Apply a pipeline step or a pipeline: ``d | pp.pupil.blinks_detect()``.

        Returns a processed copy; the object itself is not changed (see
        :class:`~pypillometry.Pipeline`).
        """
        if isinstance(other, (Step, Pipeline)):
            return other(self)
        if getattr(other, "_is_step", False):
            raise TypeError(_missing_call(other))
        return NotImplemented

    def __len__(self):
        """Return number of sampling points"""
        return len(self.tx)

    def __getitem__(self, key):
        """
        Provide dict-style access to data arrays, returning masked arrays
        (masked: unusable under the default mask policy, see :meth:`mask`;
        e.g., interpolated samples are not masked). Use :meth:`values` or
        :meth:`mask` for other policies.
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
            normalized_key = _join_key(key)

        if isinstance(normalized_key, str):
            if normalized_key.startswith("time"):
                prefix, _, suffix = normalized_key.partition("_")
                if prefix == "time":
                    return self._get_time_array(suffix or None)

        # Delegate to EyeDataDict and wrap using the default mask policy
        data_array = self.data[normalized_key]
        mask = self.data.get_policy_mask(self._channel_key(normalized_key, None))
        return np.ma.masked_array(data_array, mask=mask, fill_value=np.nan)

    @keephistory
    def __setitem__(self, key, value):
        """
        Set data arrays using dict-style access.
        
        Accepts numpy arrays (mask will be all zeros, except for NaN, and
        the layers are cleared) or masked arrays (mask and data stored
        separately in EyeDataDict: the mask of the variable becomes the
        mask of the array, samples masked in addition are added to the
        ``"user"`` layer, see :meth:`~pypillometry.EyeDataDict.set_mask`).
        
        Special handling for "time" keys:
        - data["time"] = array: Updates tx (assumes milliseconds; the steps
          must match the sampling rate `fs`)
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
        >>> import pypillometry as pp
        >>> d = pp.EyeData(time=[0, 10, 20], left_x=[1., 2., 3.],
        ...                left_y=[1., 2., 3.], left_pupil=[5., 5., 5.])
        >>> d["left_x"] = np.array([1, 2, 3])
        >>> d["left", "pupil"] = np.ma.masked_array([10, 20, 30], mask=[0, 1, 0])
        >>> d["time_sec"] = np.array([0, 0.01, 0.02])  # converted to ms
        >>> d.tx
        array([ 0., 10., 20.])
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
            normalized_key = _join_key(key)
        
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
            # Extract data and mask from masked array (layers kept where consistent)
            self.data.set_with_mask(normalized_key, value.data, preserve_mask=True)
            self.data.set_mask(normalized_key, np.ma.getmaskarray(value), layer="user")
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
            self._set_timebase(value.astype(float))
            return
        
        # Normalize unit (handles all aliases)
        unit = normalize_unit(unit)
        
        if unit == "ms":
            # Already in milliseconds
            self._set_timebase(value.astype(float))
        else:
            # Convert to milliseconds
            # _unit_fac gives factor to convert FROM ms to target unit
            # So to convert TO ms, we divide by the factor
            factor = self._unit_fac(unit)
            self._set_timebase((value / factor).astype(float))

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
        >>> import pypillometry as pp
        >>> d = pp.EyeData(time=[0, 10, 20], left_x=[1., 2., 3.],
        ...                left_y=[1., 2., 3.], left_pupil=[5., 5., 5.])
        >>> del d["left_x"]
        >>> del d["left", "pupil"]
        >>> list(d.data.keys())
        ['left_y']
        
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
            normalized_key = _join_key(key)
                
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

        if unit == "ms":
            return self.tx

        # Use _unit_fac which already handles all valid units
        factor = self._unit_fac(unit)
        return self.tx * factor

    def nevents(self) -> int:
        """Return number of events in data."""
        return len(self._events)

    def _eyes_with(self, eyes, variables):
        """Private helper: the eyes that have data for all `variables`.

        Eyes are taken from `eyes` (all available eyes if empty); without
        explicitly requested eyes, eyes that lack one of the variables are
        left out. Raises a KeyError if a requested eye/variable combination
        does not exist (see :meth:`~pypillometry.EyeDataDict.select`).
        """
        if isinstance(variables, str):
            variables = [variables]
        pairs = self.data._select(eyes, variables)
        found = dict.fromkeys(eye for eye, _ in pairs)
        eyes = [eye for eye in found if all((eye, var) in pairs for var in variables)]
        # name the channels in the log of the calling method
        logger.opt(depth=1).debug("channels: %s" % ", ".join(
            f"{eye}_{var}" for eye in eyes for var in variables))
        return eyes

    def _require(self, *variables):
        """Private helper: raise a ValueError unless there is data for each
        of `variables` (in at least one eye).

        Used by methods that need specific data (e.g., pupil or gaze), so
        that they fail with a clear message.
        """
        available = self.data.get_available_variables()
        missing = [var for var in variables if var not in available]
        if missing:
            raise ValueError("no %s data; available channels: %s"
                             % (", ".join(missing), ", ".join(self.data.keys()) or "none"))

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
            dictionary with blinks (Intervals in ms) for each
            eye/variable, keys like "left_pupil" (same as :meth:`get_blinks`)
        """
        return self.get_blinks()

    def _set_blink_layer(self, key, intervals):
        """Replace the blink layer of `key` by `intervals` (None: no blinks)."""
        blink = (np.zeros(len(self.tx), dtype=bool) if intervals is None
                 else intervals.to_mask(self).astype(bool))
        self.data.set_layer(key, "blink", blink, replace=True)
    
    @step
    @keephistory
    def set_blinks(self, intervals, eyes=[], variables=[]):
        """
        Set blinks for specified eyes and variables (in-place).

        The blinks replace the ``"blink"`` layer of the selected variables
        (and the mask of the blinks set before). Blinks are masked under
        the mask policies (see :meth:`mask`).

        Parameters
        ----------
        intervals : Intervals, dict or None
            Either:

            - a single :class:`~pypillometry.Intervals` object: stored for all
              specified eyes/variables;
            - a dict of Intervals (or `None`) with keys like "left_pupil",
              "right_x", etc.; when a dict is provided, `eyes` and
              `variables` are ignored;
            - `None`: remove the blinks for the specified
              eyes/variables (and their mask).
        eyes : str or list of str, optional
            eye(s) to set blinks for (e.g., ``["left", "right"]``);
            if empty, all available eyes. Ignored if `intervals` is a dict.
        variables : str or list of str, optional
            variable(s) to set blinks for (e.g., ``["pupil", "x", "y"]``);
            if empty, all available variables. Ignored if `intervals` is a dict.

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object itself (always modified in-place)
        """
        # Check if intervals is a dict
        if isinstance(intervals, dict):
            for key, interval_obj in intervals.items():
                if interval_obj is not None and not isinstance(interval_obj, Intervals):
                    raise TypeError(f"Dict values must be Intervals objects or None, got {type(interval_obj)} for key '{key}'")
            per_key = intervals
        else:
            # Single Intervals object - apply to specified eyes/variables
            if intervals is not None and not isinstance(intervals, Intervals):
                raise TypeError(f"intervals must be an Intervals object, dict, or None, got {type(intervals)}")
            per_key = {f"{eye}_{var}": intervals for eye, var in self.data.select(eyes, variables)}

        for key, interval_obj in per_key.items():
            if key not in self.data:
                logger.warning(f"Key '{key}' not found in data, skipping")
                continue
            self._set_blink_layer(key, interval_obj)
        return self

    def get_blinks(self, eyes: str|list = [], variables: str|list = [], 
                   units: str = "ms"):
        """
        Get blinks as a dict of Intervals (one per eye/variable).

        The blinks are the runs of the ``"blink"`` layer (blinks that
        overlap or touch are one blink), also those in or next to an
        excluded period (see
        :meth:`~pypillometry.EyeData.exclude_intervals`). If no
        blinks are stored for an eye/variable, an empty Intervals object is
        returned for it.

        Parameters
        ----------
        eyes : str or list of str, optional
            Eye(s) to get blinks for; if empty, all available eyes.
        variables : str or list of str, optional
            Variable(s) to get blinks for; if empty, all available variables.
        units : str, optional
            Units for intervals: "ms" (default), "sec", "min", "h", or
            "samples" for sample indices
            
        Returns
        -------
        dict
            one Intervals object per existing eye/variable combination, keys
            like "left_pupil" (also for a single eye and variable)

        Raises
        ------
        KeyError
            if a requested eye/variable combination does not exist (see
            :meth:`~pypillometry.EyeDataDict.select`)
            
        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short").pupil.blinks_detect()
        >>> blinks = d.get_blinks("left", "pupil")["left_pupil"]
        >>> blinks_sec = d.get_blinks("left", "pupil", units="sec")["left_pupil"]
        >>> blinks_dict = d.get_blinks(["left", "right"], "pupil")
        >>> list(blinks_dict)
        ['left_pupil', 'right_pupil']
        >>> blinks_idx = blinks.to_units("indices", d)
        """
        return self._layer_intervals("blink", eyes, variables, units, "blinks")

    def get_interpolated_blinks(self, eyes: str|list = [], variables: str|list = [], 
                                units: str = "ms"):
        """
        Get interpolated blinks as a dict of Intervals (one per eye/variable).
        
        Interpolated blinks are the runs of the ``"interpolated"`` layer:
        the blink regions that were successfully interpolated
        by :meth:`d.pupil.blinks_interpolate() <pypillometry.eyedata.pupil.PupilNamespace.blinks_interpolate>` (and the intervals interpolated by
        :meth:`interpolate_intervals`). These may differ from detected blinks because:

        - Blink boundaries are refined using velocity thresholds
        - Some blinks may be skipped if they're too short to interpolate

        If no interpolated blinks are stored for an eye/variable, an empty
        Intervals object is returned for it.

        Parameters
        ----------
        eyes : str or list of str, optional
            Eye(s) to get interpolated blinks for; if empty, all available eyes.
        variables : str or list of str, optional
            Variable(s) to get interpolated blinks for; if empty, all available
            variables.
        units : str, optional
            Units for intervals: "ms" (default), "sec", "min", "h", or
            "samples" for sample indices
            
        Returns
        -------
        dict
            one Intervals object per existing eye/variable combination, keys
            like "left_pupil" (also for a single eye and variable)

        Raises
        ------
        KeyError
            if a requested eye/variable combination does not exist (see
            :meth:`~pypillometry.EyeDataDict.select`)
            
        See Also
        --------
        get_blinks : Get detected (not interpolated) blinks
        pypillometry.eyedata.pupil.PupilNamespace.blinks_interpolate : Interpolate blinks
        """
        return self._layer_intervals("interpolated", eyes, variables, units,
                                     "interpolated_blinks")

    def get_layer_intervals(self, name: str, eyes: str|list = [],
                            variables: str|list = [], units: str = "ms"):
        """
        Get the runs of a mask layer (e.g., ``"offscreen"``) as a dict of Intervals.

        Each contiguous run of samples in which the layer is set becomes
        one interval. See :meth:`get_blinks` for the blinks and
        :meth:`get_excluded` for the excluded periods.

        Parameters
        ----------
        name : str
            name of the layer (e.g., ``"blink"``, ``"interpolated"``,
            ``"offscreen"``, ``"divergence"``, ``"user"``)
        eyes : str or list of str, optional
            eye(s); if empty, all available eyes
        variables : str or list of str, optional
            variable(s); if empty, all available variables
        units : str, optional
            units of the intervals: "ms" (default), "sec", "min", "h", or
            "samples" for sample indices

        Returns
        -------
        dict
            one Intervals object per existing eye/variable combination, keys
            like "left_x" (empty Intervals where the layer is not set)

        Examples
        --------
        >>> d = d.gaze.mask_offscreen()  # doctest: +SKIP
        >>> d.get_layer_intervals("offscreen", variables="x")  # doctest: +SKIP
        """
        return self._layer_intervals(name, eyes, variables, units, name)

    def _layer_runs(self, key, layer):
        """Runs of `layer` of `key` (Intervals in samples)."""
        if layer in REC_LAYERS:
            runs = Intervals.from_mask(self.data.get_rec_layer(layer)).intervals
        else:
            runs = Intervals.from_mask(self.data.get_layer(key, layer)).intervals
        return Intervals(runs, "samples")

    def _layer_intervals(self, layer, eyes, variables, units, suffix):
        """Runs of `layer` (e.g., interpolated blinks) for eyes/variables
        (see `_layer_runs`): a dict (keys like "left_pupil") over the
        existing eye/variable combinations.
        """
        result = {}
        for e, v in self.data.select(eyes, variables):
            key = f"{e}_{v}"
            runs = self._layer_runs(key, layer)
            result[key] = Intervals(runs.intervals, "samples",
                                    label=f"{key}_{suffix}").to_units(units, self)
        return result

    def _channel_key(self, eye, var):
        """Key ``"<eye>_<var>"`` of an existing channel (`var` None: `eye` is the key)."""
        key = eye if var is None else _join_key((eye, var))
        if key not in self.data:
            raise KeyError(f"No data for {key}; available channels: "
                           f"{', '.join(self.data.keys())}")
        return key

    def mask(self, eye: str, var: str|None = None, policy="default") -> np.ndarray:
        """
        Mask of a channel under a mask policy (True: unusable sample).

        A sample is unusable if its value is NaN or if one of the layers
        of the policy is set (see :class:`~pypillometry.MaskPolicy`). By
        default, blinks, off-screen gaze, eye divergences, user masks
        and excluded periods are masked, interpolated samples are not.

        Parameters
        ----------
        eye : str
            eye (e.g., "left") or the key of the channel (e.g., "left_pupil")
        var : str, optional
            variable (e.g., "pupil"); None if `eye` is the key
        policy : str, MaskPolicy or list of str
            ``"default"``, ``"strict"`` (interpolated samples are masked
            too), ``"raw"`` (nothing is masked), a
            :class:`~pypillometry.MaskPolicy` or a list of layers

        Returns
        -------
        np.ndarray
            boolean array, one value per sample

        Examples
        --------
        >>> d.mask("left", "pupil")  # doctest: +SKIP
        >>> d.mask("left_pupil", policy="strict")  # doctest: +SKIP
        >>> d.mask("left", "pupil", policy=["blink"])  # doctest: +SKIP
        """
        return self.data.get_policy_mask(self._channel_key(eye, var), policy)

    def values(self, eye: str, var: str|None = None, policy="default") -> np.ndarray:
        """
        Values of a channel with NaN where the sample is unusable under `policy`.

        Parameters
        ----------
        eye, var, policy
            see :meth:`mask`

        Returns
        -------
        np.ndarray
            copy of the values (float), NaN where :meth:`mask` is True
        """
        key = self._channel_key(eye, var)
        out = self.data[key].copy()
        out[self.data.get_policy_mask(key, policy)] = np.nan
        return out

    @step
    @keephistory
    def clear_layer(self, name: str, eyes: str|list = [], variables: str|list = [],
                    inplace=None):
        """
        Clear a mask layer (e.g., ``"blink"``) for the selected channels.

        Clearing ``"blink"`` removes the blinks and their mask (e.g., to
        undo :meth:`d.pupil.blinks_detect() <pypillometry.eyedata.pupil.PupilNamespace.blinks_detect>`).
        ``"excluded"`` is a recording-wide layer: clearing it removes the
        excluded periods (`eyes` and `variables` are ignored).

        Parameters
        ----------
        name : str
            name of the layer
        eyes : str or list of str, optional
            eye(s); if empty, all available eyes
        variables : str or list of str, optional
            variable(s); if empty, all available variables
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object without the layer (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        if name in REC_LAYERS:
            obj.data.clear_rec_layer(name)
            return obj
        for eye, var in obj.data.select(eyes, variables):
            obj.data.clear_layer(f"{eye}_{var}", name)
        return obj

    def mask_as_intervals(self, eyes: str|list = [], variables: str|list = [], 
                          units: str = "ms", label: str|None = None,
                          policy="default"):
        """
        Convert masks to Intervals objects.

        Each contiguous run of masked samples (unusable under the mask
        `policy`, see :meth:`mask`) becomes one interval.

        Parameters
        ----------
        eyes : str or list of str, optional
            Eye(s) to get masks for; if empty, all available eyes.
        variables : str or list of str, optional
            Variable(s) to get masks for; if empty, all available variables
            (combinations not present in the data are left out).
        units : str, optional
            Units for intervals: "ms" (default), "sec", "min", "h", or
            "samples" for sample indices.
        label : str, optional
            Descriptive label for the intervals. Default is None.
        policy : str, MaskPolicy or list of str
            mask policy (see :meth:`mask`); default ``"default"``
            
        Returns
        -------
        dict
            one Intervals object per existing eye/variable combination, keys
            like "left_pupil" (also for a single eye and variable)

        Raises
        ------
        KeyError
            if a requested eye/variable combination does not exist (see
            :meth:`~pypillometry.EyeDataDict.select`)
            
        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short").pupil.blinks_detect()
        >>> masked = d.mask_as_intervals("left", "pupil")["left_pupil"]
        >>> masked_sec = d.mask_as_intervals("left", "pupil", units="sec")["left_pupil"]
        >>> mask_dict = d.mask_as_intervals(["left", "right"], "pupil")
        >>> list(mask_dict)
        ['left_pupil', 'right_pupil']
        """
        result = {}
        for e, v in self.data.select(eyes, variables):
            key = f"{e}_{v}"
            mask = self.data.get_policy_mask(key, policy)
            result[key] = Intervals.from_mask(mask, self, units, label=label or f"{key}_mask")
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

    def nblinks(self, eyes=[], variables=[]) -> Dict[str, int]:
        """
        Return number of detected blinks. Should be run after
        :meth:`d.pupil.blinks_detect() <pypillometry.eyedata.pupil.PupilNamespace.blinks_detect>`.

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
            "left_pupil" (0 for combinations without blinks)
        """
        return {key: len(iv) for key, iv in self.get_blinks(eyes, variables).items()}


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
            stats dict (number, mean, sd, min, max and total duration of
            blinks; number 0 and NaN without blinks) for each eye with pupil
            data, with keys like "left_pupil"
        """
        return {key: blinks.stats()
                for key, blinks in self.get_blinks(eyes, "pupil", units=units).items()}


    def quality_report(self, thresholds: Optional[dict] = None, eyes: str|list = [],
                       policy="default"):
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
        :meth:`d.pupil.blinks_detect() <pypillometry.eyedata.pupil.PupilNamespace.blinks_detect>`), masked and
        interpolated data (after preprocessing; masked: unusable under the
        mask `policy`, see :meth:`mask`) and calibration/validation
        errors (e.g., when read from an Eyelink file).

        Where it makes sense, each measure is labeled as "good", "ok",
        "poor" or "bad". The overall label is the worst of all labels.

        Excluded periods (see :meth:`exclude_intervals`) are left out: all
        measures refer to the included time, and dropouts and
        sample-to-sample differences end at their boundaries.

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
        policy : str, MaskPolicy or list of str
            mask policy for the masked data (see :meth:`mask`); default
            ``"default"`` (interpolated samples are not masked)

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
        return _compute_quality_report(self, thresholds=thresholds, eyes=eyes, policy=policy)

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

    def get_duration(self, units="ms"):
        """Return duration of the dataset in units specified.

        Parameters
        ----------
        units : str, optional
            unit one of "ms" (default), "sec", "min", "h"

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

    @step
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
        :class:`~pypillometry.EyeData`
            the object without the original data (`self` or a copy,
            depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        obj.original=None
        return obj
    

    @step
    @keephistory
    def reset_time(self, t0: float|None=None, inplace=None):
        """
        Resets time so that time `t0` becomes time zero.

        `t0` is subtracted from the time-vector, the event onsets and the
        periods in ``params["removed_intervals"]`` (see
        :meth:`remove_intervals`).

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
        :class:`~pypillometry.EyeData`
            the object with shifted time (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        if t0 is None:
            t0 = self.tx[0]
        obj._set_timebase(self.tx-t0)
        obj._events = self._events.shift(-t0)
        # the original time of remove_intervals moves with the clock
        if "removed_intervals" in obj.params:
            obj.params["removed_intervals"] = [(a - t0, b - t0)
                                               for a, b in obj.params["removed_intervals"]]
        if "removed_shifts" in obj.params:
            obj.params["removed_shifts"] = [(t - t0, sh) for t, sh in obj.params["removed_shifts"]]
        return obj

    def _as_spans(self, intervals):
        """
        Intervals from `intervals`: an Intervals object (returned as it is),
        Epochs (their spans in ms) or a list of ``(start, end)`` tuples in ms (or with units), where
        ``"<start>"`` and ``"<end>"`` stand for the start and the end of the
        recording (as in :meth:`get_intervals`; ``-np.inf`` and ``np.inf``
        too, with a warning).
        """
        if isinstance(intervals, (Intervals, Epochs)):
            return _as_intervals(intervals)
        if not isinstance(intervals, (list, tuple, np.ndarray)):
            raise TypeError("intervals must be Intervals, Epochs or a list of (start, end) "
                            f"tuples, got {type(intervals).__name__}")
        bounds = {"<start>": float(self.tx[0]),
                  "<end>": float(_samples_to_ms(self, len(self.tx)))}

        def bound(v):
            if isinstance(v, str) and v in bounds:
                return bounds[v]
            if isinstance(v, numbers.Real) and np.isinf(v):
                logger.warning('use "<start>" and "<end>" for the start and end of the '
                               "recording instead of -np.inf and np.inf")
                return bounds["<start>" if v < 0 else "<end>"]
            # plain numbers are ms; values with units are converted (each
            # value on its own, so that e.g. (0, 2 * u.min) works)
            return parse_time(v, to="ms", default="ms")
        return Intervals([tuple(bound(v) for v in pair) for pair in intervals], units="ms")

    def _original_time(self, t):
        """
        Original time of the times `t` (ms) on the current clock.

        The original time is the clock of the data before
        :meth:`remove_intervals` closed gaps. It is the current time plus a
        shift that changes at the joins (``params["removed_shifts"]``: pairs
        ``(t, shift)``, the shift from time `t` on).
        """
        t = np.asarray(t, dtype=float)
        table = self.params.get("removed_shifts")
        if not table:
            return t.copy()
        ts, shifts = np.asarray(table, dtype=float).T
        return t + shifts[np.maximum(np.searchsorted(ts, t, side="right") - 1, 0)]

    def _set_removed_shifts(self, orig):
        """Store the shifts to the original time `orig` of the samples (one
        per sample; see :meth:`_original_time`)."""
        shifts = np.round(orig - self.tx, 6)  # without rounding errors
        change = np.flatnonzero(np.r_[True, np.diff(shifts) != 0])
        if len(shifts) == 0 or not shifts.any():
            self.params.pop("removed_shifts", None)
        else:
            self.params["removed_shifts"] = [(float(self.tx[i]), float(shifts[i])) for i in change]

    def _periods_to_original_time(self, periods):
        """
        Periods ``(start, end)`` (ms) on the current clock -> original time.

        A period across a join of :meth:`remove_intervals` is split into
        the parts before and after the join.
        """
        table = self.params.get("removed_shifts") or []
        joins = [t for t, _ in table]
        out = []
        for a, b in periods:
            cuts = [a] + [t for t in joins if a < t < b] + [b]
            for x, y in zip(cuts[:-1], cuts[1:]):
                shift = float(self._original_time([x])[0] - x) if table else 0.
                out.append((x + shift, y + shift))
        return out

    def _map_sample_aligned(self, fct, *args):
        """Replace each sample-aligned EyeDataDict `d` (see `_SAMPLE_ALIGNED`)
        by ``fct(d, *args)`` (in place)."""
        for name in self._SAMPLE_ALIGNED:
            d = getattr(self, name, None)
            if d is not None:
                setattr(self, name, fct(d, *args))

    def _keep_samples(self, keep, time_map=None, keep_events=None):
        """
        Keep only the samples where `keep` is True (in place).

        Cuts all variables, masks and layers. With `time_map` (a function
        mapping an array of times on the old clock to the new clock, e.g.
        to close the gaps), the time vector and the event onsets are
        mapped; without it, they stay valid. `keep_events` (one bool per
        event; default: all) selects the events that are kept.
        """
        keep = np.asarray(keep, dtype=bool)
        old_tx = self.tx
        self._map_sample_aligned(_keep_dict, keep)
        self._set_timebase((self.tx if time_map is None else time_map(self.tx))[keep])
        if time_map is not None:
            self._set_removed_shifts(self._original_time(old_tx)[keep])
        if keep_events is not None:
            self._events = self._events.keep(keep_events)
        if time_map is not None:
            self._events = self._events.shift(time_map)
        return self

    @step
    @keephistory
    def crop(self,
             start: float=-np.inf,
             end: float=np.inf,
             units: str="ms", inplace=None):
        """
        Crop the data to the time range from `start` to `end`.

        The result contains all data (and events) between `start` and `end`
        (inclusive), e.g. ``d.crop(3*u.min, 5*u.min)`` or
        ``d.crop("3 min", "5 min")``.
        `start` and `end` refer to the values of the time-vector (not to
        the time since the start of the recording). Masks are kept; stored
        blinks (in sample indices) are shifted and cut to the new range.

        Parameters
        ----------
        start : float, str or Quantity, optional
            start for new dataset; default: beginning of the data
        end : float, str or Quantity, optional
            end of new dataset; default: end of the data
        units : str, optional
            time units of `start` and `end` if they are plain numbers
            (one of "ms", "sec", "min", "h"); default "ms" (the units of
            the time vector).
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the shortened object (`self` or a copy, depending on `inplace`)

        Raises
        ------
        ValueError
            if `start` is after `end` or there are no samples between them

        See Also
        --------
        remove_intervals : remove periods (e.g., breaks) and join the rest
        """
        default = normalize_unit(units)
        start = parse_time(start, default=default)
        end = parse_time(end, default=default)
        if start > end:
            raise ValueError(f"crop(): start ({start:g} ms) is after end ({end:g} ms)")
        keep = (self.tx >= start) & (self.tx <= end)
        if not keep.any():
            raise ValueError(f"crop(): no samples between {start:g} and {end:g} ms "
                             f"(the data are from {self.tx[0]:g} to {self.tx[-1]:g} ms)")
        obj = self._get_inplace(inplace)
        on = obj._events.onsets
        obj._keep_samples(keep, keep_events=(on >= start) & (on <= end))
        return obj

    def sub_slice(self, start: float=-np.inf, end: float=np.inf,
                  units: str="ms", inplace=None):
        """
        Deprecated alias of :meth:`crop`.

        .. deprecated:: 2.1
           Use :meth:`crop` instead; `sub_slice` will be removed.
        """
        warnings.warn("sub_slice() is deprecated, use crop() instead",
                      DeprecationWarning, stacklevel=2)
        return self.crop(start, end, units=units, inplace=inplace)
    
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
        Reads a :class:`~pypillometry.EyeData` object from a pickle
        or HDF5 file (see :meth:`write_file`).
        Use as ``pypillometry.EyeData.from_file("yourfile.pypil")``.

        The file is read with :func:`pypillometry.io.read_pickle`, so
        `fname` can also be an HTTP(S) or SSH URL. Files written by older
        versions of pypillometry are upgraded to the current version.
        Pickle files can run arbitrary code when they are read: only read
        them from sources you trust (HDF5 files do not run code).

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
        :class:`~pypillometry.EyeData`
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

        All "messages" are stored as events (can be filtered later) and
        it tries to be smart to detect different information from the EDF file.
        Spatial calibration/validation data is automatically parsed and stored
        in the `calibration` attribute as SpatialCalibration objects.
        Eyelink codes missing pupil samples (e.g., during blinks) as 0; these
        are converted to NaN (and therefore masked).

        However, the EDF file may contain other information of interest, e.g.,
        eyelink-detected saccades, etc.
        If you need more control, use the "eyelinkio" package directly and
        create the object manually.

        Reading EDF files is slow (about a minute for a 90 MB file). Save the
        object with :meth:`write_file` and read it with :meth:`from_file`
        later, or convert the files once with the command-line program
        ``edf2pypil``.

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
            if True (default), remove all Eyelink triggers from the events
            (related to eye-tracker settings, calibration, validation etc)
        **kwargs
            additional arguments passed to :func:`pypillometry.io.read_eyelink`
            (e.g., `cache_file`, `cache_dir`, `cache_layout`, `autodelete`,
            `session`)

        Returns
        -------
        :class:`~pypillometry.EyeData` or tuple
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
            # eyelinkio gives e.g. "RIGHT_EYE"
            tracked_eye = str(edf["info"].get("eye", "left")).lower().removesuffix("_eye")
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
        
        obj = cls(time=tx, **d, calibration=calibration, experimental_setup=setup,
            event_onsets=evon, event_labels=evlab,
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
            obj = obj.set_events(obj.events.filter(eyelink_filter_func))

        if return_edf_obj:
            return obj, edf
        else:
            return obj

    def summary(self):
        """Return a summary of the dataset as a dictionary.

        Returns
        -------
        : dict
            A dictionary summarizing the dataset.
        """        
        # Experimental setup info
        if self.experimental_setup is not None:
            setup_info = self.experimental_setup.summary()
        else:
            setup_info = "not set"

        # Calibration info
        if self.calibration is not None:
            calibration_info = {
                eye: cal.get_stats() for eye, cal in self.calibration.items()
            }
        else:
            calibration_info = "not available"
        
        summary=dict(
            name=self.name, 
            n=len(self),
            sampling_rate=self.fs,
            eyes=self.eyes,
            data=list(self.data.keys()),
            nevents=self.nevents(), 
            experimental_setup=setup_info,
            calibration=calibration_info,
            duration_minutes=self.get_duration("min"),
            start_min=self.tx.min()/1000./60.,
            end_min=self.tx.max()/1000./60.,
            params=copy.deepcopy(self.params)
        )
        if self._excluded_mask().any():
            summary["excluded_minutes"] = self.excluded_duration("min")
        if "blink" in self.data.layer_names and "pupil" in self.variables:
            summary["nblinks"] = self.nblinks(variables="pupil")
            summary["blinks"] = self.blink_stats()
        else:
            summary["nblinks"] = {}
            summary["blinks"] = {}

        return _plain_values(summary)


    def __repr__(self) -> str:
        """Return a string-representation of the dataset."""
        pars=self.summary()
        del pars["name"]
        if "params" in pars:
            pars["params"] = self._strfy_params()  # short form
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
        if "excluded_minutes" in summary:
            overview.insert(3, ("Excluded", f"{summary['excluded_minutes']:.2f} min "
                                f"({len(self.get_excluded())} periods)"))

        if blink_stats_raw:
            counts = ", ".join(
                f"{key}: {stats['n']}" for key, stats in blink_stats_raw.items()
            )
            overview.insert(-1, ("Blinks", counts))

        blink_stats = []
        for key, stats in blink_stats_raw.items():
            if not stats["n"]:
                continue
            blink_stats.append(
                {
                    "eye": key,
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

    def copy(self, new_name: Optional[str]=None):
        """
        Make and return a deep-copy of the eyedata object.

        Parameters
        ----------
        new_name : str, optional
            New name for the copy. If None, "_copy" is appended to the original name.

        Returns
        -------
        :class:`~pypillometry.EyeData`
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

    @step
    @keephistory
    def drop_data(self, eyes: list = [], variables: list = [], inplace=None):
        """
        Drop specified eyes and/or variables from the data.
        
        All data of the given eyes and all data of the given variables are
        dropped (together with their mask layers). Everything else is kept
        (events, excluded periods, parameters, history, experimental setup,
        calibration); the class of the object does not change.

        Parameters
        ----------
        eyes : str or list of str, optional
            Eyes to drop (e.g., ['left'] or ['right'] or ['left', 'right']);
            if empty, no eye is dropped
        variables : str or list of str, optional
            Variables to drop (e.g., ['pupil'] or ['x', 'y']);
            if empty, no variable is dropped
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)
            
        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object without the specified data (`self` or a copy,
            depending on `inplace`)
            
        Raises
        ------
        KeyError
            If an eye or variable does not exist.
        ValueError
            If dropping would result in no data remaining.
            
        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> # Drop left eye data
        >>> right_only = d.drop_data(eyes=["left"], inplace=False)
        >>> right_only.eyes
        ['right']
        >>> # Drop pupil data, keep only gaze
        >>> gaze_only = d.drop_data(variables=["pupil"], inplace=False)
        >>> # Drop right eye and pupil data
        >>> left_gaze = d.drop_data(eyes=["right"], variables=["pupil"], inplace=False)
        >>> sorted(left_gaze.data.keys())
        ['left_x', 'left_y']
        """
        # Normalize inputs to lists
        if isinstance(eyes, str):
            eyes = [eyes]
        if isinstance(variables, str):
            variables = [variables]
        # unknown names raise (each is checked on its own)
        for eye in eyes or []:
            self.data.select(eyes=eye)
        for var in variables or []:
            self.data.select(variables=var)
        eyes, variables = eyes or [], variables or []
            
        # Get current keys
        keys_to_remove = []
        for key in list(self.data.data.keys()):
            parts = key.split("_", 1)
            if len(parts) != 2:
                continue
            eye, var = parts
            
            # Check if this key should be removed
            if eye in eyes or var in variables:
                keys_to_remove.append(key)
        if len(keys_to_remove) == len(self.data.data):
            raise ValueError("Dropping the specified data would leave no data remaining")

        # Remove the keys from data and mask
        result = self._get_inplace(inplace)
        for key in keys_to_remove:
            del result.data[key]
        return result

    def _check_events(self, events) -> Events:
        """
        `events` in ms, checked against the data.

        Raises a TypeError if `events` is not an Events object and a
        ValueError if an onset is outside the time range of the data; warns
        if the events are on another clock (`needs_sync`).
        """
        if not isinstance(events, Events):
            raise TypeError(f"events must be an Events object, got {type(events)}")
        _warn_if_needs_sync(events)
        events = events.to_units("ms")
        on = events.onsets
        t0, t1 = float(self.tx.min()), float(self.tx.max())
        outside = (on < t0) | (on > t1)
        if outside.any():
            hint = ("they are on another clock (e.g., from Events.from_table()): map them "
                    "with pp.sync_clocks() first" if events.needs_sync else
                    "events from another clock (e.g., a behaviour log) must be mapped to "
                    "the eye-tracker clock with pp.sync_clocks() first")
            raise ValueError(
                f"{outside.sum()} of {len(on)} event onsets are outside the data "
                f"({t0:.6g} to {t1:.6g} ms), e.g. at {_some(on[outside], 3)} ms; {hint}")
        # the time range of the data is not stored with the events; they are
        # on the clock of the data now
        return Events(on, events.labels, units="ms", meta=events.meta)

    @property
    def events(self) -> Events:
        """
        The events of the data (onsets in ms).

        An :class:`~pypillometry.Events` object. Events are immutable: their
        methods (e.g., :meth:`~pypillometry.Events.filter`,
        :meth:`~pypillometry.Events.parse_labels`) return new Events, which
        are stored with :meth:`set_events`. Operations on the data that
        change the time (e.g., :meth:`crop`, :meth:`remove_intervals`,
        :meth:`reset_time`) move the events with it.

        Examples
        --------
        >>> d.events.filter("stim")  # doctest: +SKIP
        >>> d = d.set_events(d.events.parse_labels(r"_(?P<trial>\\d+)"))  # doctest: +SKIP
        """
        return self._events

    @step
    @keephistory
    def set_experimental_setup(self, **kwargs):
        """
        Set or update the experimental setup geometry (in-place).
        
        Creates a new :class:`~pypillometry.ExperimentalSetup` or updates the
        existing one with the provided parameters (parameters that are not
        given are taken from the existing setup). All parameters are passed
        through to the ExperimentalSetup constructor.

        Parameters
        ----------
        **kwargs : dict
            Any parameters accepted by :class:`~pypillometry.ExperimentalSetup`:

            - screen_resolution: tuple (width, height) in pixels
            - physical_screen_size: tuple (width, height) with units
            - eye_to_screen_center: distance from the eye to the screen center
            - screen_offset: tuple (delta_x, delta_y) - offset of screen center from eye
            - screen_pitch: tilt angle (alpha_tilt)
            - screen_yaw: tilt angle (beta_tilt)
            - camera_offset: tuple (x, y, z) in eye-centric frame
            - camera_spherical: tuple (theta, phi, r)
            - ipd: inter-pupillary distance

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object itself (always modified in-place)
        
        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> d = d.set_experimental_setup(
        ...     screen_resolution=(1920, 1080),
        ...     physical_screen_size=("52 cm", "29 cm"),
        ...     eye_to_screen_center="65 cm",
        ... )
        """
        if self.experimental_setup is None:
            # Create new setup
            self.experimental_setup = ExperimentalSetup(**kwargs)
        else:
            # Merge with existing setup (stored values get their units
            # back, so they are not parsed as plain numbers)
            old = ExperimentalSetup._constructor_kwargs(self.experimental_setup.to_dict())
            if 'camera_offset' in kwargs or 'camera_spherical' in kwargs:
                # a new camera position replaces the old one completely
                old['camera_offset'], old['camera_spherical'] = None, None
            merged = {**old, **kwargs}

            self.experimental_setup = ExperimentalSetup(**merged)
        return self

    @step
    @keephistory
    def set_events(self, events):
        """
        Replace the events of the dataset (in-place).

        The onsets are converted to ms (the internal storage format) if
        needed. The time range of the data is not stored with the events.
        A warning is given for events that are marked as being on another
        clock (`needs_sync`, see :func:`~pypillometry.sync_clocks`).

        Parameters
        ----------
        events : Events
            the new events (onsets, labels and metadata)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object itself, with the new events (use ``d.copy()`` first
            to keep the old events)

        Raises
        ------
        TypeError
            if `events` is not an Events object
        ValueError
            if an onset is outside the time range of the data

        Examples
        --------
        >>> d = d.set_events(d.events.filter("stim"))  # doctest: +SKIP
        >>> d = d.set_events(d.events.parse_labels(r"_(?P<trial>\\d+)"))  # doctest: +SKIP

        See Also
        --------
        events : the events of the dataset
        add_events : add events to the existing ones
        """
        self._events = self._check_events(events)
        return self

    @step
    @keephistory
    def add_events(self, events, inplace=None):
        """
        Add events to the events of the dataset.

        The new events are merged with the existing ones and all events are
        sorted by onset (events at the same time keep their order, existing
        ones first). Metadata columns that only the existing or only the new
        events have get missing values (NaN or "") for the other events.
        Events from a behaviour log on another clock must be mapped to the
        eye-tracker clock with :func:`~pypillometry.sync_clocks` first; a
        warning is given for events that are marked as being on another
        clock (`needs_sync`, e.g., from
        :meth:`~pypillometry.Events.from_table`).

        Parameters
        ----------
        events : Events
            the events to add (converted to ms if needed)
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object with the added events (`self` or a copy, depending on `inplace`)

        Raises
        ------
        TypeError
            if `events` is not an Events object
        ValueError
            if an onset is outside the time range of the data

        Examples
        --------
        Events from a behaviour log ``behav`` (a table with times in sec,
        already on the eye-tracker clock; otherwise, see
        :func:`~pypillometry.sync_clocks`):

        >>> ev = pp.Events.from_table(behav, ["stim_on", "response"], units="sec",
        ...                           needs_sync=False)  # doctest: +SKIP
        >>> d = d.add_events(ev)  # doctest: +SKIP

        See Also
        --------
        set_events : replace all events
        """
        obj = self._get_inplace(inplace)
        if not isinstance(events, Events):
            raise TypeError(f"events must be an Events object, got {type(events)}")
        obj._events = obj._check_events(obj._events + events)
        return obj

    def event_onset(self, label: str, units: str = "ms") -> float:
        """
        Onset of the one event with the given label.

        Parameters
        ----------
        label : str
            label of the event; only a label equal to it matches
        units : str, optional
            units of the returned onset: "ms" (default), "sec", "min" or "h"

        Returns
        -------
        float
            onset of the event

        Raises
        ------
        ValueError
            if no event or more than one event has this label

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> d.event_onset("F_2", units="sec")
        8.776

        See Also
        --------
        events : all events (e.g., ``events.filter(label)`` for labels
            that occur several times)
        """
        events = self._events.to_units(units)
        onsets = events.onsets[events.labels == label]
        if len(onsets) != 1:
            raise ValueError(f"expected one event with label '{label}', "
                             f"found {len(onsets)}")
        return float(onsets[0])

    @step
    @keephistory
    def fill_time_discontinuities(self, yval=np.nan, print_info=True, inplace=None):
        """
        Find gaps in the time-vector and fill them in.

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
            so the filled-in samples are masked as missing data (other
            values: masked by the ``"user"`` layer)
        print_info : bool, optional
            if True (default), log the number of gaps that are filled in
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object without gaps (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        tx=obj.tx
        stepsize=np.median(np.diff(tx))
        n=len(obj)
        gaps_end_ix=np.where(np.r_[stepsize,np.diff(tx)]>2*stepsize)[0]
        ngaps=gaps_end_ix.size
        if ngaps!=0:
            ## at least one gap here
            if print_info:
                logger.info("Filling in %i gaps"%ngaps)
            gaps_start_ix=gaps_end_ix-1
            logger.info( str(((tx[gaps_end_ix]-tx[gaps_start_ix])/1000))+ " seconds" )
            
            ## build new time-vector: keep all valid segments and insert
            ## regularly spaced time points in each gap
            bounds=np.r_[0, gaps_end_ix, n]
            ntx=[]
            for i in range(ngaps+1):
                ntx.append( tx[bounds[i]:bounds[i+1]] )
                if i<ngaps:
                    start,end=gaps_start_ix[i], gaps_end_ix[i]
                    nfill=int((tx[end]-tx[start])/stepsize)-1
                    ntx.append( tx[start]+stepsize*np.arange(1, nfill+1) )
            # new position of each original sample (ntx alternates kept
            # segments and filled gaps)
            lens=[len(t) for t in ntx]
            offsets=np.cumsum([0]+lens[:-1])
            newpos=np.concatenate([offsets[2*i]+np.arange(lens[2*i]) for i in range(ngaps+1)])
            obj._insert_samples(np.concatenate(ntx), newpos, yval)
        return obj

    def _insert_samples(self, tx, pos, value=np.nan):
        """
        Insert new samples (in place).

        Parameters
        ----------
        tx : np.ndarray
            the new time vector (ms), with the old and the new samples
        pos : np.ndarray
            the index in `tx` of each old sample (increasing)
        value : float, optional
            value of all variables at the new samples (default NaN); the
            new samples are masked as missing (NaN, or the ``"user"`` layer
            if `value` is not NaN)

        Existing masks and layers are kept at the old samples; events
        stay valid.
        """
        pos=np.asarray(pos, dtype=int)
        self._map_sample_aligned(_insert_dict, len(tx), pos, value)
        self._set_timebase(tx)
        return self
    
    def _parse_window(self, window, units):
        """`window` as a tuple of plain numbers in `units` (and the units): a
        window with units is converted to `units` (or keeps its own), plain
        numbers are in `units` (default ms)."""
        if not isinstance(window, Iterable) or isinstance(window, str):
            raise ValueError("window must be iterable, e.g., (-200, 1000) * u.ms")
        if _has_units(window):
            window, units = _time_values_and_units(window, units, self.fs)
            if units == "samples":
                window = np.round(window).astype(int)
            window = tuple(window.tolist())
        elif units is UNSET:
            units = "ms"  # plain numbers are ms
        window = tuple(window)
        if len(window) != 2:
            raise ValueError(f"window must be (start, end), got {window}")
        return window, normalize_unit(units)

    def _select_events(self, evsel, exact, kwargs):
        """Onsets (ms), samples, indices and labels of the events matching the
        selector `evsel` (str or function); ``"<start>"`` and ``"<end>"`` are the
        start and the end of the recording (without indices)."""
        n = len(self.tx)
        # start and end of the recording as (onset in ms, sample); the end is
        # one sample past the last one, as intervals are end-exclusive
        if isinstance(evsel, str) and evsel in ("<start>", "<end>"):
            t, smp = (float(self.tx[0]), 0) if evsel == "<start>" \
                else (float(_samples_to_ms(self, n)), n)
            return np.array([t]), np.array([smp]), None, [evsel]
        if callable(evsel):
            ix = np.array([bool(evsel(evlab, **kwargs)) for evlab in self._events.labels],
                          dtype=bool)
        elif isinstance(evsel, str):
            ix = np.array([evlab == evsel if exact else evsel in evlab
                           for evlab in self._events.labels], dtype=bool)
        else:
            raise TypeError(f"event selectors must be a str or a function, "
                            f"got {type(evsel).__name__}")
        if np.sum(ix) == 0:
            raise ValueError(f"no events found matching {_history_arg(evsel)}")
        on = self._events.onsets[ix]
        return (on, np.clip(_ms_to_samples(self, on), 0, n-1), np.where(ix)[0],
                [self._events.labels[i] for i in np.where(ix)[0]])

    def _meta_of(self, indices):
        """Metadata of the events with `indices` (None if there are none)."""
        if indices is None or not self._events.meta:
            return None
        return {name: col[indices] for name, col in self._events.meta.items()}

    def get_epochs(self, event_select, window, units: str = UNSET,
                   label: str | None = None, exact: bool = False, **kwargs):
        """
        Return the time windows around events (Epochs).

        Selects events, e.g., all stimuli, and one time window around each
        of them, e.g., from 200 ms before to 1000 ms after the stimulus. The
        result is the input of :meth:`get_segments` (event-related pupil
        dilations) and can be used wherever intervals are taken (e.g.,
        :meth:`stat_per_event`, :meth:`mask_intervals`). For spans between two
        different events (e.g., from trial start to response), use
        :meth:`get_intervals`.

        Parameters
        ----------
        event_select : str, function or Events
            which events to select:

            - str: all events whose label contains the string (or is equal
              to it, if `exact` is True); ``"<start>"`` and ``"<end>"`` stand
              for the start and the end of the recording
            - function: applied to all labels, the events where it returns
              True are used
            - Events: the events of this :class:`~pypillometry.Events`
              object (e.g., ``d.events.filter(cond="easy")``)
        window : tuple (min, max) or Quantity
            time window relative to the event onset (0 is the onset; negative
            numbers are before it), e.g. ``(-200, 800) * u.ms`` or
            ``("-0.2 sec", "0.8 sec")``. Plain numbers are in `units`.
        units : str, optional
            units of plain numbers in `window` ("ms", "sec", "min", "h", or
            "samples" for sample indices, converted to ms with the sampling
            rate); default: the units of `window`, or "ms" for plain numbers
        label : str or None, optional
            label of the Epochs; if None, it is taken from `event_select`
        exact : bool, optional
            if True, a string selector matches only labels that are equal
            to it; if False (default), labels that contain it
        **kwargs
            passed on to the `event_select` function; an error is raised if
            it is not a function

        Returns
        -------
        Epochs
            the selected events (onsets in ms, labels and metadata, see
            :class:`~pypillometry.Events`) and the window (in ms). Windows
            that extend beyond the data are not clipped.

        Examples
        --------
        >>> import pypillometry as pp
        >>> from pypillometry import u
        >>> d = pp.EyeData(time=range(10000), left_pupil=range(10000),
        ...                event_onsets=[2000, 6000], event_labels=["stim", "stim"])
        >>> d.get_epochs("stim", (-200, 800) * u.ms)
        Epochs 'stim' | 2 epochs, window (-200, 800) ms
        """
        if kwargs and (isinstance(event_select, Events) or not callable(event_select)):
            raise TypeError("get_epochs() got unexpected keyword argument(s) "
                            f"{', '.join(kwargs)} (only passed to a function selector)")
        if isinstance(event_select, tuple):
            raise TypeError("get_epochs() takes one event selector; for the spans "
                            "between two events, use get_intervals(start=..., end=...)")
        window, units = self._parse_window(window, units)
        if window[1] <= window[0]:
            raise ValueError(f"window must be (min, max) with min < max, got {window}")

        if isinstance(event_select, Events):
            _warn_if_needs_sync(event_select)
            onsets = event_select.to_units("ms").onsets
            labels = event_select.labels.tolist()
            meta = dict(event_select.meta) or None
            if label is None:
                label = f"events_{len(event_select)}"
        else:
            onsets, _, indices, labels = self._select_events(event_select, exact, kwargs)
            meta = self._meta_of(indices)
            if label is None:
                label = event_select if isinstance(event_select, str) else "custom_events"

        # the window in ms
        fac = 1000. / self.fs if units == "samples" else 1. / self._unit_fac(units)
        return Epochs(onsets, (window[0] * fac, window[1] * fac), labels=labels,
                      meta=meta, label=label)

    def get_intervals(self, *, start, end, window=(0, 0), units: str = UNSET,
                      label: str | None = None, exact: bool = False, **kwargs):
        """
        Return the spans between two kinds of events (Intervals).

        For example, the time from each trial start to the response, or the
        breaks between blocks. For one time window around single events
        (e.g., stimuli), use :meth:`get_epochs`.

        Each `start` event is paired with the next `end` event after it (an
        `end` event at the same time does not count). If several `start`
        events come before the same `end` event, the last of them is used.
        Events without a partner are skipped with a warning that lists them
        (not for ``"<start>"`` and ``"<end>"``, which are paired with the
        first or last event only).

        Parameters
        ----------
        start, end : str or function
            the events at the start and at the end of the spans:

            - str: all events whose label contains the string (or is equal
              to it, if `exact` is True); ``"<start>"`` and ``"<end>"`` stand
              for the start and the end of the recording (the first sample
              and one sample past the last one, as intervals are
              end-exclusive)
            - function: applied to all labels, the events where it returns
              True are used

            For example, ``start="End Block", end="Start Block"`` gives the
            breaks between blocks and ``start="<start>", end="Start Block"``
            the time before the first block.
        window : tuple or Quantity, optional
            `window[0]` is added to the `start` events and `window[1]` to
            the `end` events; default ``(0, 0)``: exactly the spans between
            them. Plain numbers are in `units`.
        units : str, optional
            units of plain numbers in `window` and of the returned Intervals
            ("ms", "sec", "min", "h", or "samples" for sample indices: each
            event is placed at the first sample at or after its onset);
            default: the units of `window`, or "ms"
        label : str or None, optional
            label of the Intervals; if None, ``"<start> to <end>"`` (for two
            strings)
        exact : bool, optional
            if True, string selectors match only labels that are equal
            to them; if False (default), labels that contain them
        **kwargs
            passed on to the selector functions; an error is raised if no
            selector is a function

        Returns
        -------
        Intervals
            the spans (in `units`) with, per interval, the label
            ``"<start label> to <end label>"``, the onset of the start event
            and its metadata (those of the end event if `start` is
            ``"<start>"``). Intervals that extend beyond the data are not
            clipped.

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.EyeData(time=range(10000), left_pupil=range(10000),
        ...                event_onsets=[1000, 3000, 5000, 8000],
        ...                event_labels=["trial", "resp", "trial", "resp"])
        >>> d.get_intervals(start="trial", end="resp").intervals
        [(1000.0, 3000.0), (5000.0, 8000.0)]
        """
        if kwargs and not (callable(start) or callable(end)):
            raise TypeError("get_intervals() got unexpected keyword argument(s) "
                            f"{', '.join(kwargs)} (only passed to function selectors)")
        # window[0] and window[1] shift different events, so e.g. (0, 0) is
        # valid; the resulting intervals are checked below
        window, units = self._parse_window(window, units)

        on_a, s_a, i_a, l_a = self._select_events(start, exact, kwargs)
        on_b, s_b, i_b, l_b = self._select_events(end, exact, kwargs)
        pairs, skip_a, skip_b = _pair_with_next(on_a, on_b)
        name_a, name_b = _history_arg(start), _history_arg(end)
        # "<start>"/"<end>" pair with one event only; the others are not reported
        if skip_a and i_b is not None:
            logger.warning(
                f"get_intervals(): skipped {len(skip_a)} {name_a} event(s) with no "
                f"{name_b} event after them (before the next {name_a} event or "
                f"the end), at {_some(on_a[skip_a])} ms")
        if skip_b and i_a is not None:
            logger.warning(
                f"get_intervals(): skipped {len(skip_b)} {name_b} event(s) with no "
                f"{name_a} event before them, at {_some(on_b[skip_b])} ms")
        if not pairs:
            raise ValueError(f"no {name_a} event is followed by a {name_b} event")
        pa = np.array([p[0] for p in pairs])
        pb = np.array([p[1] for p in pairs])
        if label is None:
            label = f"{start} to {end}" if isinstance(start, str) and isinstance(end, str) \
                else "between_events"
        labels = [f"{l_a[i]} to {l_b[j]}" for i, j in pairs]
        meta = self._meta_of(i_a[pa]) if i_a is not None \
            else self._meta_of(None if i_b is None else i_b[pb])

        if units == "samples":
            # sample of each event, the window in samples around it
            sti = s_a[pa] + window[0]
            ste = s_b[pb] + window[1]
            onsets = np.atleast_1d(s_a[pa]).tolist()
        else:
            fac = self._unit_fac(units)
            sti = on_a[pa] * fac + window[0]
            ste = on_b[pb] * fac + window[1]
            onsets = (on_a[pa] * fac).tolist()
        bad = np.where(ste <= sti)[0]
        if len(bad) > 0:
            raise ValueError(f"window {window} gives empty intervals for pair(s) "
                             f"{bad.tolist()}")
        return Intervals(
            intervals=list(zip(np.atleast_1d(sti).tolist(), np.atleast_1d(ste).tolist())),
            units=units, label=label, event_labels=labels, event_onsets=onsets,
            event_meta=meta)

    def get_segments(self, intervals, variable: str, name: str = None, policy="default"):
        """
        Extract segments from data using intervals.
        
        Creates a SegmentedEyeData object containing equal-length segments
        of a single variable extracted at the specified intervals.
        
        Parameters
        ----------
        intervals : Epochs or Intervals
            the segments to extract: :class:`~pypillometry.Epochs` (from
            :meth:`get_epochs`; length
            and time axis ``window[0] + arange(n) * 1000/fs`` from the
            window, parts outside the data masked) or Intervals that all have
            the same length (time axis from 0 at their starts)
        variable : str
            Variable to extract in "eye_variable" format (e.g., "left_pupil", "right_x")
        name : str, optional
            Name for the segmented dataset. If None, uses intervals.label or "segments"
        policy : str, MaskPolicy or list of str
            mask policy of the segments: the samples that are unusable under
            it are masked (see :meth:`mask`); default ``"default"``
            (interpolated samples are used)
            
        Returns
        -------
        SegmentedEyeData
            Object containing the extracted segments with mask information

        Raises
        ------
        ValueError
            if `intervals` are Intervals of different lengths (e.g., between
            two events)
            
        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> # Extract pupil segments around the feedback events
        >>> epochs = d.get_epochs("F_", window=(-200, 800), units="ms")
        >>> segments = d.get_segments(epochs, "left_pupil")
        >>> segments.plot()  # doctest: +SKIP

        >>> # Apply baseline correction
        >>> segments_bc = segments.baseline_correct(window=(-200, 0))
        
        See Also
        --------
        get_epochs : Windows around events (Epochs)
        SegmentedEyeData : Class holding segmented data
        """
        from ..segmented import SegmentedEyeData
        return SegmentedEyeData.from_eyedata(self, intervals, variable, name=name,
                                             policy=policy)

    @step
    @keephistory
    def scale(self, variables=[], mean: Union[float,dict,None]=None, 
              sd: Union[float,dict,None]=None, eyes=[], policy="default", inplace=None):
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
            (of the usable samples under `policy`);
            if float, use the same value for all eyes and variables;
            if dict, provide mean for each channel as
            ``{"left_pupil": value, ...}``
        sd : None, float or dict, optional
            sd to scale with; if `None`, use the signal's std (of the
            usable samples under `policy`);
            if float, use the same value for all eyes and variables;
            if dict, provide sd for each channel as
            ``{"left_pupil": value, ...}``
        eyes : str or list of str, optional
            eye(s) to consider; if empty, all available eyes
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are used for the mean
            and sd (see :meth:`mask`); default ``"default"`` (interpolated
            samples are used)
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the scaled object (`self` or a copy, depending on `inplace`)

        Notes
        -----
        Scaling-parameters are being saved in the `params["scale"]` attribute
        (``{"mean": {"left_pupil": ...}, "sd": {...}}``) and are used by
        :meth:`unscale`. Scaling a
        variable again combines the parameters, so that :meth:`unscale`
        restores the values before the first `scale`.
        """
        obj = self._get_inplace(inplace)
        pairs = obj.data.select(eyes, variables)

        if mean is None:
            # use only the usable samples
            mean = _per_pair(pairs, lambda eye, var: np.nanmean(obj.values(eye, var, policy)))
        elif isinstance(mean, numbers.Real):
            mean = _per_pair(pairs, lambda eye, var, val=mean: val)
        elif not isinstance(mean, dict):
            raise ValueError("mean must be None, a number or dict")

        if sd is None:
            sd = _per_pair(pairs, lambda eye, var: np.nanstd(obj.values(eye, var, policy)))
        elif isinstance(sd, numbers.Real):
            sd = _per_pair(pairs, lambda eye, var, val=sd: val)
        elif not isinstance(sd, dict):
            raise ValueError("sd must be None, a number or dict")
        logger.debug("Mean: %s" % _plain_values(mean))
        logger.debug("SD: %s" % _plain_values(sd))

        # check that all eye/var combinations are present in mean/sd
        _check_per_pair(pairs, mean, sd)

        # store scaling parameters (per channel; combined with those of an
        # earlier scale() of the same channel, so that unscale() undoes both)
        stored = obj.params.get("scale") or {}
        smean, ssd = dict(stored.get("mean", {})), dict(stored.get("sd", {}))
        for eye, var in pairs:
            key = f"{eye}_{var}"
            m0, s0 = smean.get(key, 0.), ssd.get(key, 1.)
            smean[key] = m0 + mean[key] * s0
            ssd[key] = s0 * sd[key]
        obj.params["scale"] = {"mean": smean, "sd": ssd}

        for eye, var in pairs:
            # scale the masked samples too, keep the layers
            key = f"{eye}_{var}"
            obj.data.set_with_mask(key, (obj.data[key] - mean[key]) / sd[key],
                                   preserve_mask=True)
        return obj
        
    @step
    @keephistory    
    def unscale(self, variables=[], mean=None, sd=None, eyes=[], inplace=None):
        """
        Scale back to original values using either values provided as arguments
        or the values stored in `params["scale"]` by :meth:`scale`.

        The signal is multiplied by `sd` and `mean` is added. The stored
        parameters of the unscaled channels are removed.

        Parameters
        ----------
        variables : str or list of str, optional
            variables to unscale; can be "pupil", "x","y", "baseline", "response" or any other variable
            that is available in the dataset; either a string or a list of strings;
            available variables can be checked with `obj.variables`; empty list means all variables
        mean : None, float or dict, optional
            mean to add to the signal; a number (same for all eyes and
            variables) or ``{"left_pupil": value, ...}``;
            if `None`, use the mean stored by :meth:`scale`
        sd : None, float or dict, optional
            sd to multiply the signal with; a number (same for all eyes and
            variables) or ``{"left_pupil": value, ...}``;
            if `None`, use the sd stored by :meth:`scale`
        eyes : str or list of str, optional
            eye(s) to consider; if empty, all available eyes
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the unscaled object (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        pairs = obj.data.select(eyes, variables)

        # if no parameters are provided, use the stored ones (normal)
        if mean is None:
            mean=obj.params["scale"]["mean"]
        elif isinstance(mean, numbers.Real):
            mean = _per_pair(pairs, lambda eye, var, val=mean: val)
        elif not isinstance(mean, dict):
            raise ValueError("mean must be None, a number or dict")
        if sd is None:
            sd=obj.params["scale"]["sd"]
        elif isinstance(sd, numbers.Real):
            sd = _per_pair(pairs, lambda eye, var, val=sd: val)
        elif not isinstance(sd, dict):
            raise ValueError("sd must be None, a number or dict")
        
        # check whether unscaling parameters are provided
        _check_per_pair(pairs, mean, sd)

        for eye, var in pairs:
            # unscale the masked samples too, keep the layers
            key = f"{eye}_{var}"
            obj.data.set_with_mask(key, obj.data[key] * sd[key] + mean[key],
                                   preserve_mask=True)
        # the unscaled channels have no scaling parameters any more
        stored = obj.params.get("scale")
        if stored:
            keys = {f"{eye}_{var}" for eye, var in pairs}
            obj.params["scale"] = {stat: {k: v for k, v in stored.get(stat, {}).items()
                                          if k not in keys}
                                   for stat in ("mean", "sd")}
        return obj    

    
    @step
    @keephistory
    def downsample(self, fsd: float, dsfac: bool=False, policy="default", inplace=None):
        """
        Simple downsampling scheme using mean within the downsampling window.

        All data fields and the time-vector are downsampled simultaneously.
        See :func:`pypillometry.signal.baseline.downsample`. The samples
        that are unusable under the mask `policy` are left out of the window
        means (by default, interpolated samples are used, see :meth:`mask`).
        A layer is kept for a new sample if it covers more than half of its
        window. A new sample whose window has no usable sample holds the
        mean of the stored values (NaN only if they are NaN) and stays
        unusable.

        The new sampling rate is ``fs/factor``, the rate of the decimated
        data. It differs from `fsd` if `fs/fsd` is not an integer
        (e.g., 1000 Hz with ``fsd=300`` gives factor 3 and 333.3 Hz).

        Parameters
        ----------
        fsd : float
            new sampling-rate (in Hz) or decimate-factor
        dsfac : bool, optional
            if False (default), `fsd` is the requested sampling rate (the
            decimate factor is ``int(fs/fsd)``; a warning is logged if
            `fs/fsd` is not an integer);
            if True, `fsd` is the decimate factor (a positive integer)
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are used for the window
            means (see :meth:`mask`); default ``"default"``
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the downsampled object (`self` or a copy, depending on `inplace`)

        Raises
        ------
        TypeError
            if `fsd` is not a number
        ValueError
            if the sampling rate `fsd` is not above 0 and at most `fs`, or
            the decimate factor is not a positive integer
        """
        dsfac=_decimate_factor(self.fs, fsd, dsfac)
        obj = self._get_inplace(inplace)
        obj._decimate(dsfac, policy)
        return obj

    def _decimate(self, factor, policy="default"):
        """
        Decimate the samples by the integer `factor` (in place).

        Each window of `factor` samples becomes one sample (the last window
        may be shorter): values are the mean of the samples that are usable
        under the mask `policy`, the time is the mean time of the window,
        and the new sampling rate is ``fs/factor``. A layer (e.g.,
        ``"blink"``) is set for a new sample if it covers more than half of
        the samples in the window (also the recording-wide layers, e.g.
        excluded periods). A new sample whose window has no usable sample
        holds the mean of the stored values and stays unusable: if the
        layers that cover more than half of the window do not make it
        unusable, it gets the layers of its samples that make them unusable
        under `policy`.
        Events stay valid.
        """
        self._map_sample_aligned(_decimate_dict, factor, policy)
        self._set_timebase(baseline.downsample(self.tx, factor), self.fs/factor)
        return self

    @step
    @keephistory
    def merge_eyes(self, eyes=[], variables=[], method="mean", keep_eyes=True,
                   policy="default", name="mean", inplace=None):
        """Merge data from both eyes into a single variable.

        The merged data is stored under the eye name `name` (default
        "mean", e.g., ``obj["mean", "pupil"]``). Each sample is the mean over the eyes
        in which it is usable under the mask `policy` (see :meth:`mask`),
        where recorded samples come first: an interpolated sample is used
        only if no eye has a recorded one there. A sample is masked in the
        merged data only if it is unusable in all merged eyes (it then holds
        the mean of the stored values; NaN only if they are NaN). The merged
        data get the layers that are set in all eyes that went into a sample
        (e.g., ``"interpolated"`` where all of them were interpolated, or
        ``"blink"`` where all eyes are masked by a blink); masked samples
        without such a common layer get the ``"user"`` layer. Each variable
        is merged over the selected eyes that have data for it.

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
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are usable (see
            :meth:`mask`); default ``"default"``
        name : str, optional
            name of the new eye; default ``"mean"`` (as in
            :meth:`d.pupil.merge_eyes() <pypillometry.eyedata.pupil.PupilNamespace.merge_eyes>`)
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object with merged data (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        pairs = obj.data.select(eyes, variables)

        if method=="mean":            
            for var in dict.fromkeys(v for _, v in pairs):
                # stored values of all eyes with this variable and the eyes
                # that go into each merged sample
                eyes = [eye for eye, v in pairs if v == var]
                keys = [f"{eye}_{var}" for eye in eyes]
                stored = np.stack([obj.data[k] for k in keys])
                used = _merge_usage(obj.data, keys, policy)

                # Mean over the used eyes; masked only where no eye is usable
                meanval = np.ma.mean(np.ma.array(stored, mask=~used), axis=0)
                allmasked = np.ma.getmaskarray(meanval)
                # where all eyes are unusable, keep the mean of the stored
                # values, still masked
                rawmean = np.ma.mean(np.ma.masked_invalid(stored), axis=0).filled(np.nan)
                values = np.where(allmasked, rawmean, np.ma.getdata(meanval))
                _store_merged(obj.data, f"{name}_{var}", values, keys, used, allmasked, policy)
                
                if not keep_eyes:
                    for eye in eyes:
                        logger.debug("Dropping eye %s for variable %s" % (eye, var))
                        del obj.data[eye+"_"+var]
        else:
            raise ValueError("Method %s not implemented" % method)

        return obj
        
    def stat_per_event(self, 
                       intervals=None,
                       window: Tuple[float,float]=None, event_select=None,
                       eyes: str|list=[], variables: str|list=[],                       
                       statfct: Callable=np.mean, units: str="ms",
                       policy="default", **kwargs):
        """
        Return result of applying a statistical function to data in a
        given interval relative to event-onsets. For example, extract mean 
        pupil-size in interval before trial onset.

        Parameters
        ----------
        intervals : Intervals or Epochs, optional
            Intervals object (in any time units or in sample indices, e.g.,
            from :meth:`get_intervals`) or :class:`~pypillometry.Epochs`
            (e.g., from :meth:`get_epochs`). If provided, window and
            event_select are ignored.
        window : tuple (min,max) or Quantity, optional
            time-window relative to event-onset (0 is event-onset), e.g.
            ``(-200, 0) * u.ms``; plain numbers are in units given by `units`.
            Required if intervals is not provided.
        event_select : str, function or Events, optional
            the events to select (see :meth:`get_epochs`); required if
            intervals is not provided
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
            units for plain numbers in `window` (if used); default "ms"
        policy : str, MaskPolicy or list of str
            mask policy: the samples that are unusable under it are masked
            (see :meth:`mask`); default ``"default"`` (interpolated samples
            are used)
        **kwargs
            passed onto the event_select function

        Returns
        -------
        dict
            one array per eye/variable (keys like "left_pupil"), with one
            value per interval/event
        """
        from ..intervals import stat_event_interval
        
        pairs = self.data.select(eyes, variables)
        
        # Accept either Intervals object or old-style parameters
        if isinstance(intervals, str):
            raise TypeError(f"intervals must be Intervals or Epochs, got the string "
                            f"{intervals!r}; to select events by their label, use "
                            f"event_select={intervals!r} (and window=...)")
        if intervals is not None:
            intervs = _as_intervals(intervals)
        else:
            if event_select is None:
                raise ValueError("Must provide either 'intervals' or 'event_select' "
                                 "(and 'window')")
            if window is None:
                raise ValueError("give a window around the events, e.g., "
                                 "window=(-200, 0) * u.ms")
            intervs = _as_intervals(self.get_epochs(event_select, window, units=units,
                                                    **kwargs))

        # bounds in sample indices, matched against the sample numbers
        tx = np.arange(len(self.tx))
        intervals_list = intervs.to_units("samples", self).intervals

        _warn_if_ignores_mask(statfct, "statfct")
        stat={}
        for eye,var in pairs:
            # masked array, so that the unusable samples are ignored
            key = f"{eye}_{var}"
            x = np.ma.masked_array(self.data[key], self.data.get_policy_mask(key, policy))
            stat[key]=stat_event_interval(tx, x, intervals_list, statfct)

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
        other_size += self._events.onsets.nbytes + self._events.labels.nbytes
        other_size += sum(col.nbytes for col in self._events.meta.values())
        
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

    @step
    @keephistory
    def merge_masks(self, inplace=None):
        """Merge masks of all variables into a single joint mask.
        
        This function creates a joint mask by taking the logical OR of all individual masks
        (under the default mask policy, see `get_mask()`), then applies this joint mask to
        all variables (the newly masked samples get the ``"user"`` layer).
        
        Parameters
        ----------
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object with merged masks (`self` or a copy, depending on `inplace`)
        """
        obj = self._get_inplace(inplace)
        
        # Get joint mask across all variables
        joint_mask = obj.data.get_mask()
        
        # Apply joint mask to all variables
        for key in obj.data.keys():
            obj.data.set_mask(key, joint_mask)
            
        return obj

    @step
    @keephistory
    def mask_intervals(self, intervals, eyes=[], variables=[], inplace=None):
        """Apply Intervals as masks to selected data.
        
        Takes detected intervals (e.g., blinks, artifacts) and applies them
        as masks to specified eyes and variables: they are added to the
        ``"user"`` layer (existing masks are kept).

        Parameters
        ----------
        intervals : Intervals, Epochs, list of tuple or dict
            Either:

            - A single Intervals (or :class:`~pypillometry.Epochs`) object
              or a list of ``(start, end)`` tuples (in ms, or with units):
              applied to all specified eyes/variables
            - A dict of Intervals (or Epochs): keys like "left_pupil",
              "right_x", etc. When dict is provided, `eyes` and `variables`
              are ignored.
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
        :class:`~pypillometry.EyeData`
            The object with masks applied (`self` or a copy, depending on `inplace`).
            
        Examples
        --------
        Apply blinks from a single Intervals object to left eye pupil data:
        
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short").pupil.blinks_detect()
        >>> blinks = d.get_blinks()
        >>> d = d.mask_intervals(blinks["left_pupil"], eyes=["left"], variables=["pupil"])

        Apply a dict of intervals (one per eye/variable):

        >>> d = d.mask_intervals(blinks)  # eyes and variables ignored

        Chain with other operations:

        >>> d = d.mask_intervals(blinks).pupil.lowpass_filter(cutoff=5)
        
        See Also
        --------
        pypillometry.eyedata.pupil.PupilNamespace.blinks_detect : Detect blinks
        get_blinks : Retrieve stored blink intervals
        """
        obj = self._get_inplace(inplace)

        if isinstance(intervals, dict):
            # one Intervals object per eye_variable
            per_key = intervals
        else:
            intervals = obj._as_spans(intervals)
            per_key = {f"{eye}_{var}": intervals for eye, var in obj.data.select(eyes, variables)}

        n = len(obj.tx)
        for key, interval_obj in per_key.items():
            interval_obj = _as_intervals(interval_obj, f"intervals for '{key}'")
            if key not in obj.data.data:
                logger.warning(f"Key '{key}' not found in data, skipping")
                continue
            mask = np.zeros(n, dtype=bool)
            for start, end in interval_obj.to_units("samples", obj).intervals:
                mask[min(max(int(start), 0), n):min(max(int(end), 0), n)] = True
            obj.data.set_layer(key, "user", mask)
        
        return obj

    @step
    @keephistory
    def interpolate_intervals(self, intervals, eyes=[], variables=[], 
                             method="linear", margin=(-20, 20),
                             store_as_suffix=None, policy="default", inplace=None):
        """
        Interpolate data within specified intervals.
        
        This function replaces data within the specified intervals with interpolated
        values based on boundary points outside each interval. The
        interpolated samples get the ``"interpolated"`` layer (see
        :meth:`get_interpolated_blinks`), so they are used under the default
        mask policy and masked under the ``"strict"`` one (see :meth:`mask`).
        Intervals without usable boundary points on both sides are skipped
        with a warning.

        Parameters
        ----------
        intervals : Intervals, Epochs, list of tuple or dict
            Either:

            - A single Intervals (or :class:`~pypillometry.Epochs`) object
              or a list of ``(start, end)`` tuples (in ms, or with units):
              applied to all specified eyes/variables
            - A dict of Intervals (or Epochs): keys like "left_pupil",
              "right_x", etc. When dict is provided, `eyes` and `variables`
              are ignored.
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
            <= 0 and `margin[1]` >= 0. Samples that are unusable under
            `policy` (masked or NaN) are not used.
            For higher-order methods like "cubic" or "quadratic", more
            boundary points from this margin are used to fit the
            interpolation curve.
        store_as_suffix : str or None, default None
            If None, replace the original data.
            If a string, store interpolated data in a new variable with this suffix
            (e.g., "_interp" creates "left_pupil_interp").
        policy : str, MaskPolicy or list of str
            mask policy that decides which boundary points are usable (see
            :meth:`mask`); default ``"default"``
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            The object with interpolated data (`self` or a copy, depending on `inplace`).
            
        Examples
        --------
        Interpolate blinks for left eye pupil data:
        
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short").pupil.blinks_detect()
        >>> blinks = d.get_blinks(eyes="left", variables="pupil")["left_pupil"]
        >>> d1 = d.interpolate_intervals(blinks, eyes=["left"], variables=["pupil"],
        ...                              inplace=False)

        Interpolate using a dict of intervals:

        >>> blinks = d.get_blinks()  # returns dict
        >>> d2 = d.interpolate_intervals(blinks, inplace=False)

        Store interpolated data in a new variable:

        >>> d3 = d.interpolate_intervals(blinks, store_as_suffix="_interp",
        ...                              inplace=False)
        >>> "left_pupil_interp" in d3.data
        True

        Use cubic interpolation with wider margin:

        >>> d4 = d.interpolate_intervals(blinks, method="cubic", margin=(-50, 50),
        ...                              inplace=False)
        
        See Also
        --------
        mask_intervals : Apply intervals as masks without interpolation
        get_blinks : Retrieve stored blink intervals
        """
        from scipy.interpolate import interp1d
        
        obj = self._get_inplace(inplace)
        
        margin = parse_time(margin)
        if margin[0] > 0 or margin[1] < 0:
            raise ValueError(f"margin must be (<=0, >=0) in ms, got {tuple(margin)}")

        # Convert margin from ms to samples
        margin_left_ix, margin_right_ix = obj._samples((-margin[0], margin[1]))
        
        def interpolate_single_key(key, interval_obj):
            """Local helper to interpolate a single data key."""
            # Convert to index-based intervals
            intervals_array = np.array(interval_obj.to_units("samples", obj)).astype(int)
            
            # Get the data and the unusable samples
            data = obj.data[key].copy()
            mask = obj.data.get_policy_mask(key, policy)
            interpolated = np.zeros(len(data), dtype=bool)
            
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
                # `mask` also covers the intervals interpolated in this call
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
                interpolated[start_idx:end_idx] = True
            
            # Store result (with the masks and layers of `key`)
            target_key = key if store_as_suffix is None else f"{key}{store_as_suffix}"
            if target_key == key:
                obj.data.set_with_mask(key, data, preserve_mask=True)
            else:
                obj.data._set_channel(target_key, data, obj.data.layers[key])
            obj.data.set_layer(target_key, "interpolated", interpolated)
        
        # Handle dict vs single Intervals
        if isinstance(intervals, dict):
            for key, interval_obj in intervals.items():
                interval_obj = _as_intervals(interval_obj, f"intervals for '{key}'")
                if key not in obj.data.data:
                    logger.warning(f"Key '{key}' not found in data, skipping")
                    continue
                
                interpolate_single_key(key, interval_obj)
        else:
            intervals = obj._as_spans(intervals)
            for eye, var in obj.data.select(eyes, variables):
                interpolate_single_key(f"{eye}_{var}", intervals)
        
        return obj

    # --- excluded periods ------------------------------------------------

    def _excluded_mask(self) -> np.ndarray:
        """Boolean array that is True for the samples in an excluded period."""
        return self.data.get_rec_layer("excluded")

    @step
    @keephistory
    def exclude_intervals(self, intervals, inplace=None):
        """
        Exclude periods that are not part of the analysis (e.g., breaks or
        the calibration at the start).

        The periods are stored with the object as the recording-wide
        ``"excluded"`` layer (see :meth:`get_excluded`), which masks all
        variables (also those added later), so that functions that
        ignore masked data (e.g., :meth:`scale`, :meth:`get_segments`) also
        ignore the excluded periods. Unlike other masked data (blinks,
        artifacts), excluded periods are not treated as missing data: blink
        detection cuts blinks at them, and blinks that end in an excluded
        period are not interpolated. Blinks detected before keep their
        layer.

        The layer moves with the samples (:meth:`crop`, :meth:`downsample`,
        :meth:`remove_intervals`, ...). Calling the method again adds to
        the excluded periods; ``clear_layer("excluded")`` removes them.

        Parameters
        ----------
        intervals : Intervals, Epochs or list of tuple
            the periods to exclude: an :class:`~pypillometry.Intervals`
            object (e.g., from :meth:`get_intervals`; in time units or sample
            indices), :class:`~pypillometry.Epochs` or a list of ``(start, end)`` tuples (in ms, or with
            units, e.g., ``[(0, 2 * u.min)]``; ``"<start>"`` and ``"<end>"``
            stand for the start and end of the recording, e.g.,
            ``[(10 * u.min, "<end>")]``); ends are exclusive
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object with the excluded periods (`self` or a copy,
            depending on `inplace`)

        Examples
        --------
        >>> breaks = d.get_intervals(start="break_start", end="break_end", units="ms")  # doctest: +SKIP
        >>> d = d.exclude_intervals(breaks)  # doctest: +SKIP
        >>> d = d.exclude_intervals([(0, 2 * u.min)])  # doctest: +SKIP

        See Also
        --------
        keep_intervals : exclude everything outside the given periods
        """
        obj = self._get_inplace(inplace)
        intervals = obj._as_spans(intervals)
        new = intervals.to_units("ms", obj)
        new = [(float(s), float(e)) for s, e in new.intervals if e > s]
        obj.data.set_rec_layer("excluded", Intervals(new, units="ms").to_mask(obj))
        return obj

    @step
    @keephistory
    def keep_intervals(self, intervals, inplace=None):
        """
        Exclude everything outside the given periods (e.g., keep only the
        task blocks or the fMRI run).

        Same as ``exclude_intervals(intervals.complement(d))``: the time
        before, between and after the periods is excluded (see
        :meth:`exclude_intervals`). Periods that are already excluded stay
        excluded, also if they are within `intervals`.

        Parameters
        ----------
        intervals : Intervals, Epochs or list of tuple
            the periods to keep: an :class:`~pypillometry.Intervals`
            object (e.g., from :meth:`get_intervals`; in time units or sample
            indices), :class:`~pypillometry.Epochs` or a list of ``(start, end)`` tuples (in ms, or with
            units; ``"<start>"`` and ``"<end>"`` stand for the start and end
            of the recording); ends are exclusive
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object with the excluded periods (`self` or a copy,
            depending on `inplace`)

        Examples
        --------
        >>> run = d.get_intervals(start="rsfMRI_ONSET", end="rsfMRI_END", units="ms")  # doctest: +SKIP
        >>> d = d.keep_intervals(run)  # doctest: +SKIP
        """
        intervals = self._as_spans(intervals)
        return self.exclude_intervals(intervals.complement(self), inplace=inplace)

    def get_excluded(self, units: str = "ms") -> Intervals:
        """
        Return the excluded periods (see :meth:`exclude_intervals`).

        The periods are the runs of samples of the ``"excluded"`` layer
        (from the time of their first sample to the time of the sample
        after the last one).

        Parameters
        ----------
        units : str, optional
            units of the returned intervals ("ms" (default), "sec", "min",
            "h", or "samples" for sample indices)

        Returns
        -------
        :class:`~pypillometry.Intervals`
            the excluded periods (empty if there are none)
        """
        return Intervals.from_mask(self._excluded_mask(), self, units, label="excluded")

    def get_included(self, units: str = "ms") -> Intervals:
        """
        Return the periods that are not excluded (see :meth:`exclude_intervals`).

        These are the contiguous runs of samples outside the excluded
        periods (the whole recording if nothing is excluded).

        Parameters
        ----------
        units : str, optional
            units of the returned intervals ("ms" (default), "sec", "min",
            "h", or "samples" for sample indices)

        Returns
        -------
        :class:`~pypillometry.Intervals`
            the included periods (end-exclusive)
        """
        return Intervals.from_mask(~self._excluded_mask(), self, units, label="included")

    def split_excluded(self) -> list:
        """
        Split the data at the excluded periods (see :meth:`exclude_intervals`).

        Returns one object per included period (see :meth:`get_included`),
        each with the samples, masks, blinks and events of that period and
        the original time vector. The objects are copies; `self` is not
        changed.

        Returns
        -------
        list of :class:`~pypillometry.EyeData`
            the parts of the recording between the excluded periods

        See Also
        --------
        remove_intervals : remove periods and join the rest, e.g.,
            ``d.remove_intervals(d.get_excluded())``
        """
        parts = []
        for start, end in self.get_included("samples").intervals:
            # crop includes both ends: from the first to the last sample
            parts.append(self.crop(self.tx[start], self.tx[end - 1], units="ms",
                                   inplace=False))
        return parts

    @step
    @keephistory
    def remove_intervals(self, intervals, inplace=None):
        """
        Remove the data in the given periods (e.g., breaks) and join the rest.

        The samples in the periods are deleted. Gaps between the remaining
        data are closed: the data after a removed period are shifted back in
        time, so that they continue where the removed period started (the
        time vector has no gap). Removing data at the beginning or end does
        not change the times, so removing everything before `start` and
        after `end` is the same as :meth:`crop`. Event onsets are shifted
        with the data; events within removed periods are deleted. Stored
        blinks and excluded periods are moved to the new positions. The
        removed periods are added to ``params["removed_intervals"]``, in
        the original time (the time before any periods were removed; a
        period across a gap closed by an earlier call is split at the gap).
        ``params["removed_shifts"]`` holds the shifts between the current
        and the original time.

        To remove the periods marked with :meth:`exclude_intervals`, use
        ``d.remove_intervals(d.get_excluded())``.

        Filters or interpolation applied after joining treat the data on
        both sides of a removed period as neighbors; apply them before, or
        use :meth:`split_excluded` to process the parts separately.

        Parameters
        ----------
        intervals : Intervals, Epochs or list of tuple
            the periods to remove: an :class:`~pypillometry.Intervals`
            object (in time units or sample indices),
            :class:`~pypillometry.Epochs` or a list of
            ``(start, end)`` tuples (in ms, or with units, e.g.,
            ``[(10 * u.min, 12 * u.min)]``; ``"<start>"`` and ``"<end>"``
            stand for the start and end of the recording); ends are exclusive
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor)

        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object without the removed periods (`self` or a copy,
            depending on `inplace`)

        Raises
        ------
        ValueError
            if all samples would be removed

        See Also
        --------
        crop : keep only the data between two times
        split_excluded : split the data at the excluded periods

        Examples
        --------
        >>> breaks = d.get_intervals(start="break_start", end="break_end", units="ms")  # doctest: +SKIP
        >>> d = d.remove_intervals(breaks)  # doctest: +SKIP
        >>> d = d.remove_intervals(d.get_excluded())  # doctest: +SKIP
        """
        obj = self._get_inplace(inplace)
        intervals = obj._as_spans(intervals)
        n = len(obj.tx)
        tx = obj.tx
        tend = tx[-1] + 1000.0 / obj.fs
        # the periods in ms, within the data (also handles infinite bounds)
        periods = [(max(float(a), tx[0]), min(float(b), tend))
                   for a, b in intervals.to_units("ms", obj).intervals]
        periods = Intervals([p for p in periods if p[1] > p[0]], "ms").merge()
        removed = periods.to_mask(obj).astype(bool)
        if removed.all():
            raise ValueError("all samples would be removed")
        if not removed.any():
            return obj

        # close the gaps between kept data: the samples after a removed run
        # of samples (s, e) are shifted back by tx[e] - tx[s]
        runs = Intervals.from_mask(removed).intervals
        gaps = [(s, e) for s, e in runs if s > 0 and e < n]

        def time_map(t):
            # times after the start of a removed run move back with the data
            t_shift = np.zeros(len(t))
            for s, e in gaps:
                t_shift[t >= tx[s]] += tx[e] - tx[s]
            return t - t_shift

        on = obj._events.onsets
        in_removed = np.zeros(len(on), dtype=bool)
        for a, b in periods.intervals:
            in_removed |= (on >= a) & (on < b)
        removed_periods = obj.params.get("removed_intervals", []) + \
            obj._periods_to_original_time(periods.intervals)
        obj._keep_samples(~removed, time_map=time_map, keep_events=~in_removed)
        obj.params["removed_intervals"] = [
            (float(a), float(b)) for a, b in Intervals(removed_periods, "ms").merge().intervals]
        return obj

    def excluded_duration(self, units: str = "ms") -> float:
        """
        Total duration of the excluded periods within the data.

        Parameters
        ----------
        units : str, optional
            units of the result ("ms" (default), "sec", "min", "h")

        Returns
        -------
        float
            the excluded time (number of excluded samples / sampling rate)
        """
        return self._excluded_mask().sum() / self.fs * 1000 * self._unit_fac(units)
