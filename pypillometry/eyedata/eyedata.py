from .generic import GenericEyeData, keephistory
from .gaze import GazeNamespace
from .eyedatadict import EyeDataDict
from ..plot import PlotNamespace
import numpy as np
from loguru import logger

from .pupil import PupilNamespace
from .namespace import accessor
import numpy as np
from collections.abc import Iterable
import copy
from ..convenience import _plain_values
from typing import Optional, Dict, Union
from .spatial_calibration import SpatialCalibration
from .experimental_setup import ExperimentalSetup
from ..events import Events



class EyeData(GenericEyeData):
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
    such data. The generic methods (e.g.,
    :meth:`~pypillometry.GenericEyeData.crop`) are methods of the object.
    All transforming methods can also be used as pipeline steps, e.g.
    ``d | pp.pupil.blinks_detect() | pp.crop(0, "10 min")`` (see
    :class:`~pypillometry.Pipeline`).

    Properties such as :attr:`~pypillometry.GenericEyeData.eyes` and
    :attr:`~pypillometry.GenericEyeData.variables` are inherited from
    :class:`~pypillometry.GenericEyeData`.

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
    >>> from pypillometry.eyedata import EyeData
    >>> d=EyeData(time=np.arange(0, 1000, 1),
    ...         left_x=np.random.randn(1000),
    ...         left_y=np.random.randn(1000))
    >>> print(d)
    """
    pupil = accessor(PupilNamespace)
    gaze = accessor(GazeNamespace)
    plot = accessor(PlotNamespace)

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
