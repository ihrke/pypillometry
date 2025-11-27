from loguru import logger
from .generic import GenericEyeData, keephistory
from .eyedatadict import EyeDataDict
from ..plot import GazePlotter
from ..intervals import Intervals
import numpy as np
import json
from typing import Optional, Dict
from .spatial_calibration import SpatialCalibration
from .experimental_setup import ExperimentalSetup


class GazeData(GenericEyeData):
    """
    Class representing Eye-Tracking data (x,y) from one or two eyes.

    If you also want to store pupil-size, use the `EyeData` class.
    If you only want to store pupil-size, use the `PupilData` class.

    Parameters
    ----------
    time: 
        timing array or `None`, in which case the time-array goes from [0,maxT]
        using `sampling_rate` (in ms)
    left_x, left_y:
        data from left eye (at least one of the eyes must be provided, both x and y)
    right_x, right_y:
        data from right eye (at least one of the eyes must be provided, both x and y)
    sampling_rate: float
        sampling-rate of the signal in Hz; if None, 
    experimental_setup: ExperimentalSetup, optional
        Geometric model of the experimental setup including screen geometry,
        eye position, camera position, and screen orientation.
    name: 
        name of the dataset or `None` (in which case a random string is selected)
    event_onsets: 
        time-onsets of any events in the data (in ms, matched in `time` vector)
    event_labels:
        for each event in `event_onsets`, provide a label
    calibration: dict, optional
        Dictionary of SpatialCalibration objects, one per eye (keys: 'left', 'right')
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
                    left_x: np.ndarray = None,
                    left_y: np.ndarray = None,
                    right_x: np.ndarray = None,
                    right_y: np.ndarray = None,
                    event_onsets: np.ndarray = None,
                    event_labels: np.ndarray = None,
                    sampling_rate: float = None,
                    experimental_setup: Optional[ExperimentalSetup] = None,
                    name: str = None,
                    calibration: Optional[Dict[str, SpatialCalibration]] = None,
                    fill_time_discontinuities: bool = True,
                    keep_orig: bool = False,
                    info: dict = None,
                    inplace: bool = False):
        """Constructor for the GazeData class.
        """
        if (left_x is None and left_y is None and right_x is None and right_y is None):
            raise ValueError("At least one eye-trace (both x and y coordinates) must be provided")
            
        # Initialize data dictionary
        self.data=EyeDataDict(left_x=left_x, left_y=left_y, right_x=right_x, right_y=right_y)
        
        self._init_common(time, sampling_rate, 
                          event_onsets, event_labels, 
                          name, fill_time_discontinuities, 
                          info=info, inplace=inplace)
        
        # Experimental setup (geometry)
        self.experimental_setup = experimental_setup
        
        # Spatial calibration data
        self.calibration = calibration

        self.original=None
        if keep_orig: 
            self.original=self.copy()

    @property
    def plot(self):
        return GazePlotter(self)

    def summary(self):
        """
        Return a summary of the dataset as a dictionary.

        Returns
        -------
        dict
            dictionary containing description of dataset
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
            n=len(self.data),
            sampling_rate=self.fs,
            data=list(self.data.keys()),
            nevents=self.nevents(), 
            experimental_setup=setup_info,
            calibration=calibration_info,
            duration_minutes=self.get_duration("min"),
            start_min=self.tx.min()/1000./60.,
            end_min=self.tx.max()/1000./60.,
            parameters=self._strfy_params(),
            glimpse=repr(self.data)
        )
        
        return summary            
    
    def set_experimental_setup(self, **kwargs):
        """
        Set or update the experimental setup geometry.
        
        Creates a new ExperimentalSetup or updates the existing one with
        the provided parameters. All parameters are passed through to
        ExperimentalSetup constructor.

        Parameters
        ----------
        **kwargs : 
            Any parameters accepted by ExperimentalSetup:
            - screen_resolution: tuple (width, height) in pixels
            - physical_screen_size: tuple (width, height) with units
            - eye_to_screen_perpendicular: distance (d)
            - eye_offset: tuple (delta_x, delta_y)
            - eye_to_screen_center: alternative distance specification
            - screen_pitch: tilt angle (alpha_tilt)
            - screen_yaw: tilt angle (beta_tilt)
            - camera_position_relative_to: "screen" or "eye"
            - camera_offset: tuple (x, y, z)
            - camera_spherical: tuple (theta, phi, r)
            - ipd: inter-pupillary distance
        
        Examples
        --------
        >>> data.set_experimental_setup(
        ...     screen_resolution=(1920, 1080),
        ...     physical_screen_size=("52 cm", "29 cm"),
        ...     eye_to_screen_perpendicular="65 cm",
        ... )
        """
        if self.experimental_setup is None:
            # Create new setup
            self.experimental_setup = ExperimentalSetup(**kwargs)
        else:
            # Merge with existing setup
            old = self.experimental_setup.to_dict()
            
            # Map old dict keys to ExperimentalSetup constructor params
            merged = {
                'screen_resolution': kwargs.get('screen_resolution', old.get('screen_resolution')),
                'physical_screen_size': kwargs.get('physical_screen_size', old.get('physical_screen_size')),
                'eye_to_screen_perpendicular': kwargs.get('eye_to_screen_perpendicular', old.get('d')),
                'eye_offset': kwargs.get('eye_offset', old.get('eye_offset')),
                'eye_to_screen_center': kwargs.get('eye_to_screen_center'),
                'screen_pitch': kwargs.get('screen_pitch', old.get('alpha_tilt')),
                'screen_yaw': kwargs.get('screen_yaw', old.get('beta_tilt')),
                'camera_position_relative_to': kwargs.get('camera_position_relative_to', old.get('camera_position_relative_to')),
                'camera_offset': kwargs.get('camera_offset', old.get('camera_offset')),
                'camera_spherical': kwargs.get('camera_spherical', old.get('camera_spherical')),
                'ipd': kwargs.get('ipd', old.get('ipd')),
            }
            
            self.experimental_setup = ExperimentalSetup(**merged)

    # =========================================================================
    # Methods
    # =========================================================================

    @keephistory
    def mask_eye_divergences(self, threshold: float = .99, thr_type: str = "percentile", 
                           store_as: str|None = None, apply_mask: bool = True, inplace=None):
        """
        Calculate Euclidean distance between left and right eye coordinates and detect divergences.
        
        This method computes the distance between corresponding (x,y) coordinates in the left
        and right eyes. Points where the distance exceeds a threshold are detected as divergences.
        When apply_mask=True, these points are masked. When apply_mask=False, the divergences
        are returned as Intervals objects.
        
        Parameters
        ----------
        threshold : float, default=.99
            If thr_type is "percentile": percentile value (0-1) for threshold calculation.
            If thr_type is "pixel": maximum allowed distance in pixels.
        thr_type : str, default="percentile"
            Type of threshold to apply:
            - "percentile": threshold is calculated as the given percentile of the distance distribution
            - "pixel": threshold is the maximum allowed distance in pixels
        store_as : str or None, default=None
            Variable name to store the distance timeseries. If None, distance is not stored.
        apply_mask : bool, default=True
            If True, apply detected divergences as masks to the data and return self.
            If False, return detected divergences as dict of Intervals objects.
        inplace : bool or None
            If True, make change in-place and return the object.
            If False, make and return copy before making changes.
            If None, use the object-level setting.
        
        Returns
        -------
        GazeData or Intervals
            If apply_mask=True: returns self for chaining.
            If apply_mask=False: returns Intervals object containing detected divergence intervals.
        
        Raises
        ------
        ValueError
            If both left and right eye data are not available.
            If thr_type is not "percentile" or "pixel".
        
        Examples
        --------
        >>> # Mask divergences at 99th percentile
        >>> gaze_data = gaze_data.mask_eye_divergences(threshold=0.99, thr_type="percentile")
        
        >>> # Detect divergences without masking
        >>> divergences = gaze_data.mask_eye_divergences(threshold=0.99, apply_mask=False)
        
        >>> # Mask divergences beyond 100 pixels
        >>> gaze_data = gaze_data.mask_eye_divergences(threshold=100, thr_type="pixel")
        """
        obj = self._get_inplace(inplace)
        
        # Check that both eyes are available
        if 'left' not in obj.eyes or 'right' not in obj.eyes:
            raise ValueError(
                "Both left and right eye data are required for mask_eye_divergences. "
                f"Available eyes: {obj.eyes}"
            )
        
        # Validate thr_type
        if thr_type not in ["percentile", "pixel"]:
            raise ValueError(
                f"thr_type must be 'percentile' or 'pixel', got '{thr_type}'"
            )
        
        # Calculate Euclidean distance
        dist = np.ma.sqrt((obj['left_x'] - obj['right_x'])**2 + (obj['left_y'] - obj['right_y'])**2)
        
        # Calculate threshold
        if thr_type == "percentile":
            # threshold should be between 0 and 1 for percentile
            if not 0 <= threshold <= 1:
                raise ValueError(
                    f"For thr_type='percentile', threshold must be between 0 and 1, got {threshold}"
                )
            thr = np.percentile(dist.compressed(), threshold * 100)
        else:  # thr_type == "pixel"
            thr = threshold
        
        logger.debug(f"Detecting divergent points with threshold {thr}")
        
        # Create a mask for divergent points (where distance exceeds threshold)
        divergence_mask = (dist > thr).filled(False)
        
        # Convert mask to intervals
        intervals_list = obj._mask_to_intervals_list(divergence_mask)
        
        # Store distance as a timeseries if requested
        if store_as is not None:
            dist_copy = dist.copy()
            dist_copy.mask |= divergence_mask
            obj[store_as] = dist_copy
        
        # Create Intervals object for the detected divergences
        intervals_obj = Intervals(
            intervals=intervals_list,
            units=None,  # Using index units
            label="eye_divergences",
            data_time_range=(0, len(obj.tx))
        )
        
        # Apply mask if requested
        if apply_mask:
            obj.mask_intervals(intervals_obj, eyes=['left', 'right'], variables=['x', 'y'])
            return obj
        else:
            return intervals_obj
    
    @keephistory
    def mask_offscreen_coords(self, eyes: str|list = [], apply_mask: bool = True, 
                             ignore_existing_mask: bool = False, inplace=None):
        """
        Detect and mask gaze coordinates that fall outside the screen boundaries.
        
        This method identifies points where the (x,y) coordinates fall outside the 
        defined screen limits. When apply_mask=True, these points are masked. 
        When apply_mask=False, the offscreen periods are returned as Intervals objects.
        
        Parameters
        ----------
        eyes : str or list, default=[]
            Eye(s) to check for offscreen coordinates. If empty, checks all available eyes.
            Can be a single eye name (e.g., "left") or a list (e.g., ["left", "right"]).
        apply_mask : bool, default=True
            If True, apply detected offscreen periods as masks to the data and return self.
            If False, return detected offscreen periods as dict of Intervals objects.
        ignore_existing_mask : bool, default=False
            If False (default), only check non-masked data points for offscreen coordinates.
            If True, check all data points including those already masked (e.g., from blinks).
        inplace : bool or None
            If True, make change in-place and return the object.
            If False, make and return copy before making changes.
            If None, use the object-level setting.
        
        Returns
        -------
        GazeData or dict of Intervals
            If apply_mask=True: returns self for chaining.
            If apply_mask=False: returns dict of Intervals objects (one per eye) containing 
            detected offscreen intervals.
        
        Examples
        --------
        >>> # Mask offscreen coordinates for all eyes
        >>> gaze_data = gaze_data.mask_offscreen_coords()
        
        >>> # Detect offscreen periods without masking
        >>> offscreen = gaze_data.mask_offscreen_coords(apply_mask=False)
        
        >>> # Mask offscreen coordinates for left eye only
        >>> gaze_data = gaze_data.mask_offscreen_coords(eyes='left')
        
        >>> # Check all data including already masked points (e.g., during blinks)
        >>> gaze_data = gaze_data.mask_offscreen_coords(ignore_existing_mask=True)
        """
        obj = self._get_inplace(inplace)
        
        # Require experimental_setup for screen limits
        if obj.experimental_setup is None:
            raise ValueError(
                "Cannot mask offscreen coords: experimental_setup not set. "
                "Use set_experimental_setup() to configure it."
            )
        
        # Get eyes to check
        eyes_list, _ = obj._get_eye_var(eyes, [])
        
        # Dictionary to store intervals per eye
        intervals_dict = {}
        
        # Get screen limits from experimental_setup
        xlim = obj.experimental_setup.screen_xlim
        ylim = obj.experimental_setup.screen_ylim
        
        # Check each eye
        for eye in eyes_list:
            vx = f"{eye}_x"
            vy = f"{eye}_y"
            
            # Check if this eye has x and y data
            if vx not in obj.data.keys() or vy not in obj.data.keys():
                logger.warning(f"Eye '{eye}' does not have both x and y coordinates. Skipping.")
                continue
            
            # Get coordinates
            if ignore_existing_mask:
                # Use raw data, ignoring existing masks
                x = obj.data[vx]
                y = obj.data[vy]
            else:
                # Use masked arrays (respects existing masks)
                x = obj[vx]
                y = obj[vy]

            # Check all points including already masked ones
            offscreen_mask = (
                (x < xlim[0]) | 
                (x > xlim[1]) | 
                (y < ylim[0]) | 
                (y > ylim[1])
            )
            
            # Convert masked array to regular boolean array if needed
            if isinstance(offscreen_mask, np.ma.MaskedArray):
                offscreen_mask = offscreen_mask.filled(False)

            # Convert mask to intervals
            intervals_list = obj._mask_to_intervals_list(offscreen_mask)
            
            # Create Intervals object for this eye
            intervals_obj = Intervals(
                intervals=intervals_list,
                units=None,  # Using index units
                label=f"{eye}_offscreen",
                data_time_range=(0, len(obj.tx))
            )
            
            intervals_dict[eye] = intervals_obj
            
            logger.debug(f"Detected {len(intervals_obj)} offscreen intervals for {eye} eye")
        
        # Apply mask if requested
        if apply_mask:
            for eye, intervals_obj in intervals_dict.items():
                obj.mask_intervals(intervals_obj, eyes=[eye], variables=['x', 'y'])
            return obj
        else:
            return intervals_dict
