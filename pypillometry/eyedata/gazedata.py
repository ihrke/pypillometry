from loguru import logger
from .generic import keephistory
from .eyedatadict import _without_layer
from ..intervals import Intervals
import numpy as np
import json
from .experimental_setup import ExperimentalSetup


class _GazeMethods:
    """
    Methods of :class:`~pypillometry.EyeData` for gaze data.

    Internal: not a data class (no state and no constructor). The methods
    that need gaze data (x and y) raise a ValueError if there is none.
    """

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
        
        Examples
        --------
        >>> data.set_experimental_setup(
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

    # =========================================================================
    # Methods
    # =========================================================================

    @keephistory
    def mask_eye_divergences(self, threshold: float = .99, thr_type: str = "percentile", 
                           store_as: str|None = None, apply_mask: bool = True, 
                           ignore_existing_mask: bool = True, inplace=None):
        """
        Calculate Euclidean distance between left and right eye coordinates and detect divergences.
        
        This method computes the distance between corresponding (x,y) coordinates in the left
        and right eyes. Points where the distance exceeds a threshold are detected as divergences.
        When apply_mask=True, these points are masked: they replace the
        ``"divergence"`` layer of x and y of both eyes (so running the method
        again with another threshold gives the same result as running it
        once). When apply_mask=False, the divergences
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
            The distance is stored for both eyes (``left_<store_as>`` and
            ``right_<store_as>``, the same values), masked where one of the
            coordinates is masked or the eyes diverge.
        apply_mask : bool, default=True
            If True, apply detected divergences as masks to the data and return self.
            If False, return detected divergences as an Intervals object.
        ignore_existing_mask : bool, default=True
            If True (default), check all data points including those already masked (e.g., from blinks).
            If False, only check the data points that are usable under the
            default mask policy for divergences (the divergence layer itself
            does not count).
        inplace : bool or None
            If True, make change in-place and return the object.
            If False, make and return copy before making changes.
            If None, use the object-level setting.
        
        Returns
        -------
        EyeData or Intervals
            If apply_mask=True: returns the object for chaining.
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
        
        >>> # Only check non-masked data points
        >>> gaze_data = gaze_data.mask_eye_divergences(ignore_existing_mask=False)
        """
        self._require("x", "y")
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
        if ignore_existing_mask:
            # Use raw data, ignoring existing masks
            left_x = obj.data['left_x']
            left_y = obj.data['left_y']
            right_x = obj.data['right_x']
            right_y = obj.data['right_y']
        else:
            # unusable samples (default mask policy), without the divergence
            # layer (it is replaced)
            policy = _without_layer("default", "divergence")
            left_x, left_y, right_x, right_y = (
                np.ma.masked_array(obj.data[k], obj.data.get_policy_mask(k, policy))
                for k in ("left_x", "left_y", "right_x", "right_y"))
        
        dist = np.ma.sqrt((left_x - right_x)**2 + (left_y - right_y)**2)
        
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
        
        # Store distance as a timeseries if requested: as variable `store_as`
        # of both eyes (a key without an eye would be taken for an eye),
        # masked where any of the coordinates is masked or the eyes diverge
        if store_as is not None:
            policy = _without_layer("default", "divergence")
            source_mask = np.logical_or.reduce(
                [obj.data.get_policy_mask(k, policy)
                 for k in ("left_x", "left_y", "right_x", "right_y")])
            for eye in ("left", "right"):
                key = f"{eye}_{store_as}"
                obj.data.set_with_mask(key, np.ma.getdata(dist), mask=source_mask)
                obj.data.set_layer(key, "divergence", divergence_mask, replace=True)
        
        # Create Intervals object for the detected divergences
        intervals_obj = Intervals(
            intervals=intervals_list,
            units="samples",
            label="eye_divergences")
        
        # Apply mask if requested: replaces the divergence layer of x and y
        if apply_mask:
            m = intervals_obj.to_mask(obj).astype(bool)
            for key in ("left_x", "left_y", "right_x", "right_y"):
                obj.data.set_layer(key, "divergence", m, replace=True)
            return obj
        else:
            return intervals_obj
    
    @keephistory
    def mask_offscreen_coords(self, eyes: str|list = [], apply_mask: bool = True, 
                             ignore_existing_mask: bool = False, inplace=None):
        """
        Detect and mask gaze coordinates that fall outside the screen boundaries.
        
        This method identifies points where the (x,y) coordinates fall outside the 
        defined screen limits. When apply_mask=True, these points are masked:
        they replace the ``"offscreen"`` layer of x and y of the checked eyes
        (so running the method again with other screen limits gives the same
        result as running it once). When apply_mask=False, the offscreen
        periods are returned as Intervals objects.
        
        Parameters
        ----------
        eyes : str or list, default=[]
            Eye(s) to check for offscreen coordinates. If empty, checks all available eyes
            that have x and y data.
            Can be a single eye name (e.g., "left") or a list (e.g., ["left", "right"]).
        apply_mask : bool, default=True
            If True, apply detected offscreen periods as masks to the data and return self.
            If False, return detected offscreen periods as dict of Intervals objects.
        ignore_existing_mask : bool, default=False
            If False (default), only check the data points that are usable
            under the default mask policy for offscreen coordinates (the
            offscreen layer itself does not count).
            If True, check all data points including those already masked (e.g., from blinks).
        inplace : bool or None
            If True, make change in-place and return the object.
            If False, make and return copy before making changes.
            If None, use the object-level setting.
        
        Returns
        -------
        EyeData or dict of Intervals
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
        self._require("x", "y")
        obj = self._get_inplace(inplace)
        
        # Require experimental_setup for screen limits
        if obj.experimental_setup is None:
            raise ValueError(
                "Cannot mask offscreen coords: experimental_setup not set. "
                "Use set_experimental_setup() to configure it."
            )
        
        # Get eyes to check
        eyes_list = obj._eyes_with(eyes, ["x", "y"])
        
        # Dictionary to store intervals per eye
        intervals_dict = {}
        
        # Get screen limits from experimental_setup
        xlim = obj.experimental_setup.screen_xlim
        ylim = obj.experimental_setup.screen_ylim
        
        # Check each eye
        for eye in eyes_list:
            vx = f"{eye}_x"
            vy = f"{eye}_y"
            
            # Get coordinates
            if ignore_existing_mask:
                # Use raw data, ignoring existing masks
                x = obj.data[vx]
                y = obj.data[vy]
            else:
                # unusable samples (default mask policy), without the
                # offscreen layer (it is replaced)
                policy = _without_layer("default", "offscreen")
                x = np.ma.masked_array(obj.data[vx], obj.data.get_policy_mask(vx, policy))
                y = np.ma.masked_array(obj.data[vy], obj.data.get_policy_mask(vy, policy))

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
                units="samples",
                label=f"{eye}_offscreen")
            
            intervals_dict[eye] = intervals_obj
            
            logger.debug(f"Detected {len(intervals_obj)} offscreen intervals for {eye} eye")
        
        # Apply mask if requested: replaces the offscreen layer of x and y
        if apply_mask:
            for eye, intervals_obj in intervals_dict.items():
                m = intervals_obj.to_mask(obj).astype(bool)
                for var in ("x", "y"):
                    obj.data.set_layer(f"{eye}_{var}", "offscreen", m, replace=True)
            return obj
        else:
            return intervals_dict
