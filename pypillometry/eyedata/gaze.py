from loguru import logger
from .generic import keephistory
from .namespace import Namespace
from .pipes import step
from .eyedatadict import _without_layer
import numpy as np


class GazeNamespace(Namespace):
    """
    Methods for gaze data, accessed as ``d.gaze`` (e.g.,
    ``d.gaze.mask_offscreen()``).

    The methods need gaze data (x and y) and raise a ValueError if there is
    none. The transforming methods return the data object (see
    :class:`~pypillometry.eyedata.namespace.Namespace`).
    """

    _name = "gaze"

    # =========================================================================
    # Methods
    # =========================================================================

    @step
    @keephistory
    def mask_divergences(self, threshold: float = .99, thr_type: str = "percentile", 
                           store_as: str|None = None,
                           ignore_existing_mask: bool = True, inplace=None):
        """
        Calculate Euclidean distance between left and right eye coordinates and detect divergences.
        
        This method computes the distance between corresponding (x,y) coordinates in the left
        and right eyes. Points where the distance exceeds a threshold are detected as divergences
        and masked: they replace the
        ``"divergence"`` layer of x and y of both eyes (so running the method
        again with another threshold gives the same result as running it
        once; ``clear_layer("divergence")`` removes them). Read the
        divergences with ``get_layer_intervals("divergence")``.
        
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
            The distance is stored once, under the eye ``"both"``
            (``both_<store_as>``), masked where one of the coordinates is
            masked or the eyes diverge.
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
        :class:`~pypillometry.EyeData`
            the object with the divergences masked (`self` or a copy,
            depending on `inplace`)
        
        Raises
        ------
        ValueError
            If both left and right eye data are not available.
            If thr_type is not "percentile" or "pixel".
        
        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> # Mask divergences at 99th percentile
        >>> d = d.gaze.mask_divergences(threshold=0.99, thr_type="percentile")

        >>> # The detected divergences (in ms)
        >>> divergences = d.get_layer_intervals("divergence", "left", "x", units="ms")
        >>> list(divergences)
        ['left_x']

        >>> # Mask divergences beyond 100 pixels
        >>> d = d.gaze.mask_divergences(threshold=100, thr_type="pixel")

        >>> # Only check non-masked data points
        >>> d = d.gaze.mask_divergences(ignore_existing_mask=False)
        """
        self._d._require("x", "y")
        obj = self._d._get_inplace(inplace)
        
        # Check that both eyes are available
        if 'left' not in obj.eyes or 'right' not in obj.eyes:
            raise ValueError(
                "Both left and right eye data are required for gaze.mask_divergences. "
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
        
        # Store distance as a timeseries if requested: as variable `store_as`
        # of the eye "both" (like "mean" for merged eyes), masked where any of
        # the coordinates is masked or the eyes diverge
        if store_as is not None:
            policy = _without_layer("default", "divergence")
            source_mask = np.logical_or.reduce(
                [obj.data.get_policy_mask(k, policy)
                 for k in ("left_x", "left_y", "right_x", "right_y")])
            key = f"both_{store_as}"
            obj.data.set_with_mask(key, np.ma.getdata(dist), mask=source_mask)
            obj.data.set_layer(key, "divergence", divergence_mask, replace=True)
        
        # replace the divergence layer of x and y
        for key in ("left_x", "left_y", "right_x", "right_y"):
            obj.data.set_layer(key, "divergence", divergence_mask, replace=True)
        return obj
    
    @step
    @keephistory
    def mask_offscreen(self, eyes: str|list = [],
                             ignore_existing_mask: bool = False, inplace=None):
        """
        Detect and mask gaze coordinates that fall outside the screen boundaries.
        
        This method identifies points where the (x,y) coordinates fall outside the 
        defined screen limits. These points are masked:
        they replace the ``"offscreen"`` layer of x and y of the checked eyes
        (so running the method again with other screen limits gives the same
        result as running it once; ``clear_layer("offscreen")`` removes
        them). Read the off-screen periods with
        ``get_layer_intervals("offscreen")``.
        
        Parameters
        ----------
        eyes : str or list, default=[]
            Eye(s) to check for offscreen coordinates. If empty, checks all available eyes
            that have x and y data.
            Can be a single eye name (e.g., "left") or a list (e.g., ["left", "right"]).
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
        :class:`~pypillometry.EyeData`
            the object with the off-screen samples masked (`self` or a copy,
            depending on `inplace`)
        
        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short")  # has a setup
        >>> # Mask offscreen coordinates for all eyes
        >>> d = d.gaze.mask_offscreen()

        >>> # The detected off-screen periods (in ms)
        >>> offscreen = d.get_layer_intervals("offscreen", variables="x", units="ms")
        >>> list(offscreen)
        ['left_x', 'right_x']

        >>> # Mask offscreen coordinates for left eye only
        >>> d = d.gaze.mask_offscreen(eyes="left")

        >>> # Check all data including already masked points (e.g., during blinks)
        >>> d = d.gaze.mask_offscreen(ignore_existing_mask=True)
        """
        self._d._require("x", "y")
        obj = self._d._get_inplace(inplace)
        
        # Require experimental_setup for screen limits
        if obj.experimental_setup is None:
            raise ValueError(
                "Cannot mask offscreen coords: experimental_setup not set. "
                "Use set_experimental_setup() to configure it."
            )
        
        # Get eyes to check
        eyes_list = obj._eyes_with(eyes, ["x", "y"])
        
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
            logger.debug(f"Detected {offscreen_mask.sum()} offscreen samples for {eye} eye")

            # replace the offscreen layer of x and y
            for var in ("x", "y"):
                obj.data.set_layer(f"{eye}_{var}", "offscreen", offscreen_mask, replace=True)
        return obj
