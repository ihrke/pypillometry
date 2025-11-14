from .generic import GenericEyeData, keephistory
from .eyedatadict import EyeDataDict
from ..plot import GazePlotter
import numpy as np
import json
from typing import Optional

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
    screen_resolution: tuple
        (xmax, ymax) screen resolution in pixels
    physical_screen_size: tuple
        (width, height) of the screen in cm; if None, the screen size is not used
    screen_eye_distance: float
        distance from the screen to the eye in cm
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
    use_cache: bool
        Whether to use cached storage for data arrays. Default is False.
    cache_dir: str
        Directory to store cache files. If None, creates a temporary directory.
    max_memory_mb: float
        Maximum memory usage in MB when using cache. Default is 100MB.
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
                    screen_resolution: tuple = None,
                    physical_screen_size: tuple = None,
                    screen_eye_distance: float = None,
                    name: str = None,
                    fill_time_discontinuities: bool = True,
                    keep_orig: bool = False,
                    info: dict = None,
                    inplace: bool = False,
                    use_cache: bool = False,
                    cache_dir: Optional[str] = None,
                    max_memory_mb: float = 100):
        """Constructor for the GazeData class.
        
        Parameters
        ----------
        use_cache : bool, optional
            Whether to use cached storage for data arrays. Default is False.
        cache_dir : str, optional
            Directory to store cache files. If None, creates a temporary directory.
        max_memory_mb : float, optional
            Maximum memory usage in MB when using cache. Default is 100MB.
        """
        if (left_x is None and left_y is None and right_x is None and right_y is None):
            raise ValueError("At least one eye-trace (both x and y coordinates) must be provided")
            
        # Initialize data dictionary
        self.data=EyeDataDict(left_x=left_x, left_y=left_y, right_x=right_x, right_y=right_y)
        
        self._init_common(time, sampling_rate, 
                          event_onsets, event_labels, 
                          name, fill_time_discontinuities, 
                          info=info, inplace=inplace,
                          use_cache=use_cache,
                          cache_dir=cache_dir,
                          max_memory_mb=max_memory_mb)
        
        self._screen_size_set=False
        self._physical_screen_dims_set=False
        self._screen_eye_distance_set=False

        ## screen limits, physical screen size, screen-eye distance
        self.set_experiment_info(screen_resolution=screen_resolution, 
                                physical_screen_size=physical_screen_size,
                                screen_eye_distance=screen_eye_distance)

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
            if self._screen_size_set:
                screen_limits=(self.screen_xlim, self.screen_ylim)
            else:
                screen_limits="not set"
            if self._physical_screen_dims_set:
                phys_dims=self.physical_screen_dims
            else:
                phys_dims="not set"

            summary=dict(
                name=self.name, 
                n=len(self.data),
                sampling_rate=self.fs,
                data=list(self.data.keys()),
                nevents=self.nevents(), 
                screen_limits=screen_limits,
                physical_screen_size=phys_dims,
                screen_eye_distance="not set",
                duration_minutes=self.get_duration("min"),
                start_min=self.tx.min()/1000./60.,
                end_min=self.tx.max()/1000./60.,
                parameters=self._strfy_params(),
                glimpse=repr(self.data)
            )
            
            return summary            
    

    def set_experiment_info(self, 
                            screen_eye_distance: float=None,
                            screen_resolution: tuple=None,
                            physical_screen_size: tuple=None):
        """
        Set some experimental information for the dataset.

        Parameters
        ----------
        screen_eye_distance: float
            distance from the screen to the eye in cm
        screen_resolution: tuple
            (width, height) of the screen in pixels
        physical_screen_size: tuple
            (width, height) of the screen in cm
        """
        if screen_resolution is not None:
            self.screen_xlim=(0,screen_resolution[0])
            self.screen_ylim=(0,screen_resolution[1])
        if physical_screen_size is not None:
            self.physical_screen_dims=physical_screen_size
            self._physical_screen_dims_set=True
        if screen_eye_distance is not None:
            self._screen_eye_distance=screen_eye_distance
            self._screen_eye_distance_set=True

    @property
    def screen_xlim(self):
        """Limits of the screen in x-direction (pixels).

        Returns
        -------
        tuple
            xmin,xmax of the screen
        """        
        if not self._screen_size_set:
            raise ValueError("Screen size not set! Use `set_experiment_info()` to set it.")
        return self._screen_xlim

    @screen_xlim.setter
    def screen_xlim(self, value):
        """Set x limits of the screen (pixels).

        Parameters
        ----------
        value : tuple (xmin, xmax)
            new limits of the screen
        """        
        if not isinstance(value, tuple):
            raise ValueError("Screen limits must be a tuple (xmin, xmax)")
        self._screen_xlim=value
        self._screen_size_set=True
    
    @property
    def screen_ylim(self):
        """Y-limits of the screen (pixels).

        Returns
        -------
        typle
            ymin,ymax of the screen
        """        
        if not self._screen_size_set:
            raise ValueError("Screen size not set! Use `set_experiment_info()` to set it.")
        return self._screen_ylim
    
    @screen_ylim.setter
    def screen_ylim(self, value):
        """Set y-limits of the screen (pixels).

        Parameters
        ----------
        value : tuple (ymin,ymax)
            new y-limits of the screen
        """        
        if not isinstance(value, tuple):
            raise ValueError("Screen limits must be a tuple (ymin, ymax)")
        self._screen_ylim=value
        self._screen_size_set=True


    @property
    def screen_width(self):
        """Width of the screen (pixels).

        Returns
        -------
        float
            xmax-xmin
        """        
        return self.screen_xlim[1]-self.screen_xlim[0]

    @property
    def screen_height(self):
        """Height of the screen (pixels).

        Returns
        -------
        float
            ymax-ymin
        """        
        return self.screen_ylim[1]-self.screen_ylim[0]
    
    @property
    def physical_screen_width(self):
        """Physical width of screen (cm).

        Returns
        -------
        float
            width of screen in cm
        """        
        if not self._physical_screen_dims_set:
            raise ValueError("Physical screen size not set! Use `set_experiment_info()` to set it.")
        return self.physical_screen_dims[0]

    @property
    def physical_screen_height(self):
        """Physical height of screen (cm).

        Returns
        -------
        float
            height of screen in cm
        """
        if not self._physical_screen_dims_set:
            raise ValueError("Physical screen size not set! Use `set_experiment_info()` to set it.")
        return self.physical_screen_dims[1]

    @property
    def screen_eye_distance(self):
        """Distance from screen to eye (cm).

        Returns
        -------
        float
            distance from screen to eye in cm
        """
        if not self._screen_eye_distance_set:
            raise ValueError("Physical screen size not set! Use `set_experiment_info()` to set it.")
        return self._screen_eye_distance

    @keephistory
    def mask_eye_divergences(self, threshold: float = .99, thr_type: str = "percentile", store_as: str|None = None, inplace=None):
        """
        Calculate Euclidean distance between left and right eye coordinates and mask divergences.
        
        This method computes the distance between corresponding (x,y) coordinates in the left
        and right eyes. Points where the distance exceeds a threshold are masked. The distance
        is stored as a new variable in the data.
        
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
        inplace : bool or None
            If True, make change in-place and return the object.
            If False, make and return copy before making changes.
            If None, use the object-level setting.
        
        Returns
        -------
        GazeData
            Object with distance timeseries stored and divergent points masked.
        
        Raises
        ------
        ValueError
            If both left and right eye data are not available.
            If thr_type is not "percentile" or "pixel".
        
        Examples
        --------
        >>> # Mask divergences at 99th percentile
        >>> gaze_data = gaze_data.mask_eye_divergences(threshold=0.99, thr_type="percentile")
        
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
        
        # Get masked arrays for left and right eye coordinates
        left_x = obj['left_x']
        left_y = obj['left_y']
        right_x = obj['right_x']
        right_y = obj['right_y']
        
        # Calculate Euclidean distance
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
        
        # Mask divergent points (where distance exceeds threshold)
        dist_mask = dist > thr
        
        # Combine with existing mask
        final_mask = np.ma.getmaskarray(dist) | dist_mask
        
        # Create masked array with the combined mask
        dist_masked = np.ma.masked_array(dist.data, mask=final_mask)
        
        # Store distance as a timeseries if requested
        if store_as is not None:
            obj[store_as] = dist_masked
        
        # Update masks for left and right eye coordinates where divergence is detected
        for eye in ['left', 'right']:
            for var in ['x', 'y']:
                key = f"{eye}_{var}"
                existing_mask = obj.data.mask[key].copy()
                # Mark divergent points as masked
                existing_mask[dist_mask] = 1
                obj.data.mask[key] = existing_mask
        
        return obj