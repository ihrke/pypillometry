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
                    notes: str = None,
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
                          notes, inplace,
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

