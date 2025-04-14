from .generic import keephistory
from .gazedata import GazeData
from .eyedatadict import EyeDataDict
from ..plot import EyePlotter
import numpy as np
from loguru import logger

from .pupildata import PupilData
import numpy as np
from collections.abc import Iterable
from typing import Optional



class EyeData(GazeData,PupilData):
    """
    Class for handling eye-tracking data. This class is a subclass of GazeData
    and inherits all its methods and attributes. In addition to the methods
    in GazeData which implement functions for eye-tracking data, `EyeData` 
    provides methods for handling pupillometric data.

    Parameters
    ----------
    time: 
        timing array or `None`, in which case the time-array goes from [0,maxT]
        using `sampling_rate` (in ms)
    left_x, left_y, left_pupil:
        data from left eye (at least one of the eyes must be provided, both x and y)
        pupil is optional
    right_x, right_y, right_pupil:
        data from right eye (at least one of the eyes must be provided, both x and y)
        pupil is optional
    sampling_rate: float
        sampling-rate of the pupillary signal in Hz; if None, 
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
    use_cache: bool
        Whether to use cached storage for data arrays. Default is False.
    cache_dir: str, optional
        Directory to store cache files. If None, creates a temporary directory.
    max_memory_mb: float, optional
        Maximum memory usage in MB when using cache. Default is 100MB.

    Examples
    --------
    >>> import numpy as np
    >>> from pypillometry.eyedata import EyeData
    >>> d=EyeData(time=np.arange(0, 1000, 1),
    ...         left_x=np.random.randn(1000),
    ...         left_y=np.random.randn(1000))
    >>> print(d)
    """
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
        """Constructor for the EyeData class.
        """
        if (left_x is None or left_y is None) and (right_x is None or right_y is None):
            raise ValueError("At least one of the eye-traces must be provided (both x and y)")
        self.data=EyeDataDict(left_x=left_x, left_y=left_y, left_pupil=left_pupil,
                                right_x=right_x, right_y=right_y, right_pupil=right_pupil)
        
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
        return EyePlotter(self)


    def summary(self):
        """Return a summary of the dataset as a dictionary.

        Returns
        -------
        : dict
            A dictionary summarizing the dataset.
        """        
        ds = dict(GazeData.summary(self),
                  **PupilData.summary(self))
        return ds

    def get_pupildata(self, eye=None):
        """
        Return the pupil data as a PupilData object.

        Parameters
        ----------
        eye : str, optional
            Which eye to return data for.

        Returns
        -------
        : :class:`PupilData`
            A PupilData object containing the pupil data. 
        """
        if eye is None:
            if len(self.eyes)==1:
                eye=self.eyes[0]
            else:
                raise ValueError("More than one eye in the dataset. Please specify which eye to use.")
        if eye not in [k.split("_")[0] for k in self.data.keys()]:
            raise ValueError("No pupil data for eye: %s" % eye)
        kwargs = {
            "time": self.tx,
            eye+"_pupil": self.data[eye+"_pupil"],
            "sampling_rate": self.fs,
            "event_onsets": self.event_onsets,
            "event_labels": self.event_labels,
            "name": self.name+"_pd",
            "keep_orig": False,
            "inplace": False
        }
        pobj = PupilData(**kwargs)
        return pobj        
    

    @keephistory
    def correct_pupil_foreshortening(self, eyes=[], midpoint=None, store_as="pupil", inplace=None):
        """Correct pupil data using a simple algorithm to correct for foreshortening effects.

        Correct the pupil data for foreshortening effects caused
        by saccades/eye movements. This method is based on a simple algorithm
        described here: 

        :ref:`Correcting pupillary signal using a simple Foreshortening Algorithm </docs/pupil_correction_carole.rst>`

        Relevant publication (not the description of the algorithm used here):    
        https://link.springer.com/article/10.3758/s13428-015-0588-x

        - [ ] TODO: when using interpolated pupil data, x/y data may still be missing. Make sure that the 
          pupil data is not touched where x/y is missing
        

        Parameters 
        ----------
        eyes: list
            Which eyes to correct. If None, correct all available eyes.
        midpoint: tuple
            The center of the screen (x,y) where it is assumed that the pupil is completely circular.
            If None, the midpoint is taken to be the center of the screen as registered
            in the EyeData object. 
        inplace: bool
            Whether to modify the object in place or return a new object.
            `true`: modify in place
            `false`: return a new object
            `None`: use the object's default setting (initialized in __init__)

        Returns
        -------
        : :class:`EyeData`
            An EyeData object with the corrected pupil
        """
        obj=self._get_inplace(inplace)
        eyes,_=self._get_eye_var(eyes, [])

        if not isinstance(store_as, str):
            logger.warning("store_as must be a string; using 'pupil' instead")
            store_as="pupil"

        if midpoint is None:
            midpoint=(self.screen_width/2, self.screen_height/2)
        
        scaling_factor_x=self.physical_screen_width/self.screen_width
        scaling_factor_y=self.physical_screen_height/self.screen_height

        # calculate distance of x,y from midpoint
        for eye in eyes:
            vx="_".join([eye, "x"])
            vy="_".join([eye, "y"])
            xdist=np.abs(self.data[vx]-midpoint[0])*scaling_factor_x
            ydist=np.abs(self.data[vy]-midpoint[1])*scaling_factor_y
            dist=np.sqrt(xdist**2 + ydist**2)
            corr=np.sqrt( (dist**2)/(self.screen_eye_distance**2) + 1)  # correction factor
            obj.data["_".join([eye, store_as])]=self.data["_".join([eye, "pupil"])]*corr

        return obj