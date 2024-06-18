"""
eyedata.py
============

Implement EyeData class for use with the pypillometry package.
This class allows to store eyetracking and pupil data in a single object.
"""
from .eyedata_generic import GenericEyedata, EyeDataDict, keephistory, _inplace
from .pupildata import PupilData
import numpy as np
from collections.abc import Iterable
import pylab as plt
import matplotlib.patches as patches
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


class EyeData(GenericEyedata):
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
                    keep_orig: bool = False):
        """
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
        """
        if time is None and sampling_rate is None:
            raise ValueError("Either `time` or `sampling_rate` must be provided")

        if (left_x is None or left_y is None) and (right_x is None or right_y is None):
            raise ValueError("At least one of the eye-traces must be provided (both x and y)")
        self.data=EyeDataDict(left_x=left_x, left_y=left_y, left_pupil=left_pupil,
                                right_x=right_x, right_y=right_y, right_pupil=right_pupil)
        ## name
        if name is None:
            self.name = self._random_id()
        else:
            self.name=name
        
        self._screen_size_set=False
        self._physical_screen_dims_set=False
        self._screen_eye_distance_set=False

        ## screen limits, physical screen size, screen-eye distance
        self.set_experiment_info(screen_resolution=screen_resolution, 
                                 physical_screen_size=physical_screen_size,
                                 screen_eye_distance=screen_eye_distance)

        ## set time array and sampling rate
        if time is None:
            maxT=len(self.data)/sampling_rate*1000.
            self.tx=np.linspace(0,maxT, num=len(self.data))
        else:
            self.tx=np.array(time, dtype=float)

        self.missing=np.zeros_like(self.tx, dtype=bool)

        if sampling_rate is None:
            self.fs=np.round(1000./np.median(np.diff(self.tx)))
        else:
            self.fs=sampling_rate
            
        self.set_event_onsets(event_onsets, event_labels)

        ## start with empty history    
        self.history=[]            

        self.original=None
        if keep_orig: 
            self.original=self.copy()

        ## fill in time discontinuities
        if fill_time_discontinuities:
            self.fill_time_discontinuities()

    def get_available_eyes(self):
        """
        Return a list of available eyes in the dataset.
        """
        return self.data.get_available_eyes()


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
        if not self._screen_size_set:
            raise ValueError("Screen size not set! Use `set_experiment_info()` to set it.")
        return self._screen_xlim

    @screen_xlim.setter
    def screen_xlim(self, value):
        if not isinstance(value, tuple):
            raise ValueError("Screen limits must be a tuple (xmin, xmax)")
        self._screen_xlim=value
        self._screen_size_set=True
    
    @property
    def screen_ylim(self):
        if not self._screen_size_set:
            raise ValueError("Screen size not set! Use `set_experiment_info()` to set it.")
        return self._screen_ylim
    
    @screen_ylim.setter
    def screen_ylim(self, value):
        if not isinstance(value, tuple):
            raise ValueError("Screen limits must be a tuple (ymin, ymax)")
        self._screen_ylim=value
        self._screen_size_set=True


    @property
    def screen_width(self):
        return self.screen_xlim[1]-self.screen_xlim[0]

    @property
    def screen_height(self):
        return self.screen_ylim[1]-self.screen_ylim[0]
    
    @property
    def physical_screen_width(self):
        if not self._physical_screen_dims_set:
            raise ValueError("Physical screen size not set! Use `set_experiment_info()` to set it.")
        return self.physical_screen_dims[0]

    @property
    def physical_screen_height(self):
        if not self._physical_screen_dims_set:
            raise ValueError("Physical screen size not set! Use `set_experiment_info()` to set it.")
        return self.physical_screen_dims[1]

    @property
    def screen_eye_distance(self):
        if not self._screen_eye_distance_set:
            raise ValueError("Physical screen size not set! Use `set_experiment_info()` to set it.")
        return self._screen_eye_distance



    def summary(self):
        """
        Return a summary of the dataset as a dictionary.
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
            nmiss=np.sum(self.missing),
            perc_miss=np.sum(self.missing)/len(self)*100.,
            duration_minutes=self.get_duration("min"),
            start_min=self.tx.min()/1000./60.,
            end_min=self.tx.max()/1000./60.,
            glimpse=repr(self.data)
        )
           
        return summary
    
    @keephistory
    def sub_slice(self, 
                start: float=-np.inf, 
                end: float=np.inf, 
                units: str=None, inplace=_inplace):
        """
        Return a new `EyeData` object that is a shortened version
        of the current one (contains all data between `start` and
        `end` in units given by `units` (one of "ms", "sec", "min", "h").
        If units is `None`, use the units in the time vector.

        Parameters
        ----------
        
        start: float
            start for new dataset
        end: float
            end of new dataset
        units: str
            time units in which `start` and `end` are provided.
            (one of "ms", "sec", "min", "h").
            If units is `None`, use the units in the time vector.
        """
        obj=self if inplace else self.copy()
        if units is not None: 
            fac=self._unit_fac(units)
            tx = self.tx*fac
            evon=obj.event_onsets*fac
        else: 
            tx = self.tx.copy()
            evon=obj.event_onsets.copy()
        keepix=np.where(np.logical_and(tx>=start, tx<=end))

        ndata={}
        for k,v in obj.data.items():
            ndata[k]=v[keepix]
        obj.data=EyeDataDict(ndata)
        obj.tx=obj.tx[keepix]

        
        keepev=np.logical_and(evon>=start, evon<=end)
        obj.event_onsets=obj.event_onsets[keepev]
        obj.event_labels=obj.event_labels[keepev]
        
        return obj


    def get_pupildata(self, eye):
        """
        Return the pupil data as a PupilData object.

        Parameters
        ----------
        eye : str, optional
            Which eye to return data for. 
        """
        if eye not in [k.split("_")[0] for k in self.data.keys()]:
            raise ValueError("No pupil data for eye: %s" % eye)
        pobj = PupilData(self.data[eye+"_pupil"],
                        sampling_rate=self.fs,
                        time=self.tx, 
                        event_onsets=self.event_onsets,
                        event_labels=self.event_labels,
                        name=self.name+"_pd",
                        keep_orig=False)
        return pobj        

    @keephistory
    def correct_pupil_foreshortening(self, eyes=None, midpoint=None, inplace=_inplace):
        """
        Correct the pupil data for foreshortening effects caused
        by saccades/eye movements. This method is based on a simple algorithm
        described here: 

            :ref:`Correcting pupillary signal using </docs/pupil_correction_carole.rst>`

        Relevant publication (not the description of the algorithm used here):    
        https://link.springer.com/article/10.3758/s13428-015-0588-x

        Parameters: 
        -----------
        eyes: list
            Which eyes to correct. If None, correct all available eyes.
        midpoint: tuple
            The center of the screen (x,y) where it is assumed that the pupil is completely circular.
            If None, the midpoint is taken to be the center of the screen as registered
            in the EyeData object. 
        inplace: bool
            Whether to modify the object in place or return a new object.
        """
        obj=self if inplace else self.copy()
        if midpoint is None:
            midpoint=(self.screen_width/2, self.screen_height/2)
        
        if eyes is None:
            eyes=self.get_available_eyes()

        scaling_factor_x=self.physical_screen_width/self.screen_width
        scaling_factor_y=self.physical_screen_height/self.screen_height

        # calculate distance of x,y from midpoint
        if not isinstance(eyes, list):
            eyes=[eyes]
        for eye in eyes:
            vx="_".join([eye, "x"])
            vy="_".join([eye, "y"])
            xdist=np.abs(self.data[vx]-midpoint[0])*scaling_factor_x
            ydist=np.abs(self.data[vy]-midpoint[1])*scaling_factor_y
            dist=np.sqrt(xdist**2 + ydist**2)
            corr=np.sqrt( (dist**2)/(self.screen_eye_distance**2) + 1)  # correction factor
            obj.data["_".join([eye, "pupil"])]=self.data["_".join([eye, "pupil"])]*corr

        return obj