"""
eyedata.py
============

Implement EyeData class for use with the pypillometry package.
This class allows to store eyetracking and pupil data in a single object.
"""
from .eyedata_generic import GenericEyedata, keephistory
import numpy as np
from collections.abc import Iterable

from pypillometry import GenericEyedata, EyeDataDict, keephistory
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
                    screen_limits: tuple = ((0,1280),(0,1024)),
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
            sampling-rate of the pupillary signal in Hz
        screen_limits: tuple
            ((xmin,xmax), (ymin,ymax)) for screen
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
        
        ## screen limits
        self.screen_xlim=screen_limits[0]
        self.screen_ylim=screen_limits[1]

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

    @keephistory
    def fill_time_discontinuities(self, yval=0, print_info=True):
        """
        find gaps in the time-vector and fill them in
        (add zeros to the signal)
        """
        tx=self.tx
        stepsize=np.median(np.diff(tx))
        n=len(self)
        gaps_end_ix=np.where(np.r_[stepsize,np.diff(tx)]>2*stepsize)[0]
        ngaps=gaps_end_ix.size
        if ngaps!=0:
            ## at least one gap here
            if print_info:
                print("> Filling in %i gaps"%ngaps)
            gaps_start_ix=gaps_end_ix-1
            print( ((tx[gaps_end_ix]-tx[gaps_start_ix])/1000), "seconds" )
            
            ## build new time-vector
            ntx=[tx[0:gaps_start_ix[0]]] # initial
            for i in range(ngaps):
                start,end=gaps_start_ix[i], gaps_end_ix[i]
                # fill in the gap
                ntx.append( np.linspace(tx[start],tx[end], int((tx[end]-tx[start])/stepsize), endpoint=False) )

                # append valid signal
                if i==ngaps-1:
                    nstart=n
                else:
                    nstart=gaps_start_ix[i+1]
                ntx.append( tx[end:nstart] )
            ntx=np.concatenate(ntx)

            ## fill in missing data
            newd = {}
            for k,v in self.data.items():
                nv = np.zeros_like(ntx)
                nv=[v[0:gaps_start_ix[0]]]
                for i in range(ngaps):
                    start,end=gaps_start_ix[i], gaps_end_ix[i]
                    nv.append( yval*np.ones_like(ntx[start:end], dtype=float) )
                # append valid signal
                if i==ngaps-1:
                    nstart=n
                else:
                    nstart=gaps_start_ix[i+1]                    
                nv.append( v[end:nstart] )
                newd[k]=np.concatenate(nv)
            
            self.data=EyeDataDict(newd)
            self.tx=ntx
        return self

    def has_left_eye(self, ignorepupil=True): 
        """Data from left eye available?"""
        if "left_x" in self.data and "left_y" in self.data:
            if ignorepupil:
                return True
            else:
                return "left_pupil" in self.data
        else:
            return False

    def has_right_eye(self, ignorepupil=True):
        """Data from right eye available?"""
        if "right_x" in self.data and "right_y" in self.data:
            if ignorepupil:
                return True
            else:
                return "right_pupil" in self.data
        else:
            return False

    def summary(self):
        summary=dict(
            name=self.name, 
            n=len(self.data),
            sampling_rate=self.fs,
            data=list(self.data.keys()),
            nevents=self.nevents(), 
            screen_limits=(self.screen_xlim, self.screen_ylim),
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
            time units in which `start` and `end` are provided
        """
        obj=self if inplace else self.copy()
        tx=self.tx
        if units is not None: 
            fac=self._unit_fac(units)
            tx *= fac
        keepix=np.where(np.logical_and(tx>=start, tx<=end))

        ndata={}
        for k,v in obj.data.items():
            ndata[k]=v[keepix]
        obj.data=EyeDataDict(ndata)
        obj.tx=obj.tx[keepix]

        evon=obj.event_onsets*obj._unit_fac(units)
        keepev=np.logical_and(evon>=start, evon<=end)
        obj.event_onsets=obj.event_onsets[keepev]
        obj.event_labels=obj.event_labels[keepev]
        
        return obj
    
    def plot(self, 
            plot_range: tuple=(-np.infty, +np.infty),
            plot_data: list=["x","y","pupil"], 
            plot_eyes: list=["left","right"],
            units: str="sec",
            figsize: tuple=(10,5)
            ) -> None:
        """
        Plot the EyeData.
        """
        fac=self._unit_fac(units)
        xlab=units
        tx=self.tx*fac
        evon=self.event_onsets*fac

        start,end=plot_range
        if start==-np.infty:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.infty:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))
        
        tx=tx[startix:endix]
        ixx=np.logical_and(evon>=start, evon<end)
        evlab=self.event_labels[ixx]
        evon=evon[ixx]

        nplots = len(plot_data)
        fig, axs = plt.subplots(nplots,1)
        # for the case when nplots=1, make axs iterable
        if not isinstance(axs, Iterable):
            axs=[axs]
        fig.set_figheight(figsize[1]*nplots)
        fig.set_figwidth(figsize[0])
        for k,ax in zip(plot_data, axs):
            for eye in plot_eyes:
                ax.plot(tx, self.data["_".join([eye,k])][startix:endix], label=eye)
            ax.set_title(k)
            ax.legend()
        
        plt.legend()
        plt.xlabel(xlab)        

