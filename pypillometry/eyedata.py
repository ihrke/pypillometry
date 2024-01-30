"""
eyedata.py
============

Implement EyeData class for use with the pypillometry package.
This class allows to store eyetracking and pupil data in a single object.
"""
from .eyedata_generic import GenericEyedata, keephistory
import numpy as np

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
                 sampling_rate: float = None,
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
        name: 
            name of the dataset or `None` (in which case a random string is selected)

        event_onsets: 
            time-onsets of any events that are to be modelled in the pupil
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

    def has_left(self, ignorepupil=True): 
        """Data from left eye available?"""
        if "left_x" in self.data and "left_y" in self.data:
            if ignorepupil:
                return True
            else:
                return "left_pupil" in self.data
        else:
            return False

    def has_right(self, ignorepupil=True):
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
            nmiss=np.sum(self.missing),
            perc_miss=np.sum(self.missing)/len(self)*100.,
            duration_minutes=self.get_duration("min"),
            start_min=self.tx.min()/1000./60.,
            end_min=self.tx.max()/1000./60.,
            glimpse=repr(self.data)
        )
           
        return summary