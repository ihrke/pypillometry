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
            sampling-rate of the pupillary signal in Hz; if None, 
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

    @property
    def screen_width(self):
        return self.screen_xlim[1]-self.screen_xlim[0]

    @property
    def screen_height(self):
        return self.screen_ylim[1]-self.screen_ylim[0]
    
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
    
    def plot(self, 
            plot_range: tuple=(-np.infty, +np.infty),
            plot_data: list=[], 
            plot_eyes: list=[],
            plot_onsets: str="line",
            units: str=None,
            figsize: tuple=(10,5)
            ) -> None:
        """
        Plot a part of the EyeData. Each data type is plotted in a separate subplot.

        Parameters:
        -----------

        plot_range: tuple
            The time range to plot. Default is (-np.infty, +np.infty), i.e. all data.
        plot_data: list
            The data to plot. Default is [], which means all available data will be plotted.
            Available data are ["x","y","pupil"] but can be extended by the user.
        plot_eyes: list
            The eyes to plot. Default is [], which means all available data ("left", "right",
            average, regression, ...)
        plot_onsets: str
            Whether to plot markers for the event onsets. One of "line" (vertical lines),
            "label" (text label), "both" (both lines and labels), or "none" (no markers).
        units: str
            The units to plot. Default is "sec". If None, use the units in the time vector.
        figsize: tuple
            The figure size (per subplot). Default is (10,5).
        """
        if units is not None: 
            fac=self._unit_fac(units)
            tx = self.tx*fac
            evon = self.event_onsets*fac
        else:
            tx=self.tx.copy()
            evon=self.event_onsets.copy()
            units="tx"

        xlab=units

        # plot_range
        start,end=plot_range
        if start==-np.infty:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.infty:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))
        
        # which data to plot
        if len(plot_data)==0:
            plot_data=self.data.get_available_variables()

        # which eyes to plot
        if len(plot_eyes)==0:
            plot_eyes=self.data.get_available_eyes()

        # how to plot onsets
        if plot_onsets=="line":
            ev_line=True
            ev_label=False
        elif plot_onsets=="label":
            ev_line=False
            ev_label=True
        elif plot_onsets=="both":
            ev_line=True
            ev_label=True
        elif plot_onsets=="none":
            ev_line=False
            ev_label=False
        else:
            raise ValueError("plot_onsets must be one of 'line', 'label', 'both', or 'none'")
        
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
        for var,ax in zip(plot_data, axs):
            for eye in plot_eyes:
                vname = "_".join([eye,var])
                if vname in self.data.keys():
                    ax.plot(tx, self.data[vname][startix:endix], label=eye)
            if ev_line:
                ax.vlines(evon, *ax.get_ylim(), color="grey", alpha=0.5)
            if ev_label:
                ll,ul=ax.get_ylim()
                for ev,lab in zip(evon,evlab):
                    ax.text(ev, ll+(ul-ll)/2., "%s"%lab, fontsize=8, rotation=90)
            ax.set_title(var)
            ax.legend()
        
        plt.legend()
        plt.xlabel(xlab)        

    def plot_heatmap(self, 
            plot_range: tuple=(-np.infty, +np.infty), 
            plot_eyes: list=[],
            plot_screen: bool=True,
            units: str="sec",
            figsize: tuple=(10,10),
            cmap: str="jet",
            gridsize="auto"
            ) -> None:
        """
        Plot EyeData as a heatmap. Typically used for a large amount of data
        to spot the most frequent locations.
        To plot a scanpath, use :meth:`.plot_scanpath()` instead.
        Each eye is plotted in a separate subplot.

        Parameters:
        -----------

        plot_range: tuple
            The time range to plot. Default is (-np.infty, +np.infty), i.e. all data.
        plot_eyes: list
            The eyes to plot. Default is [], which means all available data ("left", "right",
            average, regression, ...)
        plot_screen: bool
            Whether to plot the screen limits. Default is True.
        units: str
            The units to plot. Default is "sec".
        figsize: tuple
            The figure size (per subplot). Default is (10,5).
        cmap: str
            The colormap to use. Default is "jet".
        gridsize: str or int
            The gridsize for the hexbin plot. Default is "auto".
        """
        fac=self._unit_fac(units)
        tx=self.tx*fac

        # plot_range
        start,end=plot_range
        if start==-np.infty:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.infty:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))    
        
        # which eyes to plot
        if len(plot_eyes)==0:
            plot_eyes=self.data.get_available_eyes()
        if not isinstance(plot_eyes, list):
            plot_eyes=[plot_eyes]

        # choose gridsize
        if gridsize=="auto":
            gridsize=int(np.sqrt(endix-startix))

        nplots = len(plot_eyes)
        fig, axs = plt.subplots(1,nplots)
        # for the case when nplots=1, make axs iterable
        if not isinstance(axs, Iterable):
            axs=[axs]
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0]*nplots)
        for eye,ax in zip(plot_eyes, axs):
            vx = "_".join([eye,"x"])
            vy = "_".join([eye,"y"])
            if vx in self.data.keys() and vy in self.data.keys():
                divider = make_axes_locatable(ax)
                im=ax.hexbin(self.data[vx][startix:endix], self.data[vy][startix:endix], 
                            gridsize=gridsize, cmap=cmap, mincnt=1)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_title(eye)
            ax.set_aspect("equal")
            #fig.colorbar()
            if plot_screen:
                screenrect=patches.Rectangle((self.screen_xlim[0], self.screen_ylim[0]), 
                                self.screen_xlim[1], self.screen_ylim[1], 
                                fill=False, edgecolor="red", linewidth=2)
                ax.add_patch(screenrect)

    def plot_scanpath(self, 
            plot_range: tuple=(-np.infty, +np.infty), 
            plot_eyes: list=[],
            plot_screen: bool=True,
            plot_onsets: bool=True,
            title: str="",
            units: str=None,
            figsize: tuple=(10,10)
            ) -> None:
        """
        Plot EyeData as a scanpath. Typically used for single trials
        or another small amount of data. To plot a larger amount of data,
        use :meth:`.plot_heatmap()` instead.

        If several eyes are available, they are plotted in the same figure 
        in different colors. You can choose between them using the `plot_eyes` 
        parameter.

        Parameters:
        -----------

        plot_range: tuple
            The time range to plot. Default is (-np.infty, +np.infty), i.e. all data.
        plot_eyes: list
            The eyes to plot. Default is [], which means all available data ("left", "right",
            average, regression, ...)
        plot_screen: bool
            Whether to plot the screen limits. Default is True.
        plot_onsets: bool
            Whether to plot the event onsets. Default is True.
        title: str
            The title of the plot. Default is "".
        units: str
            The units to plot. Default is "sec". Plotted only for the first "eye".
            If None, the units are taken from the data.
        figsize: tuple
            The figure size (per subplot). Default is (10,5).
        """
        if units is not None: 
            fac=self._unit_fac(units)
            tx = self.tx*fac
            evon = self.event_onsets*fac
        else:
            tx=self.tx.copy()
            evon=self.event_onsets.copy()

        # plot_range
        start,end=plot_range
        if start==-np.infty:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.infty:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))    
        
        # get events in range
        evonixx=np.logical_and(evon>=start, evon<end)
        evlab=self.event_labels[evonixx]
        evon=evon[evonixx]
        evontix=np.argmin(np.abs(tx-evon[:,np.newaxis]), axis=1).astype(int)

        # which eyes to plot
        if len(plot_eyes)==0:
            plot_eyes=self.data.get_available_eyes()

        fig, ax = plt.subplots(1,1)
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])

        tnorm = tx[startix:endix].copy()
        tnorm = (tnorm - tnorm.min()) / (tnorm.max() - tnorm.min())
        for eye in plot_eyes:
            vx = "_".join([eye,"x"])
            vy = "_".join([eye,"y"])
            if vx in self.data.keys() and vy in self.data.keys():
                ax.plot(self.data[vx][startix:endix], self.data[vy][startix:endix], alpha=0.3, label=eye)
                ax.scatter(self.data[vx][startix:endix], self.data[vy][startix:endix], 
                        s=1, c=cm.jet(tnorm))
            if plot_onsets and eye==plot_eyes[0]:
                for ix, lab in zip(evontix, evlab):
                    ax.text(self.data[vx][ix], self.data[vy][ix], lab, 
                            fontsize=12, ha="center", va="center")

        ax.set_aspect("equal")
        #fig.colorbar()
        if plot_screen:
            screenrect=patches.Rectangle((self.screen_xlim[0], self.screen_ylim[0]), 
                            self.screen_xlim[1], self.screen_ylim[1], 
                            fill=False, edgecolor="red", linewidth=2)
            ax.add_patch(screenrect)
        ax.legend()
        ax.set_title(title)
        fig.tight_layout()

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

