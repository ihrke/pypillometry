import itertools
from ..eyedata import GenericEyeData
from .genplotter import GenericPlotter
import numpy as np
from collections.abc import Iterable
from typing import Sequence, Union, List, TypeVar, Optional, Tuple, Callable

from loguru import logger

import pylab as plt
import matplotlib.patches as patches
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

class PupilPlotter(GenericPlotter):
    """
    Class for plotting pupil data. The class is initialized with a GenericEyeData object
    and provides methods to plot the data in various ways.
    """
    obj: GenericEyeData # link to the data object

    def __init__(self, obj: GenericEyeData):
        self.obj = obj
    
    def pupil_plot(self, 
                   eyes: list=[],
                   plot_range: Tuple[float,float]=(-np.inf, +np.inf),
                   plot_events: bool=True,
                   highlight_blinks: bool=True,
                   units: str="sec"
            ) -> None:
        """Make a plot of the pupil data using `matplotlib`.

        Has pupil-specific options which makes it different from 
        :func:`GazePlotter.plot_timeseries()`.

        Parameters
        ----------
        eyes: list  
            list of eyes to plot; if empty, all available eyes are plotted            
        plot_range: tuple (start,end)
            plot from start to end (in units of `units`)
        units: str
            one of "sec"=seconds, "ms"=millisec, "min"=minutes, "h"=hours
        plot_events: bool
            plot events as vertical lines with labels
        highlight_blinks: bool
            highlight detected blinks
        """
        if not isinstance(eyes, list):
            eyes=[eyes]
        if len(eyes)==0:
            eyes=self.obj.data.get_available_eyes(variable="pupil")
        logger.debug("Plotting eyes %s"%eyes)

        fac=self.obj._unit_fac(units)
        logger.debug("Plotting in units %s (fac=%f)"%(units, fac))
        if units=="sec":
            xlab="seconds"
        elif units=="min":
            xlab="minutes"
        elif units=="h":
            xlab="hours"
        else:
            xlab="ms"
        tx=self.obj.tx*fac
        evon=self.obj.event_onsets*fac

        start,end=plot_range
        if start==-np.inf:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.inf:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))
                
        ixx=np.logical_and(evon>=start, evon<end)
        evlab=self.obj.event_labels[ixx]
        evon=evon[ixx]

        logger.debug("Plotting from %.2f to %.2f %s, (%i to %i)"%(start,end,units,startix,endix))

        tx=tx[startix:endix]

        # plot timeseries
        for eye in eyes:
            plt.plot(tx, self.obj.data[eye,"pupil"][startix:endix], label=eye)

        # plot grey lines for events
        if plot_events:
            plt.vlines(evon, *plt.ylim(), color="grey", alpha=0.5)
            ll,ul=plt.ylim()
            for ev,lab in zip(evon,evlab):
                plt.text(ev, ll+(ul-ll)/2., "%s"%lab, fontsize=8, rotation=90)
        
        # highlight if a blink in any of the eyes
        if highlight_blinks:
            blinks = self.obj.get_blinks_merged(eyes, "pupil")
            for sblink,eblink in blinks:
                if eblink<startix or sblink>endix:
                    continue
                else:
                    sblink=min(tx.size-1, max(0,sblink-startix))
                    eblink=min(endix-startix-1,eblink-startix)
                
                plt.gca().axvspan(tx[sblink],tx[eblink],color="red", alpha=0.2)


        plt.legend()
        plt.xlabel(xlab)        


    def pupil_plot_segments(self, pdffile: Optional[str]=None, interv: float=1, figsize=(15,5), ylim=None, **kwargs):
        """
        Plot the whole dataset chunked up into segments (usually to a PDF file).

        Parameters
        ----------

        pdffile: str or None
            file name to store the PDF; if None, no PDF is written 
        interv: float
            duration of each of the segments to be plotted (in minutes)
        figsize: Tuple[int,int]
            dimensions of the figures
        kwargs: 
            arguments passed to :func:`PupilData.pupil_plot()`

        Returns
        -------

        figs: list of :class:`matplotlib.Figure` objects
        """

        # start and end in minutes
        obj=self.obj
        smins,emins=obj.tx.min()/1000./60., obj.tx.max()/1000./60.
        segments=[]
        cstart=smins
        cend=smins
        while cend<emins:
            cend=min(emins, cstart+interv)
            segments.append( (cstart,cend) )
            cstart=cend

        figs=[]
        _backend=mpl.get_backend()
        mpl.use("pdf")
        plt.ioff() ## avoid showing plots when saving to PDF 

        for start,end in segments:
            plt.figure(figsize=figsize)
            self.pupil_plot( plot_range=(start,end), units="min", **kwargs)
            if ylim is not None:
                plt.ylim(*ylim)
            figs.append(plt.gcf())


        if isinstance(pdffile, str):
            print("> Writing PDF file '%s'"%pdffile)
            with PdfPages(pdffile) as pdf:
                for fig in figs:
                    pdf.savefig(fig)         

        ## switch back to original backend and interactive mode                        
        mpl.use(_backend) 
        plt.ion()

        return figs        

    def plot_blinks(self,
                    eyes: str|list=[], variables: str|list="pupil",
                    pdf_file: Optional[str]=None, nrow: int=5, ncol: int=3, 
                    figsize: Tuple[int,int]=(10,10), 
                    pre_blink: float=200, post_blink: float=200, units: str="ms", 
                    plot_index: bool=True):
        """
        Plot the detected blinks into separate figures each with nrow x ncol subplots. 

        Parameters
        ----------
        eyes: str or list
            eyes to plot
        variables: str or list
            variables to plot (default "pupil")
        pdf_file: str or None
            if the name of a file is given, the figures are saved into a multi-page PDF file
        ncol: int
            number of columns for the blink-plots
        pre_blink: float
            extend plot a certain time before each blink (in ms)
        post_blink: float
            extend plot a certain time after each blink (in ms)
        units: str
            units in which the signal is plotted
        plot_index: bool
            plot a number with the blinks' index (e.g., for identifying abnormal blinks)

        Returns
        -------

        list of plt.Figure objects each with nrow*ncol subplots
        in Jupyter Notebook, those are displayed inline one after the other
        """
        blinks=self.obj.get_blinks_merged()
        intervals=[(max(0,int(s-pre_blink)),min(self.obj.tx.size,int(e+post_blink))) for s,e in blinks]
        return self.plot_intervals(intervals, eyes, variables, pdf_file, nrow, ncol, figsize, units, plot_index)

