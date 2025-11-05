import itertools
from ..eyedata import GenericEyeData
from .genplotter import GenericPlotter
from ..intervals import Intervals
import numpy as np
from collections.abc import Iterable
from typing import Sequence, Union, List, TypeVar, Optional, Tuple, Callable
from pathlib import Path

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
                   eyes: str|list=[],
                   plot_range: Tuple[float,float]=(-np.inf, +np.inf),
                   plot_events: bool=True,
                   highlight_blinks: bool=True,
                   units: str="sec",
                   label_prefix: str="",
                   style: dict=None
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
        label_prefix: str
            Prefix to add to labels in the legend. Useful for overlaying multiple datasets.
            Default is "" (no prefix).
        style: dict or dict of dicts
            Styling for plotted lines. Can be:
            - Single dict: Applied to all eyes with automatic differentiation (e.g., {'color': 'red'})
            - Dict of dicts: Per-eye styling (e.g., {'left': {'linestyle': '-'}, 'right': {'linestyle': '--'}})
            Default is None (uses matplotlib defaults with automatic per-eye colors and linestyles).
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

        # Determine if style is per-eye or global
        per_eye_style = (style is not None and 
                         isinstance(style, dict) and 
                         any(isinstance(v, dict) for v in style.values()))
        
        # Define differentiation properties for multiple eyes
        # These cycle through different linestyles to keep eyes distinguishable
        linestyles = ['-', '--', '-.', ':']
        
        # plot timeseries
        for idx, eye in enumerate(eyes):
            # Build label with optional prefix
            if label_prefix:
                label = f"{label_prefix}_{eye}"
            else:
                label = eye
            
            # Prepare plot kwargs with style
            plot_kwargs = {'label': label}
            
            if style is not None:
                if per_eye_style:
                    # Per-eye styling: use eye-specific style if available
                    if eye in style:
                        plot_kwargs.update(style[eye])
                else:
                    # Global style: apply to all, but add differentiation for multiple eyes
                    plot_kwargs.update(style)
                    
                    # If multiple eyes and no explicit differentiation, vary linestyle
                    if len(eyes) > 1:
                        # Only add differentiation if not explicitly set in style
                        if 'linestyle' not in style and 'ls' not in style:
                            plot_kwargs['linestyle'] = linestyles[idx % len(linestyles)]
            elif len(eyes) > 1:
                # No style provided but multiple eyes: use default colors with varied linestyles
                plot_kwargs['linestyle'] = linestyles[idx % len(linestyles)]
            
            plt.plot(tx, self.obj.data[eye,"pupil"][startix:endix], **plot_kwargs)

        # plot grey lines for events
        if plot_events:
            logger.debug("Plotting events, %i in range"%len(evon))
            plt.vlines(evon, *plt.ylim(), color="grey", alpha=0.5)
            ll,ul=plt.ylim()
            for ev,lab in zip(evon,evlab):
                plt.text(ev, ll+(ul-ll)/2., "%s"%lab, fontsize=8, rotation=90)
        
        # highlight if a blink in any of the eyes
        blinks_obj = self.obj.get_blinks(eyes, "pupil", units=None)
        blinks = blinks_obj.as_index(self.obj)

        if highlight_blinks and len(blinks)>0:
            logger.debug("Highlighting blinks, %i in range"%blinks.shape[0])
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

        for start,end in segments:
            plt.figure(figsize=figsize)
            self.pupil_plot( plot_range=(start,end), units="min", **kwargs)
            if ylim is not None:
                plt.ylim(*ylim)
            fig = plt.gcf()
            figs.append(fig)
            
            # Close figure immediately after creation if saving to PDF to prevent display and memory issues
            if pdffile is not None:
                plt.close(fig)

        if pdffile is not None:
            # Create parent directories if they don't exist
            pdf_path = Path(pdffile)
            if pdf_path.parent != Path('.') and not pdf_path.parent.exists():
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Created directory: {}", pdf_path.parent)
            
            logger.info("Writing PDF file '{}'", pdffile)
            with PdfPages(pdffile) as pdf:
                for fig in figs:
                    pdf.savefig(fig)

        return figs        

    def plot_blinks(self, eyes: str|list = [], variables: str|list = "pupil",
                   pdf_file: str|None = None, nrow: int = 5, ncol: int = 3, 
                   figsize: tuple = (10, 10),
                   pre_blink: float = 200, post_blink: float = 200, 
                   units: str = "ms", plot_index: bool = True) -> list:
        """
        Plot detected blinks in separate subplots.
        
        Parameters
        ----------
        eyes : str or list
            Eyes to plot
        variables : str or list
            Variables to plot
        pdf_file : str or None
            Save to PDF file if provided
        nrow : int
            Number of rows per figure
        ncol : int
            Number of columns per figure
        figsize : tuple
            Figure size (width, height)
        pre_blink : float
            Time before blink to include (in ms)
        post_blink : float
            Time after blink to include (in ms)
        units : str
            Display units for time axis
        plot_index : bool
            Show blink index numbers
            
        Returns
        -------
        list
            List of matplotlib Figure objects
        """
        blinks = self.obj.get_blinks(units=None)
        if len(blinks) == 0:
            logger.warning("No blinks to plot")
            return []
        
        blinks_ix = blinks.as_index(self.obj)
        
        pre_blink_ix = int(pre_blink / 1000 * self.obj.fs)
        post_blink_ix = int(post_blink / 1000 * self.obj.fs)
        
        padded = [
            (max(0, s - pre_blink_ix), min(len(self.obj.tx), e + post_blink_ix))
            for s, e in blinks_ix
        ]
        
        intervals = Intervals(padded, units=None, label="blinks")
        return self.plot_intervals(intervals, eyes, variables,
                                  pdf_file, nrow, ncol, figsize, units, plot_index)

