from ..eyedata import GenericEyeData
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

class PupilPlotter:
    """
    Class for plotting pupil data. The class is initialized with a GenericEyeData object
    and provides methods to plot the data in various ways.
    """
    obj: GenericEyeData # link to the data object

    def __init__(self, obj: GenericEyeData):
        self.obj = obj
    
    def pupil_plot(self, 
                   eyes: list=[],
                   plot_range: Tuple[float,float]=(-np.infty, +np.infty),
                   plot_events: bool=True,
                   highlight_blinks: bool=True,
                   highlight_interpolated: bool=True,
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
        highlight_interpolated: bool
            highlight interpolated data
        """
        if not isinstance(eyes, list):
            eyes=[eyes]
        if len(eyes)==0:
            eyes=self.obj.eyes
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
        if start==-np.infty:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.infty:
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

        # highlight if interpolated in any of the eyes
        if highlight_interpolated:
            ieyes = [eye for eye in eyes if eye+"_pupilinterpolated" in self.obj.data]
            interp = np.any([self.obj.data[eye,"pupilinterpolated"] for eye in ieyes], axis=0)
            a=np.diff(np.r_[0, interp[startix:endix], 0])[:-1]
            istarts=np.where(a>0)[0]
            iends=np.where(a<0)[0]
            for istart,iend in zip(istarts,iends):
                plt.gca().axvspan(tx[istart],tx[iend],color="green", alpha=0.1)
        
        # highlight if a blink in any of the eyes
        if highlight_blinks:
            blinkeyes = self.obj.blinks.keys()
            for eye in blinkeyes:
                for sblink,eblink in self.obj.blinks[eye]:
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

    def pupil_plot_blinks(self, pdf_file: Optional[str]=None, nrow: int=5, ncol: int=3, 
                    figsize: Tuple[int,int]=(10,10), 
                    pre_blink: float=500, post_blink: float=500, units: str="ms", 
                    plot_index: bool=True):
        """
        TODO
        Plot the detected blinks into separate figures each with nrow x ncol subplots. 

        Parameters
        ----------
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
        fac=self._unit_fac(units)
        pre_blink_ix=int((pre_blink/1000.)*self.fs)
        post_blink_ix=int((post_blink/1000.)*self.fs)

        nblinks=self.blinks.shape[0]
        nsubplots=nrow*ncol # number of subplots per figure
        nfig=int(np.ceil(nblinks/nsubplots))

        figs=[]
        if isinstance(pdf_file,str):
            _backend=mpl.get_backend()
            mpl.use("pdf")
            plt.ioff() ## avoid showing plots when saving to PDF 
        
        iblink=0
        for i in range(nfig):
            fig=plt.figure(figsize=figsize)
            axs = fig.subplots(nrow, ncol).flatten()

            for ix,(start,end) in enumerate(self.blinks[(i*nsubplots):(i+1)*nsubplots]):
                iblink+=1
                slic=slice(start-pre_blink_ix,end+post_blink_ix)
                ax=axs[ix]
                ax.plot(self.tx[slic]*fac,self.sy[slic])

                ## highlight interpolated data
                a=np.diff(np.r_[0,self.interpolated_mask[slic],0])[:-1]
                istarts=start-pre_blink_ix+np.where(a>0)[0]
                iends=start-pre_blink_ix+np.where(a<0)[0]
                for istart,iend in zip(istarts,iends):
                    ax.axvspan(self.tx[istart]*fac,self.tx[iend]*fac,color="green", alpha=0.1)

                ## highlight blink
                ax.axvspan(self.tx[start]*fac,self.tx[end]*fac,color="red", alpha=0.2)

                if plot_index: 
                    ax.text(0.5, 0.5, '%i'%(iblink), fontsize=12, horizontalalignment='center',     
                            verticalalignment='center', transform=ax.transAxes)
            figs.append(fig)

        if pdf_file is not None:
            print("> Saving file '%s'"%pdf_file)
            with PdfPages(pdf_file) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
            ## switch back to original backend and interactive mode                
            mpl.use(_backend) 
            plt.ion()
            
        return figs    


    