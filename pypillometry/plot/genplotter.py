import itertools
from typing import Iterable, Optional, Tuple, Union
from ..eyedata import GenericEyeData
from ..convenience import mask_to_intervals
import numpy as np
from loguru import logger

import pylab as plt
import matplotlib.patches as patches
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

class GenericPlotter:
    """
    Abstract Class for plotting eye data. This class is not meant to be used directly.
    """
    obj: GenericEyeData # link to the data object

    def plot_intervals(self, intervals: list|np.ndarray,
                       eyes: str|list=[], variables: str|list=[],
                       pdf_file: Optional[str]=None, nrow: int=5, ncol: int=3, 
                       figsize: Tuple[int,int]=(10,10), 
                       units: str="ms", plot_mask: bool=False,
                       plot_index: bool=True):
        """"
        Plotting data around intervals.

        Each interval gets a separate subplot. 
        The data is plotted for each eye and variable in different colors.
        
        Parameters
        ----------
        eyes: str or list
            eyes to plot
        variables: str or list
            variables to plot
        intervals: list of tuples
            intervals to plot
        pdf_file: str or None
            if the name of a file is given, the figures are saved into a 
            multi-page PDF file
        ncol: int
            number of columns for the subplots for the intervals
        nrow: int
            number of rows for the subplots for the intervals
        units: str
            units in which the signal is plotted
        plot_index: bool
            plot a number with the blinks' index (e.g., for identifying abnormal blinks)
        plot_mask: bool
            plot the "mask" array of the underlying data object
        """
        obj=self.obj # PupilData object
        fac=obj._unit_fac(units)
        nsubplots=nrow*ncol # number of subplots per figure

        eyes,variables=obj._get_eye_var(eyes, variables)
        if isinstance(intervals, np.ndarray):
            if intervals.ndim!=2 or intervals.shape[1]!=2:
                raise ValueError("intervals must be a list of tuples or a 2D array with 2 columns")
            intervals=intervals.tolist()
        if isinstance(intervals, Iterable):
            intervals=[tuple(i) for i in intervals]
            
        nfig=int(np.ceil(len(intervals)/nsubplots))

        figs=[]
        if isinstance(pdf_file,str):
            _backend=mpl.get_backend()
            mpl.use("pdf")
            plt.ioff() ## avoid showing plots when saving to PDF 
        
        iinterv=0
        for i in range(nfig):
            fig=plt.figure(figsize=figsize)
            axs = fig.subplots(nrow, ncol).flatten()

            for ix,(start,end) in enumerate(intervals[(i*nsubplots):(i+1)*nsubplots]):
                iinterv+=1
                slic=slice(start,end)
                ax=axs[ix]
                for eye,var in itertools.product(eyes,variables):
                    ax.plot(obj.tx[slic]*fac,obj.data[eye,var][slic], label="%s_%s"%(eye,var))

                if plot_mask:
                    mask = obj.data.mask[eye+"_"+var][slic]
                    txm = obj.tx[slic]
                    mint = mask_to_intervals(mask)
                    for sm,em in mint:
                        ax.axvspan(txm[sm]*fac,txm[em]*fac, color="grey", alpha=0.3)
                if plot_index: 
                    ax.text(0.5, 0.5, '%i'%(iinterv), fontsize=12, horizontalalignment='center',     
                            verticalalignment='center', transform=ax.transAxes)
            figs.append(fig)

        if pdf_file is not None:
            logger.info("Saving file '{}'", pdf_file)
            with PdfPages(pdf_file) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
            ## switch back to original backend and interactive mode                
            mpl.use(_backend) 
            plt.ion()
            
        return figs    

    def plot_timeseries(self, 
            plot_range: tuple=(-np.inf, +np.inf),
            variables: list=[], 
            eyes: list=[],
            plot_onsets: str="line",
            units: Union[str, None]=None,
            figsize: tuple=(10,5),
            plot_masked: bool=False
            ) -> None:
        """
        Plot a part of the EyeData. Each data type is plotted in a separate subplot.

        Parameters:
        -----------

        plot_range: tuple
            The time range to plot. Default is (-np.inf, +np.inf), i.e. all data.
        variables: list
            The data to plot. Default is [], which means all available data will be plotted.
            Available data are ["x","y","pupil"] but can be extended by the user.
        eyes: list
            The eyes to plot. Default is [], which means all available data ("left", "right",
            "average", "regression", ...)
        plot_onsets: str
            Whether to plot markers for the event onsets. One of "line" (vertical lines),
            "label" (text label), "both" (both lines and labels), or "none" (no markers).
        units: str
            The units to plot. Default is "sec". If None, use the units in the time vector.
        figsize: tuple
            The figure size (per subplot). Default is (10,5).
        plot_masked: bool
            Whether to highlight masked regions with a light red background. Default is False.
        """
        obj = self.obj
        eyes,variables=obj._get_eye_var(eyes, variables)
        if units is not None: 
            fac=obj._unit_fac(units)
            tx = obj.tx*fac
            evon = obj.event_onsets*fac
        else:
            tx=obj.tx.copy()
            evon=obj.event_onsets.copy()
            units="tx"

        xlab=units

        # plot_range
        start,end=plot_range
        if start==-np.inf:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.inf:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end))
        
        # which data to plot
        variables = [v for v in variables if v not in getattr(self, 'ignore_vars', [])]

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
        evlab=obj.event_labels[ixx]
        evon=evon[ixx]

        nplots = len(variables)
        fig, axs = plt.subplots(nplots,1)
        # for the case when nplots=1, make axs iterable
        if not isinstance(axs, Iterable):
            axs=[axs]
        fig.set_figheight(figsize[1]*nplots)
        fig.set_figwidth(figsize[0])
        for var,ax in zip(variables, axs):
            for eye in eyes:
                vname = "_".join([eye,var])
                if vname in obj.data.keys():
                    ax.plot(tx, obj.data[vname][startix:endix], label=eye)
                    if plot_masked and vname in obj.data.mask:
                        mask = obj.data.mask[vname][startix:endix]
                        ax.fill_between(tx, ax.get_ylim()[0], ax.get_ylim()[1], 
                                      where=mask, color='red', alpha=0.2)
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
        
    def plot_timeseries_segments(self, pdffile: str=None, interv: float=1, figsize=(15,5), ylim=None, **kwargs):
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
            arguments passed to :func:`plot_timeseries()`

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
            self.plot_timeseries( plot_range=(start,end), units="min", **kwargs)
            if ylim is not None:
                plt.ylim(*ylim)
            figs.append(plt.gcf())


        if isinstance(pdffile, str):
            logger.info("Writing PDF file '{}'", pdffile)
            with PdfPages(pdffile) as pdf:
                for fig in figs:
                    pdf.savefig(fig)         

        ## switch back to original backend and interactive mode                        
        mpl.use(_backend) 
        plt.ion()

        return figs
        