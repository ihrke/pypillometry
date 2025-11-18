import itertools
from typing import Iterable, Optional, Tuple, Union
from ..eyedata import GenericEyeData
from ..intervals import Intervals
import numpy as np
from loguru import logger
from pathlib import Path

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

    def plot_intervals(self, intervals: Intervals,
                       eyes: str|list=[], variables: str|list=[],
                       pdf_file: Optional[str]=None, nrow: int=5, ncol: int=3, 
                       figsize: Tuple[int,int]=(10,10), 
                       units: str="ms", plot_mask: bool=False,
                       plot_index: bool=True):
        """"
        Plotting data around intervals.

        Intervals can be extracted with EyeData.get_intervals(). 
        Each interval gets a separate subplot. 
        The data is plotted for each eye and variable in different colors.
        
        Parameters
        ----------
        intervals: Intervals
            Intervals object containing the intervals to plot (from get_intervals())
        eyes: str or list
            eyes to plot
        variables: str or list
            variables to plot
        pdf_file: str or None
            if the name of a file is given, the figures are saved into a 
            multi-page PDF file
        ncol: int
            number of columns for the subplots for the intervals
        nrow: int
            number of rows for the subplots for the intervals
        units: str
            units in which the signal is plotted (for display only)
        plot_index: bool
            plot a number with the blinks' index (e.g., for identifying abnormal blinks)
        plot_mask: bool
            plot the "mask" array of the underlying data object
        """
        if not isinstance(intervals, Intervals):
            raise TypeError("intervals must be an Intervals object. Use get_intervals() to create one.")
        
        obj=self.obj # PupilData object
        fac=obj._unit_fac(units)  # For display
        nsubplots=nrow*ncol # number of subplots per figure

        eyes,variables=obj._get_eye_var(eyes, variables)
        
        # Convert Intervals to indices for data slicing
        if intervals.units is None:
            # Already in index units
            intervals_idx = intervals.intervals
        else:
            # Convert from intervals' units to indices
            fac_to_ms = 1.0 / obj._unit_fac(intervals.units)
            intervals_idx = []
            for start, end in intervals.intervals:
                # Convert to ms
                start_ms = start * fac_to_ms
                end_ms = end * fac_to_ms
                # Find corresponding indices
                start_ix = np.argmin(np.abs(obj.tx - start_ms))
                end_ix = np.argmin(np.abs(obj.tx - end_ms))
                intervals_idx.append((start_ix, end_ix))
            
        nfig=int(np.ceil(len(intervals_idx)/nsubplots))

        figs=[]
        
        iinterv=0
        for i in range(nfig):
            fig=plt.figure(figsize=figsize)
            axs = fig.subplots(nrow, ncol)
            # Ensure axs is always iterable (flatten if array, wrap if single Axes)
            if isinstance(axs, np.ndarray):
                axs = axs.flatten()
            elif not isinstance(axs, Iterable):
                axs = [axs]

            for ix,(start,end) in enumerate(intervals_idx[(i*nsubplots):(i+1)*nsubplots]):
                iinterv+=1
                slic=slice(start,end)
                ax=axs[ix]
                for eye,var in itertools.product(eyes,variables):
                    ax.plot(obj.tx[slic]*fac,obj.data[eye,var][slic], label="%s_%s"%(eye,var))

                # Plot vertical line at event onset (first event for this interval)
                if intervals.event_onsets is not None and len(intervals.event_onsets) > (iinterv-1):
                    event_onset_idx = iinterv - 1
                    # Convert event onset from intervals units to display units
                    if intervals.units is None:
                        # Event onset is in indices, need to convert to ms then to display units
                        event_onset_ms = obj.tx[int(intervals.event_onsets[event_onset_idx])]
                    else:
                        # Event onset is in intervals units, convert to ms first
                        fac_onset_to_ms = 1.0 / obj._unit_fac(intervals.units)
                        event_onset_ms = intervals.event_onsets[event_onset_idx] * fac_onset_to_ms
                    
                    # Now convert to display units
                    event_onset_display = event_onset_ms * fac
                    ax.axvline(event_onset_display, color='red', linestyle='--', 
                              linewidth=1, alpha=0.7, zorder=1)

                if plot_mask:
                    mask = obj.data.mask[eye+"_"+var][slic]
                    txm = obj.tx[slic]
                    mint = obj._mask_to_intervals_list(mask)
                    for sm,em in mint:
                        ax.axvspan(txm[sm]*fac,txm[em]*fac, color="grey", alpha=0.3)
                if plot_index: 
                    ax.text(0.5, 0.5, '%i'%(iinterv), fontsize=12, horizontalalignment='center',     
                            verticalalignment='center', transform=ax.transAxes)
            figs.append(fig)
            
            # Close figure immediately after creation if saving to PDF to prevent display and memory issues
            if pdf_file is not None:
                plt.close(fig)

        if pdf_file is not None:
            # Create parent directories if they don't exist
            pdf_path = Path(pdf_file)
            if pdf_path.parent != Path('.') and not pdf_path.parent.exists():
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Created directory: {}", pdf_path.parent)
            
            logger.info("Saving file '{}'", pdf_file)
            with PdfPages(pdf_file) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
            
        return figs    

    def plot_timeseries(self, 
            plot_range: tuple=(-np.inf, +np.inf),
            variables: list=[], 
            eyes: list=[],
            plot_onsets: str="line",
            units: Union[str, None]=None,
            plot_masked: bool=False,
            label_prefix: str="",
            style: dict=None
            ) -> None:
        """
        Plot a part of the EyeData. Each data type is plotted in a separate subplot.
        
        Uses the current figure/axes. If no figure exists, creates one automatically.

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
        plot_masked: bool
            Whether to highlight masked regions with a light red background. Default is False.
        label_prefix: str
            Prefix to add to labels in the legend. Useful for overlaying multiple datasets.
            Default is "" (no prefix).
        style: dict or dict of dicts
            Styling for plotted lines. Can be:
            - Single dict: Applied to all eyes with automatic differentiation (e.g., {'color': 'red'})
            - Dict of dicts: Per-eye styling (e.g., {'left': {'linestyle': '-'}, 'right': {'linestyle': '--'}})
            Default is None (uses matplotlib defaults with automatic per-eye colors and linestyles).
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
        # Use current figure or create new one if none exists
        fig = plt.gcf()
        # Check if current figure is empty/new
        if len(fig.axes) == 0:
            # Create subplots in the current figure
            axs = fig.subplots(nplots, 1)
            # for the case when nplots=1, make axs iterable
            if not isinstance(axs, Iterable):
                axs=[axs]
        else:
            # Use existing axes
            axs = fig.axes
            if len(axs) != nplots:
                logger.warning(f"Current figure has {len(axs)} axes but {nplots} variables to plot. Creating new subplots.")
                fig.clear()
                axs = fig.subplots(nplots, 1)
                if not isinstance(axs, Iterable):
                    axs=[axs]
        
        # Determine if style is per-eye or global
        per_eye_style = (style is not None and 
                         isinstance(style, dict) and 
                         any(isinstance(v, dict) for v in style.values()))
        
        # Define differentiation properties for multiple eyes
        # These cycle through different linestyles to keep eyes distinguishable
        linestyles = ['-', '--', '-.', ':']
        
        for var,ax in zip(variables, axs):
            for idx, eye in enumerate(eyes):
                vname = "_".join([eye,var])
                if vname in obj.data.keys():
                    # Build label with optional prefix
                    label_parts = []
                    if label_prefix:
                        label_parts.append(label_prefix)
                    label_parts.append(eye)
                    # Add variable name only if multiple variables plotted
                    if len(variables) > 1:
                        label_parts.append(var)
                    label = "_".join(label_parts)
                    
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
                    
                    ax.plot(tx, obj.data[vname][startix:endix], **plot_kwargs)
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
        
    def plot_timeseries_segments(self, pdf_file: str=None, interv: float=1, figsize=(15,5), 
                                  ylim=None, eyes: list=[], variables: list=[], 
                                  units: str="min", **kwargs):
        """
        Plot the whole dataset chunked up into segments (usually to a PDF file).

        Parameters
        ----------

        pdf_file: str or None
            file name to store the PDF; if None, no PDF is written 
        interv: float
            duration of each of the segments to be plotted (in the units specified by `units`)
        figsize: Tuple[int,int]
            dimensions of the figures
        ylim: Tuple[float,float] or None
            y-axis limits for the plots
        eyes: list
            list of eyes to plot; if empty, all available eyes are plotted
        variables: list
            list of variables to plot; if empty, all available variables are plotted
        units: str
            units for the time axis; default is "min" (minutes). 
            The `interv` parameter is also interpreted in these units.
        kwargs: 
            additional arguments passed to :func:`plot_timeseries()`

        Returns
        -------

        figs: list of :class:`matplotlib.Figure` objects
        """

        # Calculate start and end times in the specified units
        obj=self.obj
        fac=obj._unit_fac(units)
        stime = obj.tx.min() * fac
        etime = obj.tx.max() * fac
        
        # Create segments
        segments=[]
        cstart=stime
        cend=stime
        while cend<etime:
            cend=min(etime, cstart+interv)
            segments.append( (cstart,cend) )
            cstart=cend

        figs=[]

        for start,end in segments:
            # Create a new figure for each segment
            fig = plt.figure(figsize=figsize)
            self.plot_timeseries(plot_range=(start,end), units=units, 
                               eyes=eyes, variables=variables, **kwargs)
            if ylim is not None:
                plt.ylim(*ylim)
            figs.append(fig)
            
            # Close figure immediately after creation if saving to PDF to prevent display and memory issues
            if pdf_file is not None:
                plt.close(fig)

        if pdf_file is not None:
            # Create parent directories if they don't exist
            pdf_path = Path(pdf_file)
            if pdf_path.parent != Path('.') and not pdf_path.parent.exists():
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Created directory: {}", pdf_path.parent)
            
            logger.info("Writing PDF file '{}'", pdf_file)
            with PdfPages(pdf_file) as pdf:
                for fig in figs:
                    pdf.savefig(fig)

        return figs if pdf_file is None else None
        