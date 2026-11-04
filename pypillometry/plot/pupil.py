import itertools
from .generic import GenericPlots
from ..intervals import Intervals
from ..units import parse_time
import numpy as np
from collections.abc import Iterable
from typing import Sequence, Union, List, TypeVar, Optional, Tuple, Callable
from pathlib import Path

from loguru import logger

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

class PupilPlots(GenericPlots):
    """
    Plots of pupil data (pupil size, blinks, foreshortening); part of the
    plotting namespace ``d.plot`` (:class:`~pypillometry.plot.PlotNamespace`),
    e.g. ``d.plot.pupil_plot()``.
    """

    def pupil_plot(self, 
                   plot_range: Tuple[float,float]=(-np.inf, +np.inf),
                   eyes: str|list=[],
                   show_events: bool=True,
                   show_blinks: bool=True,
                   units: str="sec",
                   label_prefix: str="",
                   style: dict=None
            ) -> None:
        """Make a plot of the pupil data using `matplotlib`.

        Has pupil-specific options which makes it different from
        :meth:`plot_timeseries`. All eyes are plotted into the current axes.

        Parameters
        ----------
        plot_range : tuple (start,end)
            plot from start to end (in ms, or with units, e.g.
            ``(3, 5) * u.min``)
        eyes : str or list
            eye(s) to plot; if empty, all eyes with pupil data are plotted
        show_events : bool
            plot events as vertical lines with labels
        show_blinks : bool
            highlight masked or interpolated regions of the pupil data (e.g.,
            detected blinks; the ``"strict"`` mask policy) with a red background
        units : str
            units of the time axis: one of "sec"=seconds (default),
            "ms"=millisec, "min"=minutes, "h"=hours
        label_prefix : str
            Prefix to add to labels in the legend. Useful for overlaying multiple datasets.
            Default is "" (no prefix).
        style : dict or dict of dicts
            Styling for plotted lines. Can be:

            - Single dict: Applied to all eyes with automatic differentiation (e.g., {'color': 'red'})
            - Dict of dicts: Per-eye styling (e.g., {'left': {'linestyle': '-'}, 'right': {'linestyle': '--'}})

            Default is None (uses matplotlib defaults with automatic per-eye colors and linestyles).

        Raises
        ------
        ValueError
            if `plot_range` does not overlap with the data

        Examples
        --------
        Detected blinks are shaded in red, events are shown as vertical lines:

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> d = d.pupil.blinks_detect()
           >>> fig = plt.figure(figsize=(10, 3))
           >>> d.plot.pupil_plot(plot_range=(0, 20) * u.sec)
        """
        self._d._require("pupil")
        eyes = self._d._eyes_with(eyes, "pupil")
        logger.debug("Plotting eyes %s"%eyes)

        # Validate plot_range
        plot_range = self._plot_range_in(plot_range, units)
        self._validate_plot_range(plot_range, units)

        fac=self._d._unit_fac(units)
        logger.debug("Plotting in units %s (fac=%f)"%(units, fac))
        if units=="sec":
            xlab="seconds"
        elif units=="min":
            xlab="minutes"
        elif units=="h":
            xlab="hours"
        else:
            xlab="ms"
        tx=self._d.tx*fac
        evon=self._d.events.onsets*fac

        start,end=plot_range
        if start==-np.inf:
            startix=0
        else:
            startix=np.argmin(np.abs(tx-start))
            
        if end==np.inf:
            endix=tx.size
        else:
            endix=np.argmin(np.abs(tx-end)) + 1  # the end sample is plotted too
                
        ixx=np.logical_and(evon>=start, evon<end)
        evlab=self._d.events.labels[ixx]
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
            
            plt.plot(tx, self._d.data[eye,"pupil"][startix:endix], **plot_kwargs)

        # plot grey lines for events
        if show_events:
            logger.debug("Plotting events, %i in range"%len(evon))
            plt.vlines(evon, *plt.ylim(), color="grey", alpha=0.5)
            ll,ul=plt.ylim()
            for ev,lab in zip(evon,evlab):
                plt.text(ev, ll+(ul-ll)/2., "%s"%lab, fontsize=8, rotation=90)
        
        # highlight if a blink in any of the eyes
        if show_blinks:
            ax = plt.gca()
            for eye in eyes:
                if f"{eye}_pupil" not in self._d.data:
                    continue
                mask = self._d.data.get_policy_mask(f"{eye}_pupil", "strict")
                masked_segment = mask[startix:endix]
                blink_spans = []
                in_blink = False
                span_start = 0
                for idx, value in enumerate(masked_segment):
                    if value and not in_blink:
                        in_blink = True
                        span_start = idx
                    elif not value and in_blink:
                        blink_spans.append((span_start, idx))
                        in_blink = False
                if in_blink:
                    blink_spans.append((span_start, len(masked_segment)))

                for sblink, eblink in blink_spans:
                    ax.axvspan(tx[sblink], tx[min(eblink - 1, len(tx) - 1)], color="red", alpha=0.2)


        plt.legend()
        plt.xlabel(xlab)        


    def pupil_plot_segments(
        self,
        pdf_file: Optional[str] = None,
        interv: float = "1 min",
        figsize=(15, 5),
        ylim=None,
        units: str = "min",
        **kwargs,
    ):
        """
        Plot the whole dataset chunked up into segments (usually to a PDF file).

        Parameters
        ----------
        pdf_file : str or None
            file name to store the PDF; if None, no PDF is written
        interv : float, str or Quantity
            duration of each of the segments to be plotted (in ms, or with
            units, e.g. ``"2 min"``); default 1 min
        figsize : Tuple[int,int]
            dimensions of the figures
        ylim : Tuple[float,float] or None
            y-axis limits for the plots; if None, matplotlib's automatic limits are used
        units : str, optional
            Units of the time axis, passed to :meth:`pupil_plot`. Default "min".
        **kwargs
            arguments passed to :meth:`pupil_plot`

        Returns
        -------
        list of matplotlib.figure.Figure or None
            one figure per segment; None if `pdf_file` is given

        Examples
        --------
        Split the 40-second recording into 10-second pieces (one figure each, here
        only the first is shown):

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> d = d.pupil.blinks_detect()
           >>> figs = d.plot.pupil_plot_segments(interv=10 * u.sec, units="sec",
           ...                                   figsize=(10, 3))
           >>> for f in figs[1:]:
           ...     plt.close(f)

        With ``pdf_file="segments.pdf"``, the figures are written to a PDF file
        instead.
        """

        self._d._require("pupil")
        obj = self._d
        unit_factor = obj._unit_fac(units)
        interv = parse_time(interv, to=units, default="ms")
        start_unit = float(np.min(obj.tx) * unit_factor)
        end_unit = float(np.max(obj.tx) * unit_factor)
        self._check_nsegments(end_unit - start_unit, interv)

        segments=[]
        cstart=start_unit
        cend=start_unit
        while cend < end_unit:
            cend = min(end_unit, cstart + interv)
            segments.append( (cstart,cend) )
            cstart=cend

        figs=[]

        for start,end in segments:
            fig = plt.figure(figsize=figsize)
            self.pupil_plot(plot_range=(start / unit_factor, end / unit_factor), units=units,
                            **kwargs)
            if ylim is not None:
                for ax in fig.axes:
                    ax.set_ylim(*ylim)
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

    def plot_blinks(self, eyes: str|list = [], variables: str|list = "pupil",
                   pdf_file: str|None = None, nrow: int = 5, ncol: int = 3, 
                   figsize: tuple = (10, 10),
                   pre_blink: float = 200, post_blink: float = 200, 
                   units: str = "min", show_index: bool = True) -> list:
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
        show_index : bool
            Show blink index numbers
            
        Returns
        -------
        list
            List of matplotlib Figure objects

        Raises
        ------
        ValueError
            if the object has no data for `variables`

        Examples
        --------
        Check the detected blinks of the left eye (the numbers are the blink
        indices):

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> d = d.pupil.blinks_detect()
           >>> figs = d.plot.plot_blinks(eyes="left", nrow=3, ncol=5, figsize=(12, 6),
           ...                           units="sec")
        """
        self._d._require(*([variables] if isinstance(variables, str) else variables))
        # Get blinks for specified eyes and variables (in ms for consistent padding)
        blinks_result = self._d.get_blinks(eyes=eyes, variables=variables, units="ms")
        
        # Merge all blinks from dict
        from ..intervals import merge_intervals
        all_blinks_list = [b for b in blinks_result.values() if len(b) > 0]
        if not all_blinks_list:
            logger.warning("No blinks to plot")
            return []
        blinks = merge_intervals(*all_blinks_list)
        
        # Pad blinks with pre/post time (pre_blink and post_blink are in ms)
        padded_blinks = blinks.pad(left=pre_blink, right=post_blink)
        
        # Convert to desired display units
        intervals = padded_blinks.to_units(units, self._d)
        
        return self.plot_intervals(intervals, eyes, variables,
                                  pdf_file, nrow, ncol, figsize, show_index=show_index)

    def plot_pupil_foreshortening_error_surface(
        self,
        eyes: str | List[str] = [],
        plot_range: tuple = (-np.inf, +np.inf),
        units: str = "sec",
        show_screen: bool = True,
        cmap: str = "jet",
        gridsize: int = 30,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        min_samples: Optional[int] = None,
        smooth: Optional[float] = None,
        limits: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        """
        Plot a heatmap showing average pupil size across x/y gaze positions.
        
        This visualizes how pupil size varies across different gaze positions,
        which can reveal foreshortening effects (pupils appear smaller when
        looking away from the camera) or other spatial artifacts.
        Each eye is plotted in a separate subplot.
        
        Parameters
        ----------
        eyes : str or list, optional
            Which eye(s) to plot: "left", "right", "average", or a list of these.
            Default is [] which plots all eyes with pupil and gaze data; a
            requested eye without them raises a KeyError.
        plot_range : tuple, optional
            The time range to include in the analysis (in ms, or with
            units, e.g. ``(3, 5) * u.min``). Default is (-np.inf, +np.inf),
            i.e. all data.
        units : str, optional
            Time units of the plot. Default is "sec".
        show_screen : bool, optional
            Whether to plot the screen limits. Default is True.
        cmap : str, optional
            Colormap to use. Default is "jet".
        gridsize : int, optional
            Number of bins in each direction. Default is 30.
        vmin : float, optional
            Minimum value for color scale. If None, uses data minimum.
        vmax : float, optional
            Maximum value for color scale. If None, uses data maximum.
        min_samples : int, optional
            Minimum number of samples required in a bin to display it. Bins with
            fewer samples will not be shown. If None (default), automatically
            calculates a threshold at the 10th percentile of bin counts, effectively
            displaying only the top 90% of bins by sample count.
        smooth : float, optional
            If provided, applies Gaussian smoothing to the binned surface. The value
            is the sigma of the Gaussian kernel in bin units (e.g., smooth=1 applies
            minimal smoothing with a 1-bin sigma, smooth=2 applies more smoothing).
            When smooth is provided, uses a rectangular grid instead of hexbin.
        limits : tuple, optional
            Axis limits as (xmin, xmax, ymin, ymax). If not provided, uses screen
            limits when available (and data falls within), otherwise uses data extent.
            
        Returns
        -------
        None
            Creates a matplotlib figure with the heatmap(s)

        Examples
        --------
        Average pupil size at each gaze position of the left eye:

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> d.plot.plot_pupil_foreshortening_error_surface(eyes="left")

        ``vmin``/``vmax`` set the color range, ``min_samples`` hides bins with
        few samples and ``smooth`` applies Gaussian smoothing.
        """
        self._d._require("pupil", "x", "y")
        obj = self._d
        
        # eyes with pupil and gaze data (KeyError for a requested eye without)
        eyes = obj._eyes_with(eyes, ["pupil", "x", "y"])
        
        # Validate plot_range
        plot_range = self._plot_range_in(plot_range, units)
        self._validate_plot_range(plot_range, units)
        
        # Handle time range
        fac = obj._unit_fac(units)
        tx = obj.tx * fac
        
        start, end = plot_range
        if start == -np.inf:
            startix = 0
        else:
            startix = np.argmin(np.abs(tx - start))
            
        if end == np.inf:
            endix = tx.size
        else:
            endix = np.argmin(np.abs(tx - end)) + 1  # the end sample is plotted too
        
        # Use current figure or create new one if none exists
        nplots = len(eyes)
        fig = plt.gcf()
        
        # Check if current figure is empty/new
        if len(fig.axes) == 0:
            # Create subplots in the current figure
            axs = fig.subplots(1, nplots)
            # Make axs iterable even for single plot
            if not isinstance(axs, Iterable):
                axs = [axs]
        else:
            # Use existing axes
            axs = fig.axes
            if len(axs) != nplots:
                logger.warning(f"Current figure has {len(axs)} axes but {nplots} eyes to plot. Creating new subplots.")
                fig.clear()
                axs = fig.subplots(1, nplots)
                if not isinstance(axs, Iterable):
                    axs = [axs]
        
        # Plot each eye
        for eye, ax in zip(eyes, axs):
            # Get variable names
            vx = f"{eye}_x"
            vy = f"{eye}_y"
            vpupil = f"{eye}_pupil"
            
            # Get masked arrays (automatically filters masked data)
            x_masked = obj[vx][startix:endix]
            y_masked = obj[vy][startix:endix]
            pupil_masked = obj[vpupil][startix:endix]
            
            # Combine masks from all three variables
            combined_mask = (
                np.ma.getmaskarray(x_masked) | 
                np.ma.getmaskarray(y_masked) | 
                np.ma.getmaskarray(pupil_masked)
            )
            
            # Extract only non-masked data
            x_plot = x_masked.data[~combined_mask]
            y_plot = y_masked.data[~combined_mask]
            pupil_plot = pupil_masked.data[~combined_mask]
            
            # Check if we have any data to plot
            if len(x_plot) == 0:
                ax.text(0.5, 0.5, f"No non-masked data\navailable for '{eye}' eye", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{eye} eye')
                continue
            
            # Determine limits
            if limits is not None:
                xmin, xmax, ymin, ymax = limits
            else:
                # Use screen limits if available and data falls within, else use data extent
                if obj.experimental_setup is not None and obj.experimental_setup.has_screen_info():
                    setup = obj.experimental_setup
                    xmin = max(x_plot.min(), setup.screen_xlim[0])
                    xmax = min(x_plot.max(), setup.screen_xlim[1])
                    ymin = max(y_plot.min(), setup.screen_ylim[0])
                    ymax = min(y_plot.max(), setup.screen_ylim[1])
                else:
                    xmin, xmax = x_plot.min(), x_plot.max()
                    ymin, ymax = y_plot.min(), y_plot.max()
            
            if smooth is not None:
                # Use rectangular binning with Gaussian smoothing
                # Compute binned statistics (mean pupil per bin)
                stat, xedges, yedges, binnumber = binned_statistic_2d(
                    x_plot, y_plot, pupil_plot,
                    statistic='mean', 
                    bins=gridsize,
                    range=[[xmin, xmax], [ymin, ymax]]
                )
                # Also get counts to mask bins with insufficient samples
                counts, _, _, _ = binned_statistic_2d(
                    x_plot, y_plot, pupil_plot,
                    statistic='count',
                    bins=gridsize,
                    range=[[xmin, xmax], [ymin, ymax]]
                )
                
                # Determine min_samples threshold for masking
                if min_samples is None:
                    valid_counts = counts[counts > 0]
                    if len(valid_counts) > 0:
                        mincnt = int(np.percentile(valid_counts, 10))
                        mincnt = max(1, mincnt)
                        logger.info(f"[{eye} eye] Auto-calculated min_samples threshold: {mincnt} "
                                   f"(10th percentile of {len(valid_counts)} bins with counts ranging "
                                   f"{int(valid_counts.min())}-{int(valid_counts.max())})")
                    else:
                        mincnt = 1
                else:
                    mincnt = min_samples
                
                # Mask bins with insufficient samples
                stat_masked = np.where(counts >= mincnt, stat, np.nan)
                
                # Apply Gaussian smoothing (only to valid values)
                # Create a mask for valid data
                valid_mask = ~np.isnan(stat_masked)
                stat_filled = np.where(valid_mask, stat_masked, 0)
                
                # Smooth the data and the mask separately, then normalize
                smoothed_data = gaussian_filter(stat_filled.T, sigma=smooth)
                smoothed_mask = gaussian_filter(valid_mask.astype(float).T, sigma=smooth)
                
                # Normalize to get proper weighted average, mask where no data
                with np.errstate(divide='ignore', invalid='ignore'):
                    stat_smooth = np.where(smoothed_mask > 0.01, 
                                          smoothed_data / smoothed_mask, 
                                          np.nan)
                
                im = ax.imshow(
                    stat_smooth,
                    origin='lower',
                    extent=[xmin, xmax, ymin, ymax],
                    cmap=cmap,
                    aspect='auto',
                    vmin=vmin,
                    vmax=vmax
                )
            else:
                # Determine min_samples threshold
                if min_samples is None:
                    # Create temporary hexbin to get bin counts
                    temp_hexbin = ax.hexbin(
                        x_plot, 
                        y_plot, 
                        gridsize=gridsize, 
                        mincnt=1
                    )
                    counts = temp_hexbin.get_array()
                    # Use 10th percentile as threshold (keeps top 90%)
                    mincnt = int(np.percentile(counts, 10))
                    mincnt = max(1, mincnt)  # Ensure at least 1
                    logger.info(f"[{eye} eye] Auto-calculated min_samples threshold: {mincnt} "
                               f"(10th percentile of {len(counts)} bins with counts ranging {int(counts.min())}-{int(counts.max())})")
                    # Clear the temporary plot
                    ax.clear()
                else:
                    mincnt = min_samples
                
                # Create hexbin plot with average pupil size
                im = ax.hexbin(
                    x_plot, 
                    y_plot, 
                    C=pupil_plot,
                    gridsize=gridsize, 
                    cmap=cmap,
                    reduce_C_function=np.mean,
                    mincnt=mincnt,
                    vmin=vmin,
                    vmax=vmax
                )
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_label('Avg Pupil Size')
            
            # Set title and labels
            ax.set_title(f'{eye} eye')
            ax.set_xlabel('Gaze X Position')
            ax.set_ylabel('Gaze Y Position')
            ax.set_aspect('equal')
            
            # Set axis limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)  # pixel y points down, as on the screen
            
            # Plot screen boundaries
            if show_screen and obj.experimental_setup is not None and obj.experimental_setup.has_screen_info():
                setup = obj.experimental_setup
                screenrect = patches.Rectangle(
                    (setup.screen_xlim[0], setup.screen_ylim[0]), 
                    setup.screen_xlim[1] - setup.screen_xlim[0],
                    setup.screen_ylim[1] - setup.screen_ylim[0],
                    fill=False, 
                    edgecolor="red", 
                    linewidth=2
                )
                ax.add_patch(screenrect)
