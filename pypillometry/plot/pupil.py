from .generic import GenericPlots
from ._axes import get_ax, get_axes, write_pdf
from ..units import parse_time, _ms_to
import numpy as np
from typing import List, Optional, Tuple

from loguru import logger


class PupilPlots(GenericPlots):
    """
    Plots of pupil data (pupil size, blinks, foreshortening); part of the
    plotting namespace ``d.plot`` (:class:`~pypillometry.plot.PlotNamespace`),
    e.g. ``d.plot.plot_pupil()``.
    """

    def plot_pupil(self, 
                   plot_range: Tuple[float,float]=(-np.inf, +np.inf),
                   eyes: str|list=[],
                   show_events: bool=True,
                   show_blinks: bool=True,
                   units: str="sec",
                   label_prefix: str="",
                   style: dict=None,
                   highlight=None,
                   highlight_color="lightblue",
                   title: str = None,
                   ax=None,
                   **kwargs):
        """Make a plot of the pupil data using `matplotlib`.

        Has pupil-specific options which makes it different from
        :meth:`plot_timeseries`. All eyes are plotted into one axes (`ax`,
        or the current axes).

        Parameters
        ----------
        plot_range : tuple (start,end)
            plot from start to end (plain numbers are in `units`, or with
            units, e.g. ``(3, 5) * u.min``)
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
        highlight : Intervals, Epochs or list of them, optional
            periods to shade in the background (e.g., trials from
            :meth:`~pypillometry.EyeData.get_intervals`), converted to the
            units of the time axis
        highlight_color : color or list of colors, optional
            color of the shading: one for all or one per element of
            `highlight`; default "lightblue"
        title : str, optional
            title of the plot
        ax : matplotlib.axes.Axes, optional
            axes to plot on; default: the current axes
        **kwargs
            passed to :meth:`matplotlib.axes.Axes.plot` for all eyes (after
            `style`)

        Returns
        -------
        matplotlib.axes.Axes
            the axes of the plot

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
           >>> fig, ax = plt.subplots(figsize=(10, 3))
           >>> ax = d.plot.plot_pupil(plot_range=(0, 20) * u.sec, ax=ax)

        Shade the trials (feedback to response) and the first second after
        each stimulus in different colors:

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> trials = d.get_intervals(start="F_", end="R_")
           >>> stim = d.get_epochs("S_", (0, 1000) * u.ms)
           >>> fig, ax = plt.subplots(figsize=(10, 3))
           >>> ax = d.plot.plot_pupil(plot_range=(0, 20) * u.sec, show_blinks=False,
           ...                        highlight=[trials, stim],
           ...                        highlight_color=["lightgrey", "green"], ax=ax)
        """
        self._d._require("pupil")
        eyes = self._d._eyes_with(eyes, "pupil")
        logger.debug("Plotting eyes %s"%eyes)

        # Validate plot_range
        plot_range = self._plot_range_in(plot_range, units)
        self._validate_plot_range(plot_range, units)

        logger.debug(f"Plotting in units {units}")
        if units=="sec":
            xlab="seconds"
        elif units=="min":
            xlab="minutes"
        elif units=="h":
            xlab="hours"
        else:
            xlab="ms"
        tx = _ms_to(self._d.tx, units)
        evon = _ms_to(self._d.events.onsets, units)

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
        ax = get_ax(ax)

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
            plot_kwargs.update(kwargs)

            ax.plot(tx, self._d.data[eye,"pupil"][startix:endix], **plot_kwargs)

        # plot grey lines for events
        if show_events:
            logger.debug("Plotting events, %i in range"%len(evon))
            ax.vlines(evon, *ax.get_ylim(), color="grey", alpha=0.5)
            ll,ul=ax.get_ylim()
            for ev,lab in zip(evon,evlab):
                ax.text(ev, ll+(ul-ll)/2., "%s"%lab, fontsize=8, rotation=90)
        
        # highlight if a blink in any of the eyes
        if show_blinks:
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


        if len(tx):
            self._draw_highlight([ax], highlight, highlight_color, units,
                                 (tx[0], tx[-1]))

        ax.legend()
        ax.set_xlabel(xlab)
        if title is not None:
            ax.set_title(title)
        return ax


    def plot_pupil_segments(
        self,
        pdf_file: Optional[str] = None,
        interv: float = "1 min",
        figsize=(15, 5),
        ylim=None,
        units: str = "sec",
        **kwargs,
    ):
        """
        Plot the whole dataset chunked up into segments (usually to a PDF file).

        Parameters
        ----------
        pdf_file : str or None
            file name to store the PDF; if None, no PDF is written
        interv : float, str or Quantity
            duration of each of the segments to be plotted (plain numbers
            are in `units`, or with units, e.g. ``"2 min"``); default 1 min
        figsize : Tuple[int,int]
            dimensions of the figures
        ylim : Tuple[float,float] or None
            y-axis limits for the plots; if None, matplotlib's automatic limits are used
        units : str, optional
            Units of the time axis, passed to :meth:`plot_pupil`. Default "sec".
        **kwargs
            arguments passed to :meth:`plot_pupil` (e.g., `highlight` and
            `highlight_color`)

        Returns
        -------
        list of matplotlib.figure.Figure
            one figure per segment (also returned when `pdf_file` is given;
            the figures are then closed)

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
           >>> figs = d.plot.plot_pupil_segments(interv=10 * u.sec, units="sec",
           ...                                   figsize=(10, 3))
           >>> for f in figs[1:]:
           ...     plt.close(f)

        With ``pdf_file="segments.pdf"``, the figures are written to a PDF file
        instead.
        """
        import matplotlib.pyplot as plt
        self._d._require("pupil")
        obj = self._d
        interv = parse_time(interv, to=units, default=units)
        start_unit = float(_ms_to(np.min(obj.tx), units))
        end_unit = float(_ms_to(np.max(obj.tx), units))
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
            fig, ax = plt.subplots(figsize=figsize)
            self.plot_pupil(plot_range=(start, end), units=units, ax=ax, **kwargs)
            if ylim is not None:
                ax.set_ylim(*ylim)
            figs.append(fig)
            
            # Close figure immediately after creation if saving to PDF to prevent display and memory issues
            if pdf_file is not None:
                plt.close(fig)

        if pdf_file is not None:
            write_pdf(figs, pdf_file)
        return figs

    def plot_blinks(self, eyes: str|list = [], variables: str|list = "pupil",
                   pdf_file: str|None = None, nrow: int = 5, ncol: int = 3, 
                   figsize: tuple = (10, 10),
                   pre_blink: float = 200, post_blink: float = 200, 
                   units: str = "sec", show_index: bool = True) -> list:
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
            Display units for time axis; default "sec"
        show_index : bool
            Show blink index numbers
            
        Returns
        -------
        list of matplotlib.figure.Figure
            one figure per `nrow` x `ncol` blinks (also returned when
            `pdf_file` is given; the figures are then closed)

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
        from ..intervals import concat_intervals
        all_blinks_list = [b for b in blinks_result.values() if len(b) > 0]
        if not all_blinks_list:
            logger.warning("No blinks to plot")
            return []
        blinks = concat_intervals(*all_blinks_list)
        
        # Pad blinks with pre/post time (pre_blink and post_blink are in ms)
        padded_blinks = blinks.pad(left=pre_blink, right=post_blink)
        
        # Convert to desired display units
        intervals = padded_blinks.to_units(units, self._d)
        
        return self.plot_intervals(intervals, eyes, variables,
                                  pdf_file, nrow, ncol, figsize, show_index=show_index)

    def plot_pupil_foreshortening_error_surface(
        self,
        plot_range: tuple = (-np.inf, +np.inf),
        eyes: str | List[str] = [],
        units: str = "sec",
        show_screen: bool = True,
        cmap: str = "jet",
        gridsize: int = 30,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        min_samples: Optional[int] = None,
        smooth: Optional[float] = None,
        limits: Optional[Tuple[float, float, float, float]] = None,
        title: str = None,
        axes=None,
        figsize: tuple = None,
    ):
        """
        Plot a heatmap showing average pupil size across x/y gaze positions.
        
        This visualizes how pupil size varies across different gaze positions,
        which can reveal foreshortening effects (pupils appear smaller when
        looking away from the camera) or other spatial artifacts.
        Each eye is plotted in a separate subplot.
        
        Parameters
        ----------
        plot_range : tuple, optional
            The time range to include in the analysis (plain numbers are
            in `units`, or with units, e.g. ``(3, 5) * u.min``). Default is (-np.inf, +np.inf),
            i.e. all data.
        eyes : str or list, optional
            Which eye(s) to plot: "left", "right", "average", or a list of these.
            Default is [] which plots all eyes with pupil and gaze data; a
            requested eye without them raises a KeyError.
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
        title : str, optional
            title of the figure
        axes : list of matplotlib.axes.Axes, optional
            axes to plot on, one per eye; default: a new figure (one
            subplot per eye, side by side)
        figsize : tuple, optional
            size of the new figure

        Returns
        -------
        matplotlib.figure.Figure
            the figure of the plot (``fig.axes`` also contains the
            colorbars)

        Examples
        --------
        Average pupil size at each gaze position of the left eye:

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> fig = d.plot.plot_pupil_foreshortening_error_surface(eyes="left")

        ``vmin``/``vmax`` set the color range, ``min_samples`` hides bins with
        few samples and ``smooth`` applies Gaussian smoothing.
        """
        import matplotlib.patches as patches
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from scipy.ndimage import gaussian_filter
        from scipy.stats import binned_statistic_2d
        self._d._require("pupil", "x", "y")
        obj = self._d
        
        # eyes with pupil and gaze data (KeyError for a requested eye without)
        eyes = obj._eyes_with(eyes, ["pupil", "x", "y"])
        
        # Validate plot_range
        plot_range = self._plot_range_in(plot_range, units)
        self._validate_plot_range(plot_range, units)
        
        # Handle time range
        tx = _ms_to(obj.tx, units)
        
        start, end = plot_range
        if start == -np.inf:
            startix = 0
        else:
            startix = np.argmin(np.abs(tx - start))
            
        if end == np.inf:
            endix = tx.size
        else:
            endix = np.argmin(np.abs(tx - end)) + 1  # the end sample is plotted too
        
        nplots = len(eyes)
        fig, axs = get_axes(axes, nplots, "eye", figsize, nrows=1, ncols=nplots)
        
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
            pupil_values = pupil_masked.data[~combined_mask]
            
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
                    x_plot, y_plot, pupil_values,
                    statistic='mean', 
                    bins=gridsize,
                    range=[[xmin, xmax], [ymin, ymax]]
                )
                # Also get counts to mask bins with insufficient samples
                counts, _, _, _ = binned_statistic_2d(
                    x_plot, y_plot, pupil_values,
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
                    C=pupil_values,
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
        if title is not None:
            fig.suptitle(title)
        return fig
