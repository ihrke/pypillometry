from .generic import GenericPlots
from ._axes import get_ax, get_axes, write_pdf
from ..units import parse_time, _ms_to
import warnings

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
                    ax.axvspan(tx[sblink], tx[min(eblink - 1, len(tx) - 1)], color="red", alpha=0.1,
                               lw=0, zorder=0)


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

    def plot_pupil_spectrum(self, eyes: str|list = [], variable: str = "pupil",
                            window="200 s", scale: str = "log%", per_decade: int = 20,
                            policy="default", ax=None, title: str|None = None, **kwargs):
        """
        Plot the power spectrum of the pupil size on logarithmic axes.

        The spectra come from
        :meth:`d.pupil.spectrum() <pypillometry.eyedata.pupil.PupilNamespace.spectrum>`
        (Welch's method; windows with many unusable samples are skipped)
        and are averaged in log-spaced frequency bands. A power law
        (e.g., 1/f) is a straight line; responses that are locked to a
        fixed rhythm of the trials show up as narrow peaks.

        Parameters
        ----------
        eyes : str or list
            eyes to plot; if empty, all eyes with the variable
        variable : str
            variable to use (default ``"pupil"``)
        window : float, str or Quantity
            length of the windows (plain numbers are ms; default 200 s)
        scale : {"log%", "original"}
            ``"log%"`` (default): spectrum of the change relative to the
            median in percent, :math:`100 \\log(x / \\mathrm{median})`;
            ``"original"``: of the signal in its units
        per_decade : int or None
            number of frequency bands per factor of 10 (default 20); None:
            all frequencies
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are usable; default
            ``"default"``
        ax : matplotlib Axes, optional
            axes to draw on; default: the current axes
        title : str, optional
            title of the plot
        **kwargs
            passed to ``ax.plot``

        Returns
        -------
        matplotlib Axes

        Examples
        --------
        .. plot::

           >>> import pypillometry as pp
           >>> d = pp.get_example_data("rlmw_002").pupil.blinks_detect()
           >>> ax = d.plot.plot_pupil_spectrum(window="100 s")
        """
        from ..signal.spectral import plot_spectrum
        ax = get_ax(ax)
        spectra = self._d.pupil.spectrum(eyes, variable, window=window, scale=scale,
                                         policy=policy)
        plot_spectrum(spectra, ax=ax, per_decade=per_decade, **kwargs)
        unit = "log-%" if scale == "log%" else "units"
        ax.set_ylabel(f"power spectral density ({unit}$^2$/Hz)")
        if title is not None:
            ax.set_title(title)
        return ax

    def plot_blink_clusters(self, eyes: str|list = [], window=(200, 500),
                            n_clusters: int = 3, min_size: float = 0.02,
                            figsize: tuple = None, title: str = None, axes=None,
                            seed=None):
        """
        Plot the detected blinks in groups of similar shape.

        A quick check of :meth:`~pypillometry.EyeData.pupil.blinks_detect`:
        if a group looks unlike blinks (e.g., no drop and rise, steps), the
        detection finds too much and its parameters should be changed.

        Blinks with missing samples: the pupil before and after the missing
        samples (the core, from the first to the last missing sample) is cut
        out with `window`. Each part is shifted to 0 at its outer end (the
        first fifth of the time before the core, the last fifth of the time
        after it), and the blinks are clustered (Ward, on the parts and the
        log of the duration of the core). The tree is cut so that there are
        `n_clusters` groups with at least `min_size` of these blinks; the
        rest are shown together as "other".

        Blinks without missing samples (found in the velocity profile, e.g.
        blinks that do not close the eye completely, or eye movements) are
        shown separately, all in one panel: the pupil in `window` around the
        middle of the blink, shifted to 0 at the start.

        One row per eye: one panel per group (the largest first), one for
        "other" and one for the blinks without missing samples. The panels
        of the groups show all blinks with missing samples in light grey,
        the blinks of the group in color and their median in black; the time
        before the core is relative to its start, the time after it relative
        to its end, and the cores (of different durations) are the shaded
        gap. Below each panel, a histogram of the durations of the cores
        (the time the eye was closed) of the group, with the one of all
        blinks with missing samples as outline (both normalized; the last
        bin has all durations above the 99th percentile), and of the
        durations of the blinks without missing samples. The percentages are
        of all detected blinks of the eye.

        Parameters
        ----------
        eyes : str or list
            eyes to plot; if empty, all eyes with pupil data
        window : float, str, Quantity or tuple
            time `(before, after)` the core, or the middle of a blink
            without missing samples (in ms, or with units)
        n_clusters : int
            number of groups (besides "other")
        min_size : float
            minimum fraction of the blinks with missing samples in a group
        figsize : tuple, optional
            size of the figure; default: 4 x 4.5 inches per panel
        title : str, optional
            title of the figure
        axes : sequence of Axes, optional
            axes to draw on, `len(eyes) * (n_clusters + 2)` (one row per
            eye); default: a new figure
        seed : int, optional
            seed for the random subset of blinks that is clustered if there
            are more than 5000 (the others go to the nearest group)

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            if the object has no pupil data or no detected blinks

        Examples
        --------
        .. plot::

           >>> import pypillometry as pp
           >>> d = pp.get_example_data("rlmw_002_short").pupil.blinks_detect()
           >>> fig = d.plot.plot_blink_clusters(n_clusters=2)
        """
        d = self._d
        d._require("pupil")
        eyes = d._eyes_with(eyes, "pupil")
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")
        if not 0 <= min_size < 1:
            raise ValueError(f"min_size must be in [0, 1), got {min_size}")
        pre, post = d._samples(window, rounding="up") if np.ndim(window) else \
            (d._samples(window, rounding="up"),) * 2
        if min(pre, post) < 5:
            raise ValueError(f"window must be at least 5 samples on each side, got {window}")
        blinks = {eye: np.array([(int(a), int(b)) for a, b in
                                 d.get_blinks(eye, "pupil", units="samples")[f"{eye}_pupil"]],
                                dtype=int).reshape(-1, 2) for eye in eyes}
        if all(len(b) == 0 for b in blinks.values()):
            raise ValueError("no detected blinks: detect the blinks first (blinks_detect())")

        ncol = n_clusters + 2
        fig, axs = get_axes(axes, len(eyes) * ncol, "group of an eye",
                            figsize=figsize or (4 * ncol, 4.5 * len(eyes)),
                            nrows=len(eyes), ncols=ncol)
        rng = np.random.default_rng(seed)
        ms = 1000 / d.fs
        gap = 0.2 * (pre + post) * ms     # width of the gap in the plot
        tpre = (np.arange(pre) - pre) * ms
        tpost = np.arange(post) * ms + gap
        tmid = np.arange(-pre, post) * ms
        # round ticks, the same on both sides of the gap
        step = next(v for v in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, np.inf)
                    if v >= max(pre, post) * ms / 4)
        step = step if np.isfinite(step) else max(pre, post) * ms / 4
        ticks_pre = -np.arange(0, pre * ms + 1e-9, step)[::-1]
        ticks_post = np.arange(0, post * ms + 1e-9, step)

        def lines(ax, t, parts, **kw):
            """all rows of `parts` as one line (NaN-separated)"""
            if len(parts):
                y = np.c_[parts, np.full(len(parts), np.nan)].ravel()
                ax.plot(np.tile(np.r_[t, np.nan], len(parts)), y, **kw)

        def hist(ax, values, background, label):
            """histogram of `values` (ms) below `ax`, with the one of
            `background` as outline, both normalized; log-spaced bins up to
            the 99th percentile of `background`, the last bin also has the
            larger values"""
            from matplotlib.ticker import FuncFormatter, NullFormatter
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            hax = make_axes_locatable(ax).append_axes("bottom", size="30%", pad=0.45)
            pos = background[background > 0]
            if len(pos):
                lo, hi = pos.min(), np.percentile(pos, 99)
                if hi <= lo:
                    lo, hi = lo / 1.5, lo * 1.5
                bins = np.geomspace(lo, hi, 26)
                over = (pos > hi).any()
                if over:   # one more bin for the larger values
                    bins = np.r_[bins, hi * (bins[1] / bins[0])]
                clip = lambda v: np.minimum(v, bins[-1] * 0.999)
                hax.hist(clip(pos), bins, density=True, histtype="step", color="0.6", lw=0.8)
                values = values[values > 0]
                if len(values):
                    hax.hist(clip(values), bins, density=True, alpha=0.7,
                             color=ax.lines[-1].get_color() if ax.lines else "C0")
                hax.set_xscale("log")
                hax.set_xlim(bins[0], bins[-1])
                if over:
                    hax.axvline(hi, color="0.4", lw=0.6, ls=":")
                    label += f" (last bin: > {hi:.0f})"
                hax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                hax.xaxis.set_minor_formatter(NullFormatter())
            hax.set_yticks([])
            hax.tick_params(axis="x", labelsize=8)
            hax.set_xlabel(label, fontsize=8)
            return hax

        def ylim(ax, *parts):
            values = np.concatenate([p.ravel() for p in parts])
            if np.isfinite(values).any():
                lo, hi = np.nanpercentile(values, [0.5, 99.5])
                if hi > lo:
                    ax.set_ylim(lo, hi)

        for row, eye in enumerate(eyes):
            ax_row = axs[row * ncol:(row + 1) * ncol]
            sy = d.data[f"{eye}_pupil"]
            nall = max(1, len(blinks[eye]))
            has_core = _has_missing(sy, blinks[eye])
            before, after, core = _blink_segments(sy, blinks[eye][has_core], pre, post)
            labels = _cluster_blinks(before, after, core, n_clusters, min_size,
                                     max(1, int(round(10 / ms))), rng)
            for g, ax in zip(list(range(n_clusters)) + [-1], ax_row):
                ax.axvspan(0, gap, color="0.93", lw=0)
                ax.axhline(0, color="0.6", lw=0.5)
                lines(ax, tpre, before, color="0.85", lw=0.3)
                lines(ax, tpost, after, color="0.85", lw=0.3)
                sel = labels == g
                name = "other" if g < 0 else f"group {g + 1}"
                if sel.any():
                    color = "C3" if g < 0 else f"C{g}"
                    lines(ax, tpre, before[sel], color=color, lw=0.4, alpha=0.5)
                    lines(ax, tpost, after[sel], color=color, lw=0.4, alpha=0.5)
                    if g >= 0:
                        ax.plot(tpre, np.nanmedian(before[sel], 0), color="k", lw=2)
                        ax.plot(tpost, np.nanmedian(after[sel], 0), color="k", lw=2)
                    ax.plot([], [], color=color)    # the color of the histogram
                    ax.set_title(f"{eye}, {name}: {sel.sum()} ({100 * sel.sum() / nall:.0f}%)\n"
                                 f"core {np.median(core[sel]) * ms:.0f} ms (median)", fontsize=9)
                else:
                    ax.set_title(f"{eye}, {name}: none", fontsize=9)
                ax.set_xticks(np.r_[ticks_pre, ticks_post + gap])
                ax.set_xticklabels([f"{t + 0:g}" for t in np.r_[ticks_pre, ticks_post]],
                                   fontsize=8)   # + 0: no "-0"
                ax.set_xlim(tpre[0], tpost[-1])
                ylim(ax, before, after)
                ax.set_xlabel("ms before | after the core", fontsize=8)
                hist(ax, core[sel] * ms if sel.any() else np.array([]), core * ms,
                     "core (closed) duration, ms")

            # blinks without missing samples
            ax = ax_row[-1]
            mid = _centered_segments(sy, blinks[eye][~has_core], pre, post)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.axvline(0, color="0.6", lw=0.5)
            name = f"{eye}, without missing samples"
            if len(mid):
                lines(ax, tmid, mid, color="C4", lw=0.4, alpha=0.5)
                ax.plot(tmid, np.nanmedian(mid, 0), color="k", lw=2)
                width = np.median(np.diff(blinks[eye][~has_core], axis=1)) * ms
                ax.set_title(f"{name}: {len(mid)} ({100 * len(mid) / nall:.0f}%)\n"
                             f"{width:.0f} ms long (median)", fontsize=9)
                ylim(ax, mid)
            else:
                ax.set_title(f"{name}: none", fontsize=9)
            ax.set_xlim(tmid[0], tmid[-1])
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel("ms from the middle of the blink", fontsize=8)
            widths = np.diff(blinks[eye][~has_core], axis=1).ravel() * ms
            ax.plot([], [], color="C4")
            hist(ax, widths, widths, "blink duration, ms")
            ax_row[0].set_ylabel(f"{eye} pupil (shifted)\n"
                                 f"{100 * (~has_core).sum() / nall:.0f}% without missing samples")
        if title:
            fig.suptitle(title)
        if axes is None:
            fig.tight_layout()
        return fig

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


def _has_missing(sy, blinks):
    """True for the `blinks` (n x 2 sample indices, end-exclusive) with
    missing (NaN or 0) samples"""
    sy = np.asarray(sy, dtype=float)
    miss = np.r_[0, np.cumsum(np.isnan(sy) | (sy == 0))]
    blinks = np.asarray(blinks, dtype=int).reshape(-1, 2)
    return miss[blinks[:, 1]] - miss[blinks[:, 0]] > 0


def _shift(part, columns):
    """`part` (in place) shifted to 0 at the mean of `columns` (or of all
    columns if they are missing)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nanmean(part[:, columns], axis=1)
        level = np.where(np.isnan(level), np.nanmean(part, axis=1), level)
    part -= level[:, None]


def _blink_segments(sy, blinks, pre, post):
    """
    The `pre` samples before and `post` samples after the core of each blink
    (`blinks`: n x 2 sample indices, end-exclusive). The core is the stretch
    from the first to the last missing (NaN or 0) sample of the blink.
    Missing samples, samples outside the data and blinks without missing
    samples are NaN. Each part is shifted to 0 at its outer end (a fifth of
    it). Returns the parts before (n x pre) and after (n x post) and the
    number of samples of the cores (0: no missing samples).
    """
    sy = np.asarray(sy, dtype=float)
    sy = np.where(sy == 0, np.nan, sy)
    n = len(sy)
    before, after = np.full((len(blinks), pre), np.nan), np.full((len(blinks), post), np.nan)
    core = np.zeros(len(blinks), dtype=int)
    for i, (a, b) in enumerate(blinks):
        miss = np.flatnonzero(np.isnan(sy[a:b]))
        if miss.size == 0:
            continue
        ca, cb = a + miss[0], a + miss[-1] + 1
        core[i] = cb - ca
        lo, hi = max(0, ca - pre), min(n, cb + post)
        before[i, pre - (ca - lo):] = sy[lo:ca]
        after[i, :hi - cb] = sy[cb:hi]
    _shift(before, slice(0, max(1, pre // 5)))
    _shift(after, slice(-max(1, post // 5), None))
    return before, after, core


def _centered_segments(sy, blinks, pre, post):
    """
    The `pre` samples before and `post` samples from the middle of each
    blink (`blinks`: n x 2 sample indices, end-exclusive), shifted to 0 at
    the start (a fifth of `pre`); missing samples and samples outside the
    data are NaN. Returns an array (n x (pre + post)).
    """
    sy = np.asarray(sy, dtype=float)
    sy = np.where(sy == 0, np.nan, sy)
    n = len(sy)
    out = np.full((len(blinks), pre + post), np.nan)
    for i, (a, b) in enumerate(blinks):
        m = (a + b) // 2
        lo, hi = max(0, m - pre), min(n, m + post)
        out[i, lo - (m - pre):hi - (m - pre)] = sy[lo:hi]
    _shift(out, slice(0, max(1, pre // 5)))
    return out


def _cluster_blinks(before, after, core, n_clusters, min_size, step, rng,
                    max_n=5000):
    """
    Groups of blinks (see :meth:`PupilPlots.plot_blink_clusters`): Ward
    clustering of the parts `before` and `after` the cores (means over
    `step` samples, missing values replaced by the median of all blinks,
    robustly scaled) and the log of the duration of the `core`. Returns the
    group of each blink: 0 to `n_clusters - 1` (largest first) for the
    groups with at least `min_size` of the blinks, -1 for the others. If
    there are more than `max_n` blinks, a random subset (`rng`) is
    clustered and the others go to the group with the nearest mean.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    n = len(core)
    if n < 2:
        return np.zeros(n, dtype=int)

    def blocks(x, right):
        m = x.shape[1] // step * step
        x = x[:, x.shape[1] - m:] if right else x[:, :m]
        return x.reshape(len(x), -1, step)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X = np.c_[np.nanmean(blocks(before, True), axis=2), np.nanmean(blocks(after, False), axis=2)]
        med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0, med)
    X = np.where(np.isnan(X), med, X)
    scale = np.median(np.abs(X - med))
    X = X / (scale if scale > 0 else 1)
    # the duration counts about as much as a third of the shape
    dur = np.log1p(core / step)
    sd = dur.std()
    X = np.c_[X, dur / (sd if sd > 0 else 1) * np.sqrt(X.shape[1]) / 3]

    sub = np.sort(rng.choice(n, max_n, replace=False)) if n > max_n else np.arange(n)
    tree = linkage(X[sub], "ward")
    need = max(2, int(np.ceil(min_size * len(sub))))
    for k in range(min(n_clusters, len(sub)), len(sub) + 1):
        lab = fcluster(tree, k, "maxclust")
        counts = np.bincount(lab)
        if (counts >= need).sum() >= n_clusters or k >= 100:
            break
    big = [c for c in np.argsort(-counts, kind="stable") if counts[c] >= need][:n_clusters]
    groups = np.full(lab.max() + 1, -1)
    groups[big] = np.arange(len(big))
    if len(sub) == n:
        return groups[lab]
    # the others: the cluster with the nearest mean
    ids = np.unique(lab)
    means = np.array([X[sub][lab == c].mean(axis=0) for c in ids])
    nearest = ids[np.argmin(((X[:, None, :] - means[None]) ** 2).sum(axis=2), axis=1)]
    out = groups[nearest]
    out[sub] = groups[lab]
    return out
