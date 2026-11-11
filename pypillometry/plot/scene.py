"""
Plots for data from head-mounted eye trackers (with a scene video), see
:class:`~pypillometry.HeadMountedEyeData`.
"""
import numpy as np

from ..convenience import requires_package
from ..units import _ms_to, parse_time
from ._axes import get_ax, get_axes
from .gaze import GazePlots


class ScenePlots(GazePlots):
    """
    Plots for head-mounted eye trackers: with the scene video and with the
    additional channels in ``d.aux``.
    """

    def _aux_channels(self, name: str):
        """
        The channels in `aux` that `name` stands for: the channel with this
        name, or all channels whose name starts with ``<name>_`` (e.g.,
        "gyro": gyro_x, gyro_y, gyro_z) or ends with ``_<name>`` (e.g.,
        "luminance": scene_luminance, gaze_luminance); None if there are none.
        """
        aux = self._d.aux
        if name in aux:
            return [name]
        return [key for key in aux.keys()
                if key.startswith(name + "_") or key.endswith("_" + name)] or None

    def plot_timeseries(self,
            plot_range: tuple=(-np.inf, +np.inf),
            variables: list=[],
            eyes: list=[],
            units: str="sec",
            show_onsets: str="line",
            show_masked: bool=True,
            show_mask_highlight: bool=True,
            label_prefix: str="",
            style: dict=None,
            highlight=None,
            highlight_color="lightblue",
            title: str=None,
            axes=None,
            figsize: tuple=None,
            **kwargs):
        """
        Plot a part of the data. Each variable is plotted in a separate
        subplot (one below the other), all eyes in the same subplot.

        As :meth:`PlotNamespace.plot_timeseries
        <pypillometry.plot.PlotNamespace.plot_timeseries>`, and `variables`
        can also name channels in ``d.aux``:

        - the name of a channel (e.g., ``"gaze_luminance"``)
        - the first or the last part of the names of several channels,
          which are plotted in one subplot (e.g., ``"gyro"`` for
          ``gyro_x``, ``gyro_y`` and ``gyro_z``; ``"luminance"`` for
          ``scene_luminance`` and ``gaze_luminance``)

        Parameters
        ----------
        plot_range : tuple
            The time range to plot; plain numbers are in `units`, values with
            units are converted (e.g. ``(0, 20) * u.sec``). Default is
            (-np.inf, +np.inf), i.e. all data.
        variables : list
            The data to plot, one subplot each in this order: variables of
            the eyes ("x", "y", "pupil") and channels in ``d.aux`` (see
            above). Default is [], which means all variables of the eyes
            (and no channels of ``d.aux``).
        eyes : list
            The eyes to plot. Default is [], which means all available data
            ("left", "right", "both", ...)
        units : str
            The units of the time axis and of plain numbers in `plot_range`
            ("ms", "sec", "min", "h"). Default "sec".
        show_onsets : str
            Whether to plot markers for the event onsets. One of "line" (vertical lines),
            "label" (text label), "both" (both lines and labels), or "none" (no markers).
            Default is "line".
        show_masked : bool
            Whether to plot the data of the eyes in masked regions (default True).
        show_mask_highlight : bool
            Whether to highlight masked regions of the data of the eyes
            with a light red background. Default is True.
        label_prefix : str
            Prefix to add to labels in the legend. Default is "" (no prefix).
        style : dict or dict of dicts
            Styling for the lines of the eyes: one dict for all eyes or one
            dict per eye (e.g., ``{'left': {'linestyle': '-'}}``).
        highlight : Intervals, Epochs or list of them, optional
            periods to shade in the background, converted to the units of
            the time axis
        highlight_color : color or list of colors, optional
            color of the shading: one for all or one per element of
            `highlight`; default "lightblue"
        title : str, optional
            title of the figure
        axes : list of matplotlib.axes.Axes, optional
            axes to plot on, one per variable; default: a new figure
        figsize : tuple, optional
            size of the new figure
        **kwargs
            passed to :meth:`matplotlib.axes.Axes.plot` for all lines

        Returns
        -------
        matplotlib.figure.Figure
            the figure of the plot

        Raises
        ------
        ValueError
            if a name in `variables` is neither a variable of the eyes nor
            a channel in ``d.aux``, if `show_onsets` is not one of the
            allowed values, if `plot_range` does not overlap with the data
            or if `axes` does not contain one axes per variable

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.HeadMountedEyeData.from_tobii_prolab("rec.xlsx")  # doctest: +SKIP
        >>> d = d.scene.luminance().scene.head_movement()  # doctest: +SKIP
        >>> fig = d.plot.plot_timeseries((60, 90), variables=["pupil", "luminance",
        ...                                                   "head_speed", "gyro"])  # doctest: +SKIP
        """
        obj = self._d
        variables = [variables] if isinstance(variables, str) else list(variables)
        common = dict(eyes=eyes, units=units, show_onsets=show_onsets, show_masked=show_masked,
                      show_mask_highlight=show_mask_highlight, label_prefix=label_prefix,
                      style=style, highlight=highlight, highlight_color=highlight_color)
        datavars = obj.data.get_available_variables()
        # one panel per variable: its channels in aux, or None for a variable of the eyes
        panels = [(var, None if var in datavars else self._aux_channels(var)) for var in variables]
        unknown = [var for var, channels in panels if var not in datavars and channels is None]
        if unknown:
            raise ValueError(f"no variable or aux channel {', '.join(unknown)}; available: "
                             f"{', '.join(list(datavars) + list(obj.aux.keys()))}")
        if all(channels is None for _, channels in panels):
            return super().plot_timeseries(plot_range, variables=variables, title=title,
                                           axes=axes, figsize=figsize, **common, **kwargs)

        start, end = self._plot_range_in(plot_range, units)
        self._validate_plot_range((start, end), units)
        if show_onsets not in ("line", "label", "both", "none"):
            raise ValueError("show_onsets must be one of 'line', 'label', 'both', or 'none'")
        fig, axs = get_axes(axes, len(panels), "variable", figsize)
        eye_axs = [ax for ax, (_, channels) in zip(axs, panels) if channels is None]
        if eye_axs:
            super().plot_timeseries((start, end), axes=eye_axs, **common, **kwargs,
                                    variables=[var for var, channels in panels if channels is None])
            eye_axs[-1].set_xlabel("")

        tx = _ms_to(obj.tx, units)
        sel = (tx >= start) & (tx <= end)
        evon = _ms_to(obj.events.onsets, units)
        shown = (evon >= start) & (evon < end)
        for ax, (name, channels) in zip(axs, panels):
            if channels is None:
                continue
            for key in channels:
                ax.plot(tx[sel], obj.aux[key][sel], **kwargs,
                        label=f"{label_prefix}_{key}" if label_prefix else key)
            if sel.any():
                self._draw_highlight([ax], highlight, highlight_color, units,
                                     (tx[sel][0], tx[sel][-1]))
            if show_onsets in ("line", "both"):
                ax.vlines(evon[shown], *ax.get_ylim(), color="grey", alpha=0.5)
            if show_onsets in ("label", "both"):
                ll, ul = ax.get_ylim()
                for ev, lab in zip(evon[shown], obj.events.labels[shown]):
                    ax.text(ev, ll + (ul - ll) / 2., "%s" % lab, fontsize=8, rotation=90)
            ax.set_title(name)
            ax.legend()
        axs[-1].set_xlabel(units)
        if title is not None:
            fig.suptitle(title)
        return fig

    def plot_pupil_covariate(self,
            channel: str,
            plot_range: tuple=(-np.inf, +np.inf),
            eyes: list=[],
            log: bool=False,
            title: str=None,
            units: str="sec",
            ax=None,
            **kwargs):
        """
        Plot the pupil size against a channel in ``d.aux`` (e.g., the
        luminance of the scene video or the viewing distance).

        One dot per sample with a usable pupil size and a value in the
        channel, in one color per eye. The legend gives the correlation
        (Pearson's r) of the pupil size of each eye with the channel. The
        samples are taken as they are: the pupil follows changes of the
        light with a delay, which the plot does not account for.

        Parameters
        ----------
        channel : str
            name of the channel in ``d.aux`` (e.g., ``"gaze_luminance"``)
        plot_range : tuple
            The time range to plot (plain numbers are in `units`, or with
            units, e.g. ``(3, 5) * u.min``). Default is (-np.inf, +np.inf), i.e. all data.
        eyes : list
            The eyes to plot. Default is [], which means all eyes with
            pupil data.
        log : bool
            if True, the axis of the channel is logarithmic and the
            correlation is that with the logarithm of the channel (only
            positive values are used); e.g. for the viewing distance.
            Default False.
        title : str, optional
            The title of the plot.
        units : str
            The units of plain numbers in `plot_range` ("ms", "sec",
            "min", "h"). Default "sec".
        ax : matplotlib.axes.Axes, optional
            axes to plot on; default: the current axes
        **kwargs
            passed to :meth:`matplotlib.axes.Axes.scatter` (e.g., `s`, `alpha`)

        Returns
        -------
        matplotlib.axes.Axes
            the axes of the plot

        Raises
        ------
        ValueError
            if the object has no pupil data or no channel `channel` in
            ``d.aux``, or `plot_range` does not overlap with the data

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.HeadMountedEyeData.from_tobii_prolab("rec.xlsx")  # doctest: +SKIP
        >>> d = d.scene.luminance().scene.distance()  # doctest: +SKIP
        >>> ax = d.plot.plot_pupil_covariate("gaze_luminance")  # doctest: +SKIP
        >>> ax = d.plot.plot_pupil_covariate("gaze_distance", log=True)  # doctest: +SKIP
        """
        obj = self._d
        obj._require("pupil")
        obj._require_aux(channel)
        start, end = self._plot_range_in(plot_range, units)
        self._validate_plot_range((start, end), units)
        tx = _ms_to(obj.tx, units)
        values = obj.aux[channel]
        sel = (tx >= start) & (tx <= end) & ~np.isnan(values)
        if log:
            sel[sel] = values[sel] > 0

        ax = get_ax(ax)
        scatter_kwargs = dict(s=2, alpha=0.2, linewidths=0)
        scatter_kwargs.update(kwargs)
        for eye, _ in obj.data.select(eyes, "pupil"):
            pupil = obj.get_values(eye, "pupil")
            ok = sel & ~np.isnan(pupil)
            x = values[ok]
            r = np.corrcoef(np.log(x) if log else x, pupil[ok])[0, 1] if ok.sum() > 2 else np.nan
            points = ax.scatter(x, pupil[ok], **scatter_kwargs)
            # legend entry without the transparency of the dots
            ax.scatter([], [], color=points.get_facecolor()[0][:3] if ok.any() else None,
                       s=20, label=f"{eye} (r = {r:.2f})")
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(channel)
        ax.set_ylabel("pupil")
        ax.legend()
        if title is not None:
            ax.set_title(title)
        return ax

    @requires_package("av")
    def plot_scene_scanpath(self,
            plot_range: tuple=(-np.inf, +np.inf),
            frame="middle",
            show_onsets: bool=False,
            title: str=None,
            units: str="sec",
            ax=None):
        """
        Plot the gaze position during a time interval as a scanpath on a
        frame of the scene video.

        The usable samples of the gaze position (``both_x``, ``both_y``)
        are drawn as dots colored by time (from blue to red), as in
        :meth:`plot_scanpath`. The scene camera
        moves with the head, so the frame shows what the person looked
        at only as long as the head did not move: use short intervals.

        Requires the optional package `av`.

        Parameters
        ----------
        plot_range : tuple
            The time range to plot (plain numbers are in `units`, or with
            units, e.g. ``(3, 5) * u.min``). Default is (-np.inf, +np.inf), i.e. all data.
        frame : str, float or Quantity
            which frame of the video is shown: the one at the "start", in
            the "middle" (default) or at the "end" of `plot_range`, or at
            a time (plain numbers are in `units`)
        show_onsets : bool
            Whether to plot the event labels at the gaze position at event
            onset. Default is False.
        title : str, optional
            The title of the plot.
        units : str
            The units of plain numbers in `plot_range` and `frame`
            ("ms", "sec", "min", "h"). Default "sec".
        ax : matplotlib.axes.Axes, optional
            axes to plot on; default: the current axes

        Returns
        -------
        matplotlib.axes.Axes
            the axes of the plot

        Raises
        ------
        ValueError
            if `plot_range` does not overlap with the data, or the object
            has no scene video or no frame at the requested time

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.HeadMountedEyeData.from_tobii_prolab("rec.xlsx")  # doctest: +SKIP
        >>> ax = d.plot.plot_scene_scanpath((10, 12))  # doctest: +SKIP
        """
        obj = self._d
        obj._require("x", "y")
        start, end = self._plot_range_in(plot_range, units)
        self._validate_plot_range((start, end), units)
        tx = _ms_to(obj.tx, units)
        first, last = max(start, tx[0]), min(end, tx[-1])
        if isinstance(frame, str) and frame in ("start", "middle", "end"):
            tframe = {"start": first, "middle": (first + last) / 2., "end": last}[frame]
        else:
            tframe = float(parse_time(frame, to=units, default=units))
        img = obj.scene.frame(tframe, units=units)

        ax = get_ax(ax)
        height, width = img.shape[:2]
        ax.imshow(img, extent=(0, width, height, 0))
        sel = (tx >= start) & (tx <= end)
        x = np.where(obj.mask("both", "x"), np.nan, obj.data["both_x"])[sel]
        y = np.where(obj.mask("both", "y"), np.nan, obj.data["both_y"])[sel]
        ax.plot(x, y, color="white", linewidth=1, alpha=0.6)
        ax.scatter(x, y, s=12, c=tx[sel], cmap="jet", zorder=3)
        if show_onsets:
            evon = _ms_to(obj.events.onsets, units)
            for on, lab in zip(evon, obj.events.labels):
                if start <= on <= end:
                    ix = np.argmin(np.abs(tx - on))
                    ax.text(obj.data["both_x"][ix], obj.data["both_y"][ix], lab,
                            fontsize=12, ha="center", va="center", color="white")
        # the frame fills the axes (y pointing down), also if the gaze leaves it
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        if title is not None:
            ax.set_title(title)
        return ax
