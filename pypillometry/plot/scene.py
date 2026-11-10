"""
Plots for data from head-mounted eye trackers (with a scene video), see
:class:`~pypillometry.HeadMountedEyeData`.
"""
import numpy as np

from ..convenience import requires_package
from ..units import _ms_to, parse_time
from ._axes import get_ax
from .gaze import GazePlots


class ScenePlots(GazePlots):
    """
    Plots that use the scene video of a head-mounted eye tracker.
    """

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
