"""
:class:`SimEyeData`: simulated eye-tracking data with its ground truth.
"""
import numpy as np

from ..eyedata.eyedata import EyeData
from ..eyedata.eyedatadict import EyeDataDict
from ..eyedata.namespace import accessor
from ..plot import PlotNamespace
from ..plot._axes import get_ax, get_axes
from ..units import parse_time

__all__ = ["SimEyeData", "SimPlotNamespace"]


class SimPlotNamespace(PlotNamespace):
    """
    Plotting methods of a :class:`SimEyeData` object (``d.plot``): those
    of :class:`~pypillometry.plot.PlotNamespace` and plots of the data
    together with the ground truth.
    """

    def _time(self, plot_range, units):
        """selection of the samples in `plot_range` and their times in `units`"""
        t = self._d.tx
        lo, hi = (-np.inf, np.inf) if plot_range is None else \
            (parse_time(v, to="ms", default=units) for v in plot_range)
        sel = (t >= lo) & (t <= hi)
        return sel, parse_time(t[sel], to=units)

    def plot_truth(self, eyes=[], plot_range=None, units="sec", ax=None):
        """
        The pupil data with the true pupil: as measured without noise and,
        if they differ (e.g., by foreshortening), the true pupil size.

        Parameters
        ----------
        eyes : str or list
            eyes to plot; if empty, all eyes with pupil data
        plot_range : tuple, optional
            (start, end) of the part of the recording to plot (plain numbers
            are in `units`)
        units : str
            unit of the time axis (default "sec")
        ax : Axes, optional
            axes to draw on (default: the current axes)

        Returns
        -------
        Axes
        """
        d = self._d
        ax = get_ax(ax)
        sel, t = self._time(plot_range, units)
        for i, eye in enumerate(d._eyes_with(eyes, "pupil")):
            lab = f"{eye}: " if len(d.eyes) > 1 else ""
            ax.plot(t, np.asarray(d.data[eye, "pupil"])[sel], color=f"C{i}", lw=0.5,
                    alpha=0.5, label=f"{lab}data")
            if f"{eye}_pupil" in d.truth:
                ax.plot(t, d.truth[eye, "pupil"][sel], color=f"C{i}", lw=1.2,
                        label=f"{lab}true (without noise)")
            if f"{eye}_size" in d.truth and f"{eye}_pupil" in d.truth and \
                    not np.allclose(d.truth[eye, "size"], d.truth[eye, "pupil"]):
                ax.plot(t, d.truth[eye, "size"][sel], color=f"C{i}", lw=1.2, ls="--",
                        label=f"{lab}true size")
        ax.set_xlabel(f"time ({units})")
        ax.set_ylabel("pupil")
        ax.set_title(f"{d.name}: data and ground truth")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        return ax

    def plot_components(self, eye=None, plot_range=None, units="sec", axes=None,
                        figsize=(12, 7)):
        """
        The true components of the latent pupil size (log-%,
        :class:`~pypillometry.sim.steps.TimescalePupil`).

        Three panels: the latent signal with its slow part; the fast
        component (and the difference between the eyes); the atoms.

        Parameters
        ----------
        eye : str, optional
            eye to plot (default: the first eye)
        plot_range : tuple, optional
            (start, end) of the part of the recording to plot (plain numbers
            are in `units`)
        units : str
            unit of the time axis (default "sec")
        axes : sequence of 3 Axes, optional
            axes to draw on; default: a new figure
        figsize : tuple
            size of the new figure

        Returns
        -------
        matplotlib Figure
        """
        d = self._d
        eye = d.eyes[0] if eye is None else eye
        names = [v[4:] for v in d.truth.variables
                 if v.startswith("log_") and f"{eye}_{v}" in d.truth]
        if "size" not in names:
            raise ValueError(f"the simulation has no latent pupil of the {eye} eye")
        atoms = [n for n in names if n not in ("size", "fast", "diff", "slow")]
        sel, t = self._time(plot_range, units)
        own = axes is None
        fig, axes = get_axes(axes, 3, figsize=figsize)

        def get(n):
            return d.truth[eye, f"log_{n}"][sel]

        ax = axes[0]
        ax.plot(t, get("size"), color="0.5", lw=0.6, label="latent signal")
        if "slow" in names:
            ax.plot(t, get("slow"), color="C3", lw=1.5, label="slow part")
        ax.set_ylabel("log-%")
        ax.set_title(f"{d.name}, {eye}: true components, 100 log(size / median)")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        ax = axes[1]
        ax.plot(t, get("fast"), color="C0", lw=0.6, label="fast component")
        if "diff" in names:
            ax.plot(t, get("diff"), color="C1", lw=0.6, label="difference between the eyes")
        ax.set_ylabel("log-%")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        ax = axes[2]
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("viridis")
        for i, a in enumerate(atoms):
            ax.plot(t, get(a), color=cmap(i / max(len(atoms) - 1, 1) * 0.9), lw=1.2, label=a)
        ax.set_ylabel("atoms (log-%)")
        ax.set_xlabel(f"time ({units})")
        ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=max(len(atoms), 1))
        if own:
            fig.tight_layout()
        return fig


class SimEyeData(EyeData):
    """
    Simulated eye-tracking data together with the ground truth (made by
    :meth:`Simulation.run() <pypillometry.sim.Simulation.run>`).

    An :class:`~pypillometry.EyeData` object (all its methods work) with

    - `truth`: an :class:`~pypillometry.EyeDataDict` with the true time
      series, one value per sample. It is cut, downsampled and saved
      together with the data, but not changed by the preprocessing. Which
      variables it has depends on the steps of the simulation (see
      :mod:`pypillometry.sim.steps`), e.g. ``d.truth["left", "pupil"]``
      (the pupil without noise), ``"size"`` (the true size) and the
      components of the latent signal in log-% (``"log_size"``,
      ``"log_fast"``, ``"log_slow"``, ...).
    - `simulation`: the :class:`~pypillometry.sim.Simulation` that made the
      data, with its seed (``d.simulation.run()`` makes the same data
      again; ``d.simulation.run(seed=2)`` new data from the same recipe)
    - the plots ``d.plot.plot_truth()`` and ``d.plot.plot_components()``
      (:class:`SimPlotNamespace`)

    Parameters
    ----------
    truth : EyeDataDict, optional
        the ground truth (one value per sample)
    simulation : Simulation, optional
        the recipe of the data
    **kwargs
        parameters of :class:`~pypillometry.EyeData` (e.g., `left_pupil`,
        `sampling_rate`)
    """
    plot = accessor(SimPlotNamespace)

    _STATE_FIELDS = (("truth", "EyeDataDict"), ("simulation", "Simulation"))
    # the ground truth follows crop, downsample etc. with the data
    _SAMPLE_ALIGNED = ("data", "truth")

    def __init__(self, *, truth=None, simulation=None, **kwargs):
        # set before the constructor, which may fill in time discontinuities
        self.truth = truth if truth is not None else EyeDataDict()
        self.simulation = simulation
        super().__init__(**kwargs)
        if len(self.truth) and self.truth.length != len(self):
            raise ValueError(f"the truth has {self.truth.length} samples, the data {len(self)}")

    #: colours of the HTML display, which marks the data as simulated
    _HTML_ACCENT = {"border": "#f0a04b", "background": "#fff7ec", "badge": "#f6c48a",
                    "badge_text": "#6b3a00", "label": "SIMULATED"}

    def _html_context(self) -> dict:
        context = super()._html_context()
        context["accent"] = self._HTML_ACCENT
        context["simulation"] = repr(self.simulation) if self.simulation else None
        context["truth"] = (f"{', '.join(self.truth.variables)} ({', '.join(self.truth.eyes)})"
                            if len(self.truth) else None)
        return context

    def __repr__(self):
        head, sep, history = super().__repr__().partition(" History:")
        sim = repr(self.simulation).replace("\n", "\n   ") if self.simulation else "-"
        truth = (f"{', '.join(self.truth.variables)} ({', '.join(self.truth.eyes)})"
                 if len(self.truth) else "-")
        return head + f" Simulation: {sim}\n Truth: {truth}\n" + sep + history
