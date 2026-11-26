"""
plots.py
========

Plots of a :class:`~pypillometry.TimescaleModel` (``m.plot``) and of a
:class:`~pypillometry.TimescaleGroup` (``g.plot``). They follow the
conventions of the package: single-panel plots draw on `ax` and return it,
multi-panel plots draw on `axes` (or a new figure) and return the figure.
"""
import numpy as np

from ..plot._axes import get_ax, get_axes
from ..units import parse_time

__all__ = ["TimescalePlots", "TimescaleGroupPlots"]


def _colors(n):
    import matplotlib.pyplot as plt
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [cycle[i % len(cycle)] for i in range(n)]


def _atom_colors(names):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    return {a: cmap(i / max(len(names) - 1, 1) * 0.9) for i, a in enumerate(names)}


class TimescalePlots:
    """
    Plots of a :class:`~pypillometry.TimescaleModel`, used as ``m.plot``
    (e.g., ``m.plot.decomposition()``).

    Parameters
    ----------
    model : TimescaleModel
    """

    def __init__(self, model):
        self._m = model

    def __repr__(self):
        names = sorted(n for n in dir(self) if not n.startswith("_"))
        return f"<plots of {self._m.name!r}: {', '.join(names)}>"

    # -- helpers ---------------------------------------------------------------
    def _range(self, plot_range, units):
        """boolean selection of the model samples and their times in `units`"""
        t = self._m.time
        sel = np.ones(len(t), dtype=bool)
        if plot_range is not None:
            lo, hi = (parse_time(v, to="ms", default=units) for v in plot_range)
            sel = (t >= lo) & (t <= hi)
        return sel, parse_time(t, to=units)

    def _lines(self, ax, t, y, sel, **kw):
        """plot y against t with a break between the segments"""
        m = self._m
        first = True
        for a, b in m.segments:
            s = slice(a, b)
            if sel[s].any():
                ax.plot(t[s][sel[s]], y[s][sel[s]], **(kw if first else {**kw, "label": None}))
                first = False

    def _band(self, ax, t, mean, sd, sel, color, alpha=0.25):
        for a, b in self._m.segments:
            s = slice(a, b)
            if sel[s].any():
                ax.fill_between(t[s][sel[s]], (mean - 2 * sd)[s][sel[s]],
                                (mean + 2 * sd)[s][sel[s]], color=color, alpha=alpha, lw=0)

    # -- plots -----------------------------------------------------------------
    def decomposition(self, eye=None, cutoff="10 s", plot_range=None, units="min",
                      axes=None, figsize=(12, 8)):
        """
        The signal split into its slow part, the fast component and the atoms.

        Three panels (in log-%): the data of the model with the slow part
        (the mean plus the atoms with a time scale of at least `cutoff`)
        and two posterior standard deviations around it; the fast
        component; and the atoms.

        Parameters
        ----------
        eye : str, optional
            eye to plot (default: the first eye of the model)
        cutoff : float, str or Quantity
            time scale from which on the atoms count as slow (plain numbers
            are ms; default 10 s)
        plot_range : tuple, optional
            (start, end) of the part of the recording to plot, in `units`
        units : str
            unit of the time axis (default "min")
        axes : sequence of 3 Axes, optional
            axes to draw on; default: a new figure
        figsize : tuple
            size of the new figure

        Returns
        -------
        matplotlib Figure
        """
        m = self._m
        eye = m._eye(eye)
        parts = m.components(eye, cutoff)[eye]
        sel, t = self._range(plot_range, units)
        fig, axes = get_axes(axes, 3, figsize=figsize)
        ax = axes[0]
        self._lines(ax, t, m.y[eye], sel, color="0.6", lw=0.6, label="data (model rate)")
        mean, sd = parts["slow"]
        self._band(ax, t, mean, sd, sel, "C3")
        self._lines(ax, t, mean, sel, color="C3", lw=1.5,
                    label=f"slow part (time scales ≥ {parse_time(cutoff, to='sec'):g} s) ± 2 sd")
        ax.set_ylabel("log-%")
        ax.set_title(f"{m.name}, {eye}: 100 log({m.variable} / median)")
        ax.legend(frameon=False, loc="upper right", fontsize=8)

        ax = axes[1]
        mean, sd = parts["fast"]
        self._lines(ax, t, mean, sel, color="C0", lw=0.6, label="fast component")
        if m.binocular:
            self._lines(ax, t, parts["diff"][0], sel, color="C1", lw=0.6,
                        label="difference between the eyes")
        ax.set_ylabel("log-%")
        ax.legend(frameon=False, loc="upper right", fontsize=8)

        ax = axes[2]
        colors = _atom_colors(m.bank.atom_names)
        pars = m._fit(m._where(eye)[0]).params
        for a in m.bank.atom_names:
            mean, sd = parts[a]
            self._band(ax, t, mean, sd, sel, colors[a], alpha=0.15)
            self._lines(ax, t, mean, sel, color=colors[a], lw=1.2,
                        label=f"{a} (sd {pars[f'sd_{a}']:.1f})")
        ax.set_ylabel("atoms (log-%)")
        ax.set_xlabel(f"time ({units})")
        ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=len(colors))
        return fig

    def slow(self, eye=None, cutoffs=("3 s", "10 s", "30 s", "100 s"), plot_range=None,
             units="min", ax=None):
        """
        The slow part of the signal for several cut-offs.

        The slow part is the mean plus all atoms with a time scale of at
        least the cut-off; all atoms are estimated, so the cut-off can be
        chosen after the fit.

        Parameters
        ----------
        eye : str, optional
            eye to plot (default: the first eye of the model)
        cutoffs : sequence of float, str or Quantity
            cut-offs to compare (plain numbers are ms)
        plot_range : tuple, optional
            (start, end) of the part of the recording to plot, in `units`
        units : str
            unit of the time axis (default "min")
        ax : matplotlib Axes, optional
            axes to draw on; default: the current axes

        Returns
        -------
        matplotlib Axes
        """
        m = self._m
        eye = m._eye(eye)
        ax = get_ax(ax)
        sel, t = self._range(plot_range, units)
        self._lines(ax, t, m.y[eye], sel, color="0.75", lw=0.5, label="data (model rate)")
        for cutoff, color in zip(cutoffs, _colors(len(cutoffs))):
            mean, _ = m.components(eye, cutoff)[eye]["slow"]
            self._lines(ax, t, mean, sel, color=color, lw=1.3,
                        label=f"≥ {parse_time(cutoff, to='sec'):g} s")
        ax.set_xlabel(f"time ({units})")
        ax.set_ylabel("log-%")
        ax.set_title(f"{m.name}, {eye}: slow part by cut-off")
        ax.legend(frameon=False, fontsize=8, ncol=len(cutoffs) + 1)
        return ax

    def variogram(self, eye=None, check=False, n_sim=40, ax=None):
        """
        Semivariogram of the data and of the fitted model.

        The square root of the semivariance (the typical size of a change
        of the signal over a lag, in log-%) on logarithmic axes, with the
        contribution of each component.

        Parameters
        ----------
        eye : str, optional
            eye to plot (default: the first eye of the model)
        check : bool
            also draw the 95 % envelope of simulations of the model with
            the gaps of the data (see
            :meth:`TimescaleModel.variogram_check`); default False
        n_sim : int
            number of simulations of the envelope
        ax : matplotlib Axes, optional
            axes to draw on; default: the current axes

        Returns
        -------
        matplotlib Axes
        """
        m = self._m
        eye = m._eye(eye)
        unit, ch = m._where(eye)
        theta = m._fit(unit).theta
        ax = get_ax(ax)
        vg = m.variogram(eye)[eye]
        lags = vg.lags / 1000.0
        if check:
            chk = m.variogram_check(eye, n_sim=n_sim)[eye]
            ax.fill_between(chk.lags / 1000.0, np.sqrt(chk.lower), np.sqrt(chk.upper),
                            color="C3", alpha=0.2, lw=0,
                            label=f"95 % of simulations ({chk.outside:.0%} of lags outside)")
        colors = _atom_colors(m.bank.atom_names)
        for name, comp, w in m.bank._components(theta):
            if comp.sigma > 0:
                ax.plot(lags, np.sqrt(w * comp.variogram(vg.lags)), lw=0.9,
                        ls="--", color=colors.get(name, "C0" if name == "fast" else "C1"),
                        label=name)
        ax.plot(lags, np.sqrt(vg.gamma), "o", ms=3, color="k", label="data")
        ax.plot(lags, np.sqrt(m.bank.variogram(theta, vg.lags, ch)), color="C3", lw=1.5,
                label="model")
        ax.set_xscale("log")
        ax.set_yscale("log")
        g = np.sqrt(vg.gamma[np.isfinite(vg.gamma)])
        ax.set_ylim(g.min() / 3, g.max() * 1.5)
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("sqrt of semivariance (log-%)")
        ax.set_title(f"{m.name}, {eye}: variogram")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        return ax

    def spectrum(self, eye=None, window="200 s", ax=None):
        """
        Power spectrum of the data and of the fitted model.

        The spectrum of the data comes from the windows of the data of the
        model that have few gaps (see :func:`~pypillometry.signal.spectrum`).
        Peaks at the frequency of the trials (responses locked to a fixed
        rhythm) are not part of the model.

        Parameters
        ----------
        eye : str, optional
            eye to plot (default: the first eye of the model)
        window : float, str or Quantity
            length of the windows (plain numbers are ms; default 200 s)
        ax : matplotlib Axes, optional
            axes to draw on; default: the current axes

        Returns
        -------
        matplotlib Axes
        """
        m = self._m
        eye = m._eye(eye)
        unit, ch = m._where(eye)
        theta = m._fit(unit).theta
        ax = get_ax(ax)
        sp = m.spectrum(eye, window=window)[eye].at_log_freqs(20)
        colors = _atom_colors(m.bank.atom_names)
        for name, comp, w in m.bank._components(theta):
            if comp.sigma > 0:
                ax.plot(sp.freqs, w * comp.psd(sp.freqs), lw=0.9, ls="--",
                        color=colors.get(name, "C0" if name == "fast" else "C1"), label=name)
        ax.plot(sp.freqs, sp.power, color="k", lw=1, label=f"data ({sp.nwindows} windows)")
        ax.plot(sp.freqs, m.bank.psd(theta, sp.freqs, fs=m.fs, channel=ch), color="C3",
                lw=1.5, label="model")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if np.isfinite(sp.power).any():
            ax.set_ylim(np.nanmin(sp.power) / 5, np.nanmax(sp.power) * 3)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power spectral density (log-%$^2$/Hz)")
        ax.set_title(f"{m.name}, {eye}: spectrum")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        return ax

    def timescales(self, ax=None):
        """
        Standard deviation of each atom: how much the pupil varies at each
        time scale.

        The bars are approximate 95 % intervals (two standard errors of
        :math:`u = \\log(v + v_0)`). Neighbouring atoms are hard to
        separate in one recording; the sum over the slow atoms
        (:meth:`TimescaleModel.slow_sd`) is determined much better.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            axes to draw on; default: the current axes

        Returns
        -------
        matplotlib Axes
        """
        m = self._m
        ax = get_ax(ax)
        bank = m.bank
        x = np.arange(bank.K)
        width = 0.8 / len(m.units)
        to_sd = lambda u: np.sqrt(np.maximum(np.exp(u) - bank.v0, 0.0))
        for i, (unit, color) in enumerate(zip(m.units, _colors(len(m.units)))):
            fit = m._fit(unit)
            u = fit.theta[bank._iu:]
            xi = x + (i - (len(m.units) - 1) / 2) * width
            ax.bar(xi, to_sd(u), width=width * 0.9, color=color, alpha=0.7, label=unit)
            if fit.se is not None:
                se = fit.se[bank._iu:]
                ax.vlines(xi, to_sd(u - 2 * se), to_sd(u + 2 * se), color="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(bank.atom_names)
        ax.set_xlabel("time scale of the atom")
        ax.set_ylabel("standard deviation (log-%)")
        ax.set_title(f"{m.name}: variability by time scale")
        ax.legend(frameon=False, fontsize=8)
        return ax

    def diagnostics(self, eye=None, axes=None, figsize=(13, 3.8)):
        """
        Checks of the fitted model.

        Three panels: quantiles of the standardised one-step prediction
        errors against the normal distribution (heavy tails: fast changes
        or artifacts that a Gaussian model does not expect); their
        autocorrelation (should be near 0); and the standard deviation of
        the standardised forecast errors over horizons from 50 ms to 60 s
        (should be near 1, see :meth:`TimescaleModel.forecast_calibration`).

        Parameters
        ----------
        eye : str, optional
            eye to plot (default: the first eye of the model)
        axes : sequence of 3 Axes, optional
            axes to draw on; default: a new figure
        figsize : tuple
            size of the new figure

        Returns
        -------
        matplotlib Figure
        """
        from scipy.stats import norm
        m = self._m
        eye = m._eye(eye)
        fig, axes = get_axes(axes, 3, figsize=figsize, nrows=1, ncols=3)
        z = m.innovations(eye)[eye]
        zs = np.sort(z[np.isfinite(z)])
        q = norm.ppf((np.arange(len(zs)) + 0.5) / len(zs))
        ax = axes[0]
        ax.plot(q, zs, ".", ms=2, color="C0")
        lim = max(4.5, np.abs(q).max())
        ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.8)
        ax.set_xlabel("normal quantiles")
        ax.set_ylabel("standardised innovations")
        kurt = float(np.mean(zs**4) / np.mean(zs**2)**2 - 3)
        ax.set_title(f"{eye}: sd {zs.std():.2f}, excess kurtosis {kurt:.1f}", fontsize=9)

        ax = axes[1]
        nlag = int(min(round(5 * m.fs), len(z) // 4))
        z0 = np.where(np.isfinite(z), z, 0.0)
        v = np.isfinite(z).astype(float)
        acf = []
        for h in range(1, nlag + 1):    # pairs within the segments only
            num = cnt = 0.0
            for a, b in m.segments:
                num += np.dot(z0[a:b - h], z0[a + h:b])
                cnt += np.dot(v[a:b - h], v[a + h:b])
            acf.append(num / max(cnt, 1.0))
        acf = np.array(acf) / np.mean(zs**2)
        ax.plot(np.arange(1, nlag + 1) / m.fs, acf, color="C0", lw=1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("autocorrelation of the innovations")
        ax.set_ylim(-0.3, 0.3)

        ax = axes[2]
        cal = m.forecast_calibration()
        cal = cal[cal["eye"] == eye]
        ax.plot(cal["horizon"], cal["z_sd"], "o-", color="C0")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_ylim(0, 2)
        ax.set_xlabel("forecast horizon (s)")
        ax.set_ylabel("sd of the standardised forecast errors")
        fig.suptitle(f"{m.name}: checks of the model", fontsize=10)
        return fig

    def interpolation(self, eye=None, plot_range=None, units="sec", data=None,
                      interval=0.95, ax=None):
        """
        The original samples with the reconstruction by the model.

        Shows the data at the original sampling rate and in their units:
        the samples the model used, those it did not use (blinks,
        artifacts and their margins), the expected signal under the model
        with its posterior interval, and the slow part.

        Parameters
        ----------
        eye : str, optional
            eye to plot (default: the first eye of the model)
        plot_range : tuple, optional
            (start, end) of the part of the recording to plot, in `units`;
            default: the first 30 s of the first segment
        units : str
            unit of the time axis (default "sec")
        data : EyeData, optional
            the data (default: the data the model was built from)
        interval : float
            probability of the posterior interval (default 0.95)
        ax : matplotlib Axes, optional
            axes to draw on; default: the current axes

        Returns
        -------
        matplotlib Axes
        """
        m = self._m
        eye = m._eye(eye)
        data = m._check_data(data)
        ax = get_ax(ax)
        if plot_range is None:
            lo = data.tx[m._spans[0][0]]
            hi = lo + 30000.0
        else:
            lo, hi = (parse_time(v, to="ms", default=units) for v in plot_range)
        sel = (data.tx >= lo) & (data.tx <= hi)
        t = parse_time(data.tx[sel], to=units)
        raw = np.asarray(data.data[eye, m.variable], dtype=float)[sel]
        usable = m._usable(data, eye)[sel]
        est, lower, upper = (v[sel] for v in m.predict("signal", eye, interval=interval)[eye])
        slow = m.predict("slow", eye)[eye][sel]
        with np.errstate(invalid="ignore"):
            raw = np.where(raw > 0, raw, np.nan)
        ax.plot(t, np.where(usable, np.nan, raw), color="0.75", lw=0.8, label="not used")
        ax.plot(t, np.where(usable, raw, np.nan), color="k", lw=0.8, label="used by the model")
        ax.fill_between(t, lower, upper, color="C3", alpha=0.25, lw=0)
        ax.plot(t, est, color="C3", lw=1.2, label=f"model ({interval:.0%} interval)")
        ax.plot(t, slow, color="C0", lw=1.2, label="slow part")
        finite = est[np.isfinite(est)]
        if len(finite):
            pad = 0.25 * (np.nanmax(upper) - np.nanmin(lower))
            ax.set_ylim(np.nanmin(lower) - pad, np.nanmax(upper) + pad)
        ax.set_xlabel(f"time ({units})")
        ax.set_ylabel(m.variable)
        ax.set_title(f"{m.name}, {eye}")
        ax.legend(frameon=False, fontsize=8, ncol=4)
        return ax


class TimescaleGroupPlots:
    """
    Plots of a :class:`~pypillometry.TimescaleGroup`, used as ``g.plot``.

    Parameters
    ----------
    group : TimescaleGroup
    """

    def __init__(self, group):
        self._g = group

    def __repr__(self):
        names = sorted(n for n in dir(self) if not n.startswith("_"))
        return f"<plots of a group of {len(self._g.models)} models: {', '.join(names)}>"

    def params(self, ax=None):
        """
        Standard deviation of each atom for all recordings of the group.

        Grey: the estimates of the single recordings without pooling;
        blue: the shrunk estimates; black: the group mean with the
        standard deviation between recordings (on the scale of
        :math:`u = \\log(v + v_0)`).

        Parameters
        ----------
        ax : matplotlib Axes, optional
            axes to draw on; default: the current axes

        Returns
        -------
        matplotlib Axes
        """
        g = self._g
        fit = g._group()
        bank = g.bank
        ax = get_ax(ax)
        iu = bank._iu
        to_sd = lambda u: np.sqrt(np.maximum(np.exp(u) - bank.v0, 0.0))
        x = np.arange(bank.K)
        rng = np.random.default_rng(0)
        for fits, color, off, label in ((fit.individual, "0.6", -0.17, "individual"),
                                        (fit.fits, "C0", 0.17, "shrunk")):
            U = np.array([f.theta[iu:] for f in fits])
            jit = rng.uniform(-0.12, 0.12, U.shape)
            ax.plot((x + off + jit).ravel(), to_sd(U).ravel(), ".", ms=4, color=color,
                    alpha=0.6, label=label)
        mu, sd = fit.mu[iu:], fit.sd[iu:]
        ax.errorbar(x, to_sd(mu), yerr=[to_sd(mu) - to_sd(mu - sd), to_sd(mu + sd) - to_sd(mu)],
                    fmt="o", color="k", capsize=3, label="group mean ± sd", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(bank.atom_names)
        ax.set_xlabel("time scale of the atom")
        ax.set_ylabel("standard deviation (log-%)")
        ax.set_title(f"group of {len(fit.fits)}: variability by time scale")
        ax.legend(frameon=False, fontsize=8)
        return ax

    def shrinkage(self, cutoff="10 s", ax=None):
        """
        Total slow standard deviation of each recording: shrunk against
        individual estimate.

        Parameters
        ----------
        cutoff : float, str or Quantity
            time scale from which on the atoms count as slow (plain numbers
            are ms; default 10 s)
        ax : matplotlib Axes, optional
            axes to draw on; default: the current axes

        Returns
        -------
        matplotlib Axes
        """
        g = self._g
        fit = g._group()
        ax = get_ax(ax)
        c = float(parse_time(cutoff, to="ms"))
        ind = [g.bank.slow_sd(f.theta, c) for f in fit.individual]
        shr = [g.bank.slow_sd(f.theta, c) for f in fit.fits]
        ax.plot(ind, shr, "o", ms=4, color="C0", alpha=0.7)
        lim = [0, max(max(ind), max(shr)) * 1.05]
        ax.plot(lim, lim, color="k", lw=0.8)
        ax.axhline(g.bank.slow_sd(fit.mu, c), color="0.5", ls="--", lw=0.8,
                   label="group mean")
        ax.set_xlabel("individual estimate (log-%)")
        ax.set_ylabel("shrunk estimate (log-%)")
        ax.set_title(f"total slow sd (time scales ≥ {c / 1000:g} s)")
        ax.legend(frameon=False, fontsize=8)
        return ax
