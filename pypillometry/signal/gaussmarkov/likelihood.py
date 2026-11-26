"""
likelihood.py
=============

Objective functions for a :class:`GaussMarkovModel` of components: the exact Gaussian
log-likelihood of the raw signal (Kalman filter) and an approximate
log-likelihood of an empirical semivariogram; plus a small fitting helper.
"""
from typing import NamedTuple

import numpy as np

from ...units import parse_time
from ..variogram import Variogram, _valid_data
from .kalman import kalman_loglik

__all__ = ["check_zero_mean", "loglik_signal", "loglik_variogram",
           "neg_objective", "fit", "FitResult"]

_LOG2PI = np.log(2 * np.pi)


def check_zero_mean(signal, mask=None, tol=0.05):
    """
    Raise a ValueError unless the valid samples have mean ~0.

    The models have mean 0; subtract the mean (or a baseline) first.

    Parameters
    ----------
    signal : array or masked array
    mask : array of bool, optional
        True for invalid samples
    tol : float, optional
        largest allowed ``|mean| / std`` of the valid samples (default 0.05)
    """
    x, v = _valid_data(signal, mask)
    if not v.any():
        raise ValueError("signal has no valid samples")
    mean, sd = x[v].mean(), x[v].std()
    if abs(mean) > tol * sd:
        raise ValueError(f"signal is not zero-mean (mean {mean:.4g}, sd {sd:.4g}); "
                         "the models have mean 0, subtract the mean first")


def loglik_signal(model, signal, fs, mask=None, mean_tol=0.05, obs_var=None,
                  segments=None):
    """
    Exact Gaussian log-likelihood of a signal under `model` (Kalman filter).

    Parameters
    ----------
    model : GaussMarkovModel
    signal : array or masked array
        the signal; masked, NaN and inf samples are missing (gaps are
        handled exactly, not interpolated). It has to be zero-mean unless
        the model has a :class:`~pypillometry.signal.gaussmarkov.Constant`
        component, which makes the result the restricted likelihood (it
        does not depend on the mean).
    fs : float
        sampling rate of `signal` in Hz
    mask : array of bool, optional
        True for additional invalid samples
    mean_tol : float or None, optional
        tolerance of :func:`check_zero_mean`; None skips the check (it is
        also skipped for a model with a Constant component)
    obs_var : array, optional
        noise variance of each sample (instead of ``model.noise**2``)
    segments : list of (start, stop), optional
        sample indices of independent segments (with a mean each, if the
        model has a Constant component)

    Returns
    -------
    float
    """
    n_diffuse = model.n_diffuse
    if mean_tol is not None and not n_diffuse:
        check_zero_mean(signal, mask, tol=mean_tol)
    return kalman_loglik(model.to_lgssm(fs), signal, mask, obs_var=obs_var,
                         n_diffuse=n_diffuse, segments=segments)


def _vg_select(vg, min_pairs, max_lag):
    ok = (vg.npairs >= min_pairs) & np.isfinite(vg.gamma)
    if max_lag is not None:
        ok &= vg.lags <= parse_time(max_lag, to="ms")
    if not ok.any():
        raise ValueError("no lags with enough pairs of valid samples")
    return vg.lags[ok], vg.gamma[ok], vg.npairs[ok].astype(float)


def loglik_variogram(model, vg, min_pairs=100, max_lag=None, method="gauss",
                     neff_scale=1.0):
    """
    Approximate log-likelihood of an empirical semivariogram under `model`.

    Parameters
    ----------
    model : GaussMarkovModel
    vg : Variogram
        from :func:`~pypillometry.signal.semivariogram`
    min_pairs : int, optional
        ignore lags with fewer pairs of valid samples (default 100)
    max_lag : float, str or Quantity, optional
        ignore larger lags (plain numbers are ms)
    method : {"gauss", "wls"}, optional
        - "gauss" (default): independent Gaussians,
          ``gamma_hat(h) ~ N(gamma(h), 2 gamma(h)^2 / (neff_scale * N(h)))``,
          the large-sample variance of a mean of N(h) squared differences
          if they were independent. Includes the log-determinant, so it
          behaves like a likelihood.
        - "wls": Cressie's (1985) weighted least squares,
          ``-1/2 sum_h N(h) (gamma_hat(h) / gamma(h) - 1)^2``.
    neff_scale : float, optional
        effective fraction of independent pairs. The squared differences
        overlap and neighbouring lags are strongly correlated, so with 1
        (default) the objective is far too confident; scale it down (e.g.
        to (number of lags) / (number of samples)) for sensible curvature.
        It scales both methods.

    Returns
    -------
    float
    """
    if not isinstance(vg, Variogram):
        raise TypeError("vg must be a Variogram (see semivariogram())")
    lags, ghat, npairs = _vg_select(vg, min_pairs, max_lag)
    g = model.variogram(lags)
    w = neff_scale * npairs
    if method == "gauss":
        var = 2 * g**2 / w
        return float(-0.5 * np.sum(_LOG2PI + np.log(var) + (ghat - g)**2 / var))
    if method == "wls":
        return float(-0.5 * np.sum(w * (ghat / g - 1)**2))
    raise ValueError(f"method must be 'gauss' or 'wls', got '{method}'")


def neg_objective(theta, model, which="signal", signal=None, fs=None, vg=None,
                  **kw):
    """
    Negative log-likelihood as a function of `theta` (logs of the free
    parameters of `model`), for :func:`scipy.optimize.minimize`.

    Parameters
    ----------
    theta : array
    model : GaussMarkovModel
        supplies the structure, the fixed parameters and the names in theta
    which : {"signal", "variogram"}
        use :func:`loglik_signal` (needs `signal`, `fs`) or
        :func:`loglik_variogram` (needs `vg`)
    **kw
        passed to the log-likelihood function (e.g. ``mean_tol=None`` to
        skip the zero-mean check of the signal in every call)

    Returns
    -------
    float
        inf if the parameters give a numerically invalid model
    """
    mod = model.from_theta(theta)
    try:
        if which == "signal":
            ll = loglik_signal(mod, signal, fs, **kw)
        elif which == "variogram":
            ll = loglik_variogram(mod, vg, **kw)
        else:
            raise ValueError(f"which must be 'signal' or 'variogram', got '{which}'")
    except np.linalg.LinAlgError:
        return np.inf
    return -ll if np.isfinite(ll) else np.inf


class FitResult(NamedTuple):
    """
    Result of :func:`fit`.

    Attributes
    ----------
    model : GaussMarkovModel
        the model with the fitted parameters
    loglik : float
        maximised log-likelihood (of the objective in `which`)
    which : str
    opt : scipy.optimize.OptimizeResult
    n_obs : int or None
        number of valid samples (None for a variogram fit)
    """
    model: object
    loglik: float
    which: str
    opt: object
    n_obs: object = None

    @property
    def n_params(self):
        """number of free (fitted) parameters"""
        return len(self.model.free_names)

    def _check_ic(self):
        if self.which != "signal":
            raise ValueError("AIC/BIC need the likelihood of the signal; the "
                             "variogram objective is not a likelihood")

    @property
    def aic(self):
        """Akaike information criterion, ``-2 loglik + 2 k`` (lower is better)"""
        self._check_ic()
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self):
        """
        Bayesian information criterion, ``-2 loglik + k log(n_obs)``
        (lower is better)
        """
        self._check_ic()
        return -2 * self.loglik + self.n_params * np.log(self.n_obs)


def fit(model, signal=None, fs=None, vg=None, which="signal", method="L-BFGS-B",
        options=None, **kw):
    """
    Maximum-likelihood fit of the free parameters of `model`.

    Parameters
    ----------
    model : GaussMarkovModel
        starting values, fixed parameters (:meth:`GaussMarkovModel.fix`) and structure
    signal, fs : array, float
        signal and sampling rate (Hz), for ``which="signal"``
    vg : Variogram
        empirical semivariogram, for ``which="variogram"``
    which : {"signal", "variogram"}
    method : str, optional
        passed to :func:`scipy.optimize.minimize` (default "L-BFGS-B";
        "Nelder-Mead" is more robust for poor starting values)
    options : dict, optional
        passed to :func:`scipy.optimize.minimize`
    **kw
        passed to :func:`loglik_signal` (mask, mean_tol, obs_var, segments) or
        :func:`loglik_variogram` (min_pairs, max_lag, method -> use
        ``vg_method``, neff_scale)

    Returns
    -------
    FitResult
    """
    from scipy.optimize import minimize
    n_obs = None
    if which == "signal":
        if signal is None or fs is None:
            raise ValueError("which='signal' needs signal and fs")
        n_obs = int(_valid_data(signal, kw.get("mask"))[1].sum())
        # check once here, not in every evaluation
        mean_tol = kw.pop("mean_tol", 0.05)
        if mean_tol is not None and not model.n_diffuse:
            check_zero_mean(signal, kw.get("mask"), tol=mean_tol)
        kw["mean_tol"] = None
    elif which == "variogram":
        if vg is None:
            raise ValueError("which='variogram' needs vg")
        if "vg_method" in kw:
            kw["method"] = kw.pop("vg_method")
    opt = minimize(lambda th: neg_objective(th, model, which, signal, fs, vg, **kw),
                   model.theta, method=method, options=options)
    return FitResult(model.from_theta(opt.x), float(-opt.fun), which, opt, n_obs)
