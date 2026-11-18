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

__all__ = ["prepare_signal", "loglik_signal", "loglik_variogram",
           "neg_objective", "fit", "FitResult"]

_LOG2PI = np.log(2 * np.pi)


def prepare_signal(signal, mask=None, demean=True, downsample=1):
    """
    Signal as a float array with NaN at invalid samples.

    Parameters
    ----------
    signal : array or masked array
    mask : array of bool, optional
        True for invalid samples
    demean : bool, optional
        subtract the mean of the valid samples (default True; the models
        have mean 0)
    downsample : int, optional
        keep every `downsample`-th sample (no low-pass filtering; the model
        stays exact when evaluated at ``fs / downsample``)

    Returns
    -------
    np.ndarray
    """
    x, v = _valid_data(signal, mask)
    y = np.where(v, x, np.nan)
    if demean and v.any():
        y = y - x[v].mean()
    return y[::int(downsample)]


def loglik_signal(model, signal, fs, mask=None, demean=True, downsample=1):
    """
    Exact Gaussian log-likelihood of a signal under `model` (Kalman filter).

    Parameters
    ----------
    model : GaussMarkovModel
    signal : array or masked array
        the signal; masked, NaN and inf samples are missing (gaps are
        handled exactly, not interpolated)
    fs : float
        sampling rate of `signal` in Hz
    mask : array of bool, optional
        True for additional invalid samples
    demean : bool, optional
        subtract the mean of the valid samples first (default True)
    downsample : int, optional
        evaluate on every `downsample`-th sample (faster; default 1)

    Returns
    -------
    float
    """
    y = prepare_signal(signal, mask, demean=demean, downsample=downsample)
    return kalman_loglik(model.to_lgssm(fs / downsample), y)


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
        passed to the log-likelihood function; for "signal" pass an already
        prepared signal (see :func:`prepare_signal`) with ``demean=False``
        to avoid redoing that work in every call

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
    """
    model: object
    loglik: float
    which: str
    opt: object


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
        passed to :func:`loglik_signal` (mask, demean, downsample) or
        :func:`loglik_variogram` (min_pairs, max_lag, method -> use
        ``vg_method``, neff_scale)

    Returns
    -------
    FitResult
    """
    from scipy.optimize import minimize
    if which == "signal":
        if signal is None or fs is None:
            raise ValueError("which='signal' needs signal and fs")
        ds = kw.pop("downsample", 1)
        signal = prepare_signal(signal, kw.pop("mask", None),
                                demean=kw.pop("demean", True), downsample=ds)
        fs = fs / ds
        kw["demean"] = False
    elif which == "variogram":
        if vg is None:
            raise ValueError("which='variogram' needs vg")
        if "vg_method" in kw:
            kw["method"] = kw.pop("vg_method")
    opt = minimize(lambda th: neg_objective(th, model, which, signal, fs, vg, **kw),
                   model.theta, method=method, options=options)
    return FitResult(model.from_theta(opt.x), float(-opt.fun), which, opt)
