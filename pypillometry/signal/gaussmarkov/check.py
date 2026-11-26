"""
check.py
========

Checks of a fitted timescale-bank model that look at more than the
one-step predictions the likelihood is made of: prediction of held-out
blocks (cross-validation) and the semivariogram of the data against
simulations of the model.

Experimental: the API may change.
"""
from typing import NamedTuple

import numpy as np

from ...units import parse_time
from ..variogram import semivariogram
from .bank import fit_bank, _as_prior

__all__ = ["blocked_cv", "variogram_check", "VariogramCheck"]


def blocked_cv(bank, theta, y, fs, segments=None, prior=None, block="60 s", folds=5,
               refit=True, maxiter=30):
    """
    Blocked cross-validation of a timescale-bank model.

    The signal is cut into blocks; every `folds`-th block is held out in
    turn. The held-out blocks are scored by their joint predictive density
    given the rest of the signal,

    .. math::

        \\log p(y_B \\mid y_{-B}, \\hat\\theta_{-B}) =
        \\ell(\\hat\\theta_{-B}; y) - \\ell(\\hat\\theta_{-B}; y_{-B}),

    where :math:`y_{-B}` is the signal with the blocks B missing. A block
    has to be bridged from its neighbours, which is what the slow
    components are for, so the score is more sensitive to them than the
    likelihood (as long as the blocks are not much longer than their time
    scales).

    Parameters
    ----------
    bank : TimescaleBank
    theta : array
        estimate from the whole signal (the starting value of the refits)
    y, fs, segments
        signal, sampling rate (Hz) and segments, as for :func:`fit_bank`
    prior : GaussianPrior, str or None
        prior of the refits
    block : float, str or Quantity
        length of the blocks (plain numbers are ms; default 60 s)
    folds : int
        number of folds (default 5)
    refit : bool
        re-estimate the parameters without the held-out blocks (default);
        if False, `theta` is used for all folds (faster, optimistic)
    maxiter : int
        maximum number of iterations of each refit

    Returns
    -------
    float
        sum of the log predictive densities of the folds (higher is better)
    """
    y = bank._y(y)
    prior = _as_prior(prior, bank)
    nblock = max(int(round(parse_time(block, to="samples", fs=fs))), 1)
    fold = (np.arange(len(y)) // nblock) % folds
    total = 0.0
    for f in range(folds):
        ytr = np.where((fold == f)[:, None], np.nan, y)
        th = theta
        if refit:
            th = fit_bank(bank, ytr, fs, segments=segments, prior=prior, theta0=theta,
                          hessian=False, maxiter=maxiter).theta
        total += bank.loglik(th, y, fs, segments) - bank.loglik(th, ytr, fs, segments)
    return float(total)


class VariogramCheck(NamedTuple):
    """
    Result of :func:`variogram_check`.

    Attributes
    ----------
    lags : np.ndarray
        lags in ms (about log-spaced)
    gamma : np.ndarray
        semivariogram of the data (pooled within segments)
    model : np.ndarray
        semivariogram of the model (expected value for infinite data)
    lower, upper : np.ndarray
        pointwise 95 % envelope of the semivariograms of signals simulated
        from the model with the gaps and segments of the data
    outside : float
        fraction of the lags at which the data are outside the envelope
        (about 0.05 expected for an adequate model)
    """
    lags: np.ndarray
    gamma: np.ndarray
    model: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    outside: float


def variogram_check(bank, theta, y, fs, segments=None, n_sim=40, per_decade=8,
                    channel=0, rng=None):
    """
    Semivariogram of the data against that of the fitted model.

    The envelope comes from signals simulated from the model with the same
    gaps and segments as the data (parametric bootstrap). The check is
    sensitive to misfit at long lags, which the likelihood hardly sees.

    Parameters
    ----------
    bank : TimescaleBank
    theta : array
        parameter vector
    y, fs, segments
        signal, sampling rate (Hz) and segments, as for :func:`fit_bank`
    n_sim : int
        number of simulations (default 40)
    per_decade : int
        number of lags per decade (default 8)
    channel : int
        channel of the signal (default 0)
    rng : np.random.Generator or int, optional

    Returns
    -------
    VariogramCheck
    """
    y = bank._y(y)
    T = len(y)
    segments = [(0, T)] if segments is None else list(segments)
    rng = np.random.default_rng(rng)
    max_lag = max(b - a for a, b in segments) / 2.0 / fs * 1000.0

    def vg(x):
        return semivariogram(x, fs, max_lag=max_lag, segments=segments)

    emp = vg(y[:, channel]).at_log_lags(per_decade)
    idx = np.searchsorted(vg(y[:, channel]).lags, emp.lags)
    missing = ~np.isfinite(y[:, channel])
    sims = np.empty((n_sim, len(idx)))
    for i in range(n_sim):
        x = np.full(T, np.nan)
        for a, b in segments:   # independent segments
            s = bank.simulate(theta, b - a, fs, rng=rng)
            x[a:b] = s if s.ndim == 1 else s[:, channel]
        x[missing] = np.nan
        sims[i] = vg(x).gamma[idx]
    lower, upper = np.nanpercentile(sims, [2.5, 97.5], axis=0)
    outside = float(np.mean((emp.gamma < lower) | (emp.gamma > upper)))
    return VariogramCheck(emp.lags, emp.gamma, bank.variogram(theta, emp.lags, channel),
                          lower, upper, outside)
