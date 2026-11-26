"""
hierarchical.py
===============

Group-level (hierarchical) estimation of timescale-bank models by empirical
Bayes.

The parameter vectors of the recordings of a group are draws from a normal
distribution,

.. math::

    y^{(i)} \\mid \\theta_i \\sim \\text{bank model}(\\theta_i), \\qquad
    \\theta_i \\sim N(\\mu, \\Sigma),

whose mean and covariance are estimated together with the
:math:`\\theta_i` by an EM algorithm in which the posterior of each
recording is approximated by a normal distribution at its mode (Laplace
approximation). With :math:`\\ell_i` the restricted log-likelihood of
recording i:

- E-step: :math:`\\tilde\\theta_i = \\arg\\max_\\theta [\\ell_i(\\theta) -
  \\tfrac12 (\\theta - \\mu)^\\top \\Sigma^{-1} (\\theta - \\mu)]` and
  :math:`V_i = (H_i + \\Sigma^{-1})^{-1}` with :math:`H_i = -\\nabla^2
  \\ell_i(\\tilde\\theta_i)`
- M-step: :math:`\\mu = \\frac1N \\sum_i \\tilde\\theta_i`, :math:`\\Sigma =
  \\frac1N \\sum_i [(\\tilde\\theta_i - \\mu)(\\tilde\\theta_i - \\mu)^\\top + V_i]`

The estimates :math:`\\tilde\\theta_i` are shrunk towards the group mean in
the directions of the parameter space that the recording does not
determine.

Experimental: the API may change.
"""
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from .bank import GaussianPrior, BankFit, fit_bank, _as_prior, _psd

__all__ = ["GroupFit", "fit_group", "reference_prior"]


def reference_prior(bank):
    """
    Wide prior for fits of single recordings without pooling (signals in
    log-%): it only keeps parameters that the data do not determine from
    drifting to the bounds.

    Fast standard deviation 10 (give or take a factor of 7.4), fast length
    scale 800 ms (factor 2.7), noise 0.2 (factor 7.4), standard deviation
    of each atom 2 (:math:`u \\sim N(\\log(4 + v_0), 3^2)`).

    Parameters
    ----------
    bank : TimescaleBank

    Returns
    -------
    GaussianPrior
    """
    head = {"log_fast_sd": (np.log(10.), 2.), "log_fast_ell": (np.log(800.), 1.),
            "log_noise_sd": (np.log(0.2), 2.), "log_diff_sd": (np.log(3.), 2.),
            "log_diff_ell": (np.log(800.), 1.)}
    mean, sd = [], []
    for name in bank.names:
        if name.startswith("u_"):
            m, s = np.log(4.0 + bank.v0), 3.0
        else:
            m, s = head[name if name in head else "log_noise_sd"]
        mean.append(m)
        sd.append(s)
    return GaussianPrior(np.array(mean), np.array(sd), list(bank.names), "reference (wide)")


@dataclass
class GroupFit:
    """
    Result of :func:`fit_group`.

    Attributes
    ----------
    bank : TimescaleBank
    mu : (p,) array
        group mean of the parameter vector
    Sigma : (p, p) array
        covariance of the parameter vector between the recordings
    fits : list of BankFit
        shrunk estimates of the recordings (prior: the group distribution)
    individual : list of BankFit
        estimates of the recordings without pooling (wide prior)
    history : list of (mu, Sigma)
        the group parameters at the start and after each iteration
    converged : bool
        whether the group mean changed by less than the tolerance in the
        last iteration
    """
    bank: object
    mu: np.ndarray
    Sigma: np.ndarray
    fits: list
    individual: list
    history: list = field(default_factory=list)
    converged: bool = False

    @property
    def sd(self):
        """standard deviations of the parameters between the recordings"""
        return np.sqrt(np.diag(self.Sigma))

    @property
    def prior(self):
        """
        The group distribution as a prior (:class:`GaussianPrior`), e.g.
        for the fit of a new recording of the same kind.
        """
        return GaussianPrior(self.mu.copy(), self.Sigma.copy(), list(self.bank.names),
                             f"group of {len(self.fits)}")

    @property
    def thetas(self):
        """(N, p) array of the shrunk estimates"""
        return np.array([f.theta for f in self.fits])


def _fit_job(args):
    bank, (y, fs, segments), prior, theta0 = args
    return fit_bank(bank, y, fs, segments=segments, prior=prior, theta0=theta0)


def _map(fct, jobs, n_jobs, what):
    """results of fct for the jobs, in parallel processes if n_jobs != 1"""
    from tqdm.auto import tqdm
    if n_jobs == 1 or len(jobs) == 1:
        return [fct(j) for j in tqdm(jobs, desc=what, leave=False)]
    from concurrent.futures import ProcessPoolExecutor
    import os
    workers = min(n_jobs or max((os.cpu_count() or 2) - 1, 1), len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(fct, jobs), total=len(jobs), desc=what, leave=False))


def _post_cov(fit, precision):
    """Laplace covariance of a fit under a prior with the given precision"""
    return np.linalg.inv(_psd(fit.hessian) + precision)


def fit_group(bank, data, n_iter=5, tol=0.01, cov="diag", prior0=None, individual=None,
              n_jobs=None):
    """
    Empirical-Bayes fit of a group of recordings (see
    :mod:`~pypillometry.signal.gaussmarkov.hierarchical`).

    Parameters
    ----------
    bank : TimescaleBank
        the model, the same for all recordings
    data : list of (y, fs, segments)
        the recordings: signal (NaN where missing), sampling rate in Hz and
        segments (list of (start, stop) or None), as for :func:`fit_bank`
    n_iter : int
        maximum number of EM iterations (default 5)
    tol : float
        stop when no entry of the group mean changes by more than this
        (default 0.01)
    cov : {"diag", "full"}
        ``"diag"`` (default): the parameters vary independently between
        recordings; ``"full"``: with correlations, which need large groups
        (about 50 recordings did not determine them in tests)
    prior0 : GaussianPrior or str, optional
        prior of the fits without pooling that start the algorithm;
        default: :func:`reference_prior`
    individual : list of BankFit, optional
        fits without pooling from an earlier run (with Hessians), to skip
        the first step
    n_jobs : int, optional
        number of processes (default: one per core but one; 1: no
        parallel processes)

    Returns
    -------
    GroupFit
    """
    if cov not in ("diag", "full"):
        raise ValueError(f"cov must be 'diag' or 'full', got {cov!r}")
    data = list(data)
    N, p = len(data), bank.n_params
    if N < 2:
        raise ValueError("a group needs at least two recordings")
    prior0 = reference_prior(bank) if prior0 is None else _as_prior(prior0, bank)
    if individual is None:
        individual = _map(_fit_job, [(bank, d, prior0, None) for d in data], n_jobs,
                          "individual fits")
    elif len(individual) != N or any(f.hessian is None for f in individual):
        raise ValueError("individual must hold one fit with a Hessian per recording")
    prec0 = np.zeros((p, p)) if individual[0].prior is None else individual[0].prior.precision

    # start: covariance of the individual estimates minus their mean
    # estimation variance, floored at 20 % of the raw variances
    TH = np.array([f.theta for f in individual])
    C = np.cov(TH.T, bias=True).reshape(p, p)
    V0 = np.mean([np.linalg.pinv(_psd(f.hessian) + prec0) for f in individual], axis=0)
    mu = TH.mean(axis=0)
    Sigma = _psd(C - V0)
    Sigma += np.diag(np.maximum(0.2 * np.diag(C) - np.diag(Sigma), 0.0)) + 1e-6 * np.eye(p)
    if cov == "diag":
        Sigma = np.diag(np.diag(Sigma))
    history = [(mu.copy(), Sigma.copy())]
    fits, converged = individual, False
    for it in range(n_iter):
        prior = GaussianPrior(mu.copy(), Sigma.copy(), list(bank.names), "group")
        fits = _map(_fit_job, [(bank, d, prior, f.theta) for d, f in zip(data, fits)],
                    n_jobs, f"EM iteration {it + 1}")
        TH = np.array([f.theta for f in fits])
        prec = prior.precision
        V = np.mean([_post_cov(f, prec) for f in fits], axis=0)
        new_mu = TH.mean(axis=0)
        Sigma = np.cov(TH.T, bias=True).reshape(p, p) + V
        if cov == "diag":
            Sigma = np.diag(np.diag(Sigma))
        change = float(np.max(np.abs(new_mu - mu)))
        mu = new_mu
        history.append((mu.copy(), Sigma.copy()))
        logger.debug(f"fit_group: iteration {it + 1}, largest change of the mean {change:.3f}")
        if change < tol:
            converged = True
            break
    if not converged:
        logger.info(f"fit_group: stopped after {n_iter} iterations (largest change of "
                    f"the group mean {change:.3f}, tolerance {tol})")
    # the shrunk fits were made under the group parameters of the last E-step
    return GroupFit(bank, mu, Sigma, fits, individual, history, converged)
