"""
kalman.py
=========

Kalman filter, Rauch-Tung-Striebel smoother and simulation for a general
linear-Gaussian state-space model (:class:`LGSSM`)

.. math::

    x_0 \\sim N(m_0, P_0), \\quad
    x_t = A_t x_{t-1} + w_t,\\ w_t \\sim N(0, Q_t), \\quad
    y_t = H x_t + e_t,\\ e_t \\sim N(0, R).

The functions know nothing about OU or Matérn processes; any stack of
components (see :meth:`GaussMarkovModel.to_lgssm`) works. Missing observations (NaN) are
skipped in the update, so gaps are handled exactly; with several observed
channels, each one can be missing on its own.

The inner loops are compiled with numba if it is installed (``pip install
numba``); otherwise they run as (slow) plain Python.
"""
from dataclasses import dataclass

import numpy as np
from loguru import logger

__all__ = ["LGSSM", "FilterResult", "SmootherResult", "kalman_filter",
           "kalman_loglik", "rts_smoother", "component_states", "simulate"]

_LOG2PI = float(np.log(2 * np.pi))


# ---------------------------------------------------------------------------
# kernels (plain numpy; compiled with numba when available)
# ---------------------------------------------------------------------------
def _filter_kernel(A, Q, aidx, H, R, m0, P0, y, store,
                   mp, Pp, mf, Pf, v, S, K, llt):
    T, m = y.shape
    n = m0.shape[0]
    x = m0.copy()
    P = P0.copy()
    ll = 0.0
    for t in range(T):
        if t > 0:
            Ak = A[aidx[t]]
            x = Ak @ x
            P = Ak @ P @ Ak.T + Q[aidx[t]]
            P = 0.5 * (P + P.T)
        if store:
            mp[t] = x
            Pp[t] = P
        nv = 0
        for j in range(m):
            if not np.isnan(y[t, j]):
                nv += 1
        lt = 0.0
        if m == 1 and nv == 1:
            # scalar update (single channel): no linear-algebra calls
            h = H[0]
            ph = P @ h
            s = h @ ph + R[0, 0]
            vt = y[t, 0] - h @ x
            k = ph / s
            lt = -0.5 * (_LOG2PI + np.log(s) + vt * vt / s)
            ll += lt
            x = x + k * vt
            for a in range(n):
                for b in range(n):
                    P[a, b] -= k[a] * ph[b]
            P = 0.5 * (P + P.T)
            if store:
                v[t, 0] = vt
                K[t, :, 0] = k
        elif nv > 0:
            idx = np.empty(nv, dtype=np.int64)
            c = 0
            for j in range(m):
                if not np.isnan(y[t, j]):
                    idx[c] = j
                    c += 1
            Hs = H[idx]
            Rs = R[idx][:, idx]
            yt = y[t]
            vs = yt[idx] - Hs @ x
            PHt = P @ Hs.T
            Ss = Hs @ PHt + Rs
            Ss = 0.5 * (Ss + Ss.T)
            Lc = np.linalg.cholesky(Ss)
            Ks = np.linalg.solve(Ss, PHt.T).T
            alpha = np.linalg.solve(Ss, vs)
            logdet = 2.0 * np.sum(np.log(np.diag(Lc)))
            lt = -0.5 * (nv * _LOG2PI + logdet + vs @ alpha)
            ll += lt
            x = x + Ks @ vs
            P = P - Ks @ Ss @ Ks.T
            P = 0.5 * (P + P.T)
            if store:
                for a in range(nv):
                    v[t, idx[a]] = vs[a]
                    for b in range(n):
                        K[t, b, idx[a]] = Ks[b, a]
        if store:
            S[t] = H @ Pp[t] @ H.T + R
            mf[t] = x
            Pf[t] = P
            llt[t] = lt
    return ll


def _smoother_kernel(A, aidx, mp, Pp, mf, Pf, ms, Ps, G, Plag):
    T = mf.shape[0]
    ms[T - 1] = mf[T - 1]
    Ps[T - 1] = Pf[T - 1]
    for t in range(T - 2, -1, -1):
        Ak = A[aidx[t + 1]]
        Gt = np.linalg.solve(Pp[t + 1], Ak @ Pf[t]).T
        ms[t] = mf[t] + Gt @ (ms[t + 1] - mp[t + 1])
        Pt = Pf[t] + Gt @ (Ps[t + 1] - Pp[t + 1]) @ Gt.T
        Ps[t] = 0.5 * (Pt + Pt.T)
        G[t] = Gt
        Plag[t + 1] = Ps[t + 1] @ Gt.T


def _simulate_kernel(A, Qsqrt, aidx, H, Rsqrt, x0, ex, ey, xs, ys):
    T = ex.shape[0]
    x = x0.copy()
    for t in range(T):
        if t > 0:
            x = A[aidx[t]] @ x + Qsqrt[aidx[t]] @ ex[t]
        xs[t] = x
        ys[t] = H @ x + Rsqrt @ ey[t]


_KERNELS = None
_JIT = False


def _kernels():
    """the kernels, compiled with numba on first use if it is installed"""
    global _KERNELS, _JIT
    if _KERNELS is None:
        try:
            import numba
        except ImportError:
            logger.warning("numba is not installed: the Kalman filter runs as "
                           "plain Python and is slow for long signals")
            _KERNELS = (_filter_kernel, _smoother_kernel, _simulate_kernel)
        else:
            jit = numba.njit(cache=True)
            _KERNELS = (jit(_filter_kernel), jit(_smoother_kernel),
                        jit(_simulate_kernel))
            _JIT = True
    return _KERNELS


# rough cost per step and squared state dimension (s, with numba; measured)
_SEC_PER_STEP = {"kalman_loglik": 1e-7, "kalman_filter": 2e-7, "rts_smoother": 4e-7}
_FLOATS_PER_STEP = {"kalman_loglik": lambda n, m: 0,
                    "kalman_filter": lambda n, m: 2 * n * n + 2 * n + n * m + m * m + m,
                    "rts_smoother": lambda n, m: 3 * n * n + n}
_MAX_SEC, _MAX_BYTES = 30.0, 2e9
_WARNED = set()


def _check_cost(what, T, n, m):
    """warn (once per call signature) if a run will be slow or use much memory"""
    sec = _SEC_PER_STEP[what] * T * n * n * (1 if _JIT else 30)
    mem = 8.0 * T * _FLOATS_PER_STEP[what](n, m)
    if (sec > _MAX_SEC or mem > _MAX_BYTES) and (what, T, n, m) not in _WARNED:
        _WARNED.add((what, T, n, m))
        logger.warning(f"{what}: {T} samples with a {n}-dim state will take "
                       f"~{sec:.0f} s and ~{mem / 1e9:.1f} GB "
                       f"{'' if _JIT else '(numba not installed) '}"
                       "- consider downsampling or a shorter segment")


# ---------------------------------------------------------------------------
# model and results
# ---------------------------------------------------------------------------
@dataclass
class LGSSM:
    """
    Linear-Gaussian state-space model.

    Attributes
    ----------
    A : (n, n) or (T, n, n)
        transition matrix; if 3-d, ``A[t]`` is the transition from t-1 to t
        (``A[0]`` is unused)
    Q : (n, n) or (T, n, n)
        process-noise covariance, shaped like `A`
    H : (m, n)
        observation matrix
    R : (m, m)
        measurement-noise covariance
    m0 : (n,)
        mean of the state at the first sample
    P0 : (n, n)
        covariance of the state at the first sample
    """
    A: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: np.ndarray
    m0: np.ndarray
    P0: np.ndarray

    @property
    def dim_state(self):
        return self.m0.shape[0]

    @property
    def dim_obs(self):
        return self.H.shape[0]

    def _arrays(self, T):
        """float64 C-contiguous arrays and the index of A/Q for each step"""
        f = lambda a: np.ascontiguousarray(a, dtype=np.float64)
        A, Q = f(self.A), f(self.Q)
        if A.ndim == 2:
            A, Q = A[None], Q[None]
            aidx = np.zeros(T, dtype=np.int64)
        else:
            if A.shape[0] != T:
                raise ValueError(f"time-varying A has {A.shape[0]} steps, data {T}")
            aidx = np.arange(T, dtype=np.int64)
        H = f(np.atleast_2d(self.H))
        R = f(np.atleast_2d(self.R))
        return A, Q, aidx, H, R, f(self.m0), f(self.P0)


def _obs(y, mask, m):
    """observations as a (T, m) float array with NaN where missing"""
    if isinstance(y, np.ma.MaskedArray):
        y = np.where(np.ma.getmaskarray(y), np.nan, np.asarray(y.data, dtype=float))
    y = np.array(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[1] != m:
        raise ValueError(f"y must have shape (T,) or (T, {m}), got {y.shape}")
    y[~np.isfinite(y)] = np.nan
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y[mask.reshape(len(y), -1) if mask.ndim > 1 else mask] = np.nan
    return np.ascontiguousarray(y)


@dataclass
class FilterResult:
    """
    Result of :func:`kalman_filter`.

    ``m_pred[t], P_pred[t]``: state at t given ``y[:t]`` (``m_pred[0] = m0``);
    ``m_filt[t], P_filt[t]``: state at t given ``y[:t+1]``.

    Attributes
    ----------
    m_pred, m_filt : (T, n)
    P_pred, P_filt : (T, n, n)
    innov : (T, m)
        innovations ``y[t] - H m_pred[t]`` (NaN where y is missing)
    innov_var : (T, m, m)
        their covariance ``H P_pred[t] H' + R``
    gain : (T, n, m)
        Kalman gain (0 for missing observations)
    loglik_t : (T,)
        log-likelihood contribution of each sample (0 where missing)
    loglik : float
        total log-likelihood
    valid : (T, m)
        True where y was observed
    """
    m_pred: np.ndarray
    P_pred: np.ndarray
    m_filt: np.ndarray
    P_filt: np.ndarray
    innov: np.ndarray
    innov_var: np.ndarray
    gain: np.ndarray
    loglik_t: np.ndarray
    loglik: float
    valid: np.ndarray

    @property
    def std_innov(self):
        """
        Standardised innovations, (T, m): N(0, I) and white if the model is
        right. Each channel is divided by its own predictive standard
        deviation (NaN where missing).
        """
        sd = np.sqrt(np.diagonal(self.innov_var, axis1=1, axis2=2))
        return self.innov / sd


@dataclass
class SmootherResult:
    """
    Result of :func:`rts_smoother`.

    Attributes
    ----------
    m_smooth : (T, n)
        mean of the state given all data
    P_smooth : (T, n, n)
        its covariance
    G : (T, n, n)
        smoother gains (``G[T-1]`` is 0)
    P_lag1 : (T, n, n)
        ``Cov(x[t], x[t-1] | all data)`` (``P_lag1[0]`` is 0); needed for EM
    """
    m_smooth: np.ndarray
    P_smooth: np.ndarray
    G: np.ndarray
    P_lag1: np.ndarray


# ---------------------------------------------------------------------------
# public functions
# ---------------------------------------------------------------------------
def kalman_filter(model, y, mask=None):
    """
    Run the Kalman filter and keep everything.

    Parameters
    ----------
    model : LGSSM
    y : array, (T,) or (T, m)
        observations; NaN, inf and masked values are missing
    mask : array of bool, optional
        True for additional missing samples

    Returns
    -------
    FilterResult

    Notes
    -----
    Stores O(T n^2) numbers; use :func:`kalman_loglik` when only the
    likelihood is needed.
    """
    flt, _, _ = _kernels()
    y = _obs(y, mask, model.dim_obs)
    T, m = y.shape
    A, Q, aidx, H, R, m0, P0 = model._arrays(T)
    n = m0.shape[0]
    _check_cost("kalman_filter", T, n, m)
    mp, mf = np.zeros((T, n)), np.zeros((T, n))
    Pp, Pf = np.zeros((T, n, n)), np.zeros((T, n, n))
    v = np.full((T, m), np.nan)
    S = np.zeros((T, m, m))
    K = np.zeros((T, n, m))
    llt = np.zeros(T)
    ll = flt(A, Q, aidx, H, R, m0, P0, y, True, mp, Pp, mf, Pf, v, S, K, llt)
    return FilterResult(mp, Pp, mf, Pf, v, S, K, llt, float(ll), ~np.isnan(y))


def kalman_loglik(model, y, mask=None):
    """
    Log-likelihood of `y` under `model` (Kalman filter without storage).

    Parameters are as in :func:`kalman_filter`.

    Returns
    -------
    float
    """
    flt, _, _ = _kernels()
    y = _obs(y, mask, model.dim_obs)
    T, m = y.shape
    A, Q, aidx, H, R, m0, P0 = model._arrays(T)
    n = m0.shape[0]
    _check_cost("kalman_loglik", T, n, m)
    d2, d3 = np.zeros((1, n)), np.zeros((1, n, n))
    return float(flt(A, Q, aidx, H, R, m0, P0, y, False, d2, d3, d2, d3,
                     np.zeros((1, m)), np.zeros((1, m, m)), np.zeros((1, n, m)),
                     np.zeros(1)))


def rts_smoother(model, fres):
    """
    Rauch-Tung-Striebel smoother.

    Parameters
    ----------
    model : LGSSM
        the model that produced `fres`
    fres : FilterResult
        output of :func:`kalman_filter`

    Returns
    -------
    SmootherResult
    """
    _, smo, _ = _kernels()
    T, n = fres.m_filt.shape
    _check_cost("rts_smoother", T, n, fres.innov.shape[1])
    A, _, aidx, *_ = model._arrays(T)
    ms, Ps = np.zeros((T, n)), np.zeros((T, n, n))
    G, Plag = np.zeros((T, n, n)), np.zeros((T, n, n))
    smo(A, aidx, fres.m_pred, fres.P_pred, fres.m_filt, fres.P_filt, ms, Ps, G, Plag)
    return SmootherResult(ms, Ps, G, Plag)


def component_states(model, m, P=None):
    """
    Split the latent signal into the contributions of the components.

    Parameters
    ----------
    model : GaussMarkovModel
    m : (T, n) array, SmootherResult or FilterResult
        state means; for a result object the smoothed (or filtered) mean and
        covariance are used and `P` is ignored
    P : (T, n, n) array, optional
        state covariances

    Returns
    -------
    dict
        name of component -> (mean, var) of its output, each (T,)
        (var is None if no covariance was given)
    """
    if isinstance(m, SmootherResult):
        m, P = m.m_smooth, m.P_smooth
    elif isinstance(m, FilterResult):
        m, P = m.m_filt, m.P_filt
    out = {}
    for name, sl in model.slices.items():
        h = model.components[name].ss().H[0]
        mean = m[:, sl] @ h
        var = None if P is None else np.einsum("i,tij,j->t", h, P[:, sl, sl], h)
        out[name] = (mean, var)
    return out


def _sqrtm_psd(M):
    """symmetric square root of a PSD matrix (negative eigenvalues clipped)"""
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def simulate(model, n, fs=None, rng=None, return_states=False):
    """
    Simulate from a state-space model.

    Parameters
    ----------
    model : LGSSM or GaussMarkovModel
        a :class:`GaussMarkovModel` is discretised at `fs`
    n : int
        number of samples
    fs : float, optional
        sampling rate in Hz (needed for a GaussMarkovModel)
    rng : np.random.Generator or int, optional
    return_states : bool, optional
        also return the states

    Returns
    -------
    y : (n,) or (n, m) array
        observations ((n,) for a single channel)
    x : (n, dim) array
        states (if `return_states`)

    Examples
    --------
    >>> from pypillometry.signal.gaussmarkov import OU, GaussMarkovModel
    >>> y = simulate(GaussMarkovModel([OU(1., 50.)], noise=0.1), 1000, fs=1000, rng=0)
    >>> y.shape
    (1000,)
    """
    if not isinstance(model, LGSSM):
        if fs is None:
            raise ValueError("fs is needed to simulate from a GaussMarkovModel")
        model = model.to_lgssm(fs)
    _, _, sim = _kernels()
    rng = np.random.default_rng(rng)
    A, Q, aidx, H, R, m0, P0 = model._arrays(n)
    Qsqrt = np.ascontiguousarray([_sqrtm_psd(q) for q in Q])
    Rsqrt = np.ascontiguousarray(_sqrtm_psd(R))
    d, m = m0.shape[0], H.shape[0]
    x0 = m0 + _sqrtm_psd(P0) @ rng.standard_normal(d)
    ex, ey = rng.standard_normal((n, d)), rng.standard_normal((n, m))
    xs, ys = np.zeros((n, d)), np.zeros((n, m))
    sim(A, Qsqrt, aidx, H, Rsqrt, x0, ex, ey, xs, ys)
    y = ys[:, 0] if m == 1 else ys
    return (y, xs) if return_states else y
