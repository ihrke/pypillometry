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

Three optional arguments of :func:`kalman_filter` and :func:`kalman_loglik`
extend the model (they need a diagonal ``R``, i.e. independent noise of the
channels, which are then processed one after the other):

- ``obs_var``: a noise variance for each sample instead of the constant
  ``R`` (e.g., for means of different numbers of raw samples)
- ``segments``: independent stretches of the data; the state restarts from
  ``(m0, P0)`` at the start of each
- ``n_diffuse``: the number of observations at the start of each segment
  that only determine states with a diffuse prior (a very large variance in
  ``P0``, e.g. an unknown mean) and are left out of the likelihood, which
  is then the restricted likelihood (REML) of the other parameters

The inner loops are compiled with numba if it is installed (``pip install
numba``); otherwise they run as (slow) plain Python.
"""
from dataclasses import dataclass

import numpy as np
from loguru import logger

__all__ = ["LGSSM", "FilterResult", "SmootherResult", "kalman_filter",
           "kalman_loglik", "rts_smoother", "component_states", "simulate",
           "forecast_errors"]

_LOG2PI = float(np.log(2 * np.pi))


# ---------------------------------------------------------------------------
# kernels (plain numpy; compiled with numba when available)
# ---------------------------------------------------------------------------
def _filter_kernel(A, Q, H, R, m0, P0, y, store,
                   mp, Pp, mf, Pf, v, S, K, llt):
    T, m = y.shape
    n = m0.shape[0]
    x = m0.copy()
    P = P0.copy()
    ll = 0.0
    for t in range(T):
        if t > 0:
            x = A @ x
            P = A @ P @ A.T + Q
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


def _seq_loglik_kernel(A, Q, H, r, m0, P0, y, restart, skip):
    """
    Log-likelihood with the channels processed one after the other (needs
    independent noise of the channels): noise variance `r[t, j]` per sample,
    restart from (m0, P0) where `restart[t]`, and the first `skip[j]` valid
    observations of channel j after each restart left out of the sum
    (diffuse states). Explicit loops without temporary arrays; uses the
    band of non-zero entries in each row of A (block-diagonal models).
    """
    T, m = y.shape
    n = m0.shape[0]
    lo = np.zeros(n, np.int64)
    hi = np.zeros(n, np.int64)
    for i in range(n):
        a = 0
        while a < n and A[i, a] == 0.0:
            a += 1
        b = n
        while b > a and A[i, b - 1] == 0.0:
            b -= 1
        lo[i] = a
        hi[i] = b
    x = m0.copy()
    P = P0.copy()
    x2 = np.zeros(n)
    W = np.zeros((n, n))
    ph = np.zeros(n)
    seen = np.zeros(m, np.int64)
    ll = 0.0
    for t in range(T):
        if restart[t]:
            for i in range(n):
                x[i] = m0[i]
                for j in range(n):
                    P[i, j] = P0[i, j]
            for j in range(m):
                seen[j] = 0
        elif t > 0:
            for i in range(n):
                acc = 0.0
                for j in range(lo[i], hi[i]):
                    acc += A[i, j] * x[j]
                x2[i] = acc
            for i in range(n):
                x[i] = x2[i]
            for i in range(n):          # W = A P
                for j in range(n):
                    acc = 0.0
                    for l in range(lo[i], hi[i]):
                        acc += A[i, l] * P[l, j]
                    W[i, j] = acc
            for i in range(n):          # P = W A' + Q
                for j in range(i, n):
                    acc = Q[i, j]
                    for l in range(lo[j], hi[j]):
                        acc += W[i, l] * A[j, l]
                    P[i, j] = acc
                    P[j, i] = acc
        for c in range(m):
            yt = y[t, c]
            if np.isnan(yt):
                continue
            s = r[t, c]
            pred = 0.0
            for i in range(n):
                acc = 0.0
                for j in range(n):
                    acc += P[i, j] * H[c, j]
                ph[i] = acc
                s += H[c, i] * acc
                pred += H[c, i] * x[i]
            v = yt - pred
            if seen[c] >= skip[c]:
                ll += -0.5 * (_LOG2PI + np.log(s) + v * v / s)
            seen[c] += 1
            for i in range(n):
                x[i] += ph[i] * v / s
                for j in range(i, n):
                    P[i, j] -= ph[i] * ph[j] / s
                    P[j, i] = P[i, j]
    return ll


def _seq_filter_kernel(A, Q, H, r, m0, P0, y, restart, skip,
                       mp, Pp, mf, Pf, v, S, K, llt):
    """Like :func:`_seq_loglik_kernel`, but everything is stored. `v[t, j]`
    and `S[t, j]` are the innovation of channel j and its variance given the
    channels before it at the same sample (S is also set where y is missing)."""
    T, m = y.shape
    n = m0.shape[0]
    x = m0.copy()
    P = P0.copy()
    seen = np.zeros(m, np.int64)
    ll = 0.0
    for t in range(T):
        if restart[t]:
            x = m0.copy()
            P = P0.copy()
            seen[:] = 0
        elif t > 0:
            x = A @ x
            P = A @ P @ A.T + Q
            P = 0.5 * (P + P.T)
        mp[t] = x
        Pp[t] = P
        lt = 0.0
        for c in range(m):
            h = H[c]
            ph = P @ h
            s = h @ ph + r[t, c]
            S[t, c] = s
            if np.isnan(y[t, c]):
                continue
            vt = y[t, c] - h @ x
            v[t, c] = vt
            k = ph / s
            if seen[c] >= skip[c]:
                lt += -0.5 * (_LOG2PI + np.log(s) + vt * vt / s)
            seen[c] += 1
            x = x + k * vt
            for a in range(n):
                for b in range(n):
                    P[a, b] -= k[a] * ph[b]
            P = 0.5 * (P + P.T)
            K[t, :, c] = k
        ll += lt
        llt[t] = lt
        mf[t] = x
        Pf[t] = P
    return ll


def _forecast_kernel(A, Q, H, r, m0, P0, y, restart, horizons, step, E, Z):
    """Errors `E[o, j, c]` and standardised errors `Z[o, j, c]` of the
    forecasts of channel c made `horizons[j]` samples ahead from the
    filtered state at every `step`-th sample (the origins o); NaN where the
    origin or the target is missing or they are in different segments."""
    T, m = y.shape
    n = m0.shape[0]
    nh = horizons.shape[0]
    hmax = horizons[nh - 1]
    x = m0.copy()
    P = P0.copy()
    for t in range(T):
        if restart[t]:
            x = m0.copy()
            P = P0.copy()
        elif t > 0:
            x = A @ x
            P = A @ P @ A.T + Q
        nv = 0
        for c in range(m):
            if np.isnan(y[t, c]):
                continue
            nv += 1
            h = H[c]
            ph = P @ h
            s = h @ ph + r[t, c]
            k = ph / s
            x = x + k * (y[t, c] - h @ x)
            for a in range(n):
                for b in range(n):
                    P[a, b] -= k[a] * ph[b]
            P = 0.5 * (P + P.T)
        if t % step == 0 and nv > 0:
            xx = x.copy()
            PP = P.copy()
            j = 0
            for d in range(1, hmax + 1):
                if t + d >= T or restart[t + d]:
                    break
                xx = A @ xx
                PP = A @ PP @ A.T + Q
                if d == horizons[j]:
                    for c in range(m):
                        if not np.isnan(y[t + d, c]):
                            h = H[c]
                            e = y[t + d, c] - h @ xx
                            E[t // step, j, c] = e
                            Z[t // step, j, c] = e / np.sqrt(h @ PP @ h + r[t + d, c])
                    j += 1


def _smoother_kernel(A, mp, Pp, mf, Pf, restart, ms, Ps, G, Plag):
    T, n = mf.shape
    ms[T - 1] = mf[T - 1]
    Ps[T - 1] = Pf[T - 1]
    for t in range(T - 2, -1, -1):
        if restart[t + 1]:      # end of an independent segment
            ms[t] = mf[t]
            Ps[t] = Pf[t]
            continue
        # solve with the correlation matrix of the prediction: states of very
        # different size (or without variance) do not make it singular
        d = np.sqrt(np.diag(Pp[t + 1]))
        C = Pp[t + 1].copy()
        for i in range(n):
            if d[i] <= 0.0:
                d[i] = 1.0
                C[i, i] = 1.0
        C = C / np.outer(d, d)
        APf = A @ Pf[t]
        for i in range(n):
            APf[i] /= d[i]
        Gt = np.linalg.solve(C, APf).T
        for i in range(n):
            Gt[:, i] /= d[i]
        ms[t] = mf[t] + Gt @ (ms[t + 1] - mp[t + 1])
        Pt = Pf[t] + Gt @ (Ps[t + 1] - Pp[t + 1]) @ Gt.T
        Ps[t] = 0.5 * (Pt + Pt.T)
        G[t] = Gt
        Plag[t + 1] = Ps[t + 1] @ Gt.T


def _simulate_kernel(A, Qsqrt, H, Rsqrt, x0, ex, ey, xs, ys):
    T = ex.shape[0]
    x = x0.copy()
    for t in range(T):
        if t > 0:
            x = A @ x + Qsqrt @ ex[t]
        xs[t] = x
        ys[t] = H @ x + Rsqrt @ ey[t]


_KERNELS = None
_JIT = False


def _kernels():
    """
    The kernels (filter, smoother, simulation, sequential log-likelihood,
    sequential filter, forecasts), compiled with numba on first use if it
    is installed.
    """
    global _KERNELS, _JIT
    if _KERNELS is None:
        plain = (_filter_kernel, _smoother_kernel, _simulate_kernel,
                 _seq_loglik_kernel, _seq_filter_kernel, _forecast_kernel)
        try:
            import numba
        except ImportError:
            logger.warning("numba is not installed: the Kalman filter runs as "
                           "plain Python and is slow for long signals")
            _KERNELS = plain
        else:
            jit = numba.njit(cache=True)
            _KERNELS = tuple(jit(k) for k in plain)
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
    A : (n, n)
        transition matrix from one sample to the next
    Q : (n, n)
        process-noise covariance
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

    def _arrays(self):
        """float64 C-contiguous arrays for the kernels"""
        f = lambda a: np.ascontiguousarray(a, dtype=np.float64)
        A, Q = f(self.A), f(self.Q)
        n = self.dim_state
        if A.shape != (n, n) or Q.shape != (n, n):
            raise ValueError(f"A and Q must be ({n}, {n}), got {A.shape} and {Q.shape}")
        H = f(np.atleast_2d(self.H))
        R = f(np.atleast_2d(self.R))
        return A, Q, H, R, f(self.m0), f(self.P0)


def _signal_mean(H, m):
    """H m for each t: (T,) for one channel, else (T, m)"""
    f = m @ H.T
    return f[:, 0] if H.shape[0] == 1 else f


def _signal_var(H, P):
    """diag(H P H') for each t: (T,) for one channel, else (T, m)"""
    v = np.einsum("ij,tjk,ik->ti", H, P, H)
    return v[:, 0] if H.shape[0] == 1 else v


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
    H : (m, n)
        observation matrix of the model (for :attr:`signal`)
    restart : (T,)
        True at the first sample of each independent segment
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
    H: np.ndarray
    restart: np.ndarray = None

    @property
    def signal(self):
        """
        Filtered mean of the noise-free signal, ``H m_filt[t]``: (T,) for
        one channel, (T, m) for several.
        """
        return _signal_mean(self.H, self.m_filt)

    @property
    def signal_var(self):
        """Its variance, ``diag(H P_filt[t] H')``; shaped like :attr:`signal`."""
        return _signal_var(self.H, self.P_filt)

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
        smoother gains (0 at the last sample of each segment)
    P_lag1 : (T, n, n)
        ``Cov(x[t], x[t-1] | all data)`` (0 at the first sample of each
        segment); needed for EM
    H : (m, n)
        observation matrix of the model (for :attr:`signal`)
    """
    m_smooth: np.ndarray
    P_smooth: np.ndarray
    G: np.ndarray
    P_lag1: np.ndarray
    H: np.ndarray

    @property
    def signal(self):
        """
        Smoothed mean of the noise-free signal, ``H m_smooth[t]``: (T,) for
        one channel, (T, m) for several.
        """
        return _signal_mean(self.H, self.m_smooth)

    @property
    def signal_var(self):
        """Its variance, ``diag(H P_smooth[t] H')``; shaped like :attr:`signal`."""
        return _signal_var(self.H, self.P_smooth)


# ---------------------------------------------------------------------------
# public functions
# ---------------------------------------------------------------------------
def _restart(T, segments):
    """
    Boolean array that is True at the first sample of each segment, and the
    samples outside the segments (None: one segment, no sample outside).
    """
    restart = np.zeros(T, dtype=bool)
    if segments is None:
        restart[:1] = True
        return restart, np.zeros(T, dtype=bool)
    outside = np.ones(T, dtype=bool)
    end = 0
    for a, b in sorted((int(a), int(b)) for a, b in segments):
        if not (end <= a < b <= T):
            raise ValueError("segments must be (start, stop) sample indices "
                             f"that do not overlap, within 0 and {T}")
        restart[a] = True
        outside[a:b] = False
        end = b
    if end < T:     # the samples after the last segment do not belong to it
        restart[end] = True
    return restart, outside


def _seq_args(model, y, mask, obs_var, n_diffuse, segments):
    """arguments of the sequential kernels (see :func:`kalman_filter`)"""
    y = _obs(y, mask, model.dim_obs)
    T, m = y.shape
    A, Q, H, R, m0, P0 = model._arrays()
    if np.any(np.abs(R - np.diag(np.diag(R))) > 0):
        raise ValueError("obs_var, n_diffuse and segments need independent noise "
                         "of the channels (a diagonal R)")
    if obs_var is None:
        r = np.tile(np.diag(R), (T, 1))
    else:
        r = np.asarray(obs_var, dtype=np.float64)
        r = r[:, None] if r.ndim == 1 else r
        if r.shape != (T, m):
            raise ValueError(f"obs_var must have shape ({T},) or ({T}, {m}), got {r.shape}")
        if not np.all(r[~np.isnan(y)] >= 0):
            raise ValueError("obs_var must be >= 0 for all observed samples")
    skip = np.broadcast_to(np.asarray(n_diffuse, dtype=np.int64), (m,)).copy()
    restart, outside = _restart(T, segments)
    y[outside] = np.nan
    return A, Q, H, np.ascontiguousarray(r), m0, P0, y, restart, skip


def kalman_filter(model, y, mask=None, obs_var=None, n_diffuse=0, segments=None):
    """
    Run the Kalman filter and keep everything.

    Parameters
    ----------
    model : LGSSM
    y : array, (T,) or (T, m)
        observations; NaN, inf and masked values are missing
    mask : array of bool, optional
        True for additional missing samples
    obs_var : array, (T,) or (T, m), optional
        noise variance of each sample, instead of the diagonal of ``R``
    n_diffuse : int or sequence of int, optional
        number of valid observations (of each channel) at the start of each
        segment that are left out of the likelihood, because they only
        determine states with a diffuse prior (a very large variance in
        ``P0``); default 0
    segments : list of (start, stop), optional
        sample indices of independent segments: the state restarts from
        ``(m0, P0)`` at each start; samples outside the segments are
        missing. Default: the whole signal is one segment.

    Returns
    -------
    FilterResult

    Notes
    -----
    Stores O(T n^2) numbers; use :func:`kalman_loglik` when only the
    likelihood is needed. With `obs_var`, `n_diffuse` or `segments`, the
    channels are processed one after the other (they need a diagonal ``R``):
    the innovation of a channel and its variance are then conditional on the
    channels before it at the same sample, and ``innov_var`` is diagonal.
    """
    kern = _kernels()
    T = len(y)
    n, m = model.dim_state, model.dim_obs
    _check_cost("kalman_filter", T, n, m)
    mp, mf = np.zeros((T, n)), np.zeros((T, n))
    Pp, Pf = np.zeros((T, n, n)), np.zeros((T, n, n))
    v = np.full((T, m), np.nan)
    S = np.zeros((T, m, m))
    K = np.zeros((T, n, m))
    llt = np.zeros(T)
    if obs_var is None and segments is None and not np.any(n_diffuse):
        y = _obs(y, mask, m)
        A, Q, H, R, m0, P0 = model._arrays()
        ll = kern[0](A, Q, H, R, m0, P0, y, True, mp, Pp, mf, Pf, v, S, K, llt)
        restart = _restart(T, None)[0]
    else:
        A, Q, H, r, m0, P0, y, restart, skip = _seq_args(
            model, y, mask, obs_var, n_diffuse, segments)
        Sd = np.zeros((T, m))
        ll = kern[4](A, Q, H, r, m0, P0, y, restart, skip,
                     mp, Pp, mf, Pf, v, Sd, K, llt)
        S[:, np.arange(m), np.arange(m)] = Sd
    return FilterResult(mp, Pp, mf, Pf, v, S, K, llt, float(ll), ~np.isnan(y), H,
                        restart)


def kalman_loglik(model, y, mask=None, obs_var=None, n_diffuse=0, segments=None):
    """
    Log-likelihood of `y` under `model` (Kalman filter without storage).

    Parameters are as in :func:`kalman_filter`.

    Returns
    -------
    float
    """
    kern = _kernels()
    n, m = model.dim_state, model.dim_obs
    _check_cost("kalman_loglik", len(y), n, m)
    if obs_var is None and segments is None and not np.any(n_diffuse):
        y = _obs(y, mask, m)
        A, Q, H, R, m0, P0 = model._arrays()
        d2, d3 = np.zeros((1, n)), np.zeros((1, n, n))
        return float(kern[0](A, Q, H, R, m0, P0, y, False, d2, d3, d2, d3,
                             np.zeros((1, m)), np.zeros((1, m, m)),
                             np.zeros((1, n, m)), np.zeros(1)))
    return float(kern[3](*_seq_args(model, y, mask, obs_var, n_diffuse, segments)))


def rts_smoother(model, fres):
    """
    Rauch-Tung-Striebel smoother.

    Parameters
    ----------
    model : LGSSM
        the model that produced `fres`
    fres : FilterResult
        output of :func:`kalman_filter`; its segments are smoothed
        independently of each other

    Returns
    -------
    SmootherResult
    """
    smo = _kernels()[1]
    T, n = fres.m_filt.shape
    _check_cost("rts_smoother", T, n, fres.innov.shape[1])
    A, _, H, *_ = model._arrays()
    restart = fres.restart if fres.restart is not None else _restart(T, None)[0]
    ms, Ps = np.zeros((T, n)), np.zeros((T, n, n))
    G, Plag = np.zeros((T, n, n)), np.zeros((T, n, n))
    smo(A, fres.m_pred, fres.P_pred, fres.m_filt, fres.P_filt,
        np.ascontiguousarray(restart), ms, Ps, G, Plag)
    return SmootherResult(ms, Ps, G, Plag, H)


def forecast_errors(model, y, horizons, step=1, mask=None, obs_var=None,
                    segments=None):
    """
    Errors of forecasts several samples ahead.

    From the filtered state at every `step`-th sample (given the data up to
    that sample), the model predicts the observations ``horizons`` samples
    later. The errors measure the accuracy of the forecasts; the errors
    divided by their predicted standard deviation should have variance 1
    if the model describes the dependence over that horizon (below 1: the
    model expects more variability than there is).

    Parameters
    ----------
    model : LGSSM
        needs independent noise of the channels (a diagonal ``R``)
    y : array, (T,) or (T, m)
        observations; NaN, inf and masked values are missing
    horizons : sequence of int
        how many samples ahead to predict (positive)
    step : int, optional
        forecasts are made from every `step`-th sample (default 1)
    mask, obs_var, segments
        see :func:`kalman_filter`; forecasts do not cross the border of a
        segment

    Returns
    -------
    err : np.ndarray, (n_origins, n_horizons, m)
        forecast errors (NaN where the origin or the target is missing)
    std_err : np.ndarray, (n_origins, n_horizons, m)
        errors divided by their predicted standard deviation
    """
    kern = _kernels()[5]
    horizons = np.unique(np.asarray(horizons, dtype=np.int64))
    if len(horizons) == 0 or horizons[0] < 1:
        raise ValueError("horizons must be positive numbers of samples")
    A, Q, H, r, m0, P0, y, restart, _ = _seq_args(model, y, mask, obs_var, 0, segments)
    T, m = y.shape
    shape = ((T - 1) // int(step) + 1, len(horizons), m)
    E, Z = np.full(shape, np.nan), np.full(shape, np.nan)
    kern(A, Q, H, r, m0, P0, y, restart, horizons, int(step), E, Z)
    return E, Z


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
    sim = _kernels()[2]
    rng = np.random.default_rng(rng)
    A, Q, H, R, m0, P0 = model._arrays()
    Qsqrt = np.ascontiguousarray(_sqrtm_psd(Q))
    Rsqrt = np.ascontiguousarray(_sqrtm_psd(R))
    d, m = m0.shape[0], H.shape[0]
    x0 = m0 + _sqrtm_psd(P0) @ rng.standard_normal(d)
    ex, ey = rng.standard_normal((n, d)), rng.standard_normal((n, m))
    xs, ys = np.zeros((n, d)), np.zeros((n, m))
    sim(A, Qsqrt, H, Rsqrt, x0, ex, ey, xs, ys)
    y = ys[:, 0] if m == 1 else ys
    return (y, xs) if return_states else y
