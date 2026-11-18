"""
components.py
=============

Continuous-time Gauss-Markov processes and sums of them.

Each component is a linear SDE

.. math::

    dx = F x \\, dt + L \\, dW, \\qquad f(t) = H x(t),

with white-noise spectral density :math:`Q_c`, whose output :math:`f(t)` is a
Gaussian process. A :class:`GaussMarkovModel` stacks components
(block-diagonal state) and adds white measurement noise;
:meth:`GaussMarkovModel.to_lgssm` turns it into the discrete matrices used by
:mod:`~pypillometry.signal.gaussmarkov.kalman`.

Times (``tau``, ``ell``, lags) are in ms; rates are per ms.
"""
from dataclasses import dataclass, replace
from math import factorial, comb
from typing import NamedTuple

import numpy as np

__all__ = ["ContinuousSS", "Component", "OU", "Matern", "RandomWalk",
           "GaussMarkovModel"]


class ContinuousSS(NamedTuple):
    """
    Continuous-time state space of a component.

    Attributes
    ----------
    F : (n, n) feedback matrix
    L : (n, 1) noise-effect matrix
    Qc : (1, 1) spectral density of the white noise
    H : (1, n) observation row
    Pinf : (n, n) stationary covariance, or None for non-stationary components
    """
    F: np.ndarray
    L: np.ndarray
    Qc: np.ndarray
    H: np.ndarray
    Pinf: np.ndarray


def _van_loan(F, L, Qc, dt):
    """A = expm(F dt) and Q = int_0^dt expm(F s) L Qc L' expm(F' s) ds."""
    from scipy.linalg import expm
    n = F.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -F
    M[:n, n:] = L @ Qc @ L.T
    M[n:, n:] = F.T
    E = expm(M * dt)
    A = E[n:, n:].T
    Q = A @ E[:n, n:]
    return A, 0.5 * (Q + Q.T)


@dataclass(frozen=True)
class Component:
    """
    Base class of the components. Subclasses are frozen dataclasses whose
    fields listed in ``param_names`` are the (positive) parameters; other
    fields (e.g., ``nu``) are structural and not fitted.
    """
    param_names = ()

    @property
    def dim(self):
        """dimension of the state"""
        return self.ss().F.shape[0]

    @property
    def stationary(self):
        return True

    def ss(self):
        """Continuous-time state space (:class:`ContinuousSS`)."""
        raise NotImplementedError

    def discretise(self, dt):
        """
        Exact discretisation for a time step `dt` (ms).

        Returns
        -------
        A, Q : np.ndarray
            transition matrix and process-noise covariance,
            ``x[t+dt] = A x[t] + w, w ~ N(0, Q)``
        """
        s = self.ss()
        return _van_loan(s.F, s.L, s.Qc, dt)

    def cov(self, tau):
        """Autocovariance k(tau) of the output at lags `tau` (ms)."""
        from scipy.linalg import expm
        s = self.ss()
        if s.Pinf is None:
            raise ValueError(f"{type(self).__name__} is not stationary: no covariance")
        tau = np.abs(np.asarray(tau, dtype=float))
        out = [(s.H @ expm(s.F * t) @ s.Pinf @ s.H.T)[0, 0] for t in tau.ravel()]
        return np.reshape(out, tau.shape)

    def variogram(self, tau):
        """Semivariogram gamma(tau) = k(0) - k(tau) of the output (ms)."""
        tau = np.asarray(tau, dtype=float)
        return self.cov(0.0) - self.cov(tau)

    def get_params(self):
        """dict of the parameters"""
        return {k: getattr(self, k) for k in self.param_names}

    def with_params(self, **kw):
        """copy with some parameters (or structural fields) changed"""
        return replace(self, **kw)


@dataclass(frozen=True)
class Matern(Component):
    """
    Matérn process with half-integer smoothness ``nu = p + 1/2``.

    .. math::

        k(r) = \\sigma^2 e^{-\\lambda r} \\frac{p!}{(2p)!}
               \\sum_{i=0}^{p} \\frac{(p+i)!}{i!(p-i)!} (2\\lambda r)^{p-i},
        \\qquad \\lambda = \\sqrt{2\\nu} / \\ell

    The state holds the process and its first p derivatives
    (Hartikainen & Särkkä, 2010).

    Parameters
    ----------
    sigma : float
        standard deviation (signal units)
    ell : float
        length scale (ms)
    nu : float
        smoothness, one of 0.5, 1.5, 2.5, ...; default 1.5

    Examples
    --------
    >>> m = Matern(sigma=2., ell=100., nu=2.5)
    >>> bool(np.allclose(m.ss().Pinf[0, 0], 4.))
    True
    >>> lam = np.sqrt(5) / 100.
    >>> r = 30.
    >>> bool(np.isclose(m.cov(r), 4 * (1 + lam*r + (lam*r)**2 / 3) * np.exp(-lam*r)))
    True
    """
    sigma: float = 1.0
    ell: float = 100.0
    nu: float = 1.5
    param_names = ("sigma", "ell")

    def __post_init__(self):
        p = self.nu - 0.5
        if p < 0 or abs(p - round(p)) > 1e-12:
            raise ValueError(f"nu must be 0.5, 1.5, 2.5, ...; got {self.nu}")

    @property
    def p(self):
        return int(round(self.nu - 0.5))

    @property
    def lam(self):
        return np.sqrt(2 * self.nu) / self.ell

    def ss(self):
        from scipy.linalg import solve_continuous_lyapunov
        p, lam = self.p, self.lam
        n = p + 1
        F = np.diag(np.ones(n - 1), 1)
        F[-1, :] = [-comb(n, k) * lam**(n - k) for k in range(n)]
        L = np.zeros((n, 1))
        L[-1, 0] = 1.0
        q = (self.sigma**2 * factorial(p)**2 / factorial(2 * p)
             * (2 * lam)**(2 * p + 1))
        Qc = np.array([[q]])
        H = np.zeros((1, n))
        H[0, 0] = 1.0
        Pinf = solve_continuous_lyapunov(F, -L @ Qc @ L.T)
        return ContinuousSS(F, L, Qc, H, 0.5 * (Pinf + Pinf.T))

    def cov(self, tau):
        p, lam = self.p, self.lam
        r = np.abs(np.asarray(tau, dtype=float))
        poly = sum(factorial(p + i) / (factorial(i) * factorial(p - i))
                   * (2 * lam * r)**(p - i) for i in range(p + 1))
        return self.sigma**2 * np.exp(-lam * r) * poly * factorial(p) / factorial(2 * p)


@dataclass(frozen=True)
class OU(Component):
    """
    Ornstein-Uhlenbeck process, ``k(r) = sigma^2 exp(-r / tau)``
    (the Matérn process with ``nu = 1/2``).

    Parameters
    ----------
    sigma : float
        standard deviation (signal units)
    tau : float
        time constant (ms)

    Examples
    --------
    >>> ou = OU(sigma=1., tau=200.)
    >>> A, Q = ou.discretise(1.)
    >>> bool(np.isclose(A[0, 0], np.exp(-1 / 200.)))
    True
    """
    sigma: float = 1.0
    tau: float = 100.0
    param_names = ("sigma", "tau")

    def _matern(self):
        return Matern(sigma=self.sigma, ell=self.tau, nu=0.5)

    def ss(self):
        return self._matern().ss()

    def cov(self, tau):
        return self._matern().cov(tau)


@dataclass(frozen=True)
class RandomWalk(Component):
    """
    Random walk (Brownian motion), ``dx = sqrt(q) dW``: a slow drift.

    Not stationary: its semivariogram ``gamma(tau) = q tau / 2`` grows
    without bound and the state starts with the (large) variance `p0`.

    Parameters
    ----------
    q : float
        diffusion rate (signal units^2 per ms)
    p0 : float
        variance of the initial state (not fitted); should be large compared
        to the variance of the signal
    """
    q: float = 1e-3
    p0: float = 1e6
    param_names = ("q",)

    @property
    def stationary(self):
        return False

    def ss(self):
        one = np.ones((1, 1))
        return ContinuousSS(np.zeros((1, 1)), one, self.q * one, one, None)

    def variogram(self, tau):
        return 0.5 * self.q * np.abs(np.asarray(tau, dtype=float))


_PREFIX = {"OU": "ou", "Matern": "matern", "RandomWalk": "rw"}


class GaussMarkovModel:
    """
    Gauss-Markov model: a sum of components plus white measurement noise.

    The state is the stack of the component states (block-diagonal
    dynamics); the observation is the sum of the component outputs plus
    noise with standard deviation `noise`.

    Parameters
    ----------
    components : list of Component or dict of str to Component
        the components; a list gets names like ``"ou0"``, ``"matern1"``
    noise : float
        standard deviation of the measurement noise (the nugget is noise^2)
    fixed : iterable of str, optional
        parameters held fixed when fitting (see :meth:`fix`)

    Notes
    -----
    Parameters are named ``"<component>.<param>"`` (e.g. ``"ou0.tau"``) and
    ``"noise"``. :attr:`theta` holds the log of the free parameters, the
    unconstrained vector used by the optimisers.

    Examples
    --------
    >>> model = GaussMarkovModel([OU(1., 50.), Matern(2., 500., nu=2.5)], noise=0.1)
    >>> model.param_names
    ['ou0.sigma', 'ou0.tau', 'matern1.sigma', 'matern1.ell', 'noise']
    >>> model.dim
    4
    >>> model.fix(["noise"]).free_names
    ['ou0.sigma', 'ou0.tau', 'matern1.sigma', 'matern1.ell']
    """

    def __init__(self, components, noise=0.0, fixed=()):
        if isinstance(components, dict):
            items = list(components.items())
        else:
            items = [(f"{_PREFIX.get(type(c).__name__, 'comp')}{i}", c)
                     for i, c in enumerate(components)]
        names = [n for n, _ in items]
        if len(set(names)) != len(names) or any("." in n for n in names):
            raise ValueError(f"component names must be unique and contain no '.': {names}")
        self._items = items
        self.noise = float(noise)
        self.fixed = frozenset(fixed)
        unknown = self.fixed - set(self.param_names)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")

    # -- structure -----------------------------------------------------------
    @property
    def components(self):
        """dict of name to component"""
        return dict(self._items)

    @property
    def names(self):
        return [n for n, _ in self._items]

    @property
    def slices(self):
        """dict of name to the slice of the state that belongs to it"""
        out, i = {}, 0
        for name, c in self._items:
            out[name] = slice(i, i + c.dim)
            i += c.dim
        return out

    @property
    def dim(self):
        return sum(c.dim for _, c in self._items)

    @property
    def stationary(self):
        return all(c.stationary for _, c in self._items)

    def __repr__(self):
        comps = ", ".join(f"{n}={c!r}" for n, c in self._items)
        return (f"GaussMarkovModel({comps}, noise={self.noise!r}, "
                f"fixed={sorted(self.fixed)})")

    # -- parameters ----------------------------------------------------------
    @property
    def params(self):
        """dict of all parameters, ``{"ou0.tau": ..., "noise": ...}``"""
        out = {f"{n}.{k}": v for n, c in self._items for k, v in c.get_params().items()}
        out["noise"] = self.noise
        return out

    @property
    def param_names(self):
        return list(self.params)

    @property
    def free_names(self):
        """names of the parameters that are not fixed (the order of theta)"""
        return [k for k in self.param_names if k not in self.fixed]

    def with_params(self, params):
        """copy with the parameters in dict `params` changed"""
        unknown = set(params) - set(self.param_names)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        items = []
        for name, c in self._items:
            kw = {k.split(".", 1)[1]: float(v) for k, v in params.items()
                  if k.split(".", 1)[0] == name}
            items.append((name, c.with_params(**kw) if kw else c))
        noise = params.get("noise", self.noise)
        return GaussMarkovModel(dict(items), noise=noise, fixed=self.fixed)

    def fix(self, params):
        """
        Copy with parameters held fixed when fitting.

        Parameters
        ----------
        params : list of str or dict
            names, or a dict of names and the values to fix them at
        """
        new = self.with_params(params) if isinstance(params, dict) else self
        unknown = set(params) - set(self.param_names)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        return GaussMarkovModel(dict(new._items), noise=new.noise,
                                fixed=self.fixed | set(params))

    def unfix(self, names):
        """copy with the parameters `names` free again"""
        return GaussMarkovModel(dict(self._items), noise=self.noise,
                                fixed=self.fixed - set(names))

    @property
    def theta(self):
        """log of the free parameters"""
        p = self.params
        with np.errstate(divide="ignore"):
            return np.log(np.array([p[k] for k in self.free_names], dtype=float))

    def from_theta(self, theta):
        """copy with the free parameters set from their logs `theta`"""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(self.free_names),):
            raise ValueError(f"theta must have length {len(self.free_names)}")
        return self.with_params(dict(zip(self.free_names, np.exp(theta))))

    # -- second-order properties --------------------------------------------
    def cov(self, tau):
        """autocovariance of the observations at lags `tau` (ms), with noise"""
        tau = np.asarray(tau, dtype=float)
        k = sum(c.cov(tau) for _, c in self._items)
        return k + self.noise**2 * (tau == 0)

    def variogram(self, tau):
        """semivariogram of the observations at lags `tau` (ms), with nugget"""
        tau = np.asarray(tau, dtype=float)
        g = sum(c.variogram(tau) for _, c in self._items)
        return g + self.noise**2 * (tau != 0)

    # -- discrete model ------------------------------------------------------
    def to_lgssm(self, fs, diffuse=True):
        """
        Discrete-time model for sampling rate `fs` (Hz).

        Parameters
        ----------
        fs : float
            sampling rate in Hz
        diffuse : bool, optional
            initial variance of non-stationary components: their `p0` (True,
            default, for filtering) or 0 (False, start at 0, for simulation)

        Returns
        -------
        LGSSM
        """
        from scipy.linalg import block_diag
        from .kalman import LGSSM
        dt = 1000.0 / fs
        As, Qs, Hs, P0s = [], [], [], []
        for _, c in self._items:
            A, Q = c.discretise(dt)
            s = c.ss()
            As.append(A)
            Qs.append(Q)
            Hs.append(s.H)
            if s.Pinf is not None:
                P0s.append(s.Pinf)
            else:
                P0s.append(np.eye(c.dim) * (c.p0 if diffuse else 0.0))
        return LGSSM(A=block_diag(*As), Q=block_diag(*Qs), H=np.hstack(Hs),
                     R=np.array([[self.noise**2]]), m0=np.zeros(self.dim),
                     P0=block_diag(*P0s))
