"""
components.py
=============

Continuous-time stationary Gauss-Markov processes and sums of them.

Each component is a linear SDE

.. math::

    dx = F x \\, dt + dW, \\qquad \\mathrm{Cov}(dW) = Q_n \\, dt,
    \\qquad f(t) = H x(t),

whose output :math:`f(t)` is a Gaussian process. A :class:`GaussMarkovModel`
stacks components (block-diagonal state) and adds white measurement noise;
:meth:`GaussMarkovModel.to_lgssm` turns it into the discrete matrices used by
:mod:`~pypillometry.signal.gaussmarkov.kalman`.

Times (``tau``, ``ell``, lags) are in ms; rates are per ms.
"""
from dataclasses import dataclass, replace
from math import factorial, comb
from typing import NamedTuple

import numpy as np

__all__ = ["ContinuousSS", "Component", "OU", "Matern", "Constant",
           "GaussMarkovModel"]


class ContinuousSS(NamedTuple):
    """
    Continuous-time state space of a component.

    Attributes
    ----------
    F : (n, n) feedback matrix
    Qn : (n, n) intensity of the white noise driving the state
        (rate at which it adds covariance)
    H : (1, n) observation row
    Pinf : (n, n) stationary covariance of the state
    """
    F: np.ndarray
    Qn: np.ndarray
    H: np.ndarray
    Pinf: np.ndarray


def _van_loan(F, Qn, dt):
    """A = expm(F dt) and Q = int_0^dt expm(F s) Qn expm(F' s) ds."""
    from scipy.linalg import expm
    n = F.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -F
    M[:n, n:] = Qn
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
        return _van_loan(s.F, s.Qn, dt)

    def cov(self, tau):
        """Autocovariance k(tau) of the output at lags `tau` (ms)."""
        from scipy.linalg import expm
        s = self.ss()
        tau = np.abs(np.asarray(tau, dtype=float))
        out = [(s.H @ expm(s.F * t) @ s.Pinf @ s.H.T)[0, 0] for t in tau.ravel()]
        return np.reshape(out, tau.shape)

    def variogram(self, tau):
        """Semivariogram gamma(tau) = k(0) - k(tau) of the output (ms)."""
        tau = np.asarray(tau, dtype=float)
        return self.cov(0.0) - self.cov(tau)

    def psd(self, freq):
        """
        One-sided power spectral density of the output at frequencies
        `freq` (Hz), in signal units squared per Hz; its integral over all
        frequencies is the variance.

        .. math::

            S(f) = \\frac{2}{1000} H (i\\omega - F)^{-1} Q_n
                   (-i\\omega - F^\\top)^{-1} H^\\top, \\qquad
                   \\omega = 2\\pi f / 1000
        """
        s = self.ss()
        freq = np.asarray(freq, dtype=float)
        w = 2 * np.pi * freq.ravel() / 1000.0        # rad per ms
        eye = np.eye(s.F.shape[0])
        G = np.linalg.solve(1j * w[:, None, None] * eye - s.F,
                            np.broadcast_to(eye, (len(w),) + eye.shape))
        hG = (s.H @ G)[:, 0, :]
        out = np.einsum("wi,ij,wj->w", hG, s.Qn, hG.conj()).real
        return (out / 500.0).reshape(freq.shape)

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

    The state holds the process and its first p derivatives; the white noise
    drives only the highest one (Hartikainen & Särkkä, 2010).

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
        Qn = np.zeros((n, n))
        Qn[-1, -1] = (self.sigma**2 * factorial(p)**2 / factorial(2 * p)
                      * (2 * lam)**(2 * p + 1))
        H = np.zeros((1, n))
        H[0, 0] = 1.0
        Pinf = solve_continuous_lyapunov(F, -Qn)
        return ContinuousSS(F, Qn, H, 0.5 * (Pinf + Pinf.T))

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
class Constant(Component):
    """
    Unknown constant (e.g., the mean of the signal) with a diffuse prior.

    A state that does not change, with a prior variance `kappa` that is
    very large against the signal. Used with ``n_diffuse=1`` in
    :func:`~pypillometry.signal.gaussmarkov.kalman_loglik`, the likelihood
    of the other parameters does not depend on the level of the signal (the
    restricted likelihood, REML), and the smoother returns the posterior of
    the constant with the other components.

    Parameters
    ----------
    kappa : float
        prior variance (signal units squared); default 1e7. Not a fitted
        parameter.

    Examples
    --------
    >>> c = Constant()
    >>> c.discretise(50.)
    (array([[1.]]), array([[0.]]))
    >>> c.param_names
    ()
    """
    kappa: float = 1e7
    param_names = ()

    def ss(self):
        return ContinuousSS(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)),
                            np.array([[float(self.kappa)]]))

    def discretise(self, dt):
        return np.eye(1), np.zeros((1, 1))

    def cov(self, tau):
        """0: the constant does not enter the covariance and the variogram."""
        return np.zeros(np.shape(tau))

    def psd(self, freq):
        return np.zeros(np.shape(freq))


_PREFIX = {"OU": "ou", "Matern": "matern", "Constant": "const"}


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

    def psd(self, freq):
        """
        One-sided power spectral density of the observations at frequencies
        `freq` (Hz), in signal units squared per Hz, without the noise
        (white noise sampled at ``fs`` adds ``noise^2 / (fs / 2)``).
        """
        return sum(c.psd(freq) for _, c in self._items)

    @property
    def n_diffuse(self):
        """number of components with a diffuse prior (:class:`Constant`)"""
        return sum(isinstance(c, Constant) for _, c in self._items)

    # -- discrete model ------------------------------------------------------
    def to_lgssm(self, fs):
        """
        Discrete-time model for sampling rate `fs` (Hz).

        Parameters
        ----------
        fs : float
            sampling rate in Hz

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
            P0s.append(s.Pinf)
        return LGSSM(A=block_diag(*As), Q=block_diag(*Qs), H=np.hstack(Hs),
                     R=np.array([[self.noise**2]]), m0=np.zeros(self.dim),
                     P0=block_diag(*P0s))
