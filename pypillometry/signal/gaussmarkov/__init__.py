"""
Stationary Gauss-Markov processes (OU, Matérn) as linear state-space
models: Kalman filter, RTS smoother, simulation and likelihoods.

Low-level machinery on numpy arrays; experimental, the API may change.

- :mod:`~pypillometry.signal.gaussmarkov.components`: components and
  their sum (:class:`GaussMarkovModel`)
- :mod:`~pypillometry.signal.gaussmarkov.kalman`: general Kalman filter,
  smoother and simulation (compiled with numba if installed)
- :mod:`~pypillometry.signal.gaussmarkov.likelihood`: log-likelihood of a
  signal or of its semivariogram, fitting
- :mod:`~pypillometry.signal.gaussmarkov.bank`: timescale-bank model (a fast
  component and slow components at fixed time scales), priors and fitting
- :mod:`~pypillometry.signal.gaussmarkov.hierarchical`: group-level
  estimation (empirical Bayes) of the bank model
- :mod:`~pypillometry.signal.gaussmarkov.check`: cross-validation and
  variogram check of a fitted bank model

Examples
--------
>>> import numpy as np
>>> from pypillometry.signal import gaussmarkov as gm
>>> model = gm.GaussMarkovModel([gm.OU(1., 50.), gm.Matern(2., 500., nu=1.5)], noise=0.1)
>>> y = gm.simulate(model, 2000, fs=500, rng=1)
>>> ssm = model.to_lgssm(fs=500)
>>> fres = gm.kalman_filter(ssm, y)
>>> sres = gm.rts_smoother(ssm, fres)
>>> parts = gm.component_states(model, sres)
>>> sorted(parts)
['matern1', 'ou0']
>>> bool(np.isclose(fres.loglik, gm.loglik_signal(model, y, fs=500, mean_tol=None)))
True
"""
from . import components, kalman, likelihood, bank, hierarchical, check
from .components import *
from .kalman import *
from .likelihood import *
from .bank import *
from .hierarchical import *
from .check import *

__all__ = (components.__all__ + kalman.__all__ + likelihood.__all__ + bank.__all__
           + hierarchical.__all__ + check.__all__)
