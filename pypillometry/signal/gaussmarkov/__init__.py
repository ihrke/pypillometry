"""
Gauss-Markov processes (OU, Matérn, random walk) as linear state-space
models: Kalman filter, RTS smoother, simulation and likelihoods.

Low-level machinery on numpy arrays; experimental, the API may change.

- :mod:`~pypillometry.signal.gaussmarkov.components`: components and
  their sum (:class:`Sum`)
- :mod:`~pypillometry.signal.gaussmarkov.kalman`: general Kalman filter,
  smoother and simulation (compiled with numba if installed)
- :mod:`~pypillometry.signal.gaussmarkov.likelihood`: log-likelihood of a
  signal or of its semivariogram, fitting

Examples
--------
>>> import numpy as np
>>> from pypillometry.signal import gaussmarkov as gm
>>> model = gm.Sum([gm.OU(1., 50.), gm.Matern(2., 500., nu=1.5)], noise=0.1)
>>> y = gm.simulate(model, 2000, fs=500, rng=1)
>>> ssm = model.to_lgssm(fs=500)
>>> fres = gm.kalman_filter(ssm, y)
>>> sres = gm.rts_smoother(ssm, fres)
>>> parts = gm.component_states(model, sres)
>>> sorted(parts)
['matern1', 'ou0']
>>> bool(np.isclose(fres.loglik, gm.loglik_signal(model, y, fs=500, demean=False)))
True
"""
from . import components, kalman, likelihood
from .components import *
from .kalman import *
from .likelihood import *

__all__ = components.__all__ + kalman.__all__ + likelihood.__all__
