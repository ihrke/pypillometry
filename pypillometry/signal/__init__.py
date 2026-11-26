"""
Stateless, low-level signal-processing functions operating on numpy arrays.

These functions are used by the data classes (e.g.,
:class:`~pypillometry.EyeData`) and generally work in samples or
milliseconds; see the individual docstrings for units.

- :mod:`~pypillometry.signal.baseline`: filtering, downsampling and
  baseline estimation
- :mod:`~pypillometry.signal.preproc`: smoothing, velocity and blink detection
- :mod:`~pypillometry.signal.pupil`: pupil-response function (PRF),
  response estimation and signal quality
- :mod:`~pypillometry.signal.fake`: generation of synthetic data
- :mod:`~pypillometry.signal.variogram`, :mod:`~pypillometry.signal.spectral`:
  semivariogram and power spectrum of signals with missing samples
- :mod:`~pypillometry.signal.decimate`: block means with counts
- :mod:`~pypillometry.signal.gaussmarkov`: Gauss-Markov (state-space) models
- :mod:`~pypillometry.signal.transplant`: transplantation of blinks into
  clean data
- :mod:`~pypillometry.signal.autotune`: tuning the interpolation of blinks
  on transplanted blinks
"""
from . import (baseline, preproc, pupil, fake, variogram, spectral, decimate,
               gaussmarkov, transplant, autotune)
from .baseline import *
from .preproc import *
from .pupil import *
from .fake import *
from .variogram import *
from .spectral import *
from .decimate import *
from .transplant import *
from .autotune import *

# Each submodule lists its public functions in its own __all__.
__all__ = []
__all__ += baseline.__all__
__all__ += preproc.__all__
__all__ += pupil.__all__
__all__ += fake.__all__
__all__ += variogram.__all__
__all__ += spectral.__all__
__all__ += decimate.__all__
__all__ += transplant.__all__
__all__ += autotune.__all__
