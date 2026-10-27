"""
Stateless, low-level signal-processing functions operating on numpy arrays.

These functions are used by the data classes (e.g.,
:class:`~pypillometry.PupilData`) and generally work in samples or
milliseconds; see the individual docstrings for units.

- :mod:`~pypillometry.signal.baseline`: filtering, downsampling and
  baseline estimation
- :mod:`~pypillometry.signal.preproc`: smoothing, velocity and blink detection
- :mod:`~pypillometry.signal.pupil`: pupil-response function (PRF),
  response estimation and signal quality
- :mod:`~pypillometry.signal.fake`: generation of synthetic data
"""
from . import baseline, preproc, pupil, fake
from .baseline import *
from .preproc import *
from .pupil import *
from .fake import *

# Each submodule lists its public functions in its own __all__.
__all__ = []
__all__ += baseline.__all__
__all__ += preproc.__all__
__all__ += pupil.__all__
__all__ += fake.__all__
