"""
history.py
==========

The history of an eyedata object: :func:`keephistory` records the calls of
the decorated methods (of :class:`~pypillometry.EyeData` and its
namespaces ``d.pupil`` and ``d.gaze``).
"""
import functools

import numpy as np

from .namespace import Namespace


def _history_arg(v):
    """Short text for an argument in the history (no reference to the object)."""
    if isinstance(v, str):
        return repr(v)
    if v is None or isinstance(v, (bool, int, float, np.number)):
        return str(v)
    if isinstance(v, np.ndarray):
        return f"array{v.shape}"
    if callable(v):
        return getattr(v, "__name__", type(v).__name__)
    r = repr(v) if isinstance(v, (list, tuple, dict)) else str(v)
    if len(r) <= 40 and "\n" not in r:
        return r  # short enough, e.g. [1, 2], (1280, 1024), 30 cm
    try:
        return f"{type(v).__name__}({len(v)})"  # e.g. Intervals(13)
    except TypeError:
        return type(v).__name__


# number of decorated methods currently running; only the outermost call
# (the one made by the user) is recorded
_history_depth = 0


def keephistory(func):
    """
    Decorator that records a method call in the history of the returned object.

    After the decorated method has run, a short text version of the call is
    added (via ``add_to_history()``) to the object it returned (or to `self`
    if it returned nothing, e.g. for setters), provided that object is an
    eyedata object (i.e., has an ``add_to_history`` method).
    Calls made from inside another decorated method are not recorded, and
    large arguments are only summarized (e.g., ``Intervals(13)``); the
    `inplace` argument is left out (it does not change the result). The history
    is a log of the operations; show it with
    :meth:`~pypillometry.EyeData.print_history`.

    Parameters
    ----------
    func : callable
        method to wrap; its first argument must be the object (`self`) or a
        :class:`~pypillometry.eyedata.namespace.Namespace` of it

    Returns
    -------
    callable
        wrapped method that returns the same value as `func`
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        global _history_depth
        _history_depth += 1
        try:
            obj=func(*args,**kwargs)
        finally:
            _history_depth -= 1
        # record in the returned object, or in `self` for in-place setters
        # that return nothing; methods of a namespace (e.g., ``d.pupil``)
        # are recorded with its name (``pupil.blinks_detect``)
        self, name = args[0], func.__name__
        if isinstance(self, Namespace):
            self, name = self._d, f"{self._name}.{name}"
        target = obj if obj is not None else self
        if _history_depth == 0 and hasattr(target, 'add_to_history'):
            allargs = [_history_arg(v) for v in args[1:]]
            allargs += [f"{k}={_history_arg(v)}" for k, v in kwargs.items()
                        if k != "inplace"]  # does not change the result
            fstr = f"{name}({','.join(allargs)})"
            target.add_to_history({"funcstring": fstr, "funcname": name})
        return obj
    return wrapper
