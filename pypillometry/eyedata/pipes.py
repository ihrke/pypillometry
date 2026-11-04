"""
pipes.py
========

Pipeline steps for the ``|`` operator::

    d = d | pp.pupil.blinks_detect() | pp.pupil.lowpass_filter(5) | pp.downsample(50)

A :class:`Step` is a deferred call of a transforming method; ``pp.pupil.*``,
``pp.gaze.*`` and the generic steps at the top level (``pp.crop``, ...) make
them. Steps combine into a :class:`Pipeline` (``step | step``), which can be
applied to any number of data objects. ``d | pipeline`` copies `d` once and
then runs all steps in place on the copy; `d` is not changed.

Python evaluates ``d | a | b`` as ``(d | a) | b``, so each ``|`` with data
on its left makes a copy. To copy only once (e.g., for large recordings),
combine the steps first: ``d | (a | b)``, or ``preproc = a | b; d | preproc``.
"""

import functools
import inspect

from .namespace import Namespace


def step(func):
    """
    Decorator for the transforming methods (those that return the data object).

    Marks the method as usable as a pipeline step. On a data object
    (``d.crop(...)``, ``d.pupil.blinks_detect()``), the method runs as
    usual. On a namespace without data (``pp.pupil.blinks_detect()``), it
    returns a :class:`Step` instead.

    Parameters
    ----------
    func : callable
        the method (of the data class or of a
        :class:`~pypillometry.eyedata.namespace.Namespace`)

    Returns
    -------
    callable
        the wrapped method
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if isinstance(self, Namespace) and self._data is None:
            return Step(wrapper, args, kwargs, namespace=self._name)
        return func(self, *args, **kwargs)
    wrapper._is_step = True
    return wrapper


def generic_step(method):
    """
    Make the top-level step factory (e.g., ``pp.crop``) for a generic method.

    The factory has the signature (without `self` and `inplace`) and the
    docstring of the method (after a line that names it) and returns a
    :class:`Step`.

    Parameters
    ----------
    method : callable
        a method of the data class decorated with :func:`step`

    Returns
    -------
    callable
        the step factory
    """
    if not getattr(method, "_is_step", False):
        raise TypeError(f"{method.__name__} is not a step (decorate it with @step)")

    @functools.wraps(method)
    def factory(*args, **kwargs):
        return Step(method, args, kwargs)
    sig = inspect.signature(method)
    params = [p for p in list(sig.parameters.values())[1:] if p.name != "inplace"]
    factory.__signature__ = sig.replace(parameters=params)
    factory.__doc__ = (f"Pipeline step for :meth:`~pypillometry.GenericEyeData.{method.__name__}`"
                       f" (``d | pp.{method.__name__}(...)``).\n\n"
                       + inspect.cleandoc(method.__doc__ or ""))
    return factory


def _arg_text(v):
    """Short text for an argument in the repr of a step."""
    from .generic import _history_arg
    return _history_arg(v)


class Step:
    """
    A pipeline step: a deferred call of a transforming method.

    Made by ``pp.pupil.<method>(...)``, ``pp.gaze.<method>(...)`` or the
    generic steps (``pp.crop(...)``, ...), not directly. Apply it with
    ``d | step``, which returns a processed copy of `d`; combine steps
    with ``step1 | step2`` into a :class:`Pipeline`.

    The arguments are checked against the method's signature when the step
    is made. Steps do not accept `inplace` (a pipeline works on a copy).

    Parameters
    ----------
    method : callable
        the method (decorated with :func:`step`)
    args : tuple
        positional arguments for the method
    kwargs : dict
        keyword arguments for the method
    namespace : str or None
        name of the namespace of the method (e.g., ``"pupil"``), or `None`
        for a method of the data object itself
    """

    def __init__(self, method, args=(), kwargs=None, namespace=None):
        kwargs = dict(kwargs or {})
        self.namespace = namespace
        self.name = method.__name__
        self.args = tuple(args)
        self.kwargs = kwargs
        params = inspect.signature(method).parameters
        self._has_inplace = "inplace" in params
        if "inplace" in kwargs:
            raise TypeError(f"{self.qualname}(): a pipeline step does not take "
                            "'inplace' (the pipeline works on a copy of the data)")
        try:
            inspect.signature(method).bind(None, *self.args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{self.qualname}(): {e}") from None

    @property
    def qualname(self):
        """Name of the step as called, e.g. ``pp.pupil.blinks_detect``."""
        ns = f"{self.namespace}." if self.namespace else ""
        return f"pp.{ns}{self.name}"

    def __repr__(self):
        args = [_arg_text(v) for v in self.args]
        args += [f"{k}={_arg_text(v)}" for k, v in self.kwargs.items()]
        return f"{self.qualname}({', '.join(args)})"

    def _apply(self, d):
        """Run the step on `d` in place and return `d`."""
        target = d if self.namespace is None else getattr(d, self.namespace)
        kwargs = dict(self.kwargs, inplace=True) if self._has_inplace else self.kwargs
        return getattr(target, self.name)(*self.args, **kwargs)

    def __call__(self, d):
        """Apply the step to a copy of `d` (same as ``d | step``)."""
        return Pipeline([self])(d)

    def __or__(self, other):
        return Pipeline([self]) | other

    def __ror__(self, other):
        return Pipeline([self]).__ror__(other)


class Pipeline:
    """
    A sequence of pipeline steps, applied with ``d | pipeline``.

    A pipeline is a value: it can be stored, printed and applied to many data
    objects (e.g., all subjects of a study). Applying it copies the data
    object once and then runs each step in place on the copy, so the
    original object is not changed. (In ``d | a | b``, which Python reads
    as ``(d | a) | b``, each ``|`` copies; ``d | (a | b)`` copies once.) Each step is recorded in the history of
    the result. If a step fails, the error says which one.

    Pipelines are usually made by combining steps with ``|``::

        preproc = pp.pupil.blinks_detect() | pp.pupil.blinks_interpolate() \\
                  | pp.pupil.lowpass_filter(5)
        data = {subj: d | preproc for subj, d in datasets.items()}

    Parameters
    ----------
    steps : iterable of Step
        the steps, in the order in which they are applied
    """

    def __init__(self, steps=()):
        steps = list(steps)
        for s in steps:
            if not isinstance(s, Step):
                raise TypeError(f"a Pipeline holds steps, not {type(s).__name__}")
        self.steps = steps

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Pipeline(self.steps[i])
        return self.steps[i]

    def __repr__(self):
        if not self.steps:
            return "Pipeline([])"
        return " | ".join(repr(s) for s in self.steps)

    def __or__(self, other):
        if isinstance(other, Step):
            return Pipeline(self.steps + [other])
        if isinstance(other, Pipeline):
            return Pipeline(self.steps + other.steps)
        if getattr(other, "_is_step", False):
            raise TypeError(_missing_call(other))
        return NotImplemented

    def __ror__(self, other):
        # data objects handle `d | pipeline` themselves (GenericEyeData.__or__)
        raise TypeError(f"cannot pipe an object of type {type(other).__name__} into "
                        f"{self!r}: only EyeData objects can be piped into steps")

    def __call__(self, d):
        """Apply the pipeline to a copy of `d` and return the copy."""
        from .generic import GenericEyeData
        if not isinstance(d, GenericEyeData):
            self.__ror__(d)
        d = d.copy(new_name=d.name)
        n = len(self.steps)
        for i, s in enumerate(self.steps, 1):
            try:
                d = s._apply(d)
            except Exception as e:
                e.add_note(f"in step {i} of {n} of the pipeline: {s!r}" if n > 1
                           else f"in the pipeline step {s!r}")
                raise
        return d


def _missing_call(f):
    """Error message for a step factory that was not called."""
    name = getattr(f, "__name__", repr(f))
    ns = getattr(getattr(f, "__self__", None), "_name", None)
    qual = f"pp.{ns}.{name}" if ns else f"pp.{name}"
    return f"{qual} is not a step yet: call it, e.g. {qual}()"
