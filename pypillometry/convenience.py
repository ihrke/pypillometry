"""
convenience.py
==============

Some convenience functions.
"""


import numpy as np
import pandas as pd
from typing import Union, Dict
from contextlib import contextmanager
import os, sys
from tqdm import tqdm

__all__ = [
    "nprange", "zscale", "p_asym_laplac", "p_asym_laplac_kappa",
    "trans_logistic_vec",
]

@contextmanager
def change_dir(path: Union[str]):
    """Temporarily change the current working directory.
    
    Parameters
    ----------
    path : str
        Directory to change to
        
    Examples
    --------
    >>> with change_dir("/path/to/dir"):
    ...     # do something in that directory
    ...     pass
    >>> # back in original directory
    """
    old_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_dir)


@contextmanager
def suppress_all_output():
    """Suppress all output including C library output."""
    # Save original file descriptors
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    
    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    
    try:
        # Redirect file descriptors
        os.dup2(devnull_fd, 1)  # stdout
        os.dup2(devnull_fd, 2)  # stderr
        sys.stdout.flush()
        sys.stderr.flush()
        yield
    finally:
        # Restore original file descriptors
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        
        # Close file descriptors
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)
        os.close(devnull_fd)

def nprange(ar):
    """
    Return minimum and maximum of an array.

    Parameters
    ----------
    ar : np.ndarray
        input array

    Returns
    -------
    tuple
        `(min, max)` of `ar` (NaNs are not ignored)
    """
    return (ar.min(),ar.max())

import pickle
from hashlib import md5

def zscale(y):
    """
    Z-transform an array, ignoring NaNs.

    Parameters
    ----------
    y : np.ndarray
        input array

    Returns
    -------
    np.ndarray
        `y` with mean subtracted and divided by its standard deviation
        (both computed with :func:`numpy.nanmean`/:func:`numpy.nanstd`);
        NaNs are kept
    """
    return (y-np.nanmean(y))/np.nanstd(y)


def p_asym_laplac(y, mu, sigma, tau):
    """
    Asymmetric laplace distribution https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution;
    Parametrization as here: https://github.com/stan-dev/stan/issues/2312

    Parameters
    ----------
    y : float or np.ndarray
        value(s) at which to evaluate the density
    mu : float
        location parameter
    sigma : float
        scale parameter (> 0)
    tau : float
        asymmetry parameter in [0,1] (0.5 is symmetric)

    Returns
    -------
    float or np.ndarray
        density evaluated at `y`
    """
    I=np.array(y<=mu, dtype=int)
    return (2*tau*(1-tau))/sigma*np.exp(-2/sigma * ( (1-tau)*I*(mu-y) + tau*(1-I)*(y-mu) ) )

def p_asym_laplac_kappa(y, mu, lam, kappa):
    """
    Asymmetric laplace distribution https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution;
    Wikipedia parametrization.

    Parameters
    ----------
    y : float or np.ndarray
        value(s) at which to evaluate the density
    mu : float
        location parameter
    lam : float
        scale parameter (lambda, > 0)
    kappa : float
        asymmetry parameter in [0, infty] where 1 means symmetry

    Returns
    -------
    float or np.ndarray
        density evaluated at `y`
    """
    I=np.array(y<=mu, dtype=int)
    return (lam)/(kappa+1./kappa)*np.exp( ( (lam/kappa)*I*(y-mu) - lam*kappa*(1-I)*(y-mu) ) )


def trans_logistic_vec(x, a, b, inverse=False):
    """
    Transform bounded variables to an unbounded range and back.

    Goes from [a,b] to [-inf,+inf] and back (elementwise, so `a` and
    `b` can be arrays of the same shape as `x`); if a or b is +/-infty,
    a logarithmic/exponential transform is used; if both are infinite,
    `x` is returned unchanged.

    Parameters
    ----------
    x : float or np.ndarray
        values to transform
    a : float or np.ndarray
        lower bound(s), can be `-np.inf`
    b : float or np.ndarray
        upper bound(s), can be `np.inf`
    inverse : bool
        if False (default): [a,b] -> [-inf, +inf];
        if True: [-inf,+inf] -> [a,b]

    Returns
    -------
    np.ndarray
        transformed values
    """
    eps=1e-15
    if inverse==False:
        # variables from [a,inf]
        x=np.where( (a>-np.inf) & (b==np.inf), np.log(np.maximum(x-a, eps)), x)
        # variables from [-inf, b]
        x=np.where( (a==-np.inf) & (b<np.inf), np.log(np.maximum(b-x, eps)), x)
        # variables from [a, b]
        x=np.where( (a>-np.inf) & (b<np.inf), -np.log( (b-a)/(x-a)-1 ), x)
    elif inverse==True:
        # variables from [-inf,inf] -> [a,inf]
        x=np.where( (a>-np.inf) & (b==np.inf), np.exp(x)+a, x)
        # variables from [-inf, inf] -> [-inf, b]
        x=np.where( (a==-np.inf) & (b<np.inf), b-np.exp(x), x)
        # variables from [-inf,inf] -> [a, b]
        x=np.where( (a>-np.inf) & (b<np.inf), (1./(1.+np.exp(-x)))*(b-a)+a, x)
    
    return x


def sizeof_fmt(num, suffix='B'):
    """
    Convert number of bytes in `num` into human-readable string representation.
    Taken from https://stackoverflow.com/a/1094933

    Parameters
    ----------
    num : int or float
        number of bytes
    suffix : str
        suffix appended after the binary prefix (default "B")

    Returns
    -------
    str
        formatted size, e.g. "1.5KiB"
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

class ByteSize(int):
    """Class to represent size in bytes with human-readable formatting.

    Parameters
    ----------
    bytes : int or dict
        number of bytes; or a dict with keys `"memory"` and `"disk"`, in
        which case the total is the sum of both and the `"disk"` part is
        reported as cached
    """
    def __new__(cls, bytes: Union[int, Dict[str, int]]):
        if isinstance(bytes, dict):
            total_bytes = bytes['memory'] + bytes['disk']
            obj = super().__new__(cls, total_bytes)
            obj.cached_bytes = bytes['disk']
        else:
            obj = super().__new__(cls, bytes)
            obj.cached_bytes = 0
        return obj
            
    def __str__(self):
        if self.cached_bytes > 0:
            return f"{sizeof_fmt(int(self) - self.cached_bytes)} (+{sizeof_fmt(self.cached_bytes)} cached)"
        else:
            return sizeof_fmt(int(self))
        
    def is_cached(self):
        """
        Check whether part of the size refers to cached (on-disk) data.

        Returns
        -------
        bool
            True if the number of cached bytes is larger than 0
        """
        return self.cached_bytes > 0
            
    def __repr__(self):
        return self.__str__()

def test_selector(obj, selfun, **kwargs):
    """
    Function to help test the output of a selector function.

    Parameters
    ----------
    obj : GenericEyeData
        object to test the selector function on
    selfun : function
        function that takes a single event label as input and returns True/False
        to indicate whether the event should be included or not
    **kwargs : dict
        arguments to pass onto the selector function

    Returns
    -------
    np.ndarray
        the event labels of `obj` for which `selfun` returns True

    Examples
    --------
    Here, events are coded as F_1, S_1, R_1, F_2, S_2, R_2, ...
    to indicate fixation cross, stimulus, response for each trial.
    We build selector functions to extract all fixation crosses etc.

    .. code-block:: python

        def fix_selector(evlab):
            return evlab.startswith("F")
        def stim_selector(evlab):
            return evlab.startswith("S")
        test_selector(obj, fix_selector)

    We can use kwargs to pass on arguments to the selector function,
    for example, to restrict to only a single trial:

    .. code-block:: python

        def fix_selector(evlab, trial=None):
            return evlab.startswith("F") and evlab.endswith("_{}".format(trial))
        test_selector(obj, fix_selector, trial=1)
    """
    if not callable(selfun):
        raise ValueError("selfun must be a function")
    return obj.event_labels[np.array([selfun(evlab, **kwargs) for evlab in obj.event_labels])]

def is_url(url):
    """Check if a string is a URL.
    
    Parameters
    ----------
    url : str
        URL to check
        
    Returns
    -------
    bool
        True if the string is a URL, False otherwise
    """
    from urllib.parse import urlparse
    if not isinstance(url, str):
        url = str(url)
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    return True



## Decorator to check for optional dependencies
import functools
from typing import Callable, Any

def requires_package(package_name: str, install_hint: str = None):
    """
    Decorator to mark functions that require optional dependencies.

    The check is done each time the decorated function is called; if
    `package_name` cannot be imported, an :class:`ImportError` with an
    installation hint is raised.

    Parameters
    ----------
    package_name : str
        Name of the required package (as used in `import`)
    install_hint : str, optional
        Custom installation instruction; default is
        ``pip install <package_name>``

    Returns
    -------
    callable
        decorator that wraps the function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                __import__(package_name)
            except ImportError:
                hint = install_hint or f"pip install {package_name}"
                raise ImportError(
                    f"Function '{func.__name__}' requires the '{package_name}' package. "
                    f"Install it with: {hint}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Unit handling utilities
UNIT_ALIASES = {
    # milliseconds
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    # seconds
    "s": "sec",
    "sec": "sec",
    "secs": "sec",
    "second": "sec",
    "seconds": "sec",
    # minutes
    "m": "min",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    # hours
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    # indices (samples)
    "index": None,
    "indices": None,
    "idx": None,
    "sample": None,
    "samples": None,
}

CANONICAL_UNITS = {"ms", "sec", "min", "h"}


def normalize_unit(unit: Union[str, None]) -> Union[str, None]:
    """
    Normalize a time unit string to its canonical form.
    
    Accepts various aliases (e.g., "seconds", "s", "secs") and returns the
    canonical unit name ("sec"). Returns None if input is None or one of
    the index aliases ("index", "indices", "idx", "sample", "samples").

    Parameters
    ----------
    unit : str or None
        Unit string to normalize (e.g., "seconds", "ms", "h"); case and
        surrounding whitespace are ignored. If None, returns None.

    Returns
    -------
    str or None
        Canonical unit name ("ms", "sec", "min", "h") or None for
        indices/samples (or if input is None)

    Raises
    ------
    ValueError
        If unit string is not recognized
    
    Examples
    --------
    >>> normalize_unit("seconds")
    'sec'
    >>> normalize_unit("ms")
    'ms'
    >>> normalize_unit("hours")
    'h'
    >>> normalize_unit("samples") is None
    True
    >>> normalize_unit("invalid")
    Traceback (most recent call last):
        ...
    ValueError: Unknown unit 'invalid'. Valid units are: h, min, ms, sec
    """
    # Handle None case
    if unit is None:
        return None
    
    # Normalize: lowercase and strip whitespace
    unit_normalized = str(unit).lower().strip()
    
    # Look up in aliases
    if unit_normalized in UNIT_ALIASES:
        return UNIT_ALIASES[unit_normalized]
    
    # Not found
    valid_units_str = ", ".join(sorted(CANONICAL_UNITS))
    raise ValueError(f"Unknown unit '{unit}'. Valid units are: {valid_units_str}")


def _warn_if_ignores_mask(fct, name, axis=None):
    """Log a warning if `fct` uses masked values of a masked array (e.g., `np.median`).

    `fct` is called on a small masked array and on its unmasked values only, as
    ``fct(x)`` or, if `axis` is given, as ``fct(x, axis=axis)``; a warning is
    logged if the results differ. Functions that fail on it are not checked.
    """
    import numpy.ma as ma
    from loguru import logger
    probe = ma.masked_array([5.0, 1.0, 4.0, 1e9], mask=[False, False, False, True])
    valid = probe.compressed()

    def call(x):
        r = fct(x) if axis is None else fct(x.reshape(1, -1), axis=axis)
        return float(np.asarray(r).ravel()[0])
    try:
        ignores_mask = not np.isclose(call(probe), call(valid))
    except Exception:
        return  # cannot check this function
    if ignores_mask:
        logger.warning(f"{name}={getattr(fct, '__name__', fct)} does not ignore masked "
                       "values; masked samples (blinks, padding) enter the result. "
                       "Use a mask-aware function such as np.mean or np.ma.median.")


def _check_accepts_axis(fct, name):
    """Raise a TypeError if `fct` cannot be called as ``fct(data, axis=...)``.

    Functions without a readable signature (e.g., some builtins) are not checked.
    """
    import inspect
    try:
        params = inspect.signature(fct).parameters.values()
    except (TypeError, ValueError):
        return
    if not any(p.name == "axis" or p.kind == p.VAR_KEYWORD for p in params):
        raise TypeError(f"{name}={getattr(fct, '__name__', fct)} must accept an `axis` "
                        f"argument; it is called as {name}(data, axis=1) "
                        "(e.g., np.mean, np.ma.median, scipy.stats.sem)")
