"""
generic.py
==========

Helpers of :class:`~pypillometry.EyeData` (history decorator, merging,
decimation, per-channel parameters) that the namespaces ``d.pupil`` and
``d.gaze`` use as well.
"""

from .eyedatadict import EyeDataDict, LAYER_DTYPE, MaskPolicy, _as_policy
from .namespace import Namespace
from .pipes import step
from ..signal import baseline

import numpy as np
import numbers
import warnings
from loguru import logger
import functools


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


#: methods renamed in version 2.2 (old name -> new name; no forwarders): the
#: pupil and gaze methods moved into the namespaces ``d.pupil`` and ``d.gaze``;
#: the plots are only available in ``d.plot`` (``d.pupil_plot()`` was a short
#: form of ``d.plot.pupil_plot()``)
_RENAMED_METHODS = {
    "pupil_lowpass_filter": "pupil.lowpass_filter",
    "pupil_smooth_window": "pupil.smooth_window",
    "pupil_merge_eyes": "pupil.merge_eyes",
    "pupil_blinks_detect": "pupil.blinks_detect",
    "pupil_blinks_interpolate": "pupil.blinks_interpolate",
    "pupil_estimate_baseline": "pupil.estimate_baseline",
    "pupil_estimate_response": "pupil.estimate_response",
    "blinks_merge_close": "pupil.blinks_merge_close",
    "correct_pupil_foreshortening": "pupil.correct_foreshortening",
    "fit_foreshortening": "pupil.fit_foreshortening",
    "mask_offscreen_coords": "gaze.mask_offscreen",
    "mask_eye_divergences": "gaze.mask_divergences",
    **{name: f"plot.{name}" for name in [
        "plot_timeseries", "plot_timeseries_segments", "plot_intervals",
        "plot_masked", "plot_experimental_setup", "pupil_plot",
        "pupil_plot_segments", "plot_blinks",
        "plot_pupil_foreshortening_error_surface", "plot_heatmap",
        "plot_scanpath", "plot_calibration"]},
}


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


def _some(values, n=10):
    """The first `n` of `values` as a list for messages (``[1.0, 2.0, ...]``)."""
    values = list(np.asarray(values).tolist())
    return str(values) if len(values) <= n else str(values[:n])[:-1] + ", ...]"


def _per_pair(pairs, fct):
    """Dict ``{"<eye>_<var>": fct(eye, var)}`` for the (eye, var) `pairs`."""
    return {f"{eye}_{var}": fct(eye, var) for eye, var in pairs}


def _check_per_pair(pairs, mean, sd):
    """Raise a ValueError if `mean` or `sd` miss a value for one of the `pairs`."""
    for name, values in (("mean", mean), ("sd", sd)):
        if any(isinstance(v, dict) for v in values.values()):
            raise ValueError(f"{name} must be keyed by channel, e.g. "
                             "{'left_pupil': 1000.}, not {'left': {'pupil': 1000.}}")
        for eye, var in pairs:
            if f"{eye}_{var}" not in values:
                raise ValueError(f"{name} has no value for {eye}_{var}")


def _merge_usage(d, keys, policy):
    """
    Which eyes (rows, one per key of `d`, an EyeDataDict) go into each
    sample of a merged channel: the samples that are usable under `policy`,
    but recorded samples first (an interpolated sample is used only where no
    eye has a usable sample that is not interpolated). A column is all False
    where no eye is usable.
    """
    usable = np.stack([~d.get_policy_mask(k, policy) for k in keys])
    recorded = usable & ~np.stack([d.get_layer(k, "interpolated") for k in keys])
    return np.where(recorded.any(axis=0), recorded, usable)


def _store_merged(d, key, values, keys, used, masked, policy):
    """
    Store merged `values` as `key` of `d` with the layers that are set in all
    eyes that went into a sample (`used`, see `_merge_usage`; in all eyes
    where none did). Samples in `masked` that are still usable under
    `policy` with these layers get the ``"user"`` layer.
    """
    layers = np.stack([d.layers[k] for k in keys])
    sel = np.where(used.any(axis=0), used, True)
    allbits = np.iinfo(LAYER_DTYPE).max
    merged = np.bitwise_and.reduce(np.where(sel, layers, allbits), axis=0)
    d._set_channel(key, values, merged.astype(LAYER_DTYPE))
    rest = masked & ~d.get_policy_mask(key, policy)
    if rest.any():
        d.set_layer(key, "user", rest)


def _decimate_factor(fs, fsd, dsfac):
    """Decimate factor for :meth:`EyeData.downsample` (checks the arguments)."""
    what = "decimate factor" if dsfac else "sampling rate in Hz"
    if isinstance(fsd, bool) or not isinstance(fsd, numbers.Real) or not np.isfinite(fsd):
        raise TypeError(f"fsd must be a finite number (the {what}), got {fsd!r}")
    if dsfac:
        if fsd < 1 or fsd != int(fsd):
            raise ValueError(f"the decimate factor must be a positive integer, got {fsd}")
        return int(fsd)
    if not 0 < fsd <= fs:
        raise ValueError(f"fsd must be a sampling rate in Hz above 0 and at most "
                         f"the current rate ({fs} Hz), got {fsd}")
    ratio = fs / fsd
    if np.isclose(ratio, round(ratio)):
        return int(round(ratio))
    factor = int(ratio)
    logger.warning(f"{fs} Hz is not a multiple of {fsd} Hz: decimating by {factor}, "
                   f"the new sampling rate is {fs/factor:g} Hz")
    return factor


def _keep_dict(d, keep):
    """EyeDataDict with the samples of `d` where `keep` is True (values,
    masks and layers)."""
    nd = EyeDataDict()
    nd.layer_names = list(d.layer_names)
    for k, v in d.items():
        nd._set_channel(k, v[keep], d.layers[k][keep])
    if d.rec_layers is not None:
        nd.rec_layers = d.rec_layers[keep].copy()
    return nd


def _decimate_dict(d, factor, policy="default"):
    """EyeDataDict with the samples of `d` decimated (see EyeData._decimate)."""
    nd = EyeDataDict()
    nd.layer_names = list(d.layer_names)
    pol = _as_policy(policy)
    # layers that make a sample unusable under the policy
    masking = LAYER_DTYPE(d.layer_bits(pol.layers) & ~d.layer_bits(pol.unless))
    if d.rec_layers is not None:
        nd.rec_layers = _decimate_layers(d.rec_layers, factor)
    for k, v in d.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
            y = baseline.downsample(np.where(d.get_policy_mask(k, policy), np.nan, v), factor)
            stored = baseline.downsample(v, factor)
        allmasked = np.isnan(y)
        y[allmasked] = stored[allmasked]
        nd._set_channel(k, y, _decimate_layers(d.layers[k], factor))
        # a window whose samples are all unusable stays unusable: if the
        # layers that cover more than half of it do not make it unusable,
        # it gets the masking layers of its samples (if there are none,
        # all its values are NaN, and so is the new value)
        todo = allmasked & ~nd.get_policy_mask(k, MaskPolicy(pol.layers, pol.unless, nan=False))
        if todo.any():
            present = np.bitwise_or.reduceat(d.layers[k] & masking,
                                             np.arange(0, len(v), factor))
            nd.layers[k][todo] |= present[todo]
    return nd


def _decimate_layers(layers, factor):
    """Decimated layer bit-field: a bit is set where it is set for more than
    half of the samples in the window (the last window may be shorter)."""
    n = len(layers)
    nw = -(-n // factor)
    out = np.zeros(nw, dtype=LAYER_DTYPE)
    if n == 0:
        return out
    windows = np.concatenate([layers, np.zeros(nw * factor - n, dtype=LAYER_DTYPE)])
    windows = windows.reshape(nw, factor)
    size = np.full(nw, factor)
    size[-1] = n - (nw - 1) * factor
    present = int(np.bitwise_or.reduce(layers))
    for i in range(present.bit_length()):
        bit = LAYER_DTYPE(1 << i)
        if present & bit:
            out[2 * ((windows & bit) != 0).sum(axis=1) > size] |= bit
    return out


def _insert_dict(d, n, pos, value):
    """EyeDataDict of length `n` with the samples of `d` at `pos`; new samples
    hold `value` and are masked as missing (NaN, or the ``"user"`` layer if
    `value` is not NaN; no other layers)."""
    nd = EyeDataDict()
    nd.layer_names = list(d.layer_names)
    missing = 0 if np.isnan(value) else nd.layer_bits("user", create=n > len(pos))
    for k, v in d.items():
        nv = np.full(n, value, dtype=float)
        nl = np.full(n, missing, dtype=LAYER_DTYPE)  # inserted = missing
        nv[pos] = v
        nl[pos] = d.layers[k]
        nd._set_channel(k, nv, nl)
    if d.rec_layers is not None:
        nd.rec_layers = np.zeros(n, dtype=LAYER_DTYPE)
        nd.rec_layers[pos] = d.rec_layers
    return nd


def _check_timebase(tx, fs, rtol=0.05):
    """
    Raise a ValueError if the sampling rate `fs` (Hz) does not fit the time
    vector `tx` (ms).

    `fs` fits if ``1000/fs`` agrees (within `rtol`) with the median or the
    mean step of `tx`. The median is robust to gaps, the mean to time
    stamps rounded to whole ms (e.g., 300 Hz with steps of 3 and 4 ms).
    """
    if isinstance(fs, bool) or not isinstance(fs, numbers.Real) or not (0 < fs < np.inf):
        raise ValueError(f"the sampling rate must be a positive number (Hz), got {fs!r}")
    if len(tx) < 2:
        return
    median = np.median(np.diff(tx))
    mean = (tx[-1] - tx[0]) / (len(tx) - 1)
    if not any(np.isclose(step, 1000. / fs, rtol=rtol, atol=0) for step in (median, mean)):
        raise ValueError(f"the sampling rate ({fs:g} Hz) does not match the time vector "
                         f"(median step {median:g} ms, i.e. {1000. / median:g} Hz)")
