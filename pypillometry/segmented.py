"""
segmented.py
============

Segmented eye data for event-related analysis.

:class:`SegmentedEyeData` holds equal-length segments of one or several
channels (e.g., ``left_pupil``, ``left_x`` and ``left_y``) cut around
events, with the epochs they were cut from once for all channels. As for
:class:`~pypillometry.EyeData`, the methods for one kind of data are in
namespaces: ``seg.pupil`` (baseline correction, plot of the event-related
pupil dilation) and ``seg.gaze`` (regions of interest).

The columns of the segments are single trials or averages of trials
(:meth:`SegmentedEyeData.average`); the segments of several participants
are combined with :func:`concat_segments` and :func:`group_segments`.
"""
import copy
from collections.abc import Mapping

import numpy as np
import numpy.ma as ma
from loguru import logger

from .io import write_file, read_file
from . import config, state
from .units import parse_time, _ms_to_samples
from .convenience import _warn_if_ignores_mask, _check_accepts_axis, _plain_values
from .eyedata.namespace import Namespace, accessor
from .templates import render, sparkline

__all__ = [
    "SegmentedEyeData", "concat_segments", "group_segments",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _extract_epochs(eyedata, epochs, variable, policy="default"):
    """
    Time axis and segments (masked array, time points x epochs) of `variable`,
    masked where the samples are unusable under the mask `policy`.

    The length and the time axis come from the window: ``w0 + arange(n) *
    1000/fs`` ms, with ``n`` the window length in samples (rounded). Each
    segment starts at the first sample at or after ``onset + w0``. Parts
    outside the data are masked (their values are NaN).
    """
    if np.isnan(epochs.onsets).any():
        raise ValueError("these epochs have no time (their onsets are NaN, e.g., the "
                         "columns of averaged segments), so no segments can be cut")
    w0, w1 = epochs.window
    n = eyedata._samples(w1 - w0, rounding="nearest")
    tx = w0 + np.arange(n) * 1000. / eyedata.fs
    starts = np.atleast_1d(_ms_to_samples(eyedata, epochs.onsets + w0))
    ix = starts[None, :] + np.arange(n)[:, None]
    inside = (ix >= 0) & (ix < len(eyedata.tx))
    values = np.full(ix.shape, np.nan)
    mask = np.ones(ix.shape, dtype=bool)
    values[inside] = eyedata.data[variable][ix[inside]]
    mask[inside] = eyedata.data.get_policy_mask(variable, policy)[ix[inside]]
    return tx, ma.masked_array(values, mask=mask)


def _epochs_from_spans(eyedata, intervals):
    """
    Epochs for Intervals of the same length: onsets at the starts, window
    (0, length) in ms; their event labels and metadata are kept.
    """
    from .epochs import Epochs
    if len(intervals) == 0:
        raise ValueError("no intervals to extract segments from")
    lengths = np.diff(np.asarray(intervals, dtype=float), axis=1)[:, 0]
    if not np.allclose(lengths, lengths[0], rtol=1e-9, atol=0):
        raise ValueError(
            f"get_segments() needs intervals of the same length, got lengths from "
            f"{lengths.min():g} to {lengths.max():g} {intervals.units}; for "
            f"event-locked segments, use a window around single events, e.g., "
            f"get_epochs(event, window=(-200, 800) * u.ms)")
    if lengths[0] <= 0:
        raise ValueError("get_segments() needs intervals that are longer than 0")
    spans = np.asarray(intervals.to_units("ms", eyedata), dtype=float)
    labels = None if intervals.event_labels is None else \
        [str(lab) for lab in intervals.event_labels]
    return Epochs(spans[:, 0], (0., float(np.mean(spans[:, 1] - spans[:, 0]))),
                  labels=labels, meta=intervals.event_meta, label=intervals.label)


def _split_channel(channel):
    """``(eye, variable)`` of a channel name ``"<eye>_<variable>"``."""
    if not isinstance(channel, str) or "_" not in channel.strip("_"):
        raise ValueError(f"channel names have the form '<eye>_<variable>' "
                         f"(e.g., 'left_pupil'), got {channel!r}")
    eye, var = channel.split("_", 1)
    if not eye or not var:
        raise ValueError(f"channel names have the form '<eye>_<variable>' "
                         f"(e.g., 'left_pupil'), got {channel!r}")
    return eye, var


def _as_list(x):
    """None -> [], str -> [str], other iterables -> list."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _select_channels(channels, eyes=None, variables=None):
    """
    The channels among `channels` of the given eyes and variables (None or
    [] means all), in the order of `channels`.

    Raises a KeyError if a requested eye or variable has no channel among
    the selected ones, or if none is selected.
    """
    eyes, variables = _as_list(eyes), _as_list(variables)
    out = [ch for ch in channels
           if (not eyes or _split_channel(ch)[0] in eyes)
           and (not variables or _split_channel(ch)[1] in variables)]
    found_eyes = {_split_channel(ch)[0] for ch in out}
    found_vars = {_split_channel(ch)[1] for ch in out}
    missing = [e for e in eyes if e not in found_eyes] + \
        [v for v in variables if v not in found_vars]
    if missing or not out:
        what = ", ".join(repr(m) for m in missing) or "the selection"
        raise KeyError(f"no channels for {what} in the segments (available "
                       f"channels: {', '.join(channels) or 'none'})")
    return out


def _keep_epochs(epochs, mask):
    """The epochs (Epochs, Intervals or None) where `mask` is True."""
    from .epochs import Epochs
    from .intervals import Intervals
    if epochs is None:
        return None
    if isinstance(epochs, Epochs):
        return epochs.keep(mask)
    # Intervals (segments from old files that were not event-locked)
    idx = np.flatnonzero(mask)

    def pick(values):
        return None if values is None else [values[i] for i in idx]
    meta = None if epochs.event_meta is None else \
        {k: np.asarray(v)[idx] for k, v in epochs.event_meta.items()}
    return Intervals(pick(list(epochs.intervals)), units=epochs.units, label=epochs.label,
                     event_labels=pick(epochs.event_labels),
                     event_onsets=pick(epochs.event_onsets), event_meta=meta)


def _check_baseline_window(tx, sampling_rate, window):
    """Raise a ValueError if the baseline `window` (ms) is not within the time axis `tx`."""
    step = 1000. / sampling_rate if sampling_rate else \
        float(np.median(np.diff(tx))) if len(tx) > 1 else 0.
    # a time point stands for the samples within half a step around it;
    # the end of a window is exclusive
    lo, hi = tx[0] - step / 2, tx[-1] + step / 2
    seg = f"the segments cover {tx[0]:g} to {tx[-1]:g} ms"
    if np.ndim(window) == 0:
        if not lo <= window <= hi:
            raise ValueError(f"baseline time {window:g} ms is outside the segments; {seg}")
        return
    if len(window) != 2 or window[0] >= window[1]:
        raise ValueError(f"baseline window must be (start, end) with start < end, "
                         f"got {tuple(window)}")
    if window[0] < lo or window[1] > hi + step / 2:
        raise ValueError(f"baseline window ({window[0]:g}, {window[1]:g}) ms is not "
                         f"within the segments; {seg}")


def _window_mask(tx, window):
    """
    Time points of `tx` within `window` (start inclusive, end exclusive; ms
    or with units); None means the whole time axis, and None as start or
    end leaves that side open.
    """
    if window is None:
        return np.ones(len(tx), dtype=bool)
    if isinstance(window, (tuple, list)) and any(w is None for w in window):
        if len(window) != 2:
            raise ValueError(f"window must be (start, end), got {window}")
        lo, hi = (None if w is None else float(parse_time(w)) for w in window)
    else:
        lo, hi = np.asarray(parse_time(window), dtype=float).reshape(2)
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    if lo >= hi:
        raise ValueError(f"window must be (start, end) with start < end, got ({lo:g}, {hi:g})")
    sel = (tx >= lo) & (tx < hi)
    if not sel.any():
        raise ValueError(f"window ({lo:g}, {hi:g}) ms contains no time point of the "
                         f"segments ({tx[0]:g} to {tx[-1]:g} ms)")
    return sel


def _time_step(tx, sampling_rate):
    """Time between two time points in ms."""
    if sampling_rate:
        return 1000. / sampling_rate
    return float(np.median(np.diff(tx))) if len(tx) > 1 else 1.


def _untimed_epochs(n, tx, sampling_rate, window=None, labels=None, meta=None, label=None):
    """Epochs without a time (NaN onsets) for `n` columns of segments with
    time axis `tx` (window: `window`, or that of the time axis)."""
    from .epochs import Epochs
    if window is None:
        window = (float(tx[0]), float(tx[-1]) + _time_step(tx, sampling_rate))
    return Epochs(np.full(n, np.nan), window, labels=labels, meta=meta, label=label)


def _all_equal(values) -> bool:
    """Whether all `values` (1-d array) are equal (NaN equal to NaN)."""
    v = np.asarray(values)
    if len(v) < 2:
        return True
    same = v == v[0]
    if v.dtype.kind == "f":
        same |= np.isnan(v) & np.isnan(v[0])
    return bool(np.all(same))


def _group_key(value):
    """`value` as a key for grouping (NaN -> None, so that NaNs form one group)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def sem(data, axis=None):
    """Standard error of the mean, ignoring masked values
    (:func:`scipy.stats.sem`; the default error of the segment plots)."""
    import scipy.stats
    return scipy.stats.sem(data, axis=axis)


# --------------------------------------------------------------------------
# Namespaces
# --------------------------------------------------------------------------

class SegmentedPupilNamespace(Namespace):
    """
    Methods for pupil segments, accessed as ``seg.pupil`` (e.g.,
    ``seg.pupil.baseline_correct((-200, 0))``).

    The methods work on the pupil channels (``<eye>_pupil``) of a
    :class:`~pypillometry.SegmentedEyeData` object (all eyes, or those given
    in `eyes`) and raise a KeyError if there is none. Averaging
    (:meth:`~pypillometry.SegmentedEyeData.average`,
    :func:`~pypillometry.group_segments`) works for all channels.
    """

    _name = "pupil"

    def _channels(self, eyes):
        return _select_channels(self._d.channels, eyes, "pupil")

    def baseline_correct(self, window=0, method: str = "subtract", eyes=[], inplace=None):
        """
        Correct each segment of the pupil channels for its baseline.

        The window is given in ms relative to the event onsets (time 0), e.g.,
        ``window=(-200, 0)`` uses the mean of the 200 ms before each event.
        `method` says how the baseline is removed: ``"subtract"`` (default;
        the change from the baseline in the units of the data),
        ``"divide"`` (the ratio to the baseline) or ``"percent"`` (the
        change in % of the baseline, ``(x - base) / base * 100``). Other
        channels (e.g., gaze) are not changed.

        Parameters
        ----------
        window : tuple (float, float), float, str, Quantity or None
            Baseline window relative to event onset (time 0), in ms or with
            units (e.g., ``(-200, 0) * u.ms``):

            - If None, no baseline correction is applied
            - If tuple (start, end), the mean value in this time window is subtracted
            - If float, the value at the time point closest to this value is used

            Default is 0 (baseline at event onset).

            The window must be within the time axis of the segments.
        method : str, optional
            ``"subtract"`` (default), ``"divide"`` or ``"percent"``;
            segments whose baseline is masked (or 0, for ``"divide"`` and
            ``"percent"``) are masked
        eyes : str or list of str, optional
            eyes whose pupil channels are corrected (default: all)
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, return a new object with the corrected data;
            if `None`, use the package-level setting (see
            :func:`~pypillometry.set_default_inplace`)

        Returns
        -------
        SegmentedEyeData
            the baseline-corrected segments (the object or a new one,
            depending on `inplace`; the object itself if `window` is None)

        Raises
        ------
        ValueError
            if the window (or time point) is not within the segments, its
            start is not before its end, or `method` is unknown
        KeyError
            if there are no pupil channels (of `eyes`)

        Examples
        --------
        >>> import pypillometry as pp
        >>> from pypillometry import u
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
        >>> seg = d.get_segments(epochs)
        >>> # Use pre-event period as baseline
        >>> seg_bc = seg.pupil.baseline_correct(window=(-200, 0), inplace=False)
        >>> seg = seg.pupil.baseline_correct(window=(-200, 0) * u.ms)

        >>> # Use single time point as baseline, left eye only
        >>> seg = seg.pupil.baseline_correct(window=-100, eyes="left")

        >>> # Change in % of the baseline
        >>> seg_pc = d.get_segments(epochs).pupil.baseline_correct((-200, 0), method="percent")
        """
        if method not in ("subtract", "divide", "percent"):
            raise ValueError(f"unknown method {method!r}; use 'subtract', 'divide' or "
                             f"'percent'")
        seg = self._d
        if window is None:
            logger.warning("No baseline correction applied")
            return seg
        window = parse_time(window)
        _check_baseline_window(seg.tx, seg.sampling_rate, window)
        channels = self._channels(eyes)

        corrected = {}
        for ch in channels:
            arr = seg[ch].copy()
            if np.ndim(window) == 0:
                # single time point
                base = arr[np.argmin(np.abs(seg.tx - window))].copy()
            else:
                # time window
                i0, i1 = (np.argmin(np.abs(bw - seg.tx)) for bw in window)
                base = arr[i0:i1].mean(axis=0)
            if method == "subtract":
                arr -= base
            elif method == "divide":
                arr = arr / base
            else:
                arr = (arr - base) / base * 100.
            corrected[ch] = arr
        logger.info(f"Baseline correction ({method}) applied using window {window}")

        if inplace is None:
            inplace = config.get_default_inplace()
        out = seg if inplace else seg._copy()
        for ch, arr in corrected.items():
            out[ch] = arr
        return out

    def plot(self, eyes=[], meanfct=np.mean, varfct=sem,
             show_missing: bool = None, show_zero_line: bool = True,
             show_legend: bool = None, title: str = None, label: str = None,
             ribbon_alpha: float = 0.3, ax=None, **kwargs):
        """
        Plot the event-related pupil dilation: mean and error ribbon of the
        pupil channels over time.

        This is :meth:`SegmentedEyeData.plot
        <pypillometry.SegmentedEyeData.plot>` for the pupil channels (one
        line per eye); see there for the parameters.

        Parameters
        ----------
        eyes : str or list of str, optional
            eyes whose pupil channels are plotted (default: all)

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object for further customization

        Examples
        --------
        Event-related pupil dilation after each stimulus. Pass ``ax=`` to overlay
        several segments (e.g., two conditions) in one plot:

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> d = d | pp.pupil.blinks_detect() | pp.pupil.blinks_interpolate()
           >>> epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
           >>> erpd = d.get_segments(epochs).pupil.baseline_correct((-500, 0))
           >>> ax = erpd.pupil.plot()
           >>> # the left eye in the first and in the second half of the trials
           >>> fig, ax = plt.subplots()
           >>> half = erpd.epochs.onsets < erpd.epochs.onsets[5]
           >>> ax = erpd.keep(half).pupil.plot("left", ax=ax, label="first half")
           >>> ax = erpd.keep(~half).pupil.plot("left", ax=ax, label="second half")
           >>> legend = ax.legend()
        """
        return self._d.plot(self._channels(eyes), meanfct=meanfct, varfct=varfct,
                            show_missing=show_missing, show_zero_line=show_zero_line,
                            show_legend=show_legend, title=title, label=label,
                            ylabel="mean pupil", ribbon_alpha=ribbon_alpha, ax=ax,
                            **kwargs)


class SegmentedGazeNamespace(Namespace):
    """
    Methods for gaze segments and regions of interest (ROIs), accessed as
    ``seg.gaze`` (e.g., ``seg.gaze.roi_hits(roi)``).

    The methods need the gaze position (``<eye>_x`` and ``<eye>_y``) of a
    :class:`~pypillometry.SegmentedEyeData` object (all eyes that have both,
    or those given in `eyes`) and raise a KeyError if there is none.

    The ROIs (:class:`~pypillometry.CircularROI`,
    :class:`~pypillometry.RectangularROI`) are in pixels, like the gaze
    position. Where the gaze is in an ROI becomes a channel with the values
    1 (in the ROI) and 0 (outside), masked where x or y are masked
    (:meth:`roi_hits`). Its mean over the segments at each time point is the
    share of the (usable) trials in which the gaze is in the ROI (a fixation
    curve, :meth:`plot_roi`); mean ± error plots and averaging
    (:meth:`~pypillometry.SegmentedEyeData.average`,
    :func:`~pypillometry.group_segments`) work for it like for any channel.
    Measures per trial (:meth:`dwell_time`, :meth:`first_entry`) are
    returned as tables with the epochs' onsets, labels and metadata.

    Where the gaze is in an ROI is a property of single trials: the ROI
    channels must be added before averaging (the mean gaze position of
    several trials says nothing about where the gaze was), and
    :meth:`roi_hits` and :meth:`first_entry` raise a ValueError for averaged
    segments.
    """

    _name = "gaze"

    def _eyes(self, eyes):
        """Eyes with both x and y channels (all, or those in `eyes`)."""
        seg = self._d
        found = [e for e in seg.eyes if f"{e}_x" in seg and f"{e}_y" in seg]
        eyes = _as_list(eyes)
        missing = [e for e in eyes if e not in found]
        if missing or not found:
            what = ", ".join(repr(e) for e in missing) or "any eye"
            raise KeyError(f"no gaze channels (x and y) for {what} in the segments "
                           f"(available channels: {', '.join(seg.channels)})")
        return eyes or found

    @staticmethod
    def _roi_name(roi, name=None):
        """Name of the ROI channels (`name`, the name of `roi`, or `roi` if a str)."""
        if isinstance(roi, str):
            name = roi
        elif name is None:
            name = roi.name
        if not isinstance(name, str) or not name:
            raise ValueError("the ROI has no name; pass name= (e.g., name='target')")
        if name in ("x", "y", "pupil"):
            raise ValueError(f"an ROI cannot be called {name!r} (a channel of the gaze or pupil)")
        return name

    def _check_single_trials(self, what):
        """Raise a ValueError if some columns of the segments are averages."""
        if self._d._averaged().any():
            raise ValueError(f"{what} needs single trials, but the segments contain "
                             f"averages (see their metadata column 'n_trials'); use "
                             f"the segments before averaging")

    def _hits(self, roi, eye):
        """1 where the gaze of `eye` is in `roi`, 0 outside (masked where x or y are)."""
        self._check_single_trials("where the gaze is in an ROI")
        seg = self._d
        x, y = seg[f"{eye}_x"], seg[f"{eye}_y"]
        xv, yv = ma.getdata(x), ma.getdata(y)
        inside = np.asarray(roi.contains(np.column_stack([xv.ravel(), yv.ravel()])), dtype=bool)
        mask = ma.getmaskarray(x) | ma.getmaskarray(y) | np.isnan(xv) | np.isnan(yv)
        return ma.masked_array(inside.reshape(x.shape).astype(float), mask=mask)

    def _roi_data(self, roi, eye):
        """ROI channel of `eye`: computed from `roi`, or the channel ``<eye>_<roi>`` if a str."""
        if not isinstance(roi, str):
            return self._hits(roi, eye)
        ch = f"{eye}_{self._roi_name(roi)}"
        if ch not in self._d:
            raise KeyError(f"no channel {ch!r} in the segments; create it with "
                           f"seg.gaze.roi_hits(roi, name={roi!r}) or pass the ROI")
        return self._d[ch]

    def _roi_eyes(self, roi, eyes):
        """Eyes for an ROI given as ROI (eyes with gaze) or name (eyes with the channel)."""
        if not isinstance(roi, str):
            return self._eyes(eyes)
        name = self._roi_name(roi)
        try:
            chs = _select_channels(self._d.channels, eyes, name)
        except KeyError:
            raise KeyError(f"no ROI channels {name!r} in the segments (available channels: "
                           f"{', '.join(self._d.channels)}); create them with "
                           f"seg.gaze.roi_hits(roi, name={name!r}) or pass the ROI") from None
        return [_split_channel(ch)[0] for ch in chs]

    def _table(self, values):
        """Table (dict of columns) with the epochs' onsets, labels and metadata
        (if the segments have Epochs) and `values`."""
        from .epochs import Epochs
        ep = self._d.epochs
        table = ep.events.to_table() if isinstance(ep, Epochs) else {}
        table.update(values)
        return table

    def roi_hits(self, roi, name: str = None, eyes=[], inplace=None):
        """
        Add channels that say where the gaze is in a region of interest.

        For each eye, the channel ``<eye>_<name>`` is 1 where the gaze
        position (``<eye>_x``, ``<eye>_y``) is in `roi` and 0 where it is
        outside; it is masked where x or y are masked. Its mean over the
        segments (e.g., with :meth:`plot_roi` or
        :meth:`SegmentedEyeData.plot <pypillometry.SegmentedEyeData.plot>`)
        is the share of the usable trials in which the gaze is in the ROI
        at each time point. An existing channel of that name is replaced.

        Parameters
        ----------
        roi : ROI
            the region of interest (e.g., :class:`~pypillometry.CircularROI`,
            in pixels)
        name : str, optional
            name of the ROI in the channel names (default: the name of `roi`)
        eyes : str or list of str, optional
            eyes to use (default: all that have x and y)
        inplace : bool or None, optional
            if `True`, add the channels to the object and return it; if
            `False`, return a new object with the channels added; if
            `None`, use the package-level setting (see
            :func:`~pypillometry.set_default_inplace`)

        Returns
        -------
        SegmentedEyeData
            the segments with the ROI channels (the object or a new one,
            depending on `inplace`)

        Raises
        ------
        ValueError
            if neither `name` nor the ROI give a name, or the name is
            ``"x"``, ``"y"`` or ``"pupil"``
        KeyError
            if there are no gaze channels (of `eyes`)

        Examples
        --------
        >>> import pypillometry as pp
        >>> from pypillometry import u, CircularROI
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> seg = d.get_segments(d.get_epochs("S_", (-500, 2500) * u.ms))
        >>> center = CircularROI((640, 512), radius=200, name="center")
        >>> seg = seg.gaze.roi_hits(center, inplace=False)
        >>> [ch for ch in seg.channels if ch.endswith("center")]
        ['left_center', 'right_center']
        """
        if isinstance(roi, str):
            raise TypeError("roi must be an ROI object (e.g., CircularROI), not a str")
        name = self._roi_name(roi, name)
        hits = {f"{eye}_{name}": self._hits(roi, eye) for eye in self._eyes(eyes)}
        if inplace is None:
            inplace = config.get_default_inplace()
        out = self._d if inplace else self._d._copy()
        for ch, values in hits.items():
            out[ch] = values
        return out

    def dwell_time(self, roi, eyes=[], window=None) -> dict:
        """
        Time the gaze spends in a region of interest in each trial (ms).

        The dwell time is the number of usable samples in the ROI times the
        sampling interval; masked samples are not counted. Trials without
        usable samples in the window get NaN. For averaged segments, whose
        ROI channels are shares of trials, it is the mean dwell time of the
        trials of each average.

        Parameters
        ----------
        roi : ROI or str
            the region of interest, or the name of ROI channels added with
            :meth:`roi_hits`
        eyes : str or list of str, optional
            eyes to use (default: all that have x and y, or the ROI channel)
        window : tuple (start, end), Quantity or None, optional
            part of the segments to use, in ms relative to the event onsets
            (or with units); the end is exclusive, None as start or end leaves
            that side open; default: the whole segments

        Returns
        -------
        dict
            a table (column name -> array with one value per trial): the
            onsets, labels and metadata of the epochs (if the segments have
            them) and the dwell time of each eye in ms (``"<eye>_<roi
            name>"``); can be passed to ``pandas.DataFrame()``

        Examples
        --------
        >>> import pypillometry as pp
        >>> from pypillometry import u, CircularROI
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> seg = d.get_segments(d.get_epochs("S_", (-500, 2500) * u.ms))
        >>> center = CircularROI((640, 512), radius=200, name="center")
        >>> table = seg.gaze.dwell_time(center, eyes="left", window=(0, 2000))
        >>> sorted(table)
        ['label', 'left_center', 'onset']
        """
        seg = self._d
        sel = _window_mask(seg.tx, window)
        step = _time_step(seg.tx, seg.sampling_rate)
        name = roi if isinstance(roi, str) else self._roi_name(roi)
        values = {}
        for eye in self._roi_eyes(roi, eyes):
            hits = self._roi_data(roi, eye)[sel]
            n_in = ma.filled(ma.sum(hits, axis=0).astype(float), np.nan)
            values[f"{eye}_{name}"] = np.asarray(n_in, dtype=float) * step
        return self._table(values)

    def first_entry(self, roi, eyes=[], window=(0, None)) -> dict:
        """
        Time of the first usable sample in a region of interest in each trial (ms).

        The time is relative to the event onset, from the start of `window`
        (by default, the onset): a trial in which the gaze is in the ROI at
        the onset gets 0. Trials in which the gaze is never in the ROI (in
        the usable samples of the window) get NaN. Averaged segments have no
        first entry (a ValueError is raised).

        Parameters
        ----------
        roi : ROI or str
            the region of interest, or the name of ROI channels added with
            :meth:`roi_hits`
        eyes : str or list of str, optional
            eyes to use (default: all that have x and y, or the ROI channel)
        window : tuple (start, end), Quantity or None, optional
            part of the segments to search, in ms relative to the event
            onsets (or with units); the end is exclusive, None as start or
            end leaves that side open, None searches the whole segments;
            default ``(0, None)``: from the onset on

        Returns
        -------
        dict
            a table (column name -> array with one value per trial): the
            onsets, labels and metadata of the epochs (if the segments have
            them) and the time of the first entry of each eye in ms
            (``"<eye>_<roi name>"``); can be passed to ``pandas.DataFrame()``

        Examples
        --------
        >>> import pypillometry as pp
        >>> from pypillometry import u, CircularROI
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> seg = d.get_segments(d.get_epochs("S_", (-500, 2500) * u.ms))
        >>> center = CircularROI((640, 512), radius=200, name="center")
        >>> table = seg.gaze.first_entry(center)
        >>> len(table["left_center"]) == seg.n_segments
        True
        """
        self._check_single_trials("first_entry()")
        seg = self._d
        sel = _window_mask(seg.tx, window)
        tx = seg.tx[sel]
        name = roi if isinstance(roi, str) else self._roi_name(roi)
        values = {}
        for eye in self._roi_eyes(roi, eyes):
            inside = ma.filled(self._roi_data(roi, eye)[sel], 0) > 0.5
            first = tx[np.argmax(inside, axis=0)]
            values[f"{eye}_{name}"] = np.where(inside.any(axis=0), first, np.nan)
        return self._table(values)

    def plot_roi(self, roi, eyes=[], meanfct=np.mean, varfct=sem,
                 show_missing: bool = None, show_zero_line: bool = True,
                 show_legend: bool = None, title: str = None, label: str = None,
                 ribbon_alpha: float = 0.3, ax=None, **kwargs):
        """
        Plot the share of trials in which the gaze is in a region of interest
        over time (fixation curve), with an error ribbon.

        This is :meth:`SegmentedEyeData.plot
        <pypillometry.SegmentedEyeData.plot>` for the ROI channels (one line
        per eye); see there for the parameters.

        Parameters
        ----------
        roi : ROI or str
            the region of interest, or the name of ROI channels added with
            :meth:`roi_hits`
        eyes : str or list of str, optional
            eyes to plot (default: all that have x and y, or the ROI channel)

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object for further customization

        Examples
        --------
        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u, CircularROI
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> seg = d.get_segments(d.get_epochs("S_", (-500, 2500) * u.ms))
           >>> center = CircularROI((640, 512), radius=200, name="center")
           >>> ax = seg.gaze.plot_roi(center)
        """
        seg = self._d
        name = self._roi_name(roi)
        eyes = self._roi_eyes(roi, eyes)
        if not isinstance(roi, str):
            seg = self.roi_hits(roi, eyes=eyes, inplace=False)
        return seg.plot([f"{eye}_{name}" for eye in eyes], meanfct=meanfct, varfct=varfct,
                        show_missing=show_missing, show_zero_line=show_zero_line,
                        show_legend=show_legend, title=title, label=label,
                        ylabel=f"share of trials in {name}", ribbon_alpha=ribbon_alpha,
                        ax=ax, **kwargs)


# --------------------------------------------------------------------------
# Segments of one dataset
# --------------------------------------------------------------------------

class SegmentedEyeData(state.VersionedState):
    """
    Segments of eye data around events (e.g., event-related pupil dilation).

    Holds equal-length segments of one or several channels (e.g.,
    ``"left_pupil"``, ``"left_x"`` and ``"left_y"``), each a masked array
    of shape (time points, segments), together with the epochs they were
    cut from (event onsets, window, labels and metadata; once for all
    channels), the time axis and the sampling rate. The mask marks missing
    or unusable samples.

    Usually created with :meth:`~pypillometry.EyeData.get_segments` from the
    :class:`~pypillometry.Epochs` returned by
    :meth:`~pypillometry.EyeData.get_epochs`, e.g.,
    ``d.get_segments(d.get_epochs("F", window=(-200, 800) * u.ms))``.

    A channel is read with ``seg["left_pupil"]`` and set with
    ``seg["left_pupil"] = values``. The methods of the object work on all
    channels: selecting trials (:meth:`keep`, :meth:`filter`) and channels
    (:meth:`select_channels`, :meth:`drop_channels`), the plot of mean and
    error over time (:meth:`plot`) and saving (:meth:`write_file`). The
    methods for one kind of data are in namespaces, as for
    :class:`~pypillometry.EyeData`:

    - ``seg.pupil`` (:class:`~pypillometry.segmented.SegmentedPupilNamespace`):
      baseline correction and the plot of the event-related pupil dilation
    - ``seg.gaze`` (:class:`~pypillometry.segmented.SegmentedGazeNamespace`):
      regions of interest (where the gaze is in an ROI, fixation curves,
      dwell time and first entry per trial)

    **Averages.** A column of the segments is a single trial or an average
    of trials: :meth:`average` averages all trials or groups of them (e.g.,
    by condition), :func:`~pypillometry.concat_segments` stacks the segments
    of several participants (with the metadata column ``"subject"``) and
    :func:`~pypillometry.group_segments` averages them per participant.
    Averages are segments like the others; their epochs have no time (NaN
    onsets) and their metadata are the grouping columns, the columns that
    were the same in all their trials, and the number of trials
    (``"n_trials"``). An average is masked where all its trials are masked;
    `valid_share` holds the share of its trials that were usable at each
    time point (used for the missing data in :meth:`plot`).

    Parameters
    ----------
    name : str
        name of the segments (e.g., "cue-locked")
    tx : np.ndarray
        time axis relative to the event onsets in ms (one value per time point)
    data : dict
        channel name (``"<eye>_<variable>"``, e.g., ``"left_pupil"``) ->
        array of shape (time points, segments); arrays that are not masked
        arrays are masked where they are NaN
    epochs : Epochs, optional
        the epochs the segments were cut from (one per segment); segments
        from old files made from intervals that were not event-locked keep
        these Intervals
    sampling_rate : float, optional
        sampling rate in Hz
    valid_share : dict, optional
        channel name -> share (0 to 1) of the trials of each value that are
        usable, an array of the shape of the channel (for averages; created
        by :meth:`average`); channels that are not in it have 1 where they
        are not masked and 0 where they are

    Raises
    ------
    ValueError
        if there are no channels, a channel name does not have the form
        ``"<eye>_<variable>"``, or the shapes of the channels, the time axis,
        the epochs and `valid_share` do not match

    Examples
    --------
    >>> import pypillometry as pp
    >>> from pypillometry import u
    >>> d = pp.get_example_data("rlmw_002_short")
    >>> seg = d.get_segments(d.get_epochs("S_", (-500, 2500) * u.ms))
    >>> seg.channels
    ['left_x', 'left_y', 'left_pupil', 'right_x', 'right_y', 'right_pupil']
    >>> seg["left_pupil"].shape  # (time points, segments)
    (1500, 10)
    >>> seg = seg.pupil.baseline_correct((-200, 0))
    >>> seg.average()["left_pupil"].shape  # (time points, averages)
    (1500, 1)
    """

    pupil = accessor(SegmentedPupilNamespace)
    gaze = accessor(SegmentedGazeNamespace)

    # state (see pypillometry.state); `epochs` is Epochs or Intervals
    _STATE_FIELDS = ("name", "tx", ("data", "dict:masked"), ("epochs", "tagged"),
                     "sampling_rate", "valid_share")

    def __init__(self, name: str, tx: np.ndarray, data: dict, epochs=None,
                 sampling_rate: float = None, valid_share: dict = None):
        if not isinstance(data, Mapping):
            raise TypeError(f"data must be a dict (channel name -> array of shape "
                            f"(time points, segments)), got {type(data).__name__}")
        if not data:
            raise ValueError("segments need at least one channel")
        self.name = name
        self.tx = np.asarray(tx, dtype=float)
        if self.tx.ndim != 1:
            raise ValueError(f"tx must be one-dimensional, got shape {self.tx.shape}")
        self.sampling_rate = None if sampling_rate is None else float(sampling_rate)
        self.data = {}
        self.valid_share = {}
        for channel, values in data.items():
            self[channel] = values
        if epochs is not None and len(epochs) != self.n_segments:
            raise ValueError(f"{len(epochs)} epochs for {self.n_segments} segments")
        self.epochs = epochs
        for channel, share in (valid_share or {}).items():
            share = np.asarray(share, dtype=float)
            if share.shape != self[channel].shape:
                raise ValueError(f"valid_share of channel {channel!r} must have the shape "
                                 f"of the channel {self[channel].shape}, got {share.shape}")
            self.valid_share[channel] = share

    # ----------------------------------------------------------------------
    # Channels and shape
    # ----------------------------------------------------------------------

    @property
    def channels(self) -> list:
        """Names of the channels (e.g., ``["left_pupil", "left_x"]``)."""
        return list(self.data)

    @property
    def eyes(self) -> list:
        """Eyes of the channels (e.g., ``["left", "right"]``)."""
        return list(dict.fromkeys(_split_channel(ch)[0] for ch in self.data))

    @property
    def variables(self) -> list:
        """Variables of the channels (e.g., ``["pupil", "x", "y"]``)."""
        return list(dict.fromkeys(_split_channel(ch)[1] for ch in self.data))

    @property
    def n_timepoints(self) -> int:
        """Number of time points per segment."""
        return len(self.tx)

    @property
    def n_segments(self) -> int:
        """Number of segments."""
        return next(iter(self.data.values())).shape[1]

    @property
    def window(self) -> tuple:
        """Time window in ms as (first, last time point)."""
        return (self.tx.min(), self.tx.max())

    def __getitem__(self, channel: str) -> ma.MaskedArray:
        """The segments of `channel` (masked array, time points x segments)."""
        try:
            return self.data[channel]
        except KeyError:
            raise KeyError(f"no channel {channel!r} in the segments (available "
                           f"channels: {', '.join(self.data)})") from None

    def __setitem__(self, channel: str, values):
        """Set (or add) `channel`: an array of shape (time points, segments);
        arrays that are not masked arrays are masked where they are NaN."""
        _split_channel(channel)
        if isinstance(values, ma.MaskedArray):
            values = ma.masked_array(np.asarray(ma.getdata(values), dtype=float),
                                     mask=ma.getmaskarray(values).copy())
        else:
            values = np.array(values, dtype=float)
            values = ma.masked_array(values, mask=np.isnan(values))
        if values.ndim != 2 or values.shape[0] != len(self.tx):
            raise ValueError(f"channel {channel!r} must have shape ({len(self.tx)}, "
                             f"segments), got {values.shape}")
        others = [ch for ch in self.data if ch != channel]
        if others and values.shape[1] != self.data[others[0]].shape[1]:
            raise ValueError(f"channel {channel!r} has {values.shape[1]} segments, "
                             f"the others {self.data[others[0]].shape[1]}")
        self.data[channel] = values
        # a channel that is replaced keeps its share of usable trials (none
        # where it is masked now)
        share = self.valid_share.get(channel)
        if share is not None:
            if share.shape == values.shape:
                self.valid_share[channel] = np.where(ma.getmaskarray(values), 0., share)
            else:
                del self.valid_share[channel]

    def __contains__(self, channel) -> bool:
        return channel in self.data

    def _copy(self):
        """Shallow copy with its own dict of channels."""
        new = copy.copy(self)
        new.data = dict(self.data)
        new.valid_share = dict(self.valid_share)
        return new

    def _with_channels(self, channels):
        """New object with `channels` only (copies of their arrays)."""
        new = self._copy()
        new.data = {ch: self.data[ch].copy() for ch in channels}
        new.valid_share = {ch: share.copy() for ch, share in self.valid_share.items()
                           if ch in channels}
        return new

    def select_channels(self, channels=None, eyes=[], variables=[]) -> 'SegmentedEyeData':
        """
        New segments with some of the channels.

        Parameters
        ----------
        channels : str or list of str, optional
            channel names (e.g., ``"left_pupil"``); if given, `eyes` and
            `variables` must be empty
        eyes : str or list of str, optional
            eyes to keep (default: all)
        variables : str or list of str, optional
            variables to keep (e.g., ``"pupil"``; default: all)

        Returns
        -------
        SegmentedEyeData
            new object with the selected channels (and the same epochs)

        Raises
        ------
        KeyError
            if a channel, eye or variable does not exist

        Examples
        --------
        >>> import pypillometry as pp
        >>> from pypillometry import u
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> seg = d.get_segments(d.get_epochs("S_", (-500, 2500) * u.ms))
        >>> seg.select_channels(variables=["x", "y"], eyes="left").channels
        ['left_x', 'left_y']
        """
        if channels is not None:
            if _as_list(eyes) or _as_list(variables):
                raise ValueError("give either channels or eyes/variables, not both")
            channels = _as_list(channels)
            for ch in channels:
                self[ch]  # raises a KeyError if missing
        else:
            channels = _select_channels(self.channels, eyes, variables)
        return self._with_channels(channels)

    def drop_channels(self, *channels: str) -> 'SegmentedEyeData':
        """
        New segments without the given channels.

        Parameters
        ----------
        *channels : str
            channel names (e.g., ``"left_x"``)

        Returns
        -------
        SegmentedEyeData
            new object with the other channels

        Raises
        ------
        KeyError
            if a channel does not exist
        ValueError
            if no channel would be left
        """
        for ch in channels:
            self[ch]  # raises a KeyError if missing
        rest = [ch for ch in self.channels if ch not in channels]
        if not rest:
            raise ValueError("cannot drop all channels")
        return self._with_channels(rest)

    # ----------------------------------------------------------------------
    # Selecting trials
    # ----------------------------------------------------------------------

    def _take(self, mask):
        """New object with the segments where `mask` (bool, one per segment) is True."""
        new = self._copy()
        new.data = {ch: arr[:, mask] for ch, arr in self.data.items()}
        new.valid_share = {ch: share[:, mask] for ch, share in self.valid_share.items()}
        new.epochs = _keep_epochs(self.epochs, mask)
        return new

    def keep(self, mask) -> 'SegmentedEyeData':
        """
        New segments with the trials where `mask` is True (in all channels).

        Parameters
        ----------
        mask : array-like of bool
            one value per segment, e.g., ``seg.epochs.meta["rt"] < 800``

        Returns
        -------
        SegmentedEyeData
            new object with the selected segments (and their epochs)

        Raises
        ------
        ValueError
            if `mask` is not a boolean array with one value per segment

        Examples
        --------
        >>> import pypillometry as pp
        >>> from pypillometry import u
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> seg = d.get_segments(d.get_epochs("S_", (-500, 2500) * u.ms))
        >>> # reject trials with more than half of the left pupil missing
        >>> ok = seg["left_pupil"].mask.mean(axis=0) < 0.5
        >>> bool(seg.keep(ok).n_segments == ok.sum())
        True
        """
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (self.n_segments,):
            raise ValueError(f"mask must be a boolean array with one value per "
                             f"segment ({self.n_segments}), got {mask.dtype} {mask.shape}")
        return self._take(mask)

    def filter(self, selector=None, exact: bool = False, **conditions) -> 'SegmentedEyeData':
        """
        New segments with the trials whose epochs match the criteria.

        The epochs are selected as with :meth:`Epochs.filter
        <pypillometry.Epochs.filter>`: by event label, onset time (in ms),
        a boolean array or metadata. All channels keep the same trials.

        Parameters
        ----------
        selector : str, callable, tuple or boolean array, optional
            criterion for the events (see :meth:`Events.filter
            <pypillometry.Events.filter>`)
        exact : bool, optional
            if True, a string selector matches only labels equal to it
        **conditions
            criteria for metadata columns, e.g. ``cond="easy"``,
            ``trial=[1, 2]`` or ``rt=lambda rt: rt < 800``

        Returns
        -------
        SegmentedEyeData
            new object with the selected segments (and their epochs)

        Raises
        ------
        ValueError
            if the segments have no Epochs (e.g., from old files)

        Examples
        --------
        >>> import pypillometry as pp
        >>> from pypillometry import u
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> ep = d.get_epochs("S_", (-500, 2500) * u.ms)
        >>> ep = ep.set_meta("half", np.where(np.arange(len(ep)) < 5, "first", "second"))
        >>> seg = d.get_segments(ep)
        >>> seg.filter(half="first").n_segments
        5
        """
        from .epochs import Epochs
        if not isinstance(self.epochs, Epochs):
            raise ValueError("the segments have no epochs to select from; use keep()")
        return self._take(self.epochs.events._filter_mask(selector, exact, **conditions))

    # ----------------------------------------------------------------------
    # Averages
    # ----------------------------------------------------------------------

    def _valid_share(self, channel) -> np.ndarray:
        """Share of usable trials of each value of `channel` (time points x columns)."""
        share = self.valid_share.get(channel)
        if share is None:
            share = 1. - ma.getmaskarray(self[channel])
        return share

    def _n_trials(self) -> np.ndarray:
        """Number of trials of each column (metadata ``n_trials``, otherwise 1;
        NaN where it is not known)."""
        from .epochs import Epochs
        if isinstance(self.epochs, Epochs) and "n_trials" in self.epochs.meta:
            return np.asarray(self.epochs.meta["n_trials"], dtype=float)
        return np.ones(self.n_segments)

    def _averaged(self) -> np.ndarray:
        """Whether each column is an average (of more than one or an unknown
        number of trials)."""
        return ~(self._n_trials() == 1)

    def average(self, by=None, meanfct=np.mean, name: str = None) -> 'SegmentedEyeData':
        """
        Average the segments: all trials, or the trials of each group.

        The groups are the combinations of the values of the metadata
        columns `by` (e.g., ``by="cond"`` or ``by=["subject", "cond"]``), in
        the order in which they first occur. The result has one column per
        group, with :class:`~pypillometry.Epochs` without a time (NaN
        onsets) whose metadata are the `by` columns, the other columns that
        have the same value in all trials of each group, and the number of
        trials (``"n_trials"``). Its label is the label of the trials if
        they all have the same, otherwise the values of `by` (joined with
        "/").

        Averaging segments that are averages themselves (e.g., participant
        means, see :func:`~pypillometry.group_segments`) gives each of them
        the same weight. An average is masked where all its trials are
        masked; its share of usable trials (`valid_share`) is kept for the
        missing data in :meth:`plot`.

        Parameters
        ----------
        by : str or list of str, optional
            metadata columns of the epochs that define the groups; ``"label"``
            uses the event labels (if there is no metadata column of this
            name); default: one average of all trials
        meanfct : callable
            function that averages the trials, called as ``meanfct(data,
            axis=1)``; it must accept an `axis` argument and ignore masked
            values (e.g., np.mean or np.ma.median, but not np.median; a
            warning is logged otherwise)
        name : str, optional
            name of the result (default: the name of the segments)

        Returns
        -------
        SegmentedEyeData
            the averages (one column per group), with all channels

        Raises
        ------
        KeyError
            if a column of `by` is not in the metadata of the epochs

        Examples
        --------
        >>> import pypillometry as pp
        >>> from pypillometry import u
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> ep = d.get_epochs("S_", (-500, 2500) * u.ms)
        >>> ep = ep.set_meta("half", np.where(np.arange(len(ep)) < 5, "first", "second"))
        >>> avg = d.get_segments(ep).average(by="half")
        >>> avg["left_pupil"].shape  # (time points, groups)
        (1500, 2)
        >>> {k: v.tolist() for k, v in avg.epochs.meta.items()}
        {'half': ['first', 'second'], 'n_trials': [5, 5]}
        """
        from .epochs import Epochs
        _check_accepts_axis(meanfct, "meanfct")
        _warn_if_ignores_mask(meanfct, "meanfct", axis=1)
        by = _as_list(by)
        ep = self.epochs if isinstance(self.epochs, Epochs) else None
        table = {} if ep is None else {"label": ep.labels, **ep.meta}
        missing = [b for b in by if b not in table]
        if missing:
            raise KeyError(f"no metadata column(s) {', '.join(map(repr, missing))} in the "
                           f"epochs of the segments (available: "
                           f"{', '.join(table) or 'none'})")

        # the groups: unique combinations of the `by` columns
        groups = {}
        for i in range(self.n_segments):
            key = tuple(_group_key(table[b][i]) for b in by)
            groups.setdefault(key, []).append(i)
        idx = [np.array(g) for g in groups.values()]
        firsts = [g[0] for g in idx]

        n_trials = self._n_trials()
        data, shares = {}, {}
        for ch, arr in self.data.items():
            share = self._valid_share(ch)
            cols, share_cols = [], []
            for g in idx:
                cols.append(ma.asarray(meanfct(arr[:, g], axis=1)))
                w = n_trials[g] if np.isfinite(n_trials[g]).all() else np.ones(len(g))
                share_cols.append(share[:, g] @ w / w.sum())
            data[ch] = ma.column_stack(cols)
            shares[ch] = np.column_stack(share_cols)
            data[ch][shares[ch] == 0] = ma.masked   # no usable trial

        meta = {col: values[firsts] for col, values in table.items()
                if col not in ("label", "n_trials")
                and (col in by or all(_all_equal(values[g]) for g in idx))}
        total = np.array([n_trials[g].sum() for g in idx])
        meta["n_trials"] = total.astype(int) if np.isfinite(total).all() else total
        if ep is not None and all(_all_equal(ep.labels[g]) for g in idx):
            labels = ep.labels[firsts]
        else:
            labels = ["/".join(str(table[b][g[0]]) for b in by) or "average" for g in idx]
        epochs = _untimed_epochs(len(idx), self.tx, self.sampling_rate,
                                 window=None if ep is None else ep.window,
                                 labels=labels, meta=meta,
                                 label=None if ep is None else ep.label)
        return type(self)(name=self.name if name is None else name, tx=self.tx.copy(),
                          data=data, epochs=epochs, sampling_rate=self.sampling_rate,
                          valid_share=shares)

    # ----------------------------------------------------------------------
    # Creating
    # ----------------------------------------------------------------------

    @classmethod
    def from_eyedata(cls, eyedata, epochs, channels=None, name: str = None,
                     policy="default", eyes=[], variables=[]) -> 'SegmentedEyeData':
        """
        Cut segments of one or several channels from an EyeData object.

        Parameters
        ----------
        eyedata : EyeData
            source eye data object
        epochs : Epochs or Intervals
            the segments to extract; for Epochs (e.g., from
            :meth:`~pypillometry.EyeData.get_epochs`),
            the length and the time axis (``window[0] + arange(n) * 1000/fs``
            ms) come from the window, and the parts outside the data are
            masked. Intervals must all have the same length; they are used
            as Epochs with the onsets at their starts and the window
            ``(0, length)`` (kept as `epochs` of the result)
        channels : str or list of str, optional
            channels to extract (e.g., ``"left_pupil"``); if None, those of
            `eyes` and `variables`
        name : str, optional
            name of the segments; if None, the label of `epochs` or "segments"
        policy : str, MaskPolicy or list of str
            mask policy of the segments: the samples that are unusable under
            it are masked (see :meth:`~pypillometry.EyeData.mask`);
            default ``"default"`` (interpolated samples are used)
        eyes : str or list of str, optional
            eyes to extract if `channels` is None (default: all)
        variables : str or list of str, optional
            variables to extract if `channels` is None (e.g., ``"pupil"``;
            default: all)

        Returns
        -------
        SegmentedEyeData
            new instance with extracted segments
        """
        from .intervals import Intervals
        from .epochs import Epochs

        if not isinstance(epochs, (Intervals, Epochs)):
            raise TypeError("epochs must be Epochs or Intervals, e.g., from get_epochs()")
        if channels is not None:
            if _as_list(eyes) or _as_list(variables):
                raise ValueError("give either channels or eyes/variables, not both")
            channels = _as_list(channels)
            for ch in channels:
                _split_channel(ch)
                if ch not in eyedata.data:
                    raise KeyError(f"no channel {ch!r} in the data (available channels: "
                                   f"{', '.join(eyedata.data.keys())})")
        else:
            channels = [f"{e}_{v}" for e, v in eyedata.data.select(eyes, variables)]

        if name is None:
            name = epochs.label if epochs.label is not None else "segments"
        if isinstance(epochs, Intervals):
            epochs = _epochs_from_spans(eyedata, epochs)
        data = {}
        for ch in channels:
            tx, data[ch] = _extract_epochs(eyedata, epochs, ch, policy)
        return cls(name=name, tx=tx, data=data, epochs=epochs, sampling_rate=eyedata.fs)

    # ----------------------------------------------------------------------
    # Display
    # ----------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a summary of the segments.

        Returns
        -------
        dict
            name, channels, n_segments, n_timepoints, window,
            percent_masked (dict: channel -> percentage of missing values;
            for averages, of missing trials), sampling_rate, n_averages
            (the number of columns that are averages) and n_trials (the
            number of trials in all columns; None if not known)
        """
        n_trials = float(np.sum(self._n_trials()))
        return _plain_values(dict(
            name=self.name,
            channels=self.channels,
            n_segments=self.n_segments,
            n_timepoints=self.n_timepoints,
            window=self.window,
            percent_masked={ch: float(np.mean(self._percent_missing(ch)))
                            for ch in self.channels},
            sampling_rate=self.sampling_rate,
            n_averages=int(self._averaged().sum()),
            n_trials=int(n_trials) if np.isfinite(n_trials) else None,
        ))

    def _columns_text(self) -> str:
        """What the columns of the data are ("segments", "averages" or both)."""
        n_avg = int(self._averaged().sum())
        if n_avg == 0:
            return "segments"
        return "averages" if n_avg == self.n_segments else "segments and averages"

    def _shape_text(self) -> str:
        n_avg = int(self._averaged().sum())
        s = f"{self.n_segments} {self._columns_text()}"
        if 0 < n_avg < self.n_segments:
            s = f"{self.n_segments - n_avg} segments and {n_avg} averages"
        n_trials = np.sum(self._n_trials())
        if self._averaged().any() and np.isfinite(n_trials):
            s += f" (of {int(n_trials)} trials)"
        return f"{s} x {self.n_timepoints} time points"

    def __repr__(self) -> str:
        """Return a string representation of the segments."""
        s = self.summary()
        w = s["window"]
        fs = f", {s['sampling_rate']:g} Hz" if s["sampling_rate"] else ""
        out = (f"{type(self).__name__}({self.name}): {self._shape_text()} "
               f"({w[0]:g} to {w[1]:g} ms{fs})\n")
        chans = ", ".join(f"{ch} ({p:.1f}% missing)" for ch, p in s["percent_masked"].items())
        out += f"  channels: {chans}\n"
        if self.epochs is not None:
            out += f"  epochs: {self.epochs!r}\n"
        return out

    def _overview(self):
        """(label, value) items of the HTML display."""
        s = self.summary()
        w = s["window"]
        items = [
            (self._columns_text().capitalize(), f"{s['n_segments']:,}"),
            ("Time points", f"{s['n_timepoints']:,}"),
            ("Window", f"{w[0]:.0f} to {w[1]:.0f} ms"),
            ("Missing", f"{np.mean(list(s['percent_masked'].values())):.1f}%"),
        ]
        if s["sampling_rate"]:
            items.append(("Sampling rate", f"{s['sampling_rate']:.0f} Hz"))
        if s["n_averages"] and s["n_trials"] is not None:
            items.insert(1, ("Trials", f"{s['n_trials']:,}"))
        if self.epochs is not None and self.epochs.label:
            items.insert(0, ("Epochs", self.epochs.label))
        return items

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks (template
        templates/segmented_repr.html)."""
        channels = [{"label": f"{ch} ({np.mean(self._percent_missing(ch)):.1f}% missing)",
                     "image": sparkline(np.mean(arr, axis=1), x=self.tx, mark_x=0)}
                    for ch, arr in self.data.items()]
        return render(
            "segmented_repr.html", class_name=type(self).__name__, name=self.name,
            overview=self._overview(), channels=channels, columns=self._columns_text(),
            epochs=None if self.epochs is None else repr(self.epochs))

    # ----------------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------------

    def _percent_missing(self, channel) -> np.ndarray:
        """Percentage of missing data of `channel` at each time point (for
        averages: the mean percentage of their missing trials)."""
        return 100. * (1. - np.mean(self._valid_share(channel), axis=1))

    def _plot_channels(self, channels):
        if channels is None:
            if len(self.variables) > 1:
                raise ValueError(
                    f"the segments have channels of several variables "
                    f"({', '.join(self.variables)}); choose them with channels= "
                    f"(e.g., seg.plot('{self.channels[0]}')) or use seg.pupil.plot()")
            return self.channels
        channels = _as_list(channels)
        for ch in channels:
            self[ch]  # raises a KeyError if missing
        return channels

    def plot(self, channels=None, meanfct=np.mean, varfct=sem,
             show_missing: bool = None, show_zero_line: bool = True,
             show_legend: bool = None, title: str = None, label: str = None,
             ylabel: str = None, ribbon_alpha: float = 0.3, ax=None, **kwargs):
        """
        Plot mean and error ribbon over time, one line per channel.

        Can be used standalone for a quick look, or with `ax` parameter to
        overlay multiple segments on the same plot or across subplots.
        ``seg.pupil.plot()`` plots the pupil channels,
        ``seg.gaze.plot_roi()`` the share of trials in a region of interest.

        Parameters
        ----------
        channels : str or list of str, optional
            channels to plot; default: all (only if they are all of the same
            variable, e.g., pupil of both eyes)
        meanfct : callable
            Function to compute mean across segments (default: np.mean)
        varfct : callable or None
            Function to compute error bands (e.g., np.std, scipy.stats.sem)
            If None, no error bands are plotted.
            Both functions are called as ``fct(data, axis=1)``, so they must
            accept an `axis` argument, and they must ignore masked values
            (e.g., np.ma.median, not np.median; a warning is logged otherwise).
        show_missing : bool or None
            Plot percentage of missing data per time point on secondary y-axis.
            If None, defaults to True in standalone mode, False in overlay mode.
        show_zero_line : bool
            Show vertical line at time 0 (default: True)
        show_legend : bool or None
            Show legend. If None, only shows legend in standalone mode (ax=None)
        title : str, optional
            Plot title. If None and ax is None, uses the dataset name.
            Ignored when ax is provided (for overlay use).
        label : str, optional
            Label for legend. If None, the name of the segments (one channel)
            or the channel names (several channels; with `label`, they are
            added to it)
        ylabel : str, optional
            label of the y-axis in standalone mode (default: "mean" and the
            variable)
        ribbon_alpha : float
            Alpha (transparency) for error ribbon (default: 0.3)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses current axes (standalone mode)
        **kwargs : dict
            Additional arguments passed to ax.plot() (e.g., color, linestyle, linewidth)

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object for further customization

        Raises
        ------
        ValueError
            if `channels` is None and the channels are of several variables

        Examples
        --------
        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> seg = d.get_segments(d.get_epochs("S_", (-500, 2500) * u.ms))
           >>> ax = seg.plot(["left_x", "right_x"], ylabel="mean x (px)")
        """
        import matplotlib.pyplot as plt

        channels = self._plot_channels(channels)
        standalone = ax is None
        ax1 = plt.gca() if standalone else ax

        _check_accepts_axis(meanfct, "meanfct")
        _warn_if_ignores_mask(meanfct, "meanfct", axis=1)
        if callable(varfct):
            _check_accepts_axis(varfct, "varfct")
            _warn_if_ignores_mask(varfct, "varfct", axis=1)

        single = len(channels) == 1
        line_color = kwargs.get('color', kwargs.get('c', None))
        lines = []
        for ch in channels:
            data = self[ch]
            mean_data = meanfct(data, axis=1)
            if single:
                plot_label = label if label is not None else self.name
            else:
                plot_label = ch if label is None else f"{label} {ch}"
            line, = ax1.plot(self.tx, mean_data, label=plot_label, **kwargs)
            lines.append(line)
            if callable(varfct):
                sd_data = varfct(data, axis=1)
                ribbon_color = line_color or line.get_color()
                ax1.fill_between(self.tx, mean_data - sd_data, mean_data + sd_data,
                                 color=ribbon_color, alpha=ribbon_alpha)

        if show_zero_line:
            ax1.axvline(x=0, color="grey", linestyle="--", alpha=0.5, zorder=0)
        ax1.set_xlabel("time (ms)")
        if standalone:
            if ylabel is None:
                variables = {_split_channel(ch)[1] for ch in channels}
                ylabel = f"mean {variables.pop()}" if len(variables) == 1 else "mean"
            ax1.set_ylabel(ylabel)
            ax1.set_title(title if title is not None else self.name)

        # missing data on a secondary axis (default: only in standalone mode)
        if show_missing is None:
            show_missing = standalone
        if show_missing:
            ax2 = ax1.twinx()
            for ch, line in zip(channels, lines):
                ax2.plot(self.tx, self._percent_missing(ch), alpha=0.3,
                         color="orange" if single else line.get_color())
            ax2.set_ylim(0, 100)
            if single:
                ax2.set_ylabel("% missing", color="orange")
            else:
                ax2.set_ylabel("% missing")
            # restore ax1 as current axes so plt.legend() works for the user
            plt.sca(ax1)

        if show_legend is None:
            show_legend = standalone
        if show_legend:
            ax1.legend()
        return ax1

    def plot_carpet(self, channel: str = None, sort_by=None, cmap: str = None,
                    vmin: float = None, vmax: float = None, center: bool = None,
                    show_zero_line: bool = True, colorbar: bool = True,
                    title: str = None, ax=None, **kwargs):
        """
        Plot all segments of one channel as an image ("carpet plot").

        Each row is one segment (e.g., a trial), the x-axis is the time
        relative to the event, and the color is the value. Masked samples
        are left empty (white). Unlike the mean in :meth:`plot`, this shows
        how the response varies between trials, e.g., trials with artifacts
        or a change over the session; sorting the rows by a property of the
        trials (`sort_by`, e.g., the reaction time) shows how the response
        depends on it.

        Parameters
        ----------
        channel : str, optional
            channel to plot (e.g., ``"left_pupil"``); default: the only
            channel of the segments
        sort_by : str or array-like, optional
            order of the rows: a metadata column of the epochs (e.g.,
            ``"rt"``) or one value per segment; the rows are sorted by it in
            ascending order (from the top), ties keep their order; default:
            the order of the segments
        cmap : str or Colormap, optional
            colormap; default ``"RdBu_r"`` if the color scale is centered at
            0, ``"viridis"`` otherwise
        vmin, vmax : float, optional
            limits of the color scale; default: the 2nd and 98th percentile
            of the usable values (so that a few extreme values do not set
            the scale), symmetric around 0 if `center`
        center : bool or None, optional
            center the color scale at 0 (e.g., for baseline-corrected data,
            so that increases and decreases have different colors); default
            (None): if the values have both signs
        show_zero_line : bool, optional
            draw a vertical line at the event onset (time 0); default True
        colorbar : bool, optional
            add a colorbar (labelled with the channel); default True
        title : str, optional
            title; default: the name of the segments if `ax` is None, none
            otherwise
        ax : matplotlib.axes.Axes, optional
            axes to plot into; default: the current axes
        **kwargs
            passed to :meth:`matplotlib.axes.Axes.imshow`

        Returns
        -------
        ax : matplotlib.axes.Axes
            the axes of the plot

        Raises
        ------
        ValueError
            if `channel` is None and the segments have several channels, or
            `sort_by` does not have one value per segment
        KeyError
            if the channel or the metadata column does not exist

        Examples
        --------
        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> d = d | pp.pupil.blinks_detect() | pp.pupil.lowpass_filter(5)
           >>> seg = d.get_segments(d.get_epochs("S_", (-500, 2500) * u.ms), "left_pupil")
           >>> seg = seg.pupil.baseline_correct((-200, 0))
           >>> ax = seg.plot_carpet()
        """
        import matplotlib.pyplot as plt
        from .epochs import Epochs

        if channel is None:
            if len(self.channels) > 1:
                raise ValueError(f"the segments have several channels "
                                 f"({', '.join(self.channels)}); choose one with channel=")
            channel = self.channels[0]
        data = self[channel]                    # time points x segments
        order = np.arange(self.n_segments)
        if sort_by is not None:
            if isinstance(sort_by, str):
                if not isinstance(self.epochs, Epochs) or sort_by not in self.epochs.meta:
                    available = [] if not isinstance(self.epochs, Epochs) else list(self.epochs.meta)
                    raise KeyError(f"no metadata column {sort_by!r} in the epochs of the "
                                   f"segments (available: {', '.join(available) or 'none'})")
                values = np.asarray(self.epochs.meta[sort_by])
            else:
                values = np.asarray(sort_by)
            if values.shape != (self.n_segments,):
                raise ValueError(f"sort_by must have one value per segment "
                                 f"({self.n_segments}), got shape {values.shape}")
            order = np.argsort(values, kind="stable")
        image = data[:, order].T                # segments x time points

        usable = np.ma.compressed(image)
        usable = usable[np.isfinite(usable)]
        lo, hi = np.percentile(usable, [2, 98]) if usable.size else (0., 1.)
        if center is None:
            center = bool(lo < 0 < hi)
        if center:
            lim = max(abs(lo), abs(hi))
            lo, hi = -lim, lim
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
        if cmap is None:
            cmap = "RdBu_r" if center else "viridis"

        standalone = ax is None
        ax = plt.gca() if standalone else ax
        step = float(np.median(np.diff(self.tx))) if self.n_timepoints > 1 else 1.
        extent = [self.tx[0] - step / 2, self.tx[-1] + step / 2, self.n_segments, 0]
        kwargs.setdefault("interpolation", "nearest")
        im = ax.imshow(np.ma.masked_invalid(image), aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, extent=extent, **kwargs)
        if show_zero_line:
            ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("segment" if sort_by is None else
                      f"segment (sorted by {sort_by if isinstance(sort_by, str) else 'sort_by'})")
        if title is not None or standalone:
            ax.set_title(title if title is not None else self.name)
        if colorbar:
            ax.figure.colorbar(im, ax=ax, label=channel)
        return ax

    # ----------------------------------------------------------------------
    # Files
    # ----------------------------------------------------------------------

    def write_file(self, fname: str, compress: bool = True):
        """
        Save to file (see :func:`~pypillometry.io.write_file`).

        The format follows from the extension: HDF5 for ``.h5`` and
        ``.hdf5``, pickle otherwise.

        Parameters
        ----------
        fname : str
            Filename to save to
        compress : bool, optional
            compress the file (default True; pickle: zstd, HDF5: gzip)
        """
        write_file(self, fname, compress=compress)

    @classmethod
    def from_file(cls, fname: str, **kwargs) -> 'SegmentedEyeData':
        """
        Read a SegmentedEyeData object from a pickle or HDF5 file.

        Parameters
        ----------
        fname : str
            Filename (or HTTP(S)/SSH URL) to read from
        **kwargs
            passed to :func:`~pypillometry.io.read_file` (e.g.,
            `cache_file`, `cache_dir`, `cache_layout`, `autodelete`,
            `session`)

        Returns
        -------
        SegmentedEyeData
            Loaded instance
        """
        return read_file(fname, **kwargs)


# --------------------------------------------------------------------------
# Segments of several participants
# --------------------------------------------------------------------------

def _named_segments(segments):
    """(names, list of segments) of a list or dict of SegmentedEyeData: the
    dict keys, otherwise the names of the segments, or numbers if these are
    not unique."""
    if isinstance(segments, Mapping):
        return [str(k) for k in segments], list(segments.values())
    segments = list(segments)
    names = [str(s.name) for s in segments]
    if len(set(names)) != len(names):
        names = [str(i + 1) for i in range(len(segments))]
    return names, segments


def concat_segments(segments, name: str = None, channels=None,
                    key: str = "subject") -> SegmentedEyeData:
    """
    Stack the segments of several participants (or sessions) into one object.

    The columns (trials or averages) of all segments are put one after the
    other, with their epochs; the metadata column `key` (default
    ``"subject"``) says which segments each column comes from. The result
    can be selected and averaged like the segments of one participant, e.g.,
    ``concat_segments(segs).average(by=["subject", "cond"])``.

    Parameters
    ----------
    segments : list or dict of SegmentedEyeData
        the segments, e.g., one object per participant; for a dict, the
        keys are the values of `key` (otherwise the names of the segments,
        or numbers if these are not unique). All must have the same time
        axis.
    name : str, optional
        name of the result (default: the name of the first segments)
    channels : str or list of str, optional
        channels to combine (default: those of the first segments); all
        segments must have them
    key : str or None, optional
        metadata column for the names of the segments (default
        ``"subject"``); None adds no column

    Returns
    -------
    SegmentedEyeData
        the stacked segments

    Raises
    ------
    ValueError
        if `segments` is empty, the time axes differ, or the epochs already
        have a metadata column `key`
    KeyError
        if segments lack a channel

    Examples
    --------
    One SegmentedEyeData per participant (here: the example recording twice):

    >>> import pypillometry as pp
    >>> from pypillometry import u
    >>> d = pp.get_example_data("rlmw_002_short")
    >>> epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
    >>> seg = d.get_segments(epochs, variables="pupil")
    >>> both = pp.concat_segments({"S01": seg, "S02": seg})
    >>> both.n_segments
    20
    >>> both.filter(subject="S01").n_segments
    10
    """
    from .epochs import Epochs
    from .events import _concat_meta
    names, segments = _named_segments(segments)
    if not segments:
        raise ValueError("no segments to combine")
    first = segments[0]
    channels = first.channels if channels is None else _as_list(channels)
    for i, seg in enumerate(segments, start=1):
        missing = [ch for ch in channels if ch not in seg]
        if missing:
            raise KeyError(f"segments {i} ({seg.name}) have no channel(s) "
                           f"{', '.join(missing)} (available channels: "
                           f"{', '.join(seg.channels)})")
        if seg.n_timepoints != first.n_timepoints or not np.allclose(seg.tx, first.tx):
            raise ValueError(f"the time axes of segments 1 ({first.n_timepoints} time "
                             f"points) and {i} ({seg.n_timepoints}) differ")

    # epochs: those of the segments (without a time if they have none)
    eps = [seg.epochs if isinstance(seg.epochs, Epochs) else
           _untimed_epochs(seg.n_segments, seg.tx, seg.sampling_rate)
           for seg in segments]
    if key is not None and any(key in ep.meta for ep in eps):
        raise ValueError(f"the epochs already have a metadata column {key!r}; pass "
                         f"another name as key= (or key=None)")
    metas = [dict(ep.meta) for ep in eps]
    if any("n_trials" in m for m in metas):
        for m, ep in zip(metas, eps):
            m.setdefault("n_trials", np.ones(len(ep), dtype=int))
    lengths = [len(ep) for ep in eps]
    meta = _concat_meta(metas, lengths)
    if key is not None:
        meta[key] = np.repeat(np.array(names, dtype=str), lengths)
    epochs = Epochs(np.concatenate([ep.onsets for ep in eps]), eps[0].window,
                    labels=np.concatenate([ep.labels for ep in eps]), meta=meta,
                    label=eps[0].label)

    data = {ch: ma.concatenate([seg[ch] for seg in segments], axis=1) for ch in channels}
    shares = {ch: np.concatenate([seg._valid_share(ch) for seg in segments], axis=1)
              for ch in channels if any(ch in seg.valid_share for seg in segments)}
    return SegmentedEyeData(name=first.name if name is None else name, tx=first.tx.copy(),
                            data=data, epochs=epochs, sampling_rate=first.sampling_rate,
                            valid_share=shares)


def group_segments(segments, meanfct=np.mean, name: str = None,
                   channels=None) -> SegmentedEyeData:
    """
    Average the segments of several participants per participant.

    Shortcut for ``concat_segments(segments).average(by="subject")``: the
    result has one column per participant (the average of their trials),
    with the metadata column ``"subject"``. Mean and error over the columns
    (e.g., ``group.pupil.plot()``) are those across participants.

    Parameters
    ----------
    segments : list or dict of SegmentedEyeData
        one object per participant; for a dict, the keys are the names of
        the participants (otherwise the names of the segments, or numbers
        if these are not unique)
    meanfct : callable or None
        function that averages the trials of each participant, called as
        ``meanfct(data, axis=1)``; it must accept an `axis` argument and
        ignore masked values (e.g., np.mean or np.ma.median, but not
        np.median; a warning is logged otherwise). If None, the trials are
        not averaged (as :func:`concat_segments`).
    name : str, optional
        name of the result (default: ``"group_"`` and the name of the first
        segments)
    channels : str or list of str, optional
        channels to combine (default: those of the first segments)

    Returns
    -------
    SegmentedEyeData
        the averages of the participants (or all their trials)

    Examples
    --------
    One SegmentedEyeData per participant (here: the example recording twice):

    >>> import pypillometry as pp
    >>> from pypillometry import u
    >>> d = pp.get_example_data("rlmw_002_short")
    >>> epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
    >>> segments = {"S01": d.get_segments(epochs), "S02": d.get_segments(epochs)}
    >>> group = pp.group_segments(segments, channels=["left_pupil", "right_pupil"])
    >>> group["left_pupil"].shape  # (time points, participants)
    (1500, 2)
    >>> group.epochs.meta["subject"].tolist()
    ['S01', 'S02']

    Means per participant and condition, then the mean of the participants
    per condition (here, the condition is the half of the session):

    >>> ep = epochs.set_meta("half", np.where(np.arange(len(epochs)) < 5, "first", "second"))
    >>> both = pp.concat_segments({"S01": d.get_segments(ep), "S02": d.get_segments(ep)})
    >>> means = both.average(by=["subject", "half"]).average(by="half")
    >>> means.epochs.meta["n_trials"].tolist()
    [10, 10]
    """
    names, segs = _named_segments(segments)
    if name is None and segs:
        name = f"group_{segs[0].name}"
    stacked = concat_segments(dict(zip(names, segs)), name=name, channels=channels)
    if meanfct is None:
        return stacked
    return stacked.average(by="subject", meanfct=meanfct)
