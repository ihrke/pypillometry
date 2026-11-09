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
"""
import copy
from collections.abc import Mapping

import numpy as np
import numpy.ma as ma
from loguru import logger
import scipy.stats

from .io import write_pickle, read_pickle
from . import config, state
from .units import parse_time, _ms_to_samples
from .convenience import _warn_if_ignores_mask, _check_accepts_axis, _plain_values
from .eyedata.namespace import Namespace, accessor

__all__ = [
    "SegmentedEyeData", "GroupLevelSegmentedData", "group_segments",
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


def _percent_masked(arr, axis=1):
    """Percentage of masked values of `arr` along `axis`."""
    return np.mean(ma.getmaskarray(arr), axis=axis) * 100.


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


def _sparklines(tx, series):
    """
    HTML with a small PNG of the curves in `series` (label -> values), one
    row each (empty string if matplotlib cannot draw it).
    """
    import base64
    from io import BytesIO
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(3, 0.45 * len(series) + 0.2), dpi=100)
        axs = np.atleast_1d(fig.subplots(len(series), 1, sharex=True))
        for ax, (label, values) in zip(axs, series.items()):
            ax.plot(tx, values, color="#1f77b4", linewidth=1.2)
            ax.axvline(x=0, color="#d62728", linewidth=1, linestyle="--", alpha=0.7)
            ax.set_xlim(tx[0], tx[-1])
            ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=7,
                    va="top", color="#555")
            ax.axis("off")
        fig.tight_layout(pad=0.1)
        buf = BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        img = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""
    return f'''
        <div style="margin-top:10px;">
          <div style="font-size:0.85em; color:#666; margin-bottom:4px;">Mean per channel (red line = event onset)</div>
          <img src="data:image/png;base64,{img}" alt="Mean signal" style="width:100%; max-width:300px; border-radius:4px;">
        </div>'''


def _html_card(title, name, overview, extra=""):
    """HTML card with a title, a grid of (label, value) items and `extra` HTML."""
    items = "".join(
        f'''<div style="background:#f6f6f6; border-radius:4px; padding:6px 8px; border:1px solid #e0e0e0;">
              <div style="font-size:0.85em; color:#666; text-transform:uppercase;">{label}</div>
              <div style="font-weight:600;">{value}</div>
            </div>'''
        for label, value in overview)
    return f'''
    <div style="font-family:system-ui, sans-serif; border:1px solid #ddd; border-radius:6px; padding:12px; max-width:400px;">
      <h3 style="margin-top:0;">{title} — <span style="color:#555;">{name}</span></h3>
      <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:8px;">
        {items}
      </div>
      {extra}
    </div>'''


# --------------------------------------------------------------------------
# Namespaces
# --------------------------------------------------------------------------

class SegmentedPupilNamespace(Namespace):
    """
    Methods for pupil segments, accessed as ``seg.pupil`` (e.g.,
    ``seg.pupil.baseline_correct((-200, 0))``).

    The methods work on the pupil channels (``<eye>_pupil``) of a
    :class:`~pypillometry.SegmentedEyeData` object (all eyes, or those given
    in `eyes`) and raise a KeyError if there is none. Averaging across
    subjects (:func:`~pypillometry.group_segments`) works for all channels.
    """

    _name = "pupil"

    def _channels(self, eyes):
        return _select_channels(self._d.channels, eyes, "pupil")

    def baseline_correct(self, window=0, eyes=[], inplace=None):
        """
        Subtract a baseline from each segment of the pupil channels.

        The window is given in ms relative to the event onsets (time 0), e.g.,
        ``window=(-200, 0)`` uses the mean of the 200 ms before each event.
        Other channels (e.g., gaze) are not changed.

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
            if the window (or time point) is not within the segments, or its
            start is not before its end
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
        """
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
            arr -= base
            corrected[ch] = arr
        logger.info(f"Baseline correction applied using window {window}")

        if inplace is None:
            inplace = config.get_default_inplace()
        out = seg if inplace else seg._copy()
        out.data.update(corrected)
        return out

    def plot(self, eyes=[], meanfct=np.mean, varfct=scipy.stats.sem,
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
    curve, :meth:`plot_roi`); mean ± error plots and averaging across
    subjects (:func:`~pypillometry.group_segments`) work for it like for any
    channel. Measures per trial (:meth:`dwell_time`, :meth:`first_entry`)
    are returned as tables with the epochs' onsets, labels and metadata.
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

    def _hits(self, roi, eye):
        """1 where the gaze of `eye` is in `roi`, 0 outside (masked where x or y are)."""
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
        usable samples in the window get NaN.

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
        step = 1000. / seg.sampling_rate if seg.sampling_rate else \
            float(np.median(np.diff(seg.tx)))
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
        the usable samples of the window) get NaN.

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

    def plot_roi(self, roi, eyes=[], meanfct=np.mean, varfct=scipy.stats.sem,
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
    Segments of several subjects are combined with
    :func:`~pypillometry.group_segments`.

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

    Raises
    ------
    ValueError
        if there are no channels, a channel name does not have the form
        ``"<eye>_<variable>"``, or the shapes of the channels, the time axis
        and the epochs do not match

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
    """

    pupil = accessor(SegmentedPupilNamespace)
    gaze = accessor(SegmentedGazeNamespace)

    # state (see pypillometry.state); `epochs` is Epochs or Intervals
    _STATE_FIELDS = ("name", "tx", ("data", "dict:masked"), ("epochs", "tagged"),
                     "sampling_rate")

    #: label of the missing-data axis of :meth:`plot`
    _missing_label = "% missing"

    def __init__(self, name: str, tx: np.ndarray, data: dict, epochs=None,
                 sampling_rate: float = None):
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
        for channel, values in data.items():
            self[channel] = values
        if epochs is not None and len(epochs) != self.n_segments:
            raise ValueError(f"{len(epochs)} epochs for {self.n_segments} segments")
        self.epochs = epochs

    def _set_state(self, st: dict):
        """Set the attributes from a state of the current version (segments
        with one channel from development versions of 2.2 are converted)."""
        if "variable" in st:
            st = state._v3_channels(st, type(self).__name__)
        super()._set_state(st)

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

    def __contains__(self, channel) -> bool:
        return channel in self.data

    def _copy(self):
        """Shallow copy with its own dict of channels."""
        new = copy.copy(self)
        new.data = dict(self.data)
        return new

    def _with_channels(self, channels):
        """New object with `channels` only (copies of their arrays)."""
        new = self._copy()
        new.data = {ch: self.data[ch].copy() for ch in channels}
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
            if the segments have no Epochs (e.g., group averages)

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
            percent_masked (dict: channel -> percentage of masked values)
            and sampling_rate
        """
        return _plain_values(dict(
            name=self.name,
            channels=self.channels,
            n_segments=self.n_segments,
            n_timepoints=self.n_timepoints,
            window=self.window,
            percent_masked={ch: float(np.mean(self._percent_missing(ch)))
                            for ch in self.channels},
            sampling_rate=self.sampling_rate,
        ))

    def _shape_text(self) -> str:
        return f"{self.n_segments} segments x {self.n_timepoints} time points"

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
            ("Channels", ", ".join(s["channels"])),
            ("Segments", f"{s['n_segments']:,}"),
            ("Time points", f"{s['n_timepoints']:,}"),
            ("Window", f"{w[0]:.0f} to {w[1]:.0f} ms"),
            ("Missing", f"{np.mean(list(s['percent_masked'].values())):.1f}%"),
        ]
        if s["sampling_rate"]:
            items.append(("Sampling rate", f"{s['sampling_rate']:.0f} Hz"))
        if self.epochs is not None and self.epochs.label:
            items.insert(0, ("Epochs", self.epochs.label))
        return items

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        series = {ch: np.ma.filled(np.mean(arr, axis=1).astype(float), np.nan)
                  for ch, arr in list(self.data.items())[:8]}
        return _html_card(type(self).__name__, self.name, self._overview(),
                          _sparklines(self.tx, series))

    # ----------------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------------

    def _percent_missing(self, channel) -> np.ndarray:
        """Percentage of missing data of `channel` at each time point."""
        return _percent_masked(self[channel])

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

    def plot(self, channels=None, meanfct=np.mean, varfct=scipy.stats.sem,
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
                ribbon_color = line_color or ("grey" if single else line.get_color())
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
                ax2.set_ylabel(self._missing_label, color="orange")
            else:
                ax2.set_ylabel(self._missing_label)
            # restore ax1 as current axes so plt.legend() works for the user
            plt.sca(ax1)

        if show_legend is None:
            show_legend = standalone
        if show_legend:
            ax1.legend()
        return ax1

    # ----------------------------------------------------------------------
    # Files
    # ----------------------------------------------------------------------

    def write_file(self, fname: str, compress: bool = True):
        """
        Save to file (see :func:`~pypillometry.io.write_pickle`).

        The format follows from the extension: HDF5 for ``.h5`` and
        ``.hdf5``, pickle otherwise.

        Parameters
        ----------
        fname : str
            Filename to save to
        compress : bool, optional
            compress the file (default True; pickle: zstd, HDF5: gzip)
        """
        write_pickle(self, fname, compress=compress)

    @classmethod
    def from_file(cls, fname: str, **kwargs) -> 'SegmentedEyeData':
        """
        Read a SegmentedEyeData object from a pickle or HDF5 file.

        Parameters
        ----------
        fname : str
            Filename (or HTTP(S)/SSH URL) to read from
        **kwargs
            passed to :func:`~pypillometry.io.read_pickle` (e.g.,
            `cache_file`, `cache_dir`, `cache_layout`, `autodelete`,
            `session`)

        Returns
        -------
        SegmentedEyeData
            Loaded instance
        """
        return read_pickle(fname, **kwargs)


# --------------------------------------------------------------------------
# Segments of several subjects
# --------------------------------------------------------------------------

def _concat_epochs(segments, subjects):
    """
    Epochs of all `segments` one after the other, with the metadata column
    ``"subject"``, or None if they do not all have Epochs with one window.
    """
    from .epochs import Epochs
    from .events import _concat_meta
    eps = [s.epochs for s in segments]
    if not all(isinstance(e, Epochs) for e in eps) or len({e.window for e in eps}) != 1:
        return None
    lengths = [len(e) for e in eps]
    meta = _concat_meta([e.meta for e in eps], lengths)
    meta["subject"] = np.repeat(np.array(subjects, dtype=str), lengths)
    return Epochs(np.concatenate([e.onsets for e in eps]), eps[0].window,
                  labels=np.concatenate([e.labels for e in eps]), meta=meta,
                  label=eps[0].label)


class GroupLevelSegmentedData(SegmentedEyeData):
    """
    Segments of several subjects combined (e.g., group-level ERPD).

    Combines one :class:`~pypillometry.SegmentedEyeData` object per subject,
    either by averaging within each subject (one column per subject) or by
    stacking all their segments (one column per trial). It has the same
    channels, methods and namespaces as SegmentedEyeData (e.g.,
    ``group.pupil.plot()``, ``group.gaze.plot_roi("target")``); mean and
    error are taken over the columns.

    Usually created with :func:`~pypillometry.group_segments` (or
    :meth:`from_segments`) from a list or dict of
    :class:`~pypillometry.SegmentedEyeData` objects, one per subject.

    The data keep their mask: stacked segments keep their own mask, and in
    aggregate mode a subject's mean is masked where all of their segments are
    masked. In addition, `subject_mask_percent` stores the percentage of
    masked trials of each subject at each time point, and `mask_percent`
    its average across subjects.

    Stacked segments keep the epochs of all subjects, with the subject in
    the metadata column ``"subject"`` (so trials can be selected with
    :meth:`~SegmentedEyeData.filter`, e.g., ``group.filter(cond="easy")``).
    Averages have no epochs; :meth:`~SegmentedEyeData.keep` selects
    subjects.

    Parameters
    ----------
    name : str
        name of the dataset
    tx : np.ndarray
        time axis relative to the event onsets (in ms)
    data : dict
        channel name -> array of shape (time points, subjects) or (time
        points, trials of all subjects)
    subjects : list of str
        names of the subjects
    subject_mask_percent : dict
        channel name -> percentage of masked trials of each subject at each
        time point (shape: time points x subjects)
    stacked : bool, optional
        True if the columns are trials, False (default) if they are subject
        averages
    subject : np.ndarray, optional
        index (into `subjects`) of the subject of each column; default: one
        column per subject (None for stacked segments from old files, whose
        subjects are unknown)
    epochs : Epochs, optional
        the epochs of the columns (stacked segments only)
    sampling_rate : float, optional
        sampling rate in Hz
    """

    # state (see pypillometry.state): adds to that of SegmentedEyeData
    _STATE_FIELDS = ("subjects", "subject", "subject_mask_percent", "stacked")

    _missing_label = "avg % missing"

    def __init__(self, name: str, tx: np.ndarray, data: dict, subjects, subject_mask_percent: dict,
                 stacked: bool = False, subject=None, epochs=None, sampling_rate: float = None):
        super().__init__(name=name, tx=tx, data=data, epochs=epochs,
                         sampling_rate=sampling_rate)
        self.subjects = [str(s) for s in subjects]
        self.stacked = bool(stacked)
        if subject is None and not self.stacked:
            subject = np.arange(len(self.subjects))
        self.subject = None if subject is None else np.asarray(subject, dtype=int)
        if self.subject is not None and self.subject.shape != (self.n_segments,):
            raise ValueError(f"subject must have one value per column ({self.n_segments})")
        self.subject_mask_percent = {ch: np.asarray(p, dtype=float)
                                     for ch, p in subject_mask_percent.items()}

    @property
    def n_subjects(self) -> int:
        """Number of subjects."""
        return len(self.subjects)

    @property
    def mask_percent(self) -> dict:
        """Average percentage of masked trials per subject at each time point
        (channel -> array with one value per time point)."""
        return {ch: self._percent_missing(ch) for ch in self.channels}

    def _percent_missing(self, channel) -> np.ndarray:
        if channel in self.subject_mask_percent:
            return np.mean(self.subject_mask_percent[channel], axis=1)
        return _percent_masked(self[channel])

    def _subject_percent(self, arr):
        """Percentage of masked trials of each subject (stacked columns)."""
        return np.column_stack([_percent_masked(arr[:, self.subject == j])
                                for j in range(self.n_subjects)])

    def __setitem__(self, channel, values):
        new = channel not in self.data
        super().__setitem__(channel, values)
        # a new channel of stacked trials (e.g., ROI hits): masked trials per subject
        if new and getattr(self, "stacked", False) and self.subject is not None:
            self.subject_mask_percent[channel] = self._subject_percent(self.data[channel])

    def _copy(self):
        new = super()._copy()
        new.subject_mask_percent = dict(self.subject_mask_percent)
        return new

    def _with_channels(self, channels):
        new = super()._with_channels(channels)
        new.subject_mask_percent = {ch: p for ch, p in self.subject_mask_percent.items()
                                    if ch in channels}
        return new

    def _take(self, mask):
        if not self.stacked:
            # the columns are subjects
            new = super()._take(mask)
            idx = np.flatnonzero(mask)
            new.subjects = [self.subjects[i] for i in idx]
            new.subject = np.arange(len(idx))
            new.subject_mask_percent = {ch: p[:, idx] for ch, p in self.subject_mask_percent.items()}
            return new
        if self.subject is None:
            raise ValueError("the subjects of the trials are not known (segments saved "
                             "with an older version of pypillometry), so trials cannot "
                             "be selected")
        new = super()._take(mask)
        sub = self.subject[mask]
        present = [j for j in range(self.n_subjects) if j in set(sub.tolist())]
        remap = {old: i for i, old in enumerate(present)}
        new.subjects = [self.subjects[j] for j in present]
        new.subject = np.array([remap[s] for s in sub.tolist()], dtype=int)
        new.subject_mask_percent = {ch: new._subject_percent(new.data[ch])
                                    for ch in self.subject_mask_percent}
        return new

    @classmethod
    def from_segments(cls, segments, meanfct=np.mean, name: str = None,
                      channels=None) -> 'GroupLevelSegmentedData':
        """
        Combine SegmentedEyeData objects (one per subject).

        Parameters
        ----------
        segments : list or dict of SegmentedEyeData
            one object per subject; for a dict, the keys are the names of
            the subjects (otherwise the names of the segments, or numbers
            if these are not unique)
        meanfct : callable or None
            Function to aggregate within each subject, called as
            ``meanfct(data, axis=1)``; it must accept an `axis` argument and
            ignore masked values (e.g., np.mean or np.ma.median, but not
            np.median; a warning is logged otherwise).
            If None, all segments from all subjects are stacked (concatenated).
        name : str, optional
            Name for the group-level dataset. If None, uses ``"group_"`` + first
            segment's name
        channels : str or list of str, optional
            channels to combine (default: those of the first segments); all
            segments must have them

        Returns
        -------
        GroupLevelSegmentedData
            New instance with combined data

        Raises
        ------
        ValueError
            if `segments` is empty or the time axes differ
        KeyError
            if segments lack a channel

        Examples
        --------
        One SegmentedEyeData per subject (here: the example recording twice):

        >>> import pypillometry as pp
        >>> from pypillometry import u
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
        >>> subj1 = d.get_segments(epochs, variables="pupil")
        >>> subj2 = d.get_segments(epochs, variables="pupil")

        >>> # Aggregate within subjects, then combine
        >>> group = pp.GroupLevelSegmentedData.from_segments({"S01": subj1, "S02": subj2})
        >>> group["left_pupil"].shape  # (time points, subjects)
        (1500, 2)

        >>> # Stack all trials from all subjects
        >>> group = pp.GroupLevelSegmentedData.from_segments([subj1, subj2], meanfct=None)
        >>> group["left_pupil"].shape  # (time points, trials)
        (1500, 20)
        """
        if isinstance(segments, Mapping):
            subjects = [str(k) for k in segments]
            segments = list(segments.values())
        else:
            segments = list(segments)
            names = [str(s.name) for s in segments]
            subjects = names if len(set(names)) == len(names) else \
                [str(i + 1) for i in range(len(segments))]
        if not segments:
            raise ValueError("segments list cannot be empty")

        first = segments[0]
        channels = first.channels if channels is None else _as_list(channels)
        for i, seg in enumerate(segments, start=1):
            missing = [ch for ch in channels if ch not in seg]
            if missing:
                raise KeyError(f"segments {i} ({seg.name}) have no channel(s) "
                               f"{', '.join(missing)} (available channels: "
                               f"{', '.join(seg.channels)})")
            if seg.n_timepoints != first.n_timepoints:
                raise ValueError(f"Timepoint mismatch: segment 1 has {first.n_timepoints}, "
                                 f"segment {i} has {seg.n_timepoints}")
            if not np.allclose(seg.tx, first.tx):
                raise ValueError(f"Time axis mismatch between segment 1 and segment {i}")

        # percentage of masked trials of each subject at each time point
        smp = {ch: np.column_stack([_percent_masked(seg[ch]) for seg in segments])
               for ch in channels}
        if meanfct is None:
            # stack mode: all segments of all subjects (keep their masks)
            data = {ch: ma.concatenate([seg[ch] for seg in segments], axis=1)
                    for ch in channels}
            subject = np.concatenate([np.full(seg.n_segments, i, dtype=int)
                                      for i, seg in enumerate(segments)])
            epochs = _concat_epochs(segments, subjects)
        else:
            # aggregate mode: meanfct within each subject; a time point where
            # all of a subject's segments are masked stays masked
            _check_accepts_axis(meanfct, "meanfct")
            _warn_if_ignores_mask(meanfct, "meanfct", axis=1)
            data = {ch: ma.column_stack([ma.asarray(meanfct(seg[ch], axis=1))
                                         for seg in segments])
                    for ch in channels}
            subject, epochs = None, None

        return cls(name=f"group_{first.name}" if name is None else name,
                   tx=first.tx.copy(), data=data, subjects=subjects,
                   subject_mask_percent=smp, stacked=meanfct is None, subject=subject,
                   epochs=epochs, sampling_rate=first.sampling_rate)

    def summary(self) -> dict:
        """
        Return a summary of the GroupLevelSegmentedData object.

        Returns
        -------
        dict
            the summary of :meth:`SegmentedEyeData.summary` (`percent_masked`:
            the average percentage of masked trials per subject), with
            n_subjects and stacked
        """
        s = super().summary()
        s["n_subjects"] = self.n_subjects
        s["stacked"] = self.stacked
        return s

    def _shape_text(self) -> str:
        cols = "trials" if self.stacked else "subject averages"
        return (f"{self.n_subjects} subjects, {self.n_segments} {cols} x "
                f"{self.n_timepoints} time points")

    def _overview(self):
        items = super()._overview()
        i = [k for k, _ in items].index("Segments")
        items[i] = ("Trials" if self.stacked else "Columns", f"{self.n_segments:,}")
        items.insert(i, ("Subjects", f"{self.n_subjects}"))
        i = [k for k, _ in items].index("Missing")
        items[i] = ("Avg missing", items[i][1])
        return items


def group_segments(segments, meanfct=np.mean, name: str = None,
                   channels=None) -> GroupLevelSegmentedData:
    """
    Combine multiple SegmentedEyeData objects into a GroupLevelSegmentedData.

    Convenience function that wraps GroupLevelSegmentedData.from_segments().

    Parameters
    ----------
    segments : list or dict of SegmentedEyeData
        one object per subject; for a dict, the keys are the names of the
        subjects
    meanfct : callable or None
        Function to aggregate within each subject, called as
        ``meanfct(data, axis=1)``; it must accept an `axis` argument and
        ignore masked values (e.g., np.mean or np.ma.median, but not
        np.median; a warning is logged otherwise).
        If None, all segments from all subjects are stacked (concatenated).
    name : str, optional
        Name for the group-level dataset. If None, uses ``"group_"`` + first
        segment's name
    channels : str or list of str, optional
        channels to combine (default: those of the first segments)

    Returns
    -------
    GroupLevelSegmentedData
        Combined group-level data

    Examples
    --------
    One SegmentedEyeData per subject (here: the example recording twice):

    >>> import pypillometry as pp
    >>> from pypillometry import u
    >>> d = pp.get_example_data("rlmw_002_short")
    >>> epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
    >>> subj1_seg = d.get_segments(epochs, name="S01")
    >>> subj2_seg = d.get_segments(epochs, name="S02")

    >>> # From a list
    >>> group = pp.group_segments([subj1_seg, subj2_seg])
    >>> group.subjects
    ['S01', 'S02']

    >>> # From a dict, pupil channels only
    >>> segments = {"S01": subj1_seg, "S02": subj2_seg}
    >>> group = pp.group_segments(segments, channels=["left_pupil", "right_pupil"])

    >>> # Stack all trials instead of averaging
    >>> group = pp.group_segments(segments, meanfct=None)
    >>> group.filter(subject="S01").n_segments
    10
    """
    return GroupLevelSegmentedData.from_segments(segments, meanfct=meanfct, name=name,
                                                 channels=channels)
