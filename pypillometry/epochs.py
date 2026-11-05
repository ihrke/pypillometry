"""Event-locked windows (epochs): event onsets and one window around them.
"""

import types

import numpy as np

from .convenience import requires_package
from .units import parse_time
from .events import _meta_dict
from .state import Stateful

__all__ = ["Epochs"]


class Epochs(Stateful):
    """
    Event-locked windows: the onsets of events and one time window around them.

    An Epochs object describes the same window around each of several
    events, e.g., -200 to 800 ms around every stimulus. It is returned by
    :meth:`~pypillometry.EyeData.get_intervals` for single events
    and is the input of :meth:`~pypillometry.EyeData.get_segments`,
    which takes the length and the time axis of the segments from the
    window. Spans of time that are not locked to one event (e.g., between
    two events) are :class:`~pypillometry.Intervals`.

    Epochs iterate like spans (``(onset + window[0], onset + window[1])`` in
    ms), so they can be used wherever the eyedata methods take Intervals
    (e.g., :meth:`~pypillometry.EyeData.mask_intervals` or
    :meth:`~pypillometry.EyeData.stat_per_event`);
    :meth:`to_intervals` converts them explicitly. Windows that extend
    beyond the data are not clipped; the methods that read the data handle
    the parts outside of it.

    Parameters
    ----------
    onsets : array-like or Quantity
        onsets of the events in ms (or with units, e.g., ``[1, 2] * u.sec``)
    window : tuple (start, end) or Quantity
        the window relative to each onset in ms (or with units, e.g.,
        ``(-200, 800) * u.ms``); `start` < `end`
    labels : array-like of str, optional
        label of each event (default: empty strings)
    meta : dict, optional
        metadata of the events (column name -> array with one value per
        event, see :class:`~pypillometry.Events`)
    label : str, optional
        name of the epochs (e.g., the event selector)

    Attributes
    ----------
    onsets : np.ndarray
        onsets in ms (read-only)
    window : tuple of float
        window in ms
    labels : np.ndarray
        event labels (read-only)
    meta : mapping
        metadata columns (read-only)
    label : str or None
        name of the epochs

    Examples
    --------
    >>> ep = Epochs([1000, 3000], (-200, 800), labels=["stim", "stim"])
    >>> list(ep)
    [(800.0, 1800.0), (2800.0, 3800.0)]
    >>> ep.to_intervals("sec").intervals[0]
    (0.8, 1.8)
    """

    # state (see pypillometry.state)
    _STATE_FIELDS = ("onsets", "window", "labels", ("meta", None, "_meta"), "label")

    def __init__(self, onsets, window, labels=None, meta=None, label=None):
        onsets = np.atleast_1d(np.asarray(parse_time(onsets, to="ms"), dtype=float))
        if onsets.ndim != 1:
            raise ValueError(f"onsets must be one-dimensional, got shape {onsets.shape}")
        if not np.isfinite(onsets).all():
            raise ValueError("onsets must be finite")
        window = np.asarray(parse_time(window, to="ms"), dtype=float)
        if window.shape != (2,) or not np.isfinite(window).all() or window[0] >= window[1]:
            raise ValueError(f"window must be (start, end) with finite start < end, "
                             f"got {window.tolist()}")
        n = len(onsets)
        labels = np.full(n, "") if labels is None else np.asarray(labels, dtype=str)
        if labels.shape != (n,):
            raise ValueError(f"{labels.size} labels for {n} onsets")
        onsets.flags.writeable = False
        labels.flags.writeable = False
        self.onsets = onsets
        self.window = (float(window[0]), float(window[1]))
        self.labels = labels
        self._meta = _meta_dict(meta, n)
        for col in self._meta.values():
            col.flags.writeable = False
        self.label = label

    def _set_state(self, st: dict):
        """Set the attributes from a state of the current version (checked like
        the arguments of the constructor)."""
        self.__init__(st["onsets"], st["window"], labels=st["labels"], meta=st["meta"],
                      label=st["label"])

    @classmethod
    def from_events(cls, events, window, label=None) -> 'Epochs':
        """
        Epochs around all events of an :class:`~pypillometry.Events` object.

        Parameters
        ----------
        events : Events
            the events (in any time units)
        window : tuple (start, end) or Quantity
            window relative to each onset, in ms or with units
        label : str, optional
            name of the epochs

        Returns
        -------
        Epochs
        """
        return cls(events.to_units("ms").onsets, window, labels=events.labels,
                   meta=dict(events.meta), label=label)

    @property
    def meta(self):
        """Metadata of the events (read-only mapping: column name -> array)."""
        return types.MappingProxyType(self._meta)

    @property
    def duration(self) -> float:
        """Length of the window in ms."""
        return self.window[1] - self.window[0]

    def __len__(self) -> int:
        """Number of epochs."""
        return len(self.onsets)

    def _spans(self) -> np.ndarray:
        """(n, 2) array of absolute (start, end) in ms."""
        return np.stack([self.onsets + self.window[0], self.onsets + self.window[1]],
                        axis=1).reshape(-1, 2)

    def __iter__(self):
        """Iterate over the spans ``(start, end)`` in ms."""
        return iter([tuple(row) for row in self._spans().tolist()])

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """The spans as (n, 2) array in ms."""
        if copy is False:
            raise ValueError("Epochs cannot be converted to an array without a copy")
        return np.array(self._spans(), dtype=dtype)

    def keep(self, mask) -> 'Epochs':
        """
        Epochs where `mask` is True.

        Parameters
        ----------
        mask : array-like of bool
            one value per epoch

        Returns
        -------
        Epochs
            the selected epochs (with their labels and metadata)

        Examples
        --------
        >>> ep = Epochs([100, 500, 900], (0, 100), labels=["A", "B", "C"])
        >>> ep.keep(ep.onsets < 600).labels
        array(['A', 'B'], dtype='<U1')
        """
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(self),):
            raise ValueError(f"mask must be a boolean array with one value per "
                             f"epoch ({len(self)}), got {mask.dtype} {mask.shape}")
        return Epochs(self.onsets[mask], self.window, labels=self.labels[mask],
                      meta={k: v[mask] for k, v in self._meta.items()}, label=self.label)

    def to_intervals(self, units="ms", data=None):
        """
        The spans of the epochs as :class:`~pypillometry.Intervals`.

        The Intervals keep the event labels, onsets and metadata per row.

        Parameters
        ----------
        units : str, optional
            units of the Intervals ("ms" (default), "sec", "min", "h", or
            "samples" for sample indices of `data`)
        data : EyeData, optional
            data object the epochs belong to; needed for "samples"

        Returns
        -------
        Intervals
        """
        from .intervals import Intervals
        iv = Intervals([tuple(row) for row in self._spans().tolist()], units="ms",
                       label=self.label, event_labels=self.labels.tolist(),
                       event_onsets=self.onsets.copy(),
                       event_meta=dict(self._meta) or None)
        return iv.to_units(units, data)

    @requires_package("pandas")
    def as_pandas(self):
        """
        The epochs as a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            columns ``onset``, ``start``, ``end`` (in ms), ``label`` and
            the metadata columns
        """
        import pandas as pd
        spans = self._spans()
        df = pd.DataFrame({"onset": self.onsets, "start": spans[:, 0],
                           "end": spans[:, 1], "label": self.labels})
        for name, col in self._meta.items():
            df[name] = col
        return df

    def __eq__(self, other):
        if not isinstance(other, Epochs):
            return NotImplemented
        return (self.window == other.window and self.label == other.label
                and np.array_equal(self.onsets, other.onsets)
                and np.array_equal(self.labels, other.labels)
                and self._meta.keys() == other._meta.keys()
                and all(np.array_equal(v, other._meta[k], equal_nan=v.dtype.kind == "f")
                        for k, v in self._meta.items()))

    __hash__ = None

    def __repr__(self):
        name = "Epochs" if self.label is None else f"Epochs '{self.label}'"
        return (f"{name} | {len(self)} epochs, window ({self.window[0]:g}, "
                f"{self.window[1]:g}) ms")
