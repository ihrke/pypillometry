"""Event-locked windows (epochs): event onsets and one window around them.
"""

import numpy as np

from .convenience import requires_package
from .units import parse_time
from .events import Events, _warn_if_needs_sync
from .state import Stateful

__all__ = ["Epochs"]


class Epochs(Stateful):
    """
    Event-locked windows: the onsets of events and one time window around them.

    An Epochs object describes the same window around each of several
    events, e.g., -200 to 800 ms around every stimulus. It is returned by
    :meth:`~pypillometry.EyeData.get_epochs`
    and is the input of :meth:`~pypillometry.EyeData.get_segments`,
    which takes the length and the time axis of the segments from the
    window. Spans of time that are not locked to one event (e.g., between
    two events) are :class:`~pypillometry.Intervals`.

    The columns of averaged segments (see
    :meth:`~pypillometry.SegmentedEyeData.average`) have Epochs without a
    time: their onsets are NaN. Such epochs can be selected and their
    metadata used, but they have no spans in the data.

    Epochs iterate like spans (``(onset + window[0], onset + window[1])`` in
    ms), so they can be used wherever the eyedata methods take Intervals
    (e.g., :meth:`~pypillometry.EyeData.mask_intervals` or
    :meth:`~pypillometry.EyeData.stat_per_event`);
    :meth:`to_intervals` converts them explicitly. Windows that extend
    beyond the data are not clipped; the methods that read the data handle
    the parts outside of it.

    The events are kept as an :class:`~pypillometry.Events` object
    (:attr:`events`, in ms). Epochs are selected like events, by label, time
    or metadata (:meth:`filter`, :meth:`keep`), and their metadata are
    changed with :meth:`set_meta`, :meth:`drop_meta` and
    :meth:`parse_labels`; these methods return new Epochs with the same
    window.

    Parameters
    ----------
    onsets : array-like or Quantity
        onsets of the events in ms (or with units, e.g., ``[1, 2] * u.sec``);
        NaN for epochs without a time (averages)
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
    events : Events
        the events the epochs are locked to (onsets in ms)
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
    >>> ep = Epochs([1000, 3000], (-200, 800), meta={"cond": ["easy", "hard"]})
    >>> ep.filter(cond="easy").onsets
    array([1000.])
    """

    # state (see pypillometry.state)
    _STATE_FIELDS = (("events", "Events"), "window", "label")

    def __init__(self, onsets, window, labels=None, meta=None, label=None):
        onsets = np.atleast_1d(np.asarray(parse_time(onsets, to="ms"), dtype=float))
        if onsets.ndim != 1:
            raise ValueError(f"onsets must be one-dimensional, got shape {onsets.shape}")
        n = len(onsets)
        labels = np.full(n, "") if labels is None else np.asarray(labels, dtype=str)
        if labels.shape != (n,):
            raise ValueError(f"{labels.size} labels for {n} onsets")
        self._setup(Events(onsets, labels, units="ms", meta=meta), window, label)

    def _setup(self, events, window, label):
        """Set the attributes (`events` in ms) after checking them."""
        if events.units != "ms":
            raise ValueError(f"the events of Epochs must be in ms, got {events.units}")
        if np.isinf(events.onsets).any():
            raise ValueError("onsets must be finite (or NaN for epochs without a time)")
        window = np.asarray(parse_time(window, to="ms"), dtype=float)
        if window.shape != (2,) or not np.isfinite(window).all() or window[0] >= window[1]:
            raise ValueError(f"window must be (start, end) with finite start < end, "
                             f"got {window.tolist()}")
        self.events = events
        self.window = (float(window[0]), float(window[1]))
        self.label = label

    def _set_state(self, st: dict):
        """Set the attributes from a state of the current version (checked like
        the arguments of the constructor)."""
        self._setup(Events._from_state(st["events"]), st["window"], st["label"])

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
        _warn_if_needs_sync(events)
        ep = cls.__new__(cls)
        ep._setup(events.to_units("ms"), window, label)
        return ep

    def _with_events(self, events) -> 'Epochs':
        """New Epochs with the same window and label around `events`."""
        ep = type(self).__new__(type(self))
        ep._setup(events, self.window, self.label)
        return ep

    @property
    def onsets(self) -> np.ndarray:
        """Onsets of the events in ms (read-only)."""
        return self.events.onsets

    @property
    def labels(self) -> np.ndarray:
        """Labels of the events (read-only)."""
        return self.events.labels

    @property
    def meta(self):
        """Metadata of the events (read-only mapping: column name -> array)."""
        return self.events.meta

    @property
    def duration(self) -> float:
        """Length of the window in ms."""
        return self.window[1] - self.window[0]

    def __len__(self) -> int:
        """Number of epochs."""
        return len(self.events)

    def _spans(self, check: bool = True) -> np.ndarray:
        """(n, 2) array of absolute (start, end) in ms; raises a ValueError
        for epochs without a time (NaN onsets) if `check`."""
        if check and np.isnan(self.onsets).any():
            raise ValueError("these epochs have no time (their onsets are NaN, e.g., "
                             "the columns of averaged segments), so they have no "
                             "spans in the data")
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
        return self._with_events(self.events.keep(mask))

    def filter(self, selector=None, exact: bool = False, **conditions) -> 'Epochs':
        """
        Epochs whose events match the criteria.

        The events are selected as in :meth:`Events.filter
        <pypillometry.Events.filter>`: by label, onset time (in ms), a
        boolean array or metadata.

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
        Epochs
            the selected epochs (same window and label)

        Examples
        --------
        >>> ep = Epochs([100, 500, 900], (0, 100), labels=["A", "B", "A"],
        ...             meta={"rt": [300, 900, 500]})
        >>> ep.filter("A", rt=lambda rt: rt < 400).onsets
        array([100.])
        """
        return self._with_events(self.events.filter(selector, exact, **conditions))

    def set_meta(self, name: str, values, select=None, exact: bool = False) -> 'Epochs':
        """
        Epochs with a metadata column set (see :meth:`Events.set_meta
        <pypillometry.Events.set_meta>`).

        Returns
        -------
        Epochs
            new Epochs with the changed metadata
        """
        return self._with_events(self.events.set_meta(name, values, select, exact))

    def drop_meta(self, *names: str) -> 'Epochs':
        """
        Epochs without the metadata columns `names` (all, if none given; see
        :meth:`Events.drop_meta <pypillometry.Events.drop_meta>`).

        Returns
        -------
        Epochs
        """
        return self._with_events(self.events.drop_meta(*names))

    def parse_labels(self, parser) -> 'Epochs':
        """
        Epochs with metadata from the labels of the events (see
        :meth:`Events.parse_labels <pypillometry.Events.parse_labels>`).

        Returns
        -------
        Epochs
        """
        return self._with_events(self.events.parse_labels(parser))

    def merge_table(self, table, on, columns=None) -> 'Epochs':
        """
        Epochs with metadata from a table with one row per trial (see
        :meth:`Events.merge_table <pypillometry.Events.merge_table>`).

        Returns
        -------
        Epochs
        """
        return self._with_events(self.events.merge_table(table, on, columns))

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
                       event_meta=dict(self.meta) or None)
        return iv.to_units(units, data)

    @requires_package("pandas")
    def as_pandas(self):
        """
        The epochs as a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            columns ``onset``, ``start``, ``end`` (in ms; NaN for epochs
            without a time), ``label`` and the metadata columns
        """
        import pandas as pd
        spans = self._spans(check=False)
        df = pd.DataFrame({"onset": self.onsets, "start": spans[:, 0],
                           "end": spans[:, 1], "label": self.labels})
        for name, col in self.meta.items():
            df[name] = col
        return df

    def __eq__(self, other):
        if not isinstance(other, Epochs):
            return NotImplemented
        return (self.window == other.window and self.label == other.label
                and np.array_equal(self.onsets, other.onsets, equal_nan=True)
                and np.array_equal(self.labels, other.labels)
                and self.meta.keys() == other.meta.keys()
                and all(np.array_equal(v, other.meta[k], equal_nan=v.dtype.kind == "f")
                        for k, v in self.meta.items()))

    __hash__ = None

    def __repr__(self):
        name = "Epochs" if self.label is None else f"Epochs '{self.label}'"
        untimed = " without time" if len(self) and np.isnan(self.onsets).all() else ""
        return (f"{name} | {len(self)} epochs{untimed}, window ({self.window[0]:g}, "
                f"{self.window[1]:g}) ms")
