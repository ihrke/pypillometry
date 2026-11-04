"""Functions to handle events in eyetracking data.
"""

import re
from types import MappingProxyType

import numpy as np
from typing import Optional, List, Tuple

from loguru import logger
from pypillometry.convenience import normalize_unit
from pypillometry.state import Stateful
from pypillometry.units import (UNSET, Quantity, _has_units, _time_values_and_units,
                                parse_time)

__all__ = [
    "Events", "ClockSync", "sync_clocks",
]


# names that cannot be used for metadata columns
_RESERVED = ("onset", "label")


def _is_missing(v) -> bool:
    """True for None and NaN (missing metadata values)."""
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _column_from_values(values) -> np.ndarray:
    """
    Metadata column from a list of values; None and NaN are missing.

    Without missing values, bools give a bool column and integers an int
    column. Otherwise, numbers give a float column (missing: NaN) and
    anything else a string column (missing: "").
    """
    present = [v for v in values if not _is_missing(v)]
    complete = len(present) == len(values)
    if all(isinstance(v, (bool, np.bool_)) for v in present) and complete and present:
        return np.array(values, dtype=bool)
    if all(_is_int(v) for v in present) and complete and present:
        return np.array(values, dtype=int)
    if all(isinstance(v, (bool, np.bool_)) or _is_int(v) or isinstance(v, (float, np.floating))
           for v in present):
        return np.array([np.nan if _is_missing(v) else float(v) for v in values], dtype=float)
    return np.array(["" if _is_missing(v) else str(v) for v in values], dtype=str)


def _as_column(values, n: int, name: str) -> np.ndarray:
    """Metadata column `name` (n values) from an array-like."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"metadata column names must be non-empty strings, got {name!r}")
    if name in _RESERVED:
        raise ValueError(f"'{name}' cannot be a metadata column (reserved)")
    if "/" in name or name.startswith("__"):
        # names are keys in HDF5 files
        raise ValueError(f"metadata column names cannot contain '/' or start "
                         f"with '__', got {name!r}")
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = np.full(n, arr.item())
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"metadata column '{name}' has {arr.size} values, expected {n}")
    if arr.dtype.kind in "biufU":
        return arr.copy()
    return _column_from_values(arr.tolist())


def _meta_dict(meta, n: int) -> dict:
    """Metadata dict (column name -> array of n values) from `meta` (or None)."""
    if meta is None:
        return {}
    return {name: _as_column(values, n, name) for name, values in meta.items()}


def _concat_meta(metas, lengths) -> dict:
    """
    Concatenate metadata dicts with `lengths` rows each.

    Columns that are missing in some of them are filled with NaN (or "" for
    string columns); numbers and strings in one column give strings.
    """
    names = list(dict.fromkeys(name for m in metas for name in m))
    out = {}
    for name in names:
        parts = [(m.get(name), k) for m, k in zip(metas, lengths)]
        text = any(col is not None and col.dtype.kind == "U" for col, _ in parts)
        cols = []
        for col, k in parts:
            if col is None:
                if k > 0:
                    cols.append(np.full(k, "") if text else np.full(k, np.nan))
            else:
                cols.append(col.astype(str) if text else col)
        out[name] = np.concatenate(cols)
    return out


def _numbers_if_possible(values):
    """Strings converted to int or float if all of them can be (None kept)."""
    present = [v for v in values if v is not None]
    for conv in (int, float):
        try:
            converted = [conv(v) for v in present]
        except ValueError:
            continue
        it = iter(converted)
        return [None if v is None else next(it) for v in values]
    return values


def _pair_with_next(on1, on2):
    """Pair each first event with the next second event after it.

    `on1` and `on2` are the onsets of the first and the second events. Each
    event is used at most once. If several first events come before the
    same second event, the last of them (the closest one) is used. A second
    event at the same time as a first event is not paired with it.

    Returns
    -------
    pairs : list of (int, int)
        indices into `on1` and `on2` of the pairs, in time order
    unpaired1, unpaired2 : list of int
        indices of the first events without a second event (before the next
        first event or the end) and of the second events without a first
        event before them
    """
    # at the same time, second events come first, so they are not paired
    # with a first event at that time
    order = sorted([(t, 0, k) for k, t in enumerate(on2)] +
                   [(t, 1, k) for k, t in enumerate(on1)])
    pairs, unpaired1, unpaired2 = [], [], []
    pending = None  # last first event without a partner
    for _, role, k in order:
        if role == 1:
            if pending is not None:
                unpaired1.append(pending)
            pending = k
        elif pending is not None:
            pairs.append((pending, k))
            pending = None
        else:
            unpaired2.append(k)
    if pending is not None:
        unpaired1.append(pending)
    return pairs, sorted(unpaired1), sorted(unpaired2)



class Events(Stateful):
    """
    Container for events with metadata.
    
    This class wraps arrays of event onsets and labels and provides additional
    functionality such as unit conversion, filtering, and display capabilities.

    Onsets are always times (not sample indices), so they stay valid when
    the data are sliced or resampled.

    Events are immutable: onsets, labels and metadata cannot be changed in
    place; all methods return new Events.

    The events of a dataset are obtained with
    :attr:`~pypillometry.GenericEyeData.events`. An (e.g., filtered)
    Events object can be passed as `event_select` to
    :meth:`~pypillometry.GenericEyeData.get_intervals` or stored in a dataset
    with :meth:`~pypillometry.GenericEyeData.set_events` or added with
    :meth:`~pypillometry.GenericEyeData.add_events`. Events from a
    behaviour log can be created with :meth:`from_table` and mapped to the
    eye-tracker clock with :func:`~pypillometry.sync_clocks`.

    Each event can have metadata (e.g., trial, condition, reaction time),
    stored in `meta`: a read-only mapping of column names to numpy arrays
    with one value per event. Missing values are NaN in numeric columns and "" in
    string columns. Metadata come from a behaviour log (see
    :meth:`from_table`), from the labels of the events (see
    :meth:`parse_labels`) or are added with :meth:`set_meta` and
    :meth:`add_latency`; events are selected by their metadata with
    :meth:`filter`.

    Parameters
    ----------
    onsets : np.ndarray, list or Quantity
        Event onset times (stored as float array); can carry units, e.g.
        ``[100, 500] * u.ms``
    labels : np.ndarray or list
        Event labels (stored as array of strings), same length as `onsets`
    units : str, Unit or None, optional
        Time units of the onsets ("ms", "sec", "min" or "h").
        Default is "ms", or the units of `onsets` if they carry units
        (otherwise, onsets with units are converted to `units`).
    data_time_range : tuple or None, optional
        Time range (min, max) of the original dataset (in same units as onsets)
    meta : dict or None, optional
        metadata: column name -> values (one per event); "onset" and
        "label" cannot be used as names

    Raises
    ------
    ValueError
        If `onsets` and `labels` differ in length, if a metadata column
        has the wrong length, or if the units are not time units (e.g.,
        None or samples).

    Examples
    --------
    >>> events = Events(onsets=[100, 500, 1000], labels=["A", "B", "C"], units="ms")
    >>> len(events)
    3
    >>> for onset, label in events:
    ...     print(f"{label} at {onset}")
    >>> events = Events([100, 500], ["stim", "resp"],
    ...                 meta={"trial": [1, 1], "cond": ["easy", "easy"]})
    >>> events.meta["cond"]
    array(['easy', 'easy'], dtype='<U4')
    """

    # state (see pypillometry.state)
    _STATE_FIELDS = ("onsets", "labels", "units", "data_time_range", ("meta", None, "_meta"))

    def __init__(self,
                 onsets: np.ndarray | List,
                 labels: np.ndarray | List,
                 units: Optional[str] = UNSET,
                 data_time_range: Optional[Tuple[float, float]] = None,
                 meta: Optional[dict] = None):
        """
        Initialize an Events object (see class docstring for parameters).
        """
        if units is UNSET and not _has_units(onsets):
            units = "ms"
        onsets, self.units = _time_values_and_units(onsets, units)
        if self.units == "samples":
            raise ValueError("Events need time units (e.g., units='ms'), "
                             "not sample indices")
        self.onsets = np.array(onsets, dtype=float)
        self.labels = np.array(labels, dtype=str)

        self.data_time_range = data_time_range

        # Validate that onsets and labels have same length
        if len(self.onsets) != len(self.labels):
            raise ValueError(f"onsets and labels must have same length, got {len(self.onsets)} and {len(self.labels)}")
        self._meta = _meta_dict(meta, len(self.onsets))
        # Events are immutable: methods return new objects
        for arr in (self.onsets, self.labels, *self._meta.values()):
            arr.flags.writeable = False

    def _set_state(self, st: dict):
        """Set the attributes from a state of the current version (checked like
        the arguments of the constructor)."""
        self.__init__(st["onsets"], st["labels"], units=st["units"],
                      data_time_range=st["data_time_range"], meta=st["meta"])

    @property
    def meta(self):
        """
        Metadata of the events (read-only).

        A mapping of column name -> numpy array (one value per event). It
        cannot be changed in place; :meth:`set_meta`, :meth:`drop_meta`,
        :meth:`parse_labels`, :meth:`fill_meta` and :meth:`add_latency`
        return new Events with changed metadata.
        """
        return MappingProxyType(self._meta)

    def _take(self, idx) -> 'Events':
        """New Events with the events `idx` (index array, mask or slice)."""
        return Events(
            onsets=self.onsets[idx],
            labels=self.labels[idx],
            units=self.units,
            data_time_range=self.data_time_range,
            meta={name: col[idx] for name, col in self.meta.items()}
        )

    def shift(self, by) -> 'Events':
        """
        Events with shifted onsets.

        Parameters
        ----------
        by : float or callable
            a time offset (in the units of the events) that is added to all
            onsets, or a time map: a function that takes an array of times
            and returns the new times (an array of the same length)

        Returns
        -------
        Events
            new Events with the same labels and metadata; `data_time_range`
            (if set) is moved in the same way

        Raises
        ------
        ValueError
            if the time map does not return one time per onset

        Examples
        --------
        >>> Events([100, 500], ["A", "B"]).shift(-100).onsets
        array([  0., 400.])
        >>> Events([100, 500], ["A", "B"]).shift(lambda t: t * 2).onsets
        array([ 200., 1000.])
        """
        if callable(by):
            fct = by
        else:
            offset = float(by)
            fct = lambda t: t + offset  # noqa: E731
        onsets = np.asarray(fct(self.onsets), dtype=float)
        if onsets.shape != self.onsets.shape:
            raise ValueError(f"the time map returned {onsets.size} times for "
                             f"{len(self)} onsets")
        time_range = None
        if self.data_time_range is not None:
            time_range = tuple(float(t) for t in
                               fct(np.asarray(self.data_time_range, dtype=float)))
        return Events(onsets, self.labels, units=self.units,
                      data_time_range=time_range, meta=self._meta)

    def keep(self, mask) -> 'Events':
        """
        Events where `mask` is True.

        Parameters
        ----------
        mask : array-like of bool
            one value per event

        Returns
        -------
        Events
            new Events with the selected events (and their metadata)

        Raises
        ------
        ValueError
            if `mask` is not a boolean array with one value per event

        Examples
        --------
        >>> ev = Events([100, 500, 900], ["A", "B", "C"])
        >>> ev.keep(ev.onsets < 600).labels
        array(['A', 'B'], dtype='<U1')
        """
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(self),):
            raise ValueError(f"mask must be a boolean array with one value per "
                             f"event ({len(self)}), got {mask.dtype} {mask.shape}")
        return self._take(mask)

    def __len__(self) -> int:
        """Return number of events."""
        return len(self.onsets)
    
    def __iter__(self):
        """Iterate over (onset, label) pairs."""
        return zip(self.onsets, self.labels)
    
    def __getitem__(self, idx):
        """
        Get event(s) by index.
        
        Parameters
        ----------
        idx : int, slice, or array-like
            Index, slice, or boolean/integer array to retrieve
            
        Returns
        -------
        Events
            New Events object with the selected event(s)
        """
        if isinstance(idx, (int, np.integer)):
            # Wrap single index in a list to get 1-element arrays
            return self._take([idx])
        elif isinstance(idx, (slice, np.ndarray, list)):
            return self._take(idx)
        else:
            raise TypeError(f"indices must be integers, slices, or array-like, not {type(idx)}")
    
    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert events to numpy arrays.
        
        Returns
        -------
        tuple of np.ndarray
            Tuple of (onsets, labels) as numpy arrays
            
        Examples
        --------
        >>> events = Events([100, 200], ["A", "B"], units="ms")
        >>> onsets, labels = events.to_array()
        """
        return self.onsets.copy(), self.labels.copy()

    def row(self, i: int) -> dict:
        """
        One event as a dict with its onset, label and metadata.

        Parameters
        ----------
        i : int
            index of the event

        Returns
        -------
        dict
            ``{"onset": ..., "label": ..., <metadata columns>}``

        Examples
        --------
        >>> ev = Events([100, 500], ["stim", "resp"], meta={"trial": [1, 1]})
        >>> ev.row(0)
        {'onset': 100.0, 'label': 'stim', 'trial': 1}
        """
        return {"onset": self.onsets[i].item(), "label": str(self.labels[i]),
                **{name: col[i].item() for name, col in self.meta.items()}}

    def to_table(self) -> dict:
        """
        Events as a table: a dict of columns (copies).

        The columns are "onset", "label" and the metadata columns. The
        result can be passed to ``pandas.DataFrame()``.

        Returns
        -------
        dict
            column name -> numpy array

        Examples
        --------
        >>> ev = Events([100, 500], ["stim", "resp"], meta={"trial": [1, 1]})
        >>> sorted(ev.to_table())
        ['label', 'onset', 'trial']
        >>> import pandas as pd
        >>> df = pd.DataFrame(ev.to_table())
        """
        return {"onset": self.onsets.copy(), "label": self.labels.copy(),
                **{name: col.copy() for name, col in self.meta.items()}}

    @classmethod
    def from_table(cls, table, columns, units: Optional[str] = UNSET,
                   meta=None, row: Optional[str] = "row") -> 'Events':
        """
        Create Events from a table with one row per trial.

        Behaviour logs usually have one row per trial and one column per
        event time (e.g., stimulus onset, response). Each column becomes
        one kind of event; the column (or dict key) is the label. Missing
        times (NaN, e.g., no response) give no event. The events are
        sorted by onset.

        Other columns of the table (e.g., condition, reaction time) can be
        given as `meta`; every event of a row gets the values of that row
        as metadata.

        Parameters
        ----------
        table : pandas.DataFrame or dict
            table with one column of times per kind of event
        columns : list or dict
            list of column names (labels are the column names), or dict
            mapping labels to column names or to arrays of times (e.g.,
            ``{"response": df.stim_on + df.rt}``)
        units : str, Unit or None, optional
            time units of the values ("ms", "sec", "min" or "h");
            default "ms"
        meta : list or dict, optional
            columns with metadata: list of column names, or dict mapping
            metadata names to column names or to arrays of values (one per
            row)
        row : str or None, optional
            name of a metadata column with the position of the row in the
            table (0, 1, ...), which links the events of the same row;
            default "row"; None for no such column

        Returns
        -------
        Events
            all events of all columns, sorted by onset

        Raises
        ------
        ValueError
            if the columns have different lengths

        Examples
        --------
        >>> table = {"stim": [1.0, 3.0], "rt": [0.5, np.nan], "cond": ["a", "b"]}
        >>> ev = Events.from_table(table, {"stim": "stim",
        ...                                "response": np.add(table["stim"], table["rt"])},
        ...                        units="sec", meta=["cond", "rt"])
        >>> ev.labels
        array(['stim', 'response', 'stim'], dtype='<U8')
        >>> ev.meta["cond"]
        array(['a', 'a', 'b'], dtype='<U1')
        """
        if not isinstance(columns, dict):
            columns = {name: name for name in columns}
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            meta = {name: name for name in meta}
        onsets, labels, rows = [], [], []
        nrows = None
        for label, col in columns.items():
            times = table[col] if isinstance(col, str) else col
            times = np.asarray(times, dtype=float)
            if nrows is not None and len(times) != nrows:
                raise ValueError(f"column for '{label}' has {len(times)} "
                                 f"values, expected {nrows}")
            nrows = len(times)
            keep = ~np.isnan(times)
            onsets.append(times[keep])
            labels.append(np.full(keep.sum(), label, dtype=object))
            rows.append(np.flatnonzero(keep))
        onsets = np.concatenate(onsets) if onsets else np.array([])
        labels = np.concatenate(labels).astype(str) if labels else np.array([], dtype=str)
        rows = np.concatenate(rows) if rows else np.array([], dtype=int)
        # metadata of each row -> metadata of each event
        columns_meta = {}
        for name, col in meta.items():
            values = table[col] if isinstance(col, str) else col
            if nrows is None:
                nrows = len(values)
            columns_meta[name] = _as_column(values, nrows, name)[rows]
        if row is not None:
            if row in columns_meta:
                raise ValueError(f"'{row}' is a metadata column; give another "
                                 "name for the row column (or row=None)")
            columns_meta[row] = rows
        # stable sort, so events at the same time keep the column order
        order = np.argsort(onsets, kind="stable")
        return cls(onsets[order], labels[order], units=units,
                   meta={name: col[order] for name, col in columns_meta.items()})

    def __add__(self, other: 'Events') -> 'Events':
        """
        All events of both objects, sorted by onset.

        The onsets of `other` are converted to the units of this object.
        Events at the same time keep their order (those of this object
        first). Metadata columns that only one of them has are filled with
        missing values (NaN or "") for the events of the other.
        """
        if not isinstance(other, Events):
            return NotImplemented
        other = other.to_units(self.units)
        onsets = np.concatenate([self.onsets, other.onsets])
        labels = np.concatenate([self.labels, other.labels])
        meta = _concat_meta([self.meta, other.meta], [len(self), len(other)])
        order = np.argsort(onsets, kind="stable")
        return Events(onsets[order], labels[order], units=self.units,
                      data_time_range=self.data_time_range,
                      meta={name: col[order] for name, col in meta.items()})

    def _with_columns(self, values: dict) -> 'Events':
        """
        New Events with columns set from `values` (name -> list with one value
        per event, None where no value is given); events without a new value
        keep the value of an existing column.
        """
        meta = dict(self.meta)
        for name, vals in values.items():
            if name in meta:
                old = meta[name].tolist()
                vals = [o if v is None else v for v, o in zip(vals, old)]
            meta[name] = _column_from_values(vals)
        return Events(self.onsets, self.labels, units=self.units,
                      data_time_range=self.data_time_range, meta=meta)

    def parse_labels(self, parser) -> 'Events':
        r"""
        Metadata from the labels of the events.

        Triggers often code information about the trial in their label,
        e.g., ``"S_12_3_1"`` for the stimulus of trial 12 with options 3
        and 1. `parser` decomposes the labels into metadata.

        Parameters
        ----------
        parser : str or callable
            - str: regular expression with named groups, e.g.
              ``r"S_(?P<trial>\d+)_(?P<left>\d+)_(?P<right>\d+)"``; the
              groups become columns (found with :func:`re.search`, so it
              can match a part of the label). Columns whose values are
              all numbers are converted to int or float.
            - callable: function that takes a label and returns a dict
              (column name -> value) or None (no metadata for this event)

        Returns
        -------
        Events
            new Events object with the columns added; events without a
            value get a missing value (NaN or ""), unless the column exists
            already: then they keep their value (so several calls can fill
            one column, e.g., "trial", from different kinds of events)

        Examples
        --------
        >>> ev = Events([0, 100, 400], ["F_1", "S_1_3_5", "C_1_5_2"])
        >>> ev = ev.parse_labels(r"^[FSC]_(?P<trial>\d+)")
        >>> ev.meta["trial"]
        array([1, 1, 1])
        >>> ev = ev.parse_labels(lambda lab: {"chosen": int(lab.split("_")[2])}
        ...                      if lab.startswith("C_") else None)
        >>> ev.meta["chosen"]
        array([nan, nan,  5.])
        """
        if isinstance(parser, str):
            regex = re.compile(parser)
            if not regex.groupindex:
                raise ValueError("the regular expression needs named groups, "
                                 "e.g. (?P<trial>\\d+)")

            def parse(label):
                m = regex.search(label)
                return None if m is None else m.groupdict()
        elif callable(parser):
            parse = parser
        else:
            raise TypeError(f"parser must be a str or callable, got {type(parser)}")

        parsed = [parse(str(label)) for label in self.labels]
        names = list(dict.fromkeys(k for d in parsed if d for k in d))
        values = {name: [None if d is None else d.get(name) for d in parsed]
                  for name in names}
        for name in names:
            _as_column([], 0, name)  # check the name
            if isinstance(parser, str):
                values[name] = _numbers_if_possible(values[name])
        return self._with_columns(values)

    def _mask(self, selector, exact: bool = False) -> np.ndarray:
        """Boolean mask of the events selected by `selector` (see :meth:`filter`)."""
        if isinstance(selector, str):
            # String selector: substring matching
            return np.array([label == selector if exact else selector in label
                             for label in self.labels], dtype=bool)
        if callable(selector):
            # Function selector: apply to each label
            return np.array([bool(selector(label)) for label in self.labels], dtype=bool)
        if isinstance(selector, np.ndarray) and selector.dtype == bool:
            if len(selector) != len(self):
                raise ValueError(f"mask has {len(selector)} values, expected {len(self)}")
            return selector.copy()
        if isinstance(selector, (tuple, Quantity)):
            # Time range selector
            if len(selector) != 2:
                raise ValueError("Time range selector must be a tuple of (min_time, max_time)")
            if _has_units(selector):
                selector = parse_time(selector, to=self.units, default=None)

            min_time, max_time = selector
            if min_time >= max_time:
                raise ValueError(f"min_time must be < max_time, got {min_time} >= {max_time}")
            
            return np.logical_and(self.onsets >= min_time, self.onsets <= max_time)
        raise TypeError(
            f"selector must be str, callable, or tuple (or a boolean array), got {type(selector)}"
        )

    def set_meta(self, name: str, values, select=None, exact: bool = False) -> 'Events':
        """
        Set a metadata column.

        Parameters
        ----------
        name : str
            name of the column (a new or an existing one)
        values : scalar or array-like
            one value for all (selected) events, or one value per event
            (or per selected event, if `select` is given); None or NaN for
            missing values
        select : str, callable, tuple or boolean array, optional
            only set the values of these events (see :meth:`filter`); the
            other events keep their value (or get a missing value, for a
            new column)
        exact : bool, optional
            if True, a string `select` matches only labels equal to it

        Returns
        -------
        Events
            new Events object with the column set

        Examples
        --------
        >>> ev = Events([0, 100, 400], ["F_1", "S_1", "C_1"])
        >>> ev.set_meta("block", 1).meta["block"]
        array([1, 1, 1])
        >>> ev.set_meta("kind", "stimulus", select="S_").meta["kind"]
        array(['', 'stimulus', ''], dtype='<U8')
        """
        _as_column([], 0, name)  # check the name
        mask = np.ones(len(self), dtype=bool) if select is None \
            else self._mask(select, exact)
        if np.ndim(values) == 0:
            values = [values] * int(mask.sum())
        values = np.asarray(values).tolist()
        if len(values) == len(self) and select is not None and mask.sum() != len(self):
            values = [v for v, m in zip(values, mask) if m]
        if len(values) != mask.sum():
            raise ValueError(f"got {len(values)} values for {mask.sum()} events")
        # unselected events keep their value
        new = self.meta[name].tolist() if name in self.meta else [None] * len(self)
        for i, v in zip(np.flatnonzero(mask), values):
            new[i] = v
        return self._replace_column(name, new)

    def _replace_column(self, name, values) -> 'Events':
        """New Events with column `name` set to `values` (list; None, NaN or "" = missing)."""
        values = [None if _is_missing(v) or (isinstance(v, str) and v == "") else v
                  for v in values]
        meta = dict(self.meta)
        meta[name] = _column_from_values(values)
        return Events(self.onsets, self.labels, units=self.units,
                      data_time_range=self.data_time_range, meta=meta)

    def drop_meta(self, *names: str) -> 'Events':
        """
        New Events without the metadata columns `names` (all, if none given).

        Examples
        --------
        >>> ev = Events([0], ["A"], meta={"a": [1], "b": [2]})
        >>> list(ev.drop_meta("a").meta)
        ['b']
        """
        for name in names:
            if name not in self.meta:
                raise KeyError(f"no metadata column '{name}'")
        meta = {} if not names else {k: v for k, v in self.meta.items() if k not in names}
        return Events(self.onsets, self.labels, units=self.units,
                      data_time_range=self.data_time_range, meta=meta)

    def fill_meta(self, columns, by: str = "trial") -> 'Events':
        """
        Copy metadata to all events with the same value in column `by`.

        Information coded in one event (e.g., the reward in the feedback
        trigger) is often needed at another event of the same trial
        (e.g., to select the stimuli by the reward that followed). For
        each value of `by`, the missing values of each column are set to
        the value that the other events of this group have.

        Parameters
        ----------
        columns : str or list of str
            metadata columns to fill
        by : str, optional
            column that defines the groups; default "trial"

        Returns
        -------
        Events
            new Events object with the columns filled

        Raises
        ------
        ValueError
            if the events of a group have different values in a column

        Examples
        --------
        >>> ev = Events([0, 100, 400], ["S_1", "C_1", "R_1"],
        ...             meta={"trial": [1, 1, 1], "reward": [np.nan, np.nan, 15]})
        >>> ev.fill_meta("reward").meta["reward"]
        array([15., 15., 15.])
        """
        if isinstance(columns, str):
            columns = [columns]
        if by not in self.meta:
            raise KeyError(f"no metadata column '{by}'")
        group = self.meta[by]
        values = {}
        for name in columns:
            if name not in self.meta:
                raise KeyError(f"no metadata column '{name}'")
            vals = [None if _is_missing(v) or v == "" else v
                    for v in self.meta[name].tolist()]
            for g in np.unique(group):
                if _is_missing(g) or g == "":  # events without a group
                    continue
                ix = np.flatnonzero(group == g)
                present = {vals[i] for i in ix if vals[i] is not None}
                if len(present) > 1:
                    raise ValueError(f"events with {by}={g} have different values "
                                     f"in '{name}': {sorted(present, key=str)}")
                if present:
                    value = present.pop()
                    for i in ix:
                        vals[i] = value
            values[name] = vals
        return self._with_columns(values)

    def add_latency(self, first, second, name: str, by: Optional[str] = None,
                    exact: bool = False, units: Optional[str] = None) -> 'Events':
        """
        Time from each `first` event to its `second` event, as metadata.

        For example, the reaction time is the time from a stimulus to the
        response. It is stored in column `name` of the first events; first
        events without a second event (e.g., no response) get NaN, and
        the other events keep their value (or get NaN, for a new column).

        Parameters
        ----------
        first, second : str, callable or boolean array
            the first and the second events (see :meth:`filter`)
        name : str
            name of the column, e.g. "rt"
        by : str or None, optional
            if None (default), each first event is paired with the next
            second event after it (as in
            :meth:`~pypillometry.GenericEyeData.get_intervals`); if the
            name of a metadata column (e.g., "trial"), with the next second
            event with the same value in this column
        exact : bool, optional
            if True, string selectors match only labels equal to them
        units : str, optional
            units of the latencies; default: the units of the events

        Returns
        -------
        Events
            new Events object with the column set

        Examples
        --------
        >>> ev = Events([0, 450, 1000, 3000], ["S_1", "C_1", "S_2", "S_3"])
        >>> ev.add_latency("S_", "C_", "rt").meta["rt"]
        array([450.,  nan,  nan,  nan])
        """
        m1, m2 = self._mask(first, exact), self._mask(second, exact)
        i1, i2 = np.flatnonzero(m1), np.flatnonzero(m2)
        if by is None:
            groups = [(i1, i2)]
        else:
            if by not in self.meta:
                raise KeyError(f"no metadata column '{by}'")
            col = self.meta[by]
            groups = [(i1[col[i1] == v], i2[col[i2] == v])
                      for v in np.unique(col[i1]) if not (_is_missing(v) or v == "")]
        latency = np.full(len(self), np.nan)
        for g1, g2 in groups:
            pairs, _, _ = _pair_with_next(self.onsets[g1], self.onsets[g2])
            for a, b in pairs:
                latency[g1[a]] = self.onsets[g2[b]] - self.onsets[g1[a]]
        if units is not None:
            latency = latency * parse_time(1, to=normalize_unit(units), default=self.units)
        return self.set_meta(name, latency[m1], select=m1)

    def to_units(self, target_units: Optional[str]) -> 'Events':
        """
        Convert event onsets to different time units.
        
        Parameters
        ----------
        target_units : str
            Target units: "ms", "sec", "min" or "h"
            
        Returns
        -------
        Events
            New Events object with converted units
            
        Raises
        ------
        ValueError
            If `target_units` is not a time unit
        
        Examples
        --------
        >>> events = Events([1000, 2000], ["A", "B"], units="ms")
        >>> events_sec = events.to_units("sec")
        >>> events_sec.onsets
        array([1., 2.])
        """
        # Normalize target units (source units already normalized in __init__)
        target_units = normalize_unit(target_units)
        
        if target_units == "samples":
            raise ValueError("Events need time units (e.g., 'ms'), not sample "
                             "indices; use the data object to find samples")
        
        if self.units == target_units:
            # No conversion needed, return self
            return self
        
        # Define conversion factors (everything relative to ms)
        units_to_ms = {
            "ms": 1.0,
            "sec": 1000.0,
            "min": 60000.0,
            "h": 3600000.0
        }
        
        if self.units not in units_to_ms:
            raise ValueError(f"Unknown source units: {self.units}")
        if target_units not in units_to_ms:
            raise ValueError(f"Unknown target units: {target_units}")
        
        # Convert: first to ms, then to target units
        fac = units_to_ms[self.units] / units_to_ms[target_units]
        converted_onsets = self.onsets * fac
        
        # Convert data_time_range if present
        if self.data_time_range is not None:
            converted_range = (self.data_time_range[0] * fac, self.data_time_range[1] * fac)
        else:
            converted_range = None
        
        return Events(
            onsets=converted_onsets,
            labels=self.labels.copy(),
            units=target_units,
            data_time_range=converted_range,
            meta=self.meta
        )
    
    def __repr__(self) -> str:
        """
        String representation of Events.
        
        Returns
        -------
        str
            Formatted string with number of events and units
        """
        parts = []
        
        # Add basic info
        if len(self) == 0:
            parts.append("Events: 0 events")
        elif len(self) == 1:
            parts.append("Events: 1 event")
        else:
            parts.append(f"Events: {len(self)} events")
        
        parts.append(f"units={self.units}")
        
        # Add time range if available
        if self.data_time_range is not None:
            range_str = f"[{self.data_time_range[0]:.1f}, {self.data_time_range[1]:.1f}]"
            parts.append(f"range={range_str}")

        if self.meta:
            parts.append(f"meta=[{', '.join(self.meta)}]")
        
        return " | ".join(parts)
    
    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebooks.
        
        Displays events as a nicely formatted HTML table with columns for
        index, onset time, and event label.
        
        Returns
        -------
        str
            HTML string for displaying in Jupyter notebook
        """
        try:
            import pandas as pd
        except ImportError:
            # Fallback to text representation if pandas not available
            return f"<pre>{repr(self)}</pre>"
        
        if len(self) == 0:
            return f"<p><i>Events: 0 events (units={self.units})</i></p>"
        
        # Determine unit display
        unit_str = self.units
        
        # Create DataFrame
        df = pd.DataFrame({
            f'Onset ({unit_str})': self.onsets,
            'Label': self.labels,
            **self.meta
        })

        # Convert to HTML respecting current pandas display options
        import inspect

        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")

        to_html_kwargs = {
            "index": True,
            "index_names": ['#'],
        }

        to_html_signature = inspect.signature(pd.DataFrame.to_html)
        if "max_rows" in to_html_signature.parameters and max_rows is not None:
            to_html_kwargs["max_rows"] = max_rows
        if "min_rows" in to_html_signature.parameters and min_rows is not None:
            to_html_kwargs["min_rows"] = min_rows

        html = df.to_html(**to_html_kwargs)
        
        # Add summary info below table
        summary = f'<p style="font-size: 0.9em; color: #666; margin-top: 8px;">'
        summary += f'Total: {len(self)} events'
        if self.data_time_range is not None:
            summary += f' | Data range: [{self.data_time_range[0]:.1f}, {self.data_time_range[1]:.1f}] {unit_str}'
        summary += '</p>'
        
        return html + summary

    def plot(
        self,
        show_labels: str = "auto",
        marker: str = ".",
        units: Optional[str] = "min",
        **kwargs,
    ):
        """
        Plot events as points on a time axis.

        Parameters
        ----------
        show_labels : {"auto", "all", "none"}, optional
            Labeling strategy. "auto" (default) shows a subset spaced apart,
            "all" annotates every event, "none" disables labels.
        marker : str, optional
            Matplotlib marker style for the events (default '.').
        units : str or None, optional
            Units to display in the plot. Defaults to "min". Use None to keep existing units.
        **kwargs :
            Additional keyword arguments passed to :func:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        None
            Plot is drawn on the current matplotlib axes.

        Examples
        --------
        The stimulus events of the example dataset:

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> plt.figure(figsize=(10, 3))
           >>> d.events.filter("S_").plot(units="sec")
        """
        import matplotlib.pyplot as plt

        if show_labels not in {"auto", "all", "none"}:
            raise ValueError("show_labels must be one of {'auto', 'all', 'none'}")

        ax = plt.gca()

        target_units = units if units is not None else self.units
        if target_units != self.units:
            converted = self.to_units(target_units)
        else:
            converted = self

        # Always label axes even if no events so the plot remains informative.
        x_label = f"Time ({converted.units})"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Event #")

        if len(self) == 0:
            ax.set_ylim(0, 1)
            return

        y_positions = np.arange(1, len(self) + 1)

        plot_kwargs = {"linestyle": "None", "marker": marker, "markersize": 3}
        plot_kwargs.update(kwargs)

        ax.plot(converted.onsets, y_positions, **plot_kwargs)

        if converted.data_time_range is not None:
            x_min, x_max = converted.data_time_range
        else:
            x_min = float(np.min(converted.onsets))
            x_max = float(np.max(converted.onsets))

        span = x_max - x_min
        if span == 0:
            padding = max(1.0, abs(x_max) * 0.05)
        else:
            padding = span * 0.05

        x_limits = (x_min - padding, x_max + padding)
        ax.set_xlim(*x_limits)
        ax.set_ylim(0, len(self) + 1)
        ax.set_yticks([1, len(self)] if len(self) > 1 else [1])

        if show_labels != "none":
            effective_spacing = 0.0
            if show_labels == "auto":
                logger.debug("Plotting only a subset of labels, use show_labels='all' to show all labels")
                fig = ax.figure
                try:
                    fig.canvas.draw()
                except Exception:
                    pass
                bbox = ax.get_window_extent()
                axis_width_pixels = bbox.width if bbox.width > 0 else 1.0
                desired_pixels = 10.0
                span = x_limits[1] - x_limits[0]
                effective_spacing = span * desired_pixels / axis_width_pixels if axis_width_pixels else 0.0

            label_offset = padding * 0.05 if padding > 0 else 0.0
            last_label_x = -np.inf
            for onset, y_pos, label in zip(converted.onsets, y_positions, self.labels):
                spacing_requirement = (
                    show_labels == "all"
                    or effective_spacing <= 0
                    or abs(onset - last_label_x) >= effective_spacing
                )
                if spacing_requirement:
                    ax.text(
                        onset + label_offset,
                        y_pos,
                        str(label),
                        va="center",
                        ha="left",
                        fontsize=8,
                        alpha=0.8,
                        rotation=90,
                        rotation_mode="anchor",
                    )
                    last_label_x = onset
        return
    
    def filter(self, selector=None, exact: bool = False, **conditions) -> 'Events':
        """
        Filter events and return a new Events object.
        
        This method supports multiple types of selectors for flexible filtering:

        - String: matches labels containing the substring (or equal to
          it, if `exact` is True)
        - Callable: applies function to each label, keeps events where it returns True
        - Tuple: filters by time range (min_time, max_time)
        - Boolean array: keeps events where it is True

        Events can also be selected by their metadata (see `conditions`).
        All criteria must be met.
        
        Parameters
        ----------
        selector : str, callable, tuple or boolean array, optional
            Filtering criterion:

            - str: substring to search for in labels
            - callable: function that takes a label and returns bool
            - tuple: (min_time, max_time) to filter by onset time range
              (inclusive, in the units of the Events object unless it
              carries units, e.g. ``(10, 30) * u.sec``)
            - boolean array: one value per event, e.g.
              ``events.meta["rt"] < 800``
        exact : bool, optional
            if True, a string selector matches only labels that are equal
            to it; if False (default), labels that contain it
        **conditions
            criteria for metadata columns (column name = criterion):

            - a value: the column is equal to it, e.g. ``cond="easy"``
            - a list or set: the column is one of the values, e.g.
              ``trial=[1, 2, 3]``
            - a callable: function that takes the whole column (numpy
              array) and returns a boolean array, e.g.
              ``rt=lambda rt: rt < 800``

        Returns
        -------
        Events
            New Events object containing only the filtered events

        Raises
        ------
        KeyError
            if a metadata column in `conditions` does not exist
            
        Examples
        --------
        >>> events = Events([100, 500, 1000], ["stim", "resp", "stim"], units="ms")
        >>> stim_events = events.filter("stim")
        >>> len(stim_events)
        2
        
        >>> events.filter("stim", exact=True).onsets  # label equal to "stim"
        array([ 100., 1000.])

        >>> # Filter by custom function
        >>> events = Events([100, 200], ["A1", "B2"], units="ms")
        >>> numeric = events.filter(lambda label: label[-1].isdigit())
        
        >>> # Filter by time range
        >>> events = Events([100, 500, 1000], ["A", "B", "C"], units="ms")
        >>> middle = events.filter((200, 800))
        >>> len(middle)
        1

        >>> # Filter by metadata
        >>> events = Events([100, 500, 1000], ["stim", "stim", "stim"],
        ...                 meta={"cond": ["a", "b", "a"], "rt": [450, 900, 600]})
        >>> events.filter(cond="a", rt=lambda rt: rt < 500).onsets
        array([100.])
        """
        mask = np.ones(len(self), dtype=bool) if selector is None \
            else self._mask(selector, exact)
        for name, crit in conditions.items():
            if name not in self.meta:
                raise KeyError(f"no metadata column '{name}' (columns: "
                               f"{', '.join(self.meta) or 'none'})")
            col = self.meta[name]
            if callable(crit):
                m = np.asarray(crit(col), dtype=bool)
                if m.shape != col.shape:
                    raise ValueError(f"the function for '{name}' must return one "
                                     f"bool per event, got shape {m.shape}")
            elif isinstance(crit, (list, set, tuple, np.ndarray)):
                m = np.isin(col, list(crit))
            else:
                m = col == crit
            mask &= m
        return self._take(mask)



class ClockSync:
    """
    Linear map between two clocks, fitted by :func:`sync_clocks`.

    Maps times `t` on the source clock (e.g., a behaviour log) to the target
    clock (e.g., the eye tracker): ``offset + slope * t``. The slope accounts
    for the unit conversion and for the drift between the clocks.

    Calling the object maps times: plain numbers are taken to be in
    `src_units` and are returned as an array in `dst_units`; numbers with
    units and :class:`Events` are converted first and returned in
    `dst_units` (as :class:`Events` for :class:`Events`).

    Attributes
    ----------
    offset : float
        target time (in `dst_units`) of source time 0
    slope : float
        target units per source unit
    drift : float
        how much faster the target clock runs (e.g., 1e-5 is 36 ms per
        hour); 0 if only the offset was fitted
    residuals : np.ndarray
        target times minus mapped source times (in `dst_units`); they are
        small (around the timing precision) if the pairs are correct
    src, dst : np.ndarray
        the paired times (in `src_units` and `dst_units`)
    src_units, dst_units : str
        units of the source and target times

    Examples
    --------
    >>> sync = sync_clocks([10, 70, 130], [5000, 65000.6, 125001.2],
    ...                    src_units="sec", dst_units="ms")
    >>> sync(100.0)
    95000.9
    """

    def __init__(self, offset, slope, src, dst, src_units, dst_units):
        self.offset = float(offset)
        self.slope = float(slope)
        self.src = np.asarray(src, dtype=float)
        self.dst = np.asarray(dst, dtype=float)
        self.src_units = src_units
        self.dst_units = dst_units
        self.residuals = self.dst - (self.offset + self.slope * self.src)

    @property
    def drift(self) -> float:
        """Relative drift of the target clock against the source clock."""
        return self.slope / parse_time(1, to=self.dst_units,
                                       default=self.src_units) - 1

    def __call__(self, times):
        """Map times from the source clock to the target clock."""
        if isinstance(times, Events):
            ev = times.to_units(self.src_units)
            return Events(self(ev.onsets), ev.labels.copy(),
                          units=self.dst_units, meta=ev.meta)
        if _has_units(times):
            times = parse_time(times, to=self.src_units, default=None)
        mapped = self.offset + self.slope * np.asarray(times, dtype=float)
        return float(mapped) if mapped.ndim == 0 else mapped

    def summary(self) -> str:
        """One-line summary of the fit."""
        drift = self.drift * parse_time(1, to="ms", default="h")
        return (f"offset={self.offset:.6g} {self.dst_units}, "
                f"drift={drift:+.3g} ms/h, "
                f"max |residual|={np.max(np.abs(self.residuals)):.3g} "
                f"{self.dst_units} (n={len(self.src)})")

    def __repr__(self) -> str:
        lines = [f"ClockSync({self.src_units} -> {self.dst_units}): {self.summary()}",
                 f"  {'src':>12} {'dst':>14} {'residual':>10}"]
        rows = [f"  {s:12.6g} {d:14.8g} {r:10.3g}"
                for s, d, r in zip(self.src, self.dst, self.residuals)]
        if len(rows) > 10:  # first and last pairs only
            rows = rows[:5] + [f"  ... ({len(rows) - 10} more)"] + rows[-5:]
        return "\n".join(lines + rows)


def sync_clocks(src, dst, src_units=UNSET, dst_units=UNSET,
                slope: bool = True, tol=5) -> ClockSync:
    """
    Fit a linear map from one clock to another from paired times.

    Behaviour logs (e.g., from Psychtoolbox, PsychoPy or Presentation)
    and eye trackers use different clocks. They differ in their zero point
    and drift apart slowly, so an offset alone can be off by several ms
    within a session. Given the times of the same moments on both clocks
    (e.g., the start of each block), this fits
    ``dst = offset + slope * src`` by least squares.

    The pairs are matched by order. A summary of the fit is logged; the
    residuals show whether the pairing is right (large residuals mean that
    a pair is shifted or missing).

    Parameters
    ----------
    src : array-like, Quantity or Events
        times on the source clock (e.g., from the behaviour log)
    dst : array-like, Quantity or Events
        times of the same moments on the target clock, in the same order
        (e.g., ``d.events.filter("Start Block")``)
    src_units, dst_units : str or Unit, optional
        units of plain numbers in `src` and `dst` (e.g., "sec", "ms");
        needed unless the times carry units or are :class:`Events`
    slope : bool, optional
        if True (default), fit offset and slope (drift); if False, only
        the offset. With a single pair, only the offset is fitted.
    tol : float, str or Quantity, optional
        a warning is given if a residual is larger than this (plain
        numbers in ms); default 5 ms

    Returns
    -------
    ClockSync
        the fitted map; call it on times or :class:`Events` to map them
        to the target clock

    Raises
    ------
    ValueError
        if `src` and `dst` differ in length, are empty or contain NaN,
        or if units are missing

    Examples
    --------
    >>> starts = behav.groupby("block").trialstart.min()     # in sec
    >>> sync = pp.sync_clocks(starts, d.events.filter("Start Block"),
    ...                       src_units="sec")
    >>> d.add_events(sync(pp.Events.from_table(behav, ["stim_on"], units="sec")))
    """
    def values(times, units, name):
        if isinstance(times, Events):
            return times.onsets, times.units
        if not _has_units(times) and units is UNSET:
            raise ValueError(f"{name} needs units, e.g. {name}_units='sec'")
        times, units = _time_values_and_units(times, units)
        if units == "samples":
            raise ValueError(f"{name} must be times, not sample indices")
        return np.asarray(times, dtype=float), units

    src, src_units = values(src, src_units, "src")
    dst, dst_units = values(dst, dst_units, "dst")
    if len(src) != len(dst):
        raise ValueError(f"src has {len(src)} times and dst has {len(dst)}; "
                         "they must be paired one-to-one")
    if len(src) == 0:
        raise ValueError("need at least one pair of times")
    if np.isnan(src).any() or np.isnan(dst).any():
        raise ValueError("src and dst must not contain NaN")

    # slope without drift (unit conversion)
    unit_slope = parse_time(1, to=dst_units, default=src_units)
    if slope and len(src) == 1:
        logger.warning("sync_clocks(): one pair of times only gives the "
                       "offset; drift is not corrected")
        slope = False
    if slope:
        b, a = np.polyfit(src, dst, 1)
    else:
        b = unit_slope
        a = np.mean(dst - b * src)
    sync = ClockSync(a, b, src, dst, src_units, dst_units)

    logger.info(f"sync_clocks(): {sync.summary()}")
    if slope and len(src) == 2:
        logger.warning("sync_clocks(): two pairs fit exactly; the residuals "
                       "cannot check the pairing")
    tol = parse_time(tol, to=dst_units, default="ms")
    big = np.abs(sync.residuals) > tol
    if big.any():
        pairs = np.flatnonzero(big).tolist()
        pairs = str(pairs) if len(pairs) <= 5 else f"{str(pairs[:5])[:-1]}, ...]"
        logger.warning(f"sync_clocks(): {big.sum()} of {len(big)} residuals larger "
                       f"than {tol:.3g} {dst_units} (pairs {pairs}); "
                       "are the times paired correctly?")
    return sync
