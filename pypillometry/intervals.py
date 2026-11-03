"""Functions to handle intervals (blinks, erpds, etc.) in eyetracking data.
"""

import numpy as np

from pypillometry.convenience import requires_package, normalize_unit
from pypillometry.units import (UNSET, _has_units, _time_values_and_units, parse_time,
                                _ms_to_samples, _samples_to_ms)
from pypillometry.state import Stateful
from pypillometry.events import _meta_dict, _concat_meta
from pypillometry.epochs import Epochs

__all__ = [
    "Intervals", "merge_intervals",
]

_MS_PER_UNIT = {"ms": 1.0, "sec": 1000.0, "min": 60000.0, "h": 3600000.0}

def merge_intervals(*args, label: str = 'merged'):
    """Combine multiple Intervals objects into a single Intervals object.
    
    Takes multiple Intervals objects and concatenates them into one, preserving
    all intervals and their metadata. Does NOT merge overlapping intervals - 
    call .merge() on the result if you want to merge overlapping intervals.
    All input Intervals must have the same units.
    
    Parameters
    ----------
    *args : Intervals objects, list, or dict
        Variable number of Intervals objects, or a single list/dict of Intervals.
        If dict, keys are ignored and all Intervals are combined.
    label : str, optional
        Label for the resulting Intervals object. Default is "merged".
        
    Returns
    -------
    Intervals
        A single Intervals object containing all intervals from all input
        Intervals objects with their metadata preserved.
        
    Raises
    ------
    ValueError
        If no Intervals objects are provided or if units don't match.
    TypeError
        If any argument is not an Intervals object (or a list/dict of them).

    Examples
    --------
    Combine multiple Intervals objects:
    
    >>> intervals1 = Intervals([(0, 100), (200, 300)], units="ms", event_labels=["a", "b"])
    >>> intervals2 = Intervals([(50, 150), (400, 500)], units="ms", event_labels=["c", "d"])
    >>> combined = merge_intervals(intervals1, intervals2)
    >>> len(combined)
    4
    >>> combined.event_labels
    ['a', 'b', 'c', 'd']
    
    Combine from a list:
    
    >>> intervals_list = [intervals1, intervals2]
    >>> combined = merge_intervals(intervals_list)
    
    Combine from a dict:
    
    >>> intervals_dict = {"left_pupil": intervals1, "right_pupil": intervals2}
    >>> combined = merge_intervals(intervals_dict)
    
    Then merge overlapping intervals if needed:
    
    >>> combined.merge()
    """
    
    # Parse input to get list of Intervals objects
    intervals_list = []
    
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, dict):
            # Dict of Intervals
            intervals_list = list(arg.values())
        elif isinstance(arg, list):
            # List of Intervals
            intervals_list = arg
        elif isinstance(arg, Intervals):
            # Single Intervals object
            intervals_list = [arg]
        else:
            raise TypeError(f"Expected Intervals, list, or dict, got {type(arg)}")
    else:
        # Multiple Intervals objects as *args
        intervals_list = list(args)
    
    # Validate we have Intervals objects
    if not intervals_list:
        raise ValueError("No Intervals objects provided")
    
    for i, obj in enumerate(intervals_list):
        if not isinstance(obj, Intervals):
            raise TypeError(f"Argument {i} is not an Intervals object: {type(obj)}")
    
    # Check that all have the same units
    first_units = intervals_list[0].units
    for obj in intervals_list[1:]:
        if obj.units != first_units:
            raise ValueError(
                f"All Intervals must have the same units. "
                f"Found {first_units} and {obj.units}"
            )
    
    all_intervals = [pair for obj in intervals_list for pair in obj.intervals]
    if not all_intervals:
        # All were empty
        return Intervals([], units=first_units, label=label)
    labels, onsets, meta = _concat_rows(intervals_list)
    return Intervals(all_intervals, units=first_units, label=label,
                     event_labels=labels, event_onsets=onsets, event_meta=meta)


def _concat_rows(intervals_list):
    """
    The per-interval table (event labels, onsets, metadata) of several
    Intervals, concatenated row by row. A column that is missing in some of
    them is filled with "" (labels), NaN (onsets) or as in
    :func:`~pypillometry.events._concat_meta`; None if all of them lack it.
    """
    lengths = [len(obj) for obj in intervals_list]

    def column(get, fill):
        if all(get(obj) is None for obj in intervals_list):
            return None
        return [v for obj, k in zip(intervals_list, lengths)
                for v in (list(get(obj)) if get(obj) is not None else [fill] * k)]
    labels = column(lambda obj: obj.event_labels, "")
    onsets = column(lambda obj: obj.event_onsets, np.nan)
    meta = _concat_meta([obj.event_meta or {} for obj in intervals_list], lengths) or None
    return labels, onsets, meta


def _check_pairs(pairs):
    """Raise a ValueError unless `pairs` are (start, end) with finite start <= end."""
    for i, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(f"interval {i} is {pair}; intervals must be (start, end) pairs")
        start, end = pair
        try:
            finite = np.isfinite(start) and np.isfinite(end)
        except TypeError:
            raise ValueError(f"interval {i} is {pair}; start and end must be "
                             "numbers") from None
        if not finite:
            raise ValueError(f"interval {i} is {pair}; start and end must be finite")
        if start > end:
            raise ValueError(f"interval {i} is {pair}; its start is after its end")


def _as_intervals(obj, what="intervals"):
    """`obj` as Intervals: Intervals as they are, Epochs converted to spans in ms."""
    if isinstance(obj, Intervals):
        return obj
    if isinstance(obj, Epochs):
        return obj.to_intervals()
    raise TypeError(f"{what} must be Intervals or Epochs, got {type(obj).__name__}")


def stat_event_interval(tx,sy,intervals,statfct=np.mean):
    """
    Return result of applying a statistical function to pupillometric data in
    each of a list of intervals. For example, extract mean pupil-size in the
    interval before each trial onset (intervals as returned by
    :meth:`~pypillometry.GenericEyeData.get_intervals`).

    Parameters
    ----------
    tx : np.ndarray
        time-vector (e.g., in ms)
    sy : np.ndarray or np.ma.MaskedArray
        (baseline-corrected) pupil signal, same length as `tx`; for a masked
        array, `statfct` should ignore masked samples (e.g., `np.mean`), and
        intervals where all samples are masked give NaN
    intervals : list of tuples (start, end)
        absolute start and end times of the intervals, in the same units as
        `tx`; each is matched to the closest sample in `tx` (an end after
        the last sample includes it). Only the part of an interval within
        `tx` is used; intervals entirely outside give NaN
    statfct : callable
        function mapping np.array to a single number (default: `np.mean`)

    Returns
    -------
    np.ndarray
        array with one result per interval
    """
    intervals=np.array(intervals)
    starts=intervals[:,0]
    ends=intervals[:,1]

    res=np.zeros(len(starts))

    for i,interv in enumerate(zip(starts,ends)):
        if interv[1] <= tx[0] or interv[0] > tx[-1]:
            res[i]=np.nan  # no sample within the interval (end exclusive)
            continue
        start_ix=np.argmin(np.abs(interv[0]-tx))
        end_ix=len(tx) if interv[1] > tx[-1] else np.argmin(np.abs(interv[1]-tx))
        if start_ix==end_ix:
            end_ix+=1
        seg=sy[start_ix:end_ix]
        if np.ma.getmaskarray(seg).all():
            res[i]=np.nan  # all samples masked
            continue
        r=statfct(seg)
        res[i]=np.nan if r is np.ma.masked else r
    return res


class Intervals(Stateful):
    """
    Container for intervals with metadata.
    
    This class wraps a list of (start, end) intervals and provides additional
    metadata such as units, labels, and associated event information.
    Intervals are end-exclusive: ``(start, end)`` covers the samples with
    ``start <= t < end``.

    Usually, Intervals objects are not created directly but returned by
    methods of the eye-data classes, e.g.,
    :meth:`~pypillometry.GenericEyeData.get_intervals` (spans between two
    events; windows around single events are
    :class:`~pypillometry.Epochs`, see :meth:`~pypillometry.Epochs.to_intervals`),
    :meth:`~pypillometry.GenericEyeData.get_blinks` or
    :meth:`~pypillometry.GenericEyeData.mask_as_intervals`.
    Detection methods such as
    :meth:`~pypillometry.EyeData.pupil_blinks_detect` store their results
    as Intervals. Several Intervals can be combined with ``+`` or
    :func:`~pypillometry.merge_intervals` and subtracted with ``-``.

    Intervals in sample indices are only valid for the data object they
    came from. Converting between sample indices and time needs that data
    object (its time vector), e.g., ``iv.to_units("ms", d)`` or
    ``iv.to_mask(d)``; conversions between time units work without it.

    Parameters
    ----------
    intervals : list of tuples, np.ndarray or Quantity
        Intervals as list of (start, end) tuples or 2D array; can carry
        units, e.g. ``[(0, 100), (200, 300)] * u.ms`` or
        ``[("1 min", "2 min")]``
    units : str or Unit
        Units of the intervals ("ms", "sec", "min", "h", or "samples" for
        sample indices).
        Can be omitted if `intervals` carry units; otherwise, intervals with
        units are converted to `units`.
    label : str, optional
        Descriptive label for the interval collection
    event_labels : list, optional
        Event labels corresponding to each interval
    event_onsets : list or np.ndarray, optional
        Original event onset times (in same units as intervals)
    event_meta : dict, optional
        Metadata of the event corresponding to each interval (column name
        -> array with one value per interval; see
        :class:`~pypillometry.Events`)

    Examples
    --------
    >>> intervals = Intervals([(0, 100), (200, 300)], units="ms", label="stimulus")
    >>> intervals = Intervals([(0, 100), (200, 300)] * u.ms, label="stimulus")
    >>> len(intervals)
    2
    >>> for start, end in intervals:
    ...     print(f"{start}-{end}")
    """
    
    # state (see pypillometry.state); intervals as (n, 2) array
    _STATE_FIELDS = (("intervals", "pairs"), "units", "label", "event_labels",
                     "event_onsets", "event_meta")

    def __init__(self, intervals, units=UNSET, label=None, event_labels=None,
                 event_onsets=None, event_meta=None):
        """
        Initialize an Intervals object (see class docstring for parameters).

        Raises a ValueError if a start or end is not finite or a start is
        after its end.
        """
        try:
            intervals, self.units = _time_values_and_units(intervals, units)
        except ValueError as e:
            if "sampling rate" not in str(e):
                raise
            raise ValueError("intervals with time units cannot be converted to sample "
                             "indices here; use to_units() with the data object") from None
        if self.units == "samples" and isinstance(intervals, np.ndarray) \
                and intervals.dtype.kind == "f":
            intervals = np.round(intervals).astype(int).tolist()  # converted to indices
        if isinstance(intervals, np.ndarray):
            self.intervals = [tuple(row) for row in intervals]
        else:
            self.intervals = [tuple(i) for i in intervals]
        _check_pairs(self.intervals)

        self.label = label
        for name, col in (("event_labels", event_labels), ("event_onsets", event_onsets)):
            if col is not None and len(col) != len(self.intervals):
                raise ValueError(f"{name} has {len(col)} values for "
                                 f"{len(self.intervals)} intervals")
        self.event_labels = event_labels
        self.event_onsets = event_onsets
        self.event_meta = None if event_meta is None \
            else _meta_dict(event_meta, len(self.intervals))

    def _copy_with(self, **changes):
        """New Intervals with the attributes of this one, except `changes`."""
        kwargs = dict(intervals=self.intervals, units=self.units, label=self.label,
                      event_labels=self.event_labels,
                      event_onsets=self.event_onsets,
                      event_meta=self.event_meta)
        kwargs.update(changes)
        return Intervals(**kwargs)

    def _in_own_units(self, value):
        """`value` (e.g., a duration) in the units of this object; plain numbers are unchanged."""
        if value is None or not _has_units(value):
            return value
        if self.units != "samples":
            if _time_values_and_units(value)[1] == "samples":
                raise ValueError(f"{value} is in samples, but the intervals are in "
                                 f"{self.units}; give it in time units")
            return parse_time(value, to=self.units, default=None)
        try:
            value = np.round(parse_time(value, to="samples", default=None))
        except ValueError:
            raise ValueError(f"{value} has time units, but the intervals are in sample "
                             "indices; give it in samples or convert the intervals "
                             "first, e.g., to_units('ms', d)") from None
        return value.astype(int) if np.ndim(value) else int(value)

    def __len__(self):
        """Return number of intervals."""
        return len(self.intervals)
    
    def __iter__(self):
        """Iterate over intervals."""
        return iter(self.intervals)
    
    def __getitem__(self, idx):
        """Get interval(s) by index."""
        return self.intervals[idx]
    
    def to_mask(self, data) -> np.ndarray:
        """
        Convert intervals to a binary mask array for the samples of `data`.
        
        Parameters
        ----------
        data : GenericEyeData
            data object the intervals belong to (anything with a time
            vector `tx` in ms and a sampling rate `fs`); the mask has one
            entry per sample of `data`
            
        Returns
        -------
        np.ndarray
            Binary array with 1s where intervals exist, 0s elsewhere.
            
        Examples
        --------
        >>> blinks = d.get_blinks("left", "pupil")
        >>> mask = blinks.to_mask(d)
        """
        n = len(data.tx)
        mask = np.zeros(n, dtype=np.int8)
        for start, end in self.to_units("samples", data).intervals:
            mask[min(max(int(start), 0), n):min(max(int(end), 0), n)] = 1
        return mask
    
    @classmethod
    def from_mask(cls, mask: np.ndarray, data=None, units="samples",
                  label: str = None) -> 'Intervals':
        """
        Create Intervals from a binary mask array.
        
        Each contiguous run of non-zero values becomes one interval.

        Parameters
        ----------
        mask : np.ndarray
            Binary array where non-zero values indicate interval membership,
            one entry per sample.
        data : GenericEyeData, optional
            data object the mask belongs to (anything with `tx` and `fs`);
            needed for time units.
        units : str, optional
            Units for the intervals. Default is "samples" (sample indices).
        label : str, optional
            Label for the intervals.
            
        Returns
        -------
        Intervals
            New Intervals object created from the mask.
            
        Examples
        --------
        >>> mask = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        >>> intervals = Intervals.from_mask(mask)
        >>> intervals.intervals
        [(2, 5), (7, 9)]
        """
        mask_bool = np.asarray(mask).astype(bool).astype(int)
        if data is not None and len(mask_bool) != len(data.tx):
            raise ValueError(f"mask has {len(mask_bool)} entries, but the data has "
                             f"{len(data.tx)} samples")
        # Pad with zeros to detect edges
        diff = np.diff(np.concatenate([[0], mask_bool, [0]]))
        # Starts are where diff == 1, ends are where diff == -1
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        intervals = cls(list(zip(starts.tolist(), ends.tolist())), units="samples",
                        label=label)
        return intervals.to_units(units, data)

    @staticmethod
    def _union(pairs):
        """Sorted union of (start, end) pairs."""
        out = []
        for s, e in sorted(pairs):
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        return out

    def __sub__(self, other: 'Intervals') -> 'Intervals':
        """
        Subtract intervals using the - operator.
        
        Removes the regions covered by `other` from `self`; overlapping
        intervals of `self` are merged. Intervals in different time units
        are converted to the units of `self`; Intervals in sample indices
        can only be combined with Intervals in sample indices.
        
        Parameters
        ----------
        other : Intervals
            Intervals to subtract from self
            
        Returns
        -------
        Intervals
            New Intervals object with subtracted regions removed
            
        Raises
        ------
        ValueError
            If one object is in sample indices and the other in time units
            
        Examples
        --------
        >>> a = Intervals([(0, 100), (200, 300)], units="samples")
        >>> b = Intervals([(50, 150)], units="samples")
        >>> c = a - b
        >>> c.intervals  # (0, 50) and (200, 300) remain
        [(0, 50), (200, 300)]
        """
        if not isinstance(other, Intervals):
            return NotImplemented
        other = self._same_units(other)
        removed = self._union(other.intervals)
        result = []
        for s, e in self._union(self.intervals):
            for rs, re in removed:
                if re <= s or rs >= e:
                    continue
                if rs > s:
                    result.append((s, rs))
                s = max(s, re)
                if s >= e:
                    break
            if s < e:
                result.append((s, e))
        return Intervals(result, self.units,
                         label=f"{self.label or 'intervals'} - {other.label or 'intervals'}")

    def _same_units(self, other):
        """`other` converted to the units of self (only between time units)."""
        if self.units == other.units:
            return other
        if self.units == "samples" or other.units == "samples":
            raise ValueError(
                f"Cannot combine Intervals with units={self.units} and units={other.units}. "
                "Both must have time units or both must be indices; convert them "
                "with to_units(units, d).")
        return other.to_units(self.units)
    
    def __add__(self, other: 'Intervals') -> 'Intervals':
        """
        Combine two Intervals objects using the + operator.
        
        The intervals are concatenated and metadata is merged appropriately:
        - Units: other is converted to self's units if different
        - label: combined with ' + '
        - event_labels/onsets/meta: concatenated
        
        Parameters
        ----------
        other : Intervals
            Another Intervals object to combine with
            
        Returns
        -------
        Intervals
            New Intervals object containing intervals from both
            
        Raises
        ------
        TypeError
            If other is not an Intervals object
        ValueError
            If units are incompatible (one is "samples")
            
        Examples
        --------
        >>> a = Intervals([(0, 100), (200, 300)], units="ms")
        >>> b = Intervals([(500, 600)], units="ms")
        >>> c = a + b
        >>> len(c)
        3
        >>> # Different units are converted
        >>> a = Intervals([(0, 1)], units="sec")
        >>> b = Intervals([(2000, 3000)], units="ms")
        >>> c = a + b  # b converted to seconds
        >>> c.intervals
        [(0, 1), (2.0, 3.0)]
        """
        if not isinstance(other, Intervals):
            return NotImplemented
        other_converted = self._same_units(other)
        
        # Combine intervals
        combined_intervals = list(self.intervals) + list(other_converted.intervals)
        
        # Combine labels
        if self.label and other.label:
            combined_label = f"{self.label} + {other.label}"
        else:
            combined_label = self.label or other.label
        
        labels, onsets, meta = _concat_rows([self, other_converted])
        return Intervals(combined_intervals, self.units, label=combined_label,
                         event_labels=labels, event_onsets=onsets, event_meta=meta)
    
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """
        Convert intervals to numpy array.

        Parameters
        ----------
        dtype : data-type, optional
            dtype of the returned array (as in ``np.asarray(intervals, dtype=int)``)
        copy : bool or None, optional
            numpy 2 copy keyword; the array is always newly created, so
            `False` (no copy allowed) raises a `ValueError`

        Returns
        -------
        np.ndarray
            Array with shape (n, 2) containing intervals in current units
            
        Examples
        --------
        >>> intervals = data.get_intervals(("stim", "resp"), units="ms")
        >>> arr = np.array(intervals)
        """
        if copy is False:
            raise ValueError("Intervals cannot be converted to an array without a copy")
        return np.array(self.intervals, dtype=dtype)

    def _positions_to_ms(self, values, data):
        """Absolute positions (in own units) -> ms."""
        values = np.asarray(values, dtype=float)
        if self.units == "samples":
            return _samples_to_ms(data, values)
        return values * _MS_PER_UNIT[self.units]

    def to_units(self, target_units: str, data=None) -> 'Intervals':
        """
        Convert intervals to different time units or indices.
        
        Conversions between time units ("ms", "sec", "min", "h") only
        rescale the values. Conversions between sample indices and time
        use the time vector of `data`, the data object the intervals belong
        to: a sample index stands for the time of that sample, and a time
        is converted to the first sample at or after it (so ``(start, end)``
        covers the samples with ``start <= t < end``).

        Parameters
        ----------
        target_units : str
            Target units: "ms", "sec", "min", "h", or "samples" for sample
            indices (aliases such as "seconds" or "indices" work, too).
        data : GenericEyeData, optional
            data object the intervals belong to (anything with a time vector
            `tx` in ms and a sampling rate `fs`); needed to convert between
            sample indices and time
            
        Returns
        -------
        Intervals
            New Intervals object with converted units
            
        Raises
        ------
        ValueError
            If the conversion involves sample indices and `data` is not given
        
        Examples
        --------
        >>> intervals = data.get_intervals(("stim", "resp"), units="ms")
        >>> intervals_sec = intervals.to_units("sec")
        >>> intervals_seconds = intervals.to_units("seconds")  # alias works too
        
        Convert between indices and time units (needs the data object):
        
        >>> blinks = data.get_blinks("left", "pupil")
        >>> blinks_ms = blinks.to_units("ms", data)
        >>> blinks_idx = blinks_ms.to_units("samples", data)
        """
        # Normalize target units (source units already normalized in __init__)
        target_units = normalize_unit(target_units)
        
        if self.units == target_units:
            return self

        if self.units != "samples" and target_units != "samples":
            # between time units: rescale
            fac = _MS_PER_UNIT[self.units] / _MS_PER_UNIT[target_units]
            return self._copy_with(
                intervals=[(s * fac, e * fac) for s, e in self.intervals],
                units=target_units,
                event_onsets=None if self.event_onsets is None
                else np.asarray(self.event_onsets) * fac)

        if data is None:
            raise ValueError(
                "Converting between sample indices and time needs the data object "
                "the intervals belong to, e.g., intervals.to_units('ms', d)")

        # absolute positions via ms
        starts_ends = self._positions_to_ms(np.reshape(self.intervals, (-1, 2)), data)
        onsets = None if self.event_onsets is None \
            else self._positions_to_ms(self.event_onsets, data)
        if target_units == "samples":
            converted = _ms_to_samples(data, starts_ends).reshape(-1, 2)
            if onsets is not None:
                onsets = _ms_to_samples(data, onsets)
        else:
            fac = 1.0 / _MS_PER_UNIT[target_units]
            converted = starts_ends * fac
            if onsets is not None:
                onsets = onsets * fac
        return self._copy_with(
            intervals=[tuple(row) for row in np.asarray(converted).tolist()],
            units=target_units, event_onsets=onsets)
    
    def merge(self, merge_sep='_', distance: float = 0):
        """
        Merge overlapping or close intervals.
        
        When intervals are merged, metadata is handled as follows:
        - event_labels: Labels of merged intervals are joined with merge_sep
        - event_onsets: First onset of merged group is kept
        - event_meta: Metadata of the first interval of merged group is kept
        
        Parameters
        ----------
        merge_sep : str, optional
            Separator to use when joining event labels of merged intervals. 
            Default is '_'.
        distance : float, str or Quantity, optional
            Maximum gap between intervals to still merge them, in the units
            of the Intervals object unless it carries units (e.g.,
            ``50 * u.ms``; for intervals in sample indices, convert them
            to time units first or give the distance in samples).
            If two intervals are separated by less than or equal to this distance,
            they will be merged. Default is 0 (only merge overlapping intervals).
            
            Example: If distance=2 and intervals are (10, 15) and (17, 22), they
            will be merged into (10, 22) because the gap (17-15=2) is <= distance.
        
        Returns
        -------
        Intervals
            New Intervals object with merged intervals and updated metadata.
            
        Raises
        ------
        ValueError
            If `distance` has time units and the intervals are in sample
            indices (or the other way round).
        """
        if not self.intervals:
            return self
        distance = self._in_own_units(distance)
        
        # Get indices that would sort intervals by start point
        sorted_indices = sorted(range(len(self.intervals)), 
                               key=lambda i: self.intervals[i][0])
        sorted_intervals = [self.intervals[i] for i in sorted_indices]
        
        # Track which original intervals belong to each merged interval
        merged = [sorted_intervals[0]]
        merged_groups = [[sorted_indices[0]]]  # Track original indices in each group
        
        for idx, current in zip(sorted_indices[1:], sorted_intervals[1:]):
            last_merged = merged[-1]
            
            # Check if intervals overlap or are within distance
            if current[0] <= last_merged[1] + distance:
                # Merge the intervals
                merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
                merged_groups[-1].append(idx)  # Add to current merge group
            else:
                # No overlap and beyond distance, add as separate interval
                merged.append(current)
                merged_groups.append([idx])  # Start new merge group
        
        # Build merged metadata
        merged_labels = None
        merged_onsets = None
        
        if self.event_labels is not None:
            merged_labels = []
            for group in merged_groups:
                group_labels = [str(self.event_labels[i]) for i in group]
                merged_labels.append(merge_sep.join(group_labels))
        
        if self.event_onsets is not None:
            # Keep the first onset from each merge group
            merged_onsets = [self.event_onsets[group[0]] for group in merged_groups]
        
        merged_meta = None
        if self.event_meta is not None:
            first = [group[0] for group in merged_groups]
            merged_meta = {name: col[first] for name, col in self.event_meta.items()}

        return Intervals(merged, self.units, self.label, 
                        event_labels=merged_labels,
                        event_onsets=merged_onsets,
                        event_meta=merged_meta)

    def complement(self, data) -> 'Intervals':
        """
        Return the periods of the recording not covered by the intervals.

        These are the gaps between the intervals and before the first and
        after the last one, within the whole recording. Overlapping
        intervals are merged first. Event information (labels, onsets) is
        not kept.

        Parameters
        ----------
        data : GenericEyeData
            data object the intervals belong to; the complement is taken
            within the whole recording, from the first sample to the end of
            the last one (one sample past it, as intervals are
            end-exclusive)

        Returns
        -------
        Intervals
            the periods not covered, in the units of this object

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.EyeData(left_pupil=np.ones(100), sampling_rate=100)
        >>> iv = Intervals([(10, 20), (50, 60)], units="samples")
        >>> iv.complement(d).intervals
        [(0, 10), (20, 50), (60, 100)]
        """
        n = len(data.tx)
        if self.units == "samples":
            full = (0, n)
        else:
            fac = 1.0 / _MS_PER_UNIT[self.units]
            full = (float(data.tx[0]) * fac, float(_samples_to_ms(data, n)) * fac)
        result = Intervals([full], self.units) - self
        result.label = f"not {self.label}" if self.label else "complement"
        return result

    def pad(self, left: float = 0, right: float = 0):
        """
        Add padding to the left (start) and right (end) of each interval.
        
        The padding is in the current units of the Intervals object, unless
        it carries units (e.g., ``left=50 * u.ms``).
        Padding is applied independently to each interval without merging.
        Use `.merge()` after padding if you want to merge overlapping intervals.
        The intervals are not clipped to the data (e.g., a padded interval
        in sample indices may start before 0); the methods that read the
        data handle the parts outside of it.
        
        Parameters
        ----------
        left : float, optional
            Amount to subtract from the start of each interval (expand left).
            Default is 0.
        right : float, optional
            Amount to add to the end of each interval (expand right).
            Default is 0.
            
        Returns
        -------
        Intervals
            New Intervals object with padded intervals.
            
        Examples
        --------
        >>> intervals = Intervals([(100, 200), (300, 400)], units="ms")
        >>> padded = intervals.pad(left=10, right=20)
        >>> padded.intervals
        [(90, 220), (290, 420)]
        
        >>> # Pad and then merge overlapping intervals
        >>> intervals.pad(left=50, right=50).merge()
        
        Notes
        -----
        - Negative padding values will shrink intervals (left adds to start, 
          right subtracts from end); a ValueError is raised if an interval
          would end before it starts
        """
        if not self.intervals:
            return self
        left, right = self._in_own_units(left), self._in_own_units(right)
        
        for i, (start, end) in enumerate(self.intervals):
            if start - left > end + right:
                raise ValueError(
                    f"padding (left={left}, right={right}) would make interval {i} "
                    f"({start}, {end}) end before it starts")

        padded = [(start - left, end + right) for start, end in self.intervals]
        
        return Intervals(
            padded, 
            self.units, 
            self.label,
            event_labels=self.event_labels,
            event_onsets=self.event_onsets,
            event_meta=self.event_meta
        )
    
    def stats(self):
        """
        Get statistics about interval durations.
        
        Returns
        -------
        dict
            Dictionary with summary statistics (n, mean, sd, min, max, total_duration, units);
            for empty Intervals, mean/sd/min/max are NaN and total_duration is 0
        """
        durations = [i[1] - i[0] for i in self.intervals]
        if len(durations) == 0:
            return {
                "n": 0,
                "mean": np.nan,
                "sd": np.nan,
                "min": np.nan,
                "max": np.nan,
                "total_duration": 0.0,
                "units": self.units
            }
        return {
            "n": len(self.intervals),
            "mean": np.mean(durations),
            "sd": np.std(durations),
            "min": np.min(durations),
            "max": np.max(durations),
            "total_duration": np.sum(durations),
            "units": self.units
        }

    def filter(self, 
               selector: callable = None,
               min_duration: float = None,
               max_duration: float = None,
               time_range: tuple = None) -> 'Intervals':
        """
        Filter intervals and return a new Intervals object.
        
        This method supports multiple filtering criteria that can be combined:
        - Custom function: applied to each interval
        - Duration constraints: minimum and/or maximum duration
        - Time range: intervals must overlap with the specified range
        
        All specified criteria are combined with AND logic.
        
        Parameters
        ----------
        selector : callable, optional
            Function called as ``selector(start, end)`` for each interval that
            returns bool: True to keep the interval, False to discard it.
        min_duration : float, str or Quantity, optional
            Minimum duration of intervals to keep (in current units, unless
            it carries units, e.g. ``"50 ms"``)
        max_duration : float, str or Quantity, optional
            Maximum duration of intervals to keep (as `min_duration`)
        time_range : tuple, optional
            (start, end) time range (in current units, unless it carries
            units, e.g. ``(1, 2) * u.min``). Only keeps intervals that overlap with this range.
            An interval overlaps if its end > range_start AND its start < range_end.
            
        Returns
        -------
        Intervals
            New Intervals object containing only the filtered intervals
            
        Examples
        --------
        >>> intervals = Intervals([(0, 100), (200, 400), (500, 550)], units="ms")
        
        >>> # Filter by duration
        >>> short = intervals.filter(max_duration=100)
        >>> len(short)
        2
        
        >>> # Filter by minimum duration
        >>> long = intervals.filter(min_duration=150)
        >>> len(long)
        1
        
        >>> # Filter by time range (keeps intervals overlapping with range)
        >>> middle = intervals.filter(time_range=(150, 450))
        
        >>> # Filter by custom function
        >>> starts_at_multiple_of_100 = intervals.filter(
        ...     selector=lambda start, end: start % 100 == 0
        ... )
        
        >>> # Combine multiple criteria
        >>> filtered = intervals.filter(
        ...     min_duration=50,
        ...     max_duration=200,
        ...     time_range=(0, 500)
        ... )
        """
        if len(self.intervals) == 0:
            return Intervals(
                [],
                units=self.units,
                label=self.label,
                event_labels=None,
                event_onsets=None
            )
        
        min_duration = self._in_own_units(min_duration)
        max_duration = self._in_own_units(max_duration)
        time_range = self._in_own_units(time_range)
        mask = np.ones(len(self.intervals), dtype=bool)
        
        # Apply duration filters
        if min_duration is not None or max_duration is not None:
            durations = np.array([end - start for start, end in self.intervals])
            if min_duration is not None:
                mask &= durations >= min_duration
            if max_duration is not None:
                mask &= durations <= max_duration
        
        # Apply time range filter
        if time_range is not None:
            if len(time_range) != 2:
                raise ValueError("time_range must be a tuple of (start, end)")
            range_start, range_end = time_range
            if range_start >= range_end:
                raise ValueError(f"time_range start must be < end, got {range_start} >= {range_end}")
            # Keep intervals that overlap with the range
            for i, (start, end) in enumerate(self.intervals):
                # Overlap: interval end > range start AND interval start < range end
                overlaps = (end > range_start) and (start < range_end)
                mask[i] &= overlaps
        
        # Apply custom selector function
        if selector is not None:
            if not callable(selector):
                raise TypeError(f"selector must be callable, got {type(selector)}")
            for i, (start, end) in enumerate(self.intervals):
                mask[i] &= bool(selector(start, end))
        
        # Build filtered data using boolean mask
        indices = np.where(mask)[0]
        filtered_intervals = [self.intervals[i] for i in indices]
        
        filtered_event_labels = None
        if self.event_labels is not None:
            filtered_event_labels = [self.event_labels[i] for i in indices]
        
        filtered_event_onsets = None
        if self.event_onsets is not None:
            filtered_event_onsets = np.array([self.event_onsets[i] for i in indices])

        filtered_meta = None
        if self.event_meta is not None:
            filtered_meta = {name: col[indices] for name, col in self.event_meta.items()}
        
        return Intervals(
            filtered_intervals,
            units=self.units,
            label=self.label,
            event_labels=filtered_event_labels,
            event_onsets=filtered_event_onsets,
            event_meta=filtered_meta
        )

    @requires_package("pandas")
    def as_pandas(self):
        """
        Represent intervals as a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``start``, ``end``, and ``duration``. When metadata
            is available, columns ``event_label`` and ``event_onset``
            are included, and the column ``units`` gives the units. The
            metadata of the events (`event_meta`) are added as columns.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        import pandas as pd

        records = []
        for idx, (start, end) in enumerate(self.intervals):
            record = {
                "start": start,
                "end": end,
                "duration": end - start,
            }
            record["units"] = self.units
            if self.event_labels is not None and idx < len(self.event_labels):
                record["event_label"] = self.event_labels[idx]
            if self.event_onsets is not None and idx < len(self.event_onsets):
                record["event_onset"] = self.event_onsets[idx]
            for name, col in (self.event_meta or {}).items():
                record[name] = col[idx].item()
            records.append(record)

        return pd.DataFrame(records)
    
    def plot(self, show_labels: bool = True, **kwargs):
        """
        Plot intervals as horizontal lines on a timeline.
        
        Uses the current matplotlib axes to create a visualization where each 
        interval is shown as a horizontal line at a different y-level.
        
        Parameters
        ----------
        show_labels : bool
            Whether to display event labels for each interval (default True)
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plot()
            
        Returns
        -------
        None
            Modifies the current axes in place

        Examples
        --------
        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> intervals = d.get_intervals("S_", (-500, 2500) * u.ms).to_intervals("sec")
           >>> plt.figure(figsize=(8, 3))
           >>> intervals.plot()
        """
        import matplotlib.pyplot as plt
        
        if len(self.intervals) == 0:
            return
        
        ax = plt.gca()
        
        # x-axis range of the intervals
        all_values = [val for interval in self.intervals for val in interval]
        x_min, x_max = min(all_values), max(all_values)
        
        # Set x-axis limits with some padding
        padding = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - padding, x_max + padding)
        
        # Plot each interval as a horizontal line at different y-levels
        for i, (start, end) in enumerate(self.intervals):
            y_level = i + 1
            
            # Plot horizontal line for the interval
            # Only use label for the first interval to avoid duplicate legend entries
            plot_kwargs = kwargs.copy()
            if i > 0 and 'label' in plot_kwargs:
                del plot_kwargs['label']
            ax.plot([start, end], [y_level, y_level], **plot_kwargs)
            
            # Optionally add event label centered on the line
            if show_labels and self.event_labels is not None and i < len(self.event_labels):
                # Place label at the middle of the interval
                label_x = (start + end) / 2
                ax.text(label_x, y_level, self.event_labels[i], 
                       rotation=90,
                       color='grey',
                       verticalalignment='center',
                       horizontalalignment='center',
                       fontsize=8)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, len(self.intervals) + 1)
        ax.set_ylabel('Interval #')
        
        # Set x-axis label based on units
        if self.units != "samples":
            ax.set_xlabel(f'Time ({self.units})')
        else:
            ax.set_xlabel('Samples')
        
        # Set title from global label
        if self.label is not None:
            ax.set_title(self.label)
    
    def plot_highlight(self, ax=None, color='lightblue', alpha=0.3, **kwargs):
        """
        Draw highlighted backgrounds for each interval on existing plot(s).
        
        Uses matplotlib's axvspan to draw semi-transparent vertical spans
        for each interval. This is useful for highlighting regions of interest
        on top of timeseries plots.
        
        Parameters
        ----------
        ax : Axes, list of Axes, Figure, or None
            Where to draw highlights:
            - None: Apply to ALL axes in current figure (useful after plot_timeseries)
            - Single Axes: Apply only to that axes
            - List of Axes: Apply to each axes in the list
            - Figure: Apply to all axes in that figure
        color : str
            Color for the highlighted regions (default 'lightblue')
        alpha : float
            Transparency level, 0-1 (default 0.3)
        **kwargs : dict
            Additional keyword arguments passed to ax.axvspan()
            (e.g., label, linestyle, edgecolor, zorder); `zorder` defaults
            to 0 (behind the data)
            
        Returns
        -------
        list
            List of matplotlib patches created
            
        Notes
        -----
        Make sure the intervals are in the same units as the plot's x-axis.
        Use `intervals.to_units("sec")` to convert if needed.

        Examples
        --------
        Highlight the first second after each stimulus on a pupil plot:

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> intervals = d.get_intervals("S_", (0, 1000) * u.ms).to_intervals("sec")
           >>> plt.figure(figsize=(10, 3))
           >>> d.plot.pupil_plot(eyes="left", plot_range=(0, 20) * u.sec,
           ...                   show_events=False)
           >>> intervals.plot_highlight(color="green", alpha=0.2)
           >>> plt.xlim(0, 20)

        After :meth:`~pypillometry.GenericPlotter.plot_timeseries`, the intervals
        are highlighted in all subplots; use ``ax=`` to highlight only one of them.
        """
        import matplotlib.pyplot as plt
        
        if len(self.intervals) == 0:
            return []
        
        # Determine which axes to use
        if ax is None:
            # Default: all axes in current figure
            axes_list = plt.gcf().axes
        elif isinstance(ax, plt.Figure):
            axes_list = ax.axes
        elif hasattr(ax, '__iter__') and not isinstance(ax, plt.Axes):
            axes_list = list(ax)
        else:
            axes_list = [ax]
        
        if not axes_list:
            return []
        
        kwargs.setdefault("zorder", 0)
        patches = []
        for axes in axes_list:
            for start, end in self.intervals:
                patch = axes.axvspan(start, end, color=color, alpha=alpha, **kwargs)
                patches.append(patch)
        
        return patches
    
    def __repr__(self):
        """
        String representation with statistics.
        
        Returns
        -------
        str
            Formatted string with label, statistics, and units
        """
        parts = []
        
        # Add label if present
        if self.label is not None:
            parts.append(f"Intervals '{self.label}'")
        else:
            parts.append("Intervals")
        
        # Add statistics
        if len(self.intervals) > 0:
            stats = self.stats()
            parts.append(_format_interval_stats(stats))
        else:
            parts.append("0 intervals")

        return " | ".join(parts)


def _format_interval_stats(stats: dict) -> str:
    """
    Format interval statistics as a human-readable string.
    
    Parameters
    ----------
    stats : dict
        Dictionary with keys: n, mean, sd, min, max, total_duration
        
    Returns
    -------
    str
        Formatted string like "3 intervals (total duration: 600.00), 200.00 +/- 81.65, [100.00, 300.00]"
    """
    n = stats.get("n", 0)
    mean = stats.get("mean", np.nan)
    sd = stats.get("sd", np.nan)
    minv = stats.get("min", np.nan)
    maxv = stats.get("max", np.nan)
    total_duration = stats.get("total_duration", np.nan)
    units = stats.get("units")
    return "%i intervals (total duration: %.2f %s), %.2f %s +/- %.2f %s, [%.2f %s, %.2f %s]" % (
        n, total_duration, units, mean, units, sd, units, minv, units, maxv, units
    )

