"""Functions to handle intervals (blinks, erpds, etc.) in eyetracking data.
"""

import numpy as np

from pypillometry.convenience import requires_package, normalize_unit

class IntervalStats(dict):
    """
    A dictionary with a specialized repr() function
    to display summary statistics of intervals.
    """
    def __repr__(self):
        n=self["n"] if "n" in self else 0
        mean=self["mean"] if "mean" in self else np.nan
        sd=self["sd"] if "sd" in self else np.nan
        minv=self["min"] if "min" in self else np.nan
        maxv=self["max"] if "max" in self else np.nan
        r = "%i intervals, %.2f +/- %.2f, [%.2f, %.2f]" % (n, mean, sd, minv, maxv)
        return r


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
    
    >>> intervals_list = [intervals1, intervals2, intervals3]
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
    
    # Collect all intervals and metadata
    all_intervals = []
    all_event_labels = []
    all_event_indices = []
    all_event_onsets = []
    has_labels = False
    has_indices = False
    has_onsets = False
    
    for obj in intervals_list:
        all_intervals.extend(obj.intervals)
        
        # Collect metadata if present in any object
        if obj.event_labels is not None:
            has_labels = True
            all_event_labels.extend(obj.event_labels)
        else:
            # Pad with None for this object's intervals
            all_event_labels.extend([None] * len(obj.intervals))
        
        if obj.event_indices is not None:
            has_indices = True
            all_event_indices.extend(obj.event_indices)
        else:
            all_event_indices.extend([None] * len(obj.intervals))
        
        if obj.event_onsets is not None:
            has_onsets = True
            all_event_onsets.extend(obj.event_onsets)
        else:
            all_event_onsets.extend([None] * len(obj.intervals))
    
    if not all_intervals:
        # All were empty
        return Intervals([], units=first_units, label=label,
                        data_time_range=intervals_list[0].data_time_range)
    
    # Use data_time_range from first object if available
    data_time_range = intervals_list[0].data_time_range if intervals_list[0].data_time_range is not None else None
    
    return Intervals(
        all_intervals,
        units=first_units,
        label=label,
        event_labels=all_event_labels if has_labels else None,
        event_indices=all_event_indices if has_indices else None,
        event_onsets=all_event_onsets if has_onsets else None,
        data_time_range=data_time_range
    )

def get_interval_stats(intervals):
    """
    Calculate summary statistics of intervals.
    
    Parameters
    ----------
    intervals: list of tuples
        list of intervals
    
    Returns
    -------
    IntervalStats
        dictionary with summary statistics
    """
    stats=IntervalStats()
    stats["n"]=len(intervals)

    durations = [i[1]-i[0] for i in intervals]
    stats["mean"]=np.mean(durations)
    stats["sd"]=np.std(durations)
    stats["min"]=np.min(durations)
    stats["max"]=np.max(durations)
    return stats


def stat_event_interval(tx,sy,intervals,statfct=np.mean):
    """
    Return result of applying a statistical function to pupillometric data in a
    given interval relative to event-onsets. For example, extract mean 
    pupil-size in interval before trial onset.
        
    Parameters
    -----------
    
    tx : np.ndarray
        time-vector in milliseconds        
    sy : np.ndarray
        (baseline-corrected) pupil signal
    intervals : list of tuples (min,max)
        time-window in ms relative to event-onset (0 is event-onset)
    statfct : function
        function mapping np.array to a single number
    
    Returns
    --------
    
    result: np.array
        number of event-onsets long result array
    """
    intervals=np.array(intervals)
    starts=intervals[:,0]
    ends=intervals[:,1]

    res=np.zeros(len(starts))

    for i,interv in enumerate(zip(starts,ends)):
        start_ix=np.argmin(np.abs(interv[0]-tx))
        end_ix=np.argmin(np.abs(interv[1]-tx))
        if start_ix==end_ix:
            end_ix+=1
        res[i]=statfct(sy[start_ix:end_ix])
    return res


class Intervals:
    """
    Container for intervals with metadata.
    
    This class wraps a list of (start, end) intervals and provides additional
    metadata such as units, labels, and associated event information.
    
    Attributes
    ----------
    intervals : list of tuples
        List of (start, end) tuples representing intervals
    units : str or None
        Units of the intervals ("ms", "sec", "min", "h", or None for indices)
    label : str or None
        Optional descriptive label for the interval collection
    event_labels : list or None
        Event labels corresponding to each interval
    event_indices : np.ndarray or None
        Event indices corresponding to each interval
        
    Examples
    --------
    >>> intervals = Intervals([(0, 100), (200, 300)], units="ms", label="stimulus")
    >>> len(intervals)
    2
    >>> for start, end in intervals:
    ...     print(f"{start}-{end}")
    """
    
    def __init__(self, intervals, units, label=None, event_labels=None, event_indices=None, data_time_range=None, event_onsets=None):
        """
        Initialize an Intervals object.
        
        Parameters
        ----------
        intervals : list of tuples or np.ndarray
            Intervals as list of (start, end) tuples or 2D array
        units : str or None
            Units of the intervals
        label : str, optional
            Descriptive label for the intervals
        event_labels : list, optional
            Labels for each interval
        event_indices : np.ndarray, optional
            Indices for each interval
        data_time_range : tuple, optional
            Time range (min, max) of the original dataset
        event_onsets : list or np.ndarray, optional
            Original event onset times (in same units as intervals)
        """
        if isinstance(intervals, np.ndarray):
            self.intervals = [tuple(row) for row in intervals]
        else:
            self.intervals = [tuple(i) for i in intervals]
        
        # Normalize units to canonical form (handles aliases)
        self.units = normalize_unit(units)
        
        self.label = label
        self.event_labels = event_labels
        self.event_indices = event_indices
        self.data_time_range = data_time_range
        self.event_onsets = event_onsets
    
    def __len__(self):
        """Return number of intervals."""
        return len(self.intervals)
    
    def __iter__(self):
        """Iterate over intervals."""
        return iter(self.intervals)
    
    def __getitem__(self, idx):
        """Get interval(s) by index."""
        return self.intervals[idx]
    
    def to_array(self):
        """
        Convert intervals to numpy array.
        
        Returns
        -------
        np.ndarray
            Array with shape (n, 2) containing intervals
        """
        return np.array(self.intervals)
    
    def __array__(self) -> np.ndarray:
        """
        Convert intervals to numpy array.
        
        Returns
        -------
        np.ndarray
            Array with shape (n, 2) containing intervals in current units
            
        Examples
        --------
        >>> intervals = data.get_intervals("stim", units="ms")
        >>> arr = np.array(intervals)
        """
        return np.array(self.intervals)
    
    def as_index(self, eyedata_obj) -> np.ndarray:
        """
        Convert intervals to integer indices for array indexing.
        
        Parameters
        ----------
        eyedata_obj : GenericEyeData
            EyeData object with time array for conversion
            
        Returns
        -------
        np.ndarray
            Array with shape (n, 2) and dtype=int containing interval indices
            
        Examples
        --------
        >>> intervals = data.get_intervals("stim", interval=(-200, 200), units="ms")
        >>> indices = intervals.as_index(data)
        >>> for start, end in indices:
        ...     data.tx[start:end]
        """
        if self.units is None:
            return np.array(self.intervals, dtype=int)
        
        # Define conversion factors to milliseconds
        units_to_ms = {"ms": 1.0, "sec": 1000.0, "min": 60000.0, "h": 3600000.0}
        
        indices = []
        for start, end in self.intervals:
            if self.units == "ms":
                start_ms, end_ms = start, end
            else:
                start_ms = start * units_to_ms[self.units]
                end_ms = end * units_to_ms[self.units]
            
            start_ix = np.argmin(np.abs(eyedata_obj.tx - start_ms))
            end_ix = np.argmin(np.abs(eyedata_obj.tx - end_ms))
            indices.append((start_ix, end_ix))
        
        return np.array(indices, dtype=int)
    
    def to_units(self, target_units: str|None) -> 'Intervals':
        """
        Convert intervals to different time units.
        
        Parameters
        ----------
        target_units : str
            Target units: "ms", "sec", "min", or "h"
            
        Returns
        -------
        Intervals
            New Intervals object with converted units
            
        Raises
        ------
        ValueError
            If current or target units are None (indices)
        
        Examples
        --------
        >>> intervals = data.get_intervals("stim", units="ms")
        >>> intervals_sec = intervals.to_units("sec")
        >>> intervals_seconds = intervals.to_units("seconds")  # alias works too
        """
        if self.units is None:
            raise ValueError(
                "Cannot convert from indices (units=None). "
                "Use get_intervals(units='ms') or get_blinks(units='ms') instead."
            )
        
        if target_units is None:
            raise ValueError(
                "Cannot convert to indices. Use intervals.as_index(eyedata_obj) instead."
            )
        
        # Normalize target units (source units already normalized in __init__)
        target_units = normalize_unit(target_units)
        
        if self.units == target_units:
            return self
        
        units_to_ms = {"ms": 1.0, "sec": 1000.0, "min": 60000.0, "h": 3600000.0}
        
        if self.units not in units_to_ms:
            raise ValueError(f"Unknown source units: {self.units}")
        if target_units not in units_to_ms:
            raise ValueError(f"Unknown target units: {target_units}")
        
        fac = units_to_ms[self.units] / units_to_ms[target_units]
        converted = [(s * fac, e * fac) for s, e in self.intervals]
        # convert data_time_range to target units if it is not None
        if self.data_time_range is not None:
            data_time_range = (self.data_time_range[0] * fac, self.data_time_range[1] * fac)
        else:
            data_time_range = None
        # convert event_onsets to target units if it is not None
        if self.event_onsets is not None:
            event_onsets = np.array(self.event_onsets) * fac
        else:
            event_onsets = None
        return Intervals(converted, target_units, self.label,
                        self.event_labels, self.event_indices,
                        data_time_range, event_onsets)
    
    def merge(self, merge_sep='_'):
        """
        Merge overlapping intervals.
        
        When intervals are merged, metadata is handled as follows:
        - event_labels: Labels of merged intervals are joined with merge_sep
        - event_indices: First index of merged group is kept
        - event_onsets: First onset of merged group is kept
        
        Parameters
        ----------
        merge_sep : str, optional
            Separator to use when joining event labels of merged intervals. 
            Default is '_'.
        
        Returns
        -------
        Intervals
            New Intervals object with merged intervals and updated metadata.
        """
        if not self.intervals:
            return self
        
        # Get indices that would sort intervals by start point
        sorted_indices = sorted(range(len(self.intervals)), 
                               key=lambda i: self.intervals[i][0])
        sorted_intervals = [self.intervals[i] for i in sorted_indices]
        
        # Track which original intervals belong to each merged interval
        merged = [sorted_intervals[0]]
        merged_groups = [[sorted_indices[0]]]  # Track original indices in each group
        
        for idx, current in zip(sorted_indices[1:], sorted_intervals[1:]):
            last_merged = merged[-1]
            
            # Check if intervals overlap
            if current[0] <= last_merged[1]:
                # Merge the intervals
                merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
                merged_groups[-1].append(idx)  # Add to current merge group
            else:
                # No overlap, add the current interval
                merged.append(current)
                merged_groups.append([idx])  # Start new merge group
        
        # Build merged metadata
        merged_labels = None
        merged_indices = None
        merged_onsets = None
        
        if self.event_labels is not None:
            merged_labels = []
            for group in merged_groups:
                group_labels = [str(self.event_labels[i]) for i in group]
                merged_labels.append(merge_sep.join(group_labels))
        
        if self.event_indices is not None:
            # Keep the first index from each merge group
            merged_indices = [self.event_indices[group[0]] for group in merged_groups]
        
        if self.event_onsets is not None:
            # Keep the first onset from each merge group
            merged_onsets = [self.event_onsets[group[0]] for group in merged_groups]
        
        return Intervals(merged, self.units, self.label, 
                        event_labels=merged_labels,
                        event_indices=merged_indices,
                        data_time_range=self.data_time_range,
                        event_onsets=merged_onsets)
    
    def stats(self):
        """
        Get statistics about interval durations.
        
        Returns
        -------
        IntervalStats
            Dictionary with summary statistics (n, mean, sd, min, max)
        """
        return get_interval_stats(self.intervals)


    @requires_package("pandas")
    def as_pandas(self):
        """
        Represent intervals as a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``start``, ``end``, and ``duration``. When metadata
            is available, columns ``event_label``, ``event_index``, and ``event_onset``
            are included. If the intervals have units, the column ``units`` is added.

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
            if self.units is not None:
                record["units"] = self.units
            if self.event_labels is not None and idx < len(self.event_labels):
                record["event_label"] = self.event_labels[idx]
            if self.event_indices is not None and idx < len(self.event_indices):
                record["event_index"] = self.event_indices[idx]
            if self.event_onsets is not None and idx < len(self.event_onsets):
                record["event_onset"] = self.event_onsets[idx]
            records.append(record)

        return pd.DataFrame(records)
    
    def plot(self, show_labels: bool = True, **kwargs):
        """
        Plot intervals as horizontal lines on a timeline.
        
        Uses the current matplotlib axes to create a visualization where each 
        interval is shown as a horizontal black line at a different y-level.
        
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
        >>> intervals = data.get_intervals("F", interval=(-200, 200), units="ms")
        >>> plt.figure()
        >>> intervals.plot()
        >>> plt.show()
        """
        import matplotlib.pyplot as plt
        
        if len(self.intervals) == 0:
            return
        
        ax = plt.gca()
        
        # Determine x-axis range
        if self.data_time_range is not None:
            x_min, x_max = self.data_time_range
        else:
            # Fallback: use min/max of intervals
            all_values = [val for interval in self.intervals for val in interval]
            x_min, x_max = min(all_values), max(all_values)
        
        # Set x-axis limits with some padding
        padding = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - padding, x_max + padding)
        
        # Plot each interval as a horizontal line at different y-levels
        for i, (start, end) in enumerate(self.intervals):
            y_level = i + 1
            
            # Plot horizontal black line for the interval
            ax.plot([start, end], [y_level, y_level], **kwargs)
            
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
        if self.units is not None:
            ax.set_xlabel(f'Time ({self.units})')
        else:
            ax.set_xlabel('Time (indices)')
        
        # Set title from global label
        if self.label is not None:
            ax.set_title(self.label)
    
    def __repr__(self):
        """
        String representation using IntervalStats for nice formatting.
        
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
        
        # Add statistics using IntervalStats
        if len(self.intervals) > 0:
            stats = self.stats()
            parts.append(str(stats))
        else:
            parts.append("0 intervals")
        
        # Add units info
        if self.units is not None:
            parts.append(f"units={self.units}")
        else:
            parts.append("units=None (indices)")
        
        return " | ".join(parts)
