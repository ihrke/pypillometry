"""Functions to handle intervals (blinks, erpds, etc.) in eyetracking data.
"""

import numpy as np

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


def merge_intervals(intervals):
    """Merge overlapping intervals.

    Parameters
    ----------
    intervals : list of tuples
        List of intervals to merge.

    Returns
    -------
    list of tuples
        Merged intervals.
    """
    if not intervals:
        return []
    
    # Sort intervals based on the start point
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        
        # Check if intervals overlap
        if current[0] <= last_merged[1]:
            # Merge the intervals
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            # No overlap, add the current interval
            merged.append(current)
    
    return merged

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
        self.units = units
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
        
        indices = []
        for start, end in self.intervals:
            if self.units == "ms":
                start_ms, end_ms = start, end
            else:
                units_to_ms = {"sec": 1000.0, "min": 60000.0, "h": 3600000.0}
                start_ms = start * units_to_ms[self.units]
                end_ms = end * units_to_ms[self.units]
            
            start_ix = np.argmin(np.abs(eyedata_obj.tx - start_ms))
            end_ix = np.argmin(np.abs(eyedata_obj.tx - end_ms))
            indices.append((start_ix, end_ix))
        
        return np.array(indices, dtype=int)
    
    def merge(self):
        """
        Merge overlapping intervals.
        
        Returns
        -------
        Intervals
            New Intervals object with merged intervals
        """
        merged = merge_intervals(self.intervals.copy())
        return Intervals(merged, self.units, self.label, 
                        self.event_labels, self.event_indices)
    
    def stats(self):
        """
        Get statistics about interval durations.
        
        Returns
        -------
        IntervalStats
            Dictionary with summary statistics (n, mean, sd, min, max)
        """
        return get_interval_stats(self.intervals)
    
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
            ax.plot([start, end], [y_level, y_level], color='black', **kwargs)
            
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
