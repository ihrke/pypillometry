"""Functions to handle intervals (blinks, erpds, etc.) in eyetracking data.
"""

import numpy as np

from pypillometry.convenience import requires_package, normalize_unit

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
    
    # Get sampling_rate from first object that has one
    sampling_rate = None
    for obj in intervals_list:
        if obj.sampling_rate is not None:
            sampling_rate = obj.sampling_rate
            break
    
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
                        data_time_range=intervals_list[0].data_time_range,
                        sampling_rate=sampling_rate)
    
    # Use data_time_range from first object if available
    data_time_range = intervals_list[0].data_time_range if intervals_list[0].data_time_range is not None else None
    
    return Intervals(
        all_intervals,
        units=first_units,
        label=label,
        event_labels=all_event_labels if has_labels else None,
        event_indices=all_event_indices if has_indices else None,
        event_onsets=all_event_onsets if has_onsets else None,
        data_time_range=data_time_range,
        sampling_rate=sampling_rate
    )

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
    
    def __init__(self, intervals, units, label=None, event_labels=None, event_indices=None, data_time_range=None, event_onsets=None, sampling_rate=None):
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
        sampling_rate : float, optional
            Sampling rate in Hz. Required to convert index-based intervals to time units.
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
        self.sampling_rate = sampling_rate
    
    def __len__(self):
        """Return number of intervals."""
        return len(self.intervals)
    
    def __iter__(self):
        """Iterate over intervals."""
        return iter(self.intervals)
    
    def __getitem__(self, idx):
        """Get interval(s) by index."""
        return self.intervals[idx]
    
    def to_mask(self, length: int = None) -> np.ndarray:
        """
        Convert intervals to a binary mask array.
        
        If intervals have time units, they are automatically converted to indices
        using the sampling_rate.
        
        Parameters
        ----------
        length : int, optional
            Length of the mask array. If None, uses data_time_range[1] 
            (converted to indices if necessary).
            
        Returns
        -------
        np.ndarray
            Binary array with 1s where intervals exist, 0s elsewhere.
            
        Raises
        ------
        ValueError
            If length is not provided and data_time_range is not set.
            If intervals have time units but sampling_rate is not set.
            
        Examples
        --------
        >>> intervals = Intervals([(0, 10), (20, 30)], units=None, data_time_range=(0, 50))
        >>> mask = intervals.to_mask()
        >>> mask.shape
        (50,)
        
        >>> # Time-based intervals are auto-converted
        >>> intervals = Intervals([(0, 100), (200, 300)], units="ms", 
        ...                       data_time_range=(0, 500), sampling_rate=1000)
        >>> mask = intervals.to_mask()  # Converts to indices automatically
        """
        # Convert to index-based if needed
        if self.units is not None:
            if self.sampling_rate is None:
                raise ValueError(
                    "Cannot convert time-based intervals to mask without sampling_rate. "
                    "Set sampling_rate when creating the Intervals object."
                )
            intervals_idx = self.to_units(None)
        else:
            intervals_idx = self
        
        # Determine length
        # Note: We use data_time_range[1] (not the span) because intervals are stored
        # as absolute indices. The mask covers indices 0 to data_time_range[1]-1.
        if length is None:
            if intervals_idx.data_time_range is None:
                raise ValueError(
                    "Cannot create mask without length. Either provide length argument "
                    "or set data_time_range when creating the Intervals object."
                )
            length = int(intervals_idx.data_time_range[1])
        
        mask = np.zeros(length, dtype=np.int8)
        for start, end in intervals_idx.intervals:
            start_idx = max(0, int(start))
            end_idx = min(length, int(end))
            mask[start_idx:end_idx] = 1
        
        return mask
    
    @classmethod
    def from_mask(cls, mask: np.ndarray, units=None, label: str = None, 
                  sampling_rate: float = None) -> 'Intervals':
        """
        Create Intervals from a binary mask array.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary array where non-zero values indicate interval membership.
        units : str or None, optional
            Units for the intervals. Default is None (indices).
        label : str, optional
            Label for the intervals.
        sampling_rate : float, optional
            Sampling rate in Hz.
            
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
        # Find transitions in the mask
        mask_bool = mask.astype(bool).astype(int)
        # Pad with zeros to detect edges
        padded = np.concatenate([[0], mask_bool, [0]])
        diff = np.diff(padded)
        
        # Starts are where diff == 1, ends are where diff == -1
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        intervals_list = list(zip(starts.tolist(), ends.tolist()))
        
        return cls(
            intervals_list,
            units=units,
            label=label,
            data_time_range=(0, len(mask)),
            sampling_rate=sampling_rate
        )
    
    def __sub__(self, other: 'Intervals') -> 'Intervals':
        """
        Subtract intervals using the - operator.
        
        Converts both Intervals to binary masks, subtracts them (with negative 
        values clipped to 0), and converts back to Intervals. This effectively
        removes the regions covered by 'other' from 'self'.
        
        Both Intervals must be index-based (units=None) or will be converted.
        
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
        TypeError
            If other is not an Intervals object
        ValueError
            If neither object has data_time_range set
            
        Examples
        --------
        >>> a = Intervals([(0, 100), (200, 300)], units=None, data_time_range=(0, 400))
        >>> b = Intervals([(50, 150)], units=None, data_time_range=(0, 400))
        >>> c = a - b
        >>> c.intervals  # (0, 50) and (200, 300) remain
        [(0, 50), (200, 300)]
        """
        if not isinstance(other, Intervals):
            return NotImplemented
        
        # Convert to index-based if needed
        if self.units is not None:
            self_idx = self.to_units(None)
        else:
            self_idx = self
        
        if other.units is not None:
            other_idx = other.to_units(None)
        else:
            other_idx = other
        
        # Determine mask length from data_time_range
        if self_idx.data_time_range is not None:
            length = int(self_idx.data_time_range[1])
        elif other_idx.data_time_range is not None:
            length = int(other_idx.data_time_range[1])
        else:
            raise ValueError(
                "Cannot subtract Intervals without data_time_range. "
                "At least one Intervals object must have data_time_range set."
            )
        
        # Convert to masks
        mask_self = self_idx.to_mask(length)
        mask_other = other_idx.to_mask(length)
        
        # Subtract and clip negative values to 0
        result_mask = np.clip(mask_self - mask_other, 0, 1)
        
        # Convert back to Intervals
        result = Intervals.from_mask(
            result_mask,
            units=None,
            label=f"{self.label or 'intervals'} - {other.label or 'intervals'}",
            sampling_rate=self.sampling_rate or other.sampling_rate
        )
        
        # Convert back to original units if self had time units
        if self.units is not None:
            result = result.to_units(self.units)
        
        return result
    
    def __add__(self, other: 'Intervals') -> 'Intervals':
        """
        Combine two Intervals objects using the + operator.
        
        The intervals are concatenated and metadata is merged appropriately:
        - Units: other is converted to self's units if different
        - data_time_range: expanded to cover both
        - label: combined with ' + '
        - event_labels/indices/onsets: concatenated
        
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
            If units are incompatible (one is None/indices)
            
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
        
        # Handle unit conversion
        if self.units is None and other.units is None:
            # Both are indices - just concatenate
            other_converted = other
        elif self.units is None or other.units is None:
            raise ValueError(
                f"Cannot combine Intervals with units={self.units} and units={other.units}. "
                "Both must have time units or both must be indices."
            )
        elif self.units != other.units:
            # Convert other to self's units
            other_converted = other.to_units(self.units)
        else:
            other_converted = other
        
        # Combine intervals
        combined_intervals = list(self.intervals) + list(other_converted.intervals)
        
        # Combine labels
        if self.label and other.label:
            combined_label = f"{self.label} + {other.label}"
        else:
            combined_label = self.label or other.label
        
        # Combine event_labels
        if self.event_labels is not None and other.event_labels is not None:
            combined_event_labels = list(self.event_labels) + list(other.event_labels)
        elif self.event_labels is not None:
            combined_event_labels = list(self.event_labels)
        elif other.event_labels is not None:
            combined_event_labels = list(other.event_labels)
        else:
            combined_event_labels = None
        
        # Combine event_indices
        if self.event_indices is not None and other.event_indices is not None:
            combined_event_indices = np.concatenate([
                np.asarray(self.event_indices), 
                np.asarray(other.event_indices)
            ])
        elif self.event_indices is not None:
            combined_event_indices = np.asarray(self.event_indices)
        elif other.event_indices is not None:
            combined_event_indices = np.asarray(other.event_indices)
        else:
            combined_event_indices = None
        
        # Combine event_onsets (convert other's if needed)
        if self.event_onsets is not None and other_converted.event_onsets is not None:
            combined_event_onsets = np.concatenate([
                np.asarray(self.event_onsets),
                np.asarray(other_converted.event_onsets)
            ])
        elif self.event_onsets is not None:
            combined_event_onsets = np.asarray(self.event_onsets)
        elif other_converted.event_onsets is not None:
            combined_event_onsets = np.asarray(other_converted.event_onsets)
        else:
            combined_event_onsets = None
        
        # Combine data_time_range (expand to cover both)
        if self.data_time_range is not None and other_converted.data_time_range is not None:
            combined_range = (
                min(self.data_time_range[0], other_converted.data_time_range[0]),
                max(self.data_time_range[1], other_converted.data_time_range[1])
            )
        elif self.data_time_range is not None:
            combined_range = self.data_time_range
        elif other_converted.data_time_range is not None:
            combined_range = other_converted.data_time_range
        else:
            combined_range = None
        
        # Use self's sampling_rate, or other's if self doesn't have one
        combined_sampling_rate = self.sampling_rate or other.sampling_rate
        
        return Intervals(
            combined_intervals,
            self.units,
            label=combined_label,
            event_labels=combined_event_labels,
            event_indices=combined_event_indices,
            data_time_range=combined_range,
            event_onsets=combined_event_onsets,
            sampling_rate=combined_sampling_rate
        )
    
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
    
    def to_units(self, target_units: str|None) -> 'Intervals':
        """
        Convert intervals to different time units or indices.
        
        Parameters
        ----------
        target_units : str or None
            Target units: "ms", "sec", "min", "h", or "indices"/"index" for sample indices.
            Also accepts None for indices.
            
        Returns
        -------
        Intervals
            New Intervals object with converted units
            
        Raises
        ------
        ValueError
            If conversion requires sampling_rate but it is not set
        
        Examples
        --------
        >>> intervals = data.get_intervals("stim", units="ms")
        >>> intervals_sec = intervals.to_units("sec")
        >>> intervals_seconds = intervals.to_units("seconds")  # alias works too
        
        Convert from indices to time units (requires sampling_rate):
        
        >>> intervals_idx = Intervals([(0, 100), (200, 300)], units=None, sampling_rate=1000)
        >>> intervals_ms = intervals_idx.to_units("ms")
        
        Convert from time units to indices (requires sampling_rate):
        
        >>> intervals_ms = Intervals([(0, 1000), (2000, 3000)], units="ms", sampling_rate=1000)
        >>> intervals_idx = intervals_ms.to_units("indices")
        """
        # Normalize target units (source units already normalized in __init__)
        # This converts "indices", "index", "samples" etc. to None
        target_units = normalize_unit(target_units)
        
        if self.units == target_units:
            return self
        
        units_to_ms = {"ms": 1.0, "sec": 1000.0, "min": 60000.0, "h": 3600000.0}
        
        # Handle conversion TO indices
        if target_units is None:
            if self.units is None:
                return self  # Already indices
            
            if self.sampling_rate is None:
                raise ValueError(
                    "Cannot convert to indices without sampling_rate. "
                    "Set sampling_rate when creating the Intervals object."
                )
            
            if self.units not in units_to_ms:
                raise ValueError(f"Unknown source units: {self.units}")
            
            # Convert to ms first, then to indices
            # index = time_ms * fs / 1000
            fac_to_ms = units_to_ms[self.units]
            samples_per_ms = self.sampling_rate / 1000.0
            
            converted = [
                (int(round(s * fac_to_ms * samples_per_ms)), 
                 int(round(e * fac_to_ms * samples_per_ms))) 
                for s, e in self.intervals
            ]
            
            # Convert data_time_range
            if self.data_time_range is not None:
                data_time_range = (
                    int(round(self.data_time_range[0] * fac_to_ms * samples_per_ms)),
                    int(round(self.data_time_range[1] * fac_to_ms * samples_per_ms))
                )
            else:
                data_time_range = None
            
            # Convert event_onsets
            if self.event_onsets is not None:
                event_onsets = np.array(
                    [int(round(o * fac_to_ms * samples_per_ms)) for o in self.event_onsets]
                )
            else:
                event_onsets = None
            
            return Intervals(converted, None, self.label,
                            self.event_labels, self.event_indices,
                            data_time_range, event_onsets, self.sampling_rate)
        
        # Handle conversion to time units
        if target_units not in units_to_ms:
            raise ValueError(f"Unknown target units: {target_units}")
        
        # Handle conversion FROM indices
        if self.units is None:
            if self.sampling_rate is None:
                raise ValueError(
                    "Cannot convert from indices (units=None) without sampling_rate. "
                    "Set sampling_rate or use get_intervals(units='ms') instead."
                )
            # Convert indices to ms first: index / fs * 1000
            ms_per_sample = 1000.0 / self.sampling_rate
            converted_ms = [(s * ms_per_sample, e * ms_per_sample) for s, e in self.intervals]
            # Then convert ms to target units
            fac = 1.0 / units_to_ms[target_units]
            converted = [(s * fac, e * fac) for s, e in converted_ms]
            # Convert data_time_range
            if self.data_time_range is not None:
                data_time_range = (
                    self.data_time_range[0] * ms_per_sample * fac,
                    self.data_time_range[1] * ms_per_sample * fac
                )
            else:
                data_time_range = None
            # Convert event_onsets
            if self.event_onsets is not None:
                event_onsets = np.array(self.event_onsets) * ms_per_sample * fac
            else:
                event_onsets = None
        else:
            if self.units not in units_to_ms:
                raise ValueError(f"Unknown source units: {self.units}")
            
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
                        data_time_range, event_onsets, self.sampling_rate)
    
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
                        event_onsets=merged_onsets,
                        sampling_rate=self.sampling_rate)
    
    def stats(self):
        """
        Get statistics about interval durations.
        
        Returns
        -------
        dict
            Dictionary with summary statistics (n, mean, sd, min, max, total_duration, units)
        """
        durations = [i[1] - i[0] for i in self.intervals]
        return {
            "n": len(self.intervals),
            "mean": np.mean(durations),
            "sd": np.std(durations),
            "min": np.min(durations),
            "max": np.max(durations),
            "total_duration": np.sum(durations),
            "units": self.units
        }


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
        if self.units is not None:
            ax.set_xlabel(f'Time ({self.units})')
        else:
            ax.set_xlabel('Time (indices)')
        
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
            (e.g., zorder, label, linestyle, edgecolor)
            
        Returns
        -------
        list
            List of matplotlib patches created
            
        Examples
        --------
        >>> # Highlight on a single pupil plot
        >>> data.plot.pupil_plot(units="sec")
        >>> intervals = data.get_intervals("stim", units="sec")
        >>> intervals.plot_highlight(color='green', alpha=0.2)
        
        >>> # Highlight on all subplots from plot_timeseries
        >>> data.plot.plot_timeseries(units="ms")
        >>> intervals = data.get_intervals("stim", units="ms")
        >>> intervals.plot_highlight()  # Applies to all subplots
        
        >>> # Highlight only specific subplot
        >>> fig = plt.gcf()
        >>> intervals.plot_highlight(ax=fig.axes[0])  # Only first subplot
        
        Notes
        -----
        Make sure the intervals are in the same units as the plot's x-axis.
        Use `intervals.to_units("sec")` to convert if needed.
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
        
        patches = []
        for axes in axes_list:
            for start, end in self.intervals:
                patch = axes.axvspan(start, end, color=color, alpha=alpha, zorder=0, **kwargs)
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
                
        # Add sampling rate if present
        if self.sampling_rate is not None:
            parts.append(f"fs={self.sampling_rate}Hz")
        
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
    units = stats.get("units", "indices")
    return "%i intervals (total duration: %.2f %s), %.2f %s +/- %.2f %s, [%.2f %s, %.2f %s]" % (
        n, total_duration, units, mean, units, sd, units, minv, units, maxv, units
    )

