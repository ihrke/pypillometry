"""Functions to handle events in eyetracking data.
"""

import numpy as np
from typing import Optional, List, Tuple


class Events:
    """
    Container for events with metadata.
    
    This class wraps arrays of event onsets and labels and provides additional
    functionality such as unit conversion, filtering, and display capabilities.
    
    Attributes
    ----------
    onsets : np.ndarray
        Array of event onset times
    labels : np.ndarray
        Array of event labels (strings)
    units : str or None
        Units of the onsets ("ms", "sec", "min", "h", or None for indices)
    data_time_range : tuple or None
        Time range (min, max) of the original dataset
        
    Examples
    --------
    >>> events = Events(onsets=[100, 500, 1000], labels=["A", "B", "C"], units="ms")
    >>> len(events)
    3
    >>> for onset, label in events:
    ...     print(f"{label} at {onset}")
    """
    
    def __init__(self, 
                 onsets: np.ndarray | List,
                 labels: np.ndarray | List,
                 units: Optional[str] = "ms",
                 data_time_range: Optional[Tuple[float, float]] = None):
        """
        Initialize an Events object.
        
        Parameters
        ----------
        onsets : np.ndarray or list
            Event onset times
        labels : np.ndarray or list
            Event labels (strings)
        units : str or None, optional
            Units of the onsets ("ms", "sec", "min", "h", or None for indices)
            Default is "ms"
        data_time_range : tuple, optional
            Time range (min, max) of the original dataset
        """
        self.onsets = np.array(onsets, dtype=float)
        self.labels = np.array(labels, dtype=str)
        self.units = units
        self.data_time_range = data_time_range
        
        # Validate that onsets and labels have same length
        if len(self.onsets) != len(self.labels):
            raise ValueError(f"onsets and labels must have same length, got {len(self.onsets)} and {len(self.labels)}")
    
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
        idx : int or slice
            Index or slice to retrieve
            
        Returns
        -------
        tuple or Events
            If idx is int: returns (onset, label) tuple
            If idx is slice: returns new Events object with subset
        """
        if isinstance(idx, (int, np.integer)):
            return (self.onsets[idx], self.labels[idx])
        elif isinstance(idx, slice):
            return Events(
                onsets=self.onsets[idx],
                labels=self.labels[idx],
                units=self.units,
                data_time_range=self.data_time_range
            )
        else:
            raise TypeError(f"indices must be integers or slices, not {type(idx)}")
    
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
    
    def to_units(self, target_units: Optional[str]) -> 'Events':
        """
        Convert event onsets to different time units.
        
        Parameters
        ----------
        target_units : str or None
            Target units: "ms", "sec", "min", "h", or None for indices
            
        Returns
        -------
        Events
            New Events object with converted units
            
        Raises
        ------
        ValueError
            If current units are None (indices) or conversion is not possible
        
        Examples
        --------
        >>> events = Events([1000, 2000], ["A", "B"], units="ms")
        >>> events_sec = events.to_units("sec")
        >>> events_sec.onsets
        array([1., 2.])
        """
        if self.units is None:
            raise ValueError(
                "Cannot convert from indices (units=None). "
                "Events with units=None represent indices, not time values."
            )
        
        if target_units == self.units:
            # No conversion needed
            return Events(
                onsets=self.onsets.copy(),
                labels=self.labels.copy(),
                units=self.units,
                data_time_range=self.data_time_range
            )
        
        # Define conversion factors (everything relative to ms)
        units_to_ms = {
            "ms": 1.0,
            "sec": 1000.0,
            "min": 60000.0,
            "h": 3600000.0
        }
        
        if self.units not in units_to_ms:
            raise ValueError(f"Unknown source units: {self.units}")
        if target_units is not None and target_units not in units_to_ms:
            raise ValueError(f"Unknown target units: {target_units}")
        
        if target_units is None:
            raise ValueError(
                "Cannot convert to indices (units=None). "
                "Use appropriate methods on the eyedata object instead."
            )
        
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
            data_time_range=converted_range
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
        
        # Add units info
        if self.units is not None:
            parts.append(f"units={self.units}")
        else:
            parts.append("units=None (indices)")
        
        # Add time range if available
        if self.data_time_range is not None:
            range_str = f"[{self.data_time_range[0]:.1f}, {self.data_time_range[1]:.1f}]"
            parts.append(f"range={range_str}")
        
        return " | ".join(parts)

