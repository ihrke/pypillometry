"""Region selection functionality for the interactive viewer."""

import numpy as np
import pyqtgraph as pg
from typing import List, Tuple, Optional, Callable
from ..intervals import Intervals


class RegionSelector:
    """Manages interactive region selection for time intervals.
    
    Attributes
    ----------
    regions : list of pg.LinearRegionItem
        List of selected region items
    callback : callable, optional
        Callback function to call when regions change
    """
    
    def __init__(self, plot_widget, callback: Optional[Callable] = None):
        """Initialize region selector.
        
        Parameters
        ----------
        plot_widget : pg.PlotItem
            PyQtGraph plot widget to add regions to
        callback : callable, optional
            Function to call with Intervals object when regions change
        """
        self.plot_widget = plot_widget
        self.regions: List[pg.LinearRegionItem] = []
        self.callback = callback
        self._next_color_idx = 0
        self._colors = ['#00ff00', '#ff00ff', '#00ffff', '#ffff00']
    
    def add_region(self, start: Optional[float] = None, end: Optional[float] = None):
        """Add a new selection region.
        
        Parameters
        ----------
        start : float, optional
            Start time in ms. If None, uses current view range.
        end : float, optional
            End time in ms. If None, uses current view range.
        """
        # Get default range from current view if not specified
        if start is None or end is None:
            view_range = self.plot_widget.viewRange()[0]
            if start is None:
                start = view_range[0] + 0.25 * (view_range[1] - view_range[0])
            if end is None:
                end = view_range[0] + 0.75 * (view_range[1] - view_range[0])
        
        # Create region with cycling colors
        color = self._colors[self._next_color_idx % len(self._colors)]
        self._next_color_idx += 1
        
        region = pg.LinearRegionItem(
            values=[start, end],
            movable=True,
            brush=pg.mkBrush(color=color, alpha=50)
        )
        
        # Connect signal for region changes
        region.sigRegionChanged.connect(self._on_region_changed)
        
        self.plot_widget.addItem(region)
        self.regions.append(region)
        
        # Trigger callback
        self._on_region_changed()
        
        return region
    
    def remove_region(self, region):
        """Remove a selection region.
        
        Parameters
        ----------
        region : pg.LinearRegionItem
            Region to remove
        """
        if region in self.regions:
            self.plot_widget.removeItem(region)
            self.regions.remove(region)
            self._on_region_changed()
    
    def clear_regions(self):
        """Remove all selection regions."""
        for region in list(self.regions):
            self.plot_widget.removeItem(region)
        self.regions.clear()
        self._on_region_changed()
    
    def get_intervals(self) -> Optional[Intervals]:
        """Get current selections as Intervals object.
        
        Returns
        -------
        Intervals or None
            Intervals object with selected time ranges, or None if no selections
        """
        if not self.regions:
            return None
        
        # Extract time ranges
        intervals = []
        for region in self.regions:
            start, end = region.getRegion()
            intervals.append((start, end))
        
        # Sort by start time
        intervals.sort(key=lambda x: x[0])
        
        # Create Intervals object
        return Intervals(intervals, units='ms', label='viewer_selection')
    
    def set_intervals(self, intervals: Intervals):
        """Set selection regions from Intervals object.
        
        Parameters
        ----------
        intervals : Intervals
            Intervals object to load (should be in ms)
        """
        self.clear_regions()
        
        # Assume intervals are in ms (viewer always uses ms)
        # Intervals stores data as list of (start, end) tuples
        for start, end in intervals.intervals:
            self.add_region(start, end)
    
    def _on_region_changed(self):
        """Internal callback when regions change."""
        if self.callback is not None:
            intervals = self.get_intervals()
            self.callback(intervals)

