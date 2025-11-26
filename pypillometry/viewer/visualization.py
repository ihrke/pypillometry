"""Visualization helper functions for the interactive viewer."""

import numpy as np
import numpy.ma as ma
import pyqtgraph as pg
from typing import Optional, Dict, List, Tuple

# PyQt5/PyQt6 compatibility
try:
    # PyQt6 style enums
    _DashLine = pg.QtCore.Qt.PenStyle.DashLine
except AttributeError:
    # PyQt5 style enums
    _DashLine = _DashLine


# Color scheme for different data modalities (consistent left=blue, right=red)
MODALITY_COLORS = {
    'left_pupil': '#0000FF',    # Blue (left eye)
    'right_pupil': '#FF0000',   # Red (right eye)
    'left_x': '#0000FF',        # Blue (left eye)
    'left_y': '#0000FF',        # Blue (left eye)
    'right_x': '#FF0000',       # Red (right eye)
    'right_y': '#FF0000',       # Red (right eye)
}


def get_color_for_modality(modality: str) -> str:
    """Get color for a data modality.
    
    Parameters
    ----------
    modality : str
        Data modality name (e.g., 'left_pupil', 'right_x')
    
    Returns
    -------
    str
        Hex color code
    """
    return MODALITY_COLORS.get(modality, '#7f7f7f')  # Default gray


def plot_timeseries(
    plot_widget,
    time: np.ndarray,
    data: np.ndarray,
    modality: str,
    connect_mode: str = 'finite',
    show_masked_data: bool = False
) -> pg.PlotDataItem:
    """Plot a timeseries with proper handling of masked arrays.
    
    Parameters
    ----------
    plot_widget : pg.PlotItem
        PyQtGraph plot widget
    time : np.ndarray
        Time vector in ms
    data : np.ndarray
        Data values (can be masked array)
    modality : str
        Data modality name for coloring
    connect_mode : str
        Connection mode: 'finite' for gaps at masked values, 'all' for continuous
    show_masked_data : bool
        If True, plot masked data values (for shaded mode).
        If False, replace masked values with NaN to create gaps (for gaps mode).
    
    Returns
    -------
    pg.PlotDataItem
        The created plot item
    """
    color = get_color_for_modality(modality)
    
    # Handle masked arrays
    if ma.is_masked(data):
        if show_masked_data:
            # Show all data including masked values (shading will indicate mask)
            plot_data = data.data  # Get underlying array without applying mask
        else:
            # Replace masked values with NaN to create gaps
            plot_data = data.filled(np.nan)
    else:
        plot_data = data
    
    # Create plot with performance optimizations
    # Use thinner pen for large datasets to improve rendering speed
    pen_width = 1 if len(time) > 500000 else 2
    pen = pg.mkPen(color=color, width=pen_width)
    curve = plot_widget.plot(
        time, plot_data,
        pen=pen,
        name=modality,
        connect=connect_mode
    )
    
    # Set z-value to ensure curve appears above masked regions
    curve.setZValue(1000)
    
    # Enable aggressive downsampling and clipping for performance
    # Calculate downsampling factor based on data size
    n_points = len(time)
    if n_points > 10000:
        # For large datasets, downsample aggressively to ~1000-2000 points max
        ds_factor = max(1, int(n_points / 1500))
    else:
        ds_factor = 1
    
    try:
        # Set explicit downsampling factor for better performance
        curve.setDownsampling(ds=ds_factor, auto=False, method='subsample')
    except:
        try:
            curve.setDownsampling(auto=True, method='subsample')
        except:
            pass
    
    curve.setClipToView(True)
    
    # Disable anti-aliasing for faster rendering
    curve.opts['antialias'] = False
    
    # Skip finite check for even faster rendering (if supported)
    try:
        curve.opts['skipFiniteCheck'] = True
    except:
        pass
    
    return curve


def add_event_markers(
    plot_widget,
    event_onsets: np.ndarray,
    event_labels: List[str],
    color: str = '#FFA500'  # Orange for visibility on light background
) -> List[pg.InfiniteLine]:
    """Add vertical lines for event markers.
    
    Parameters
    ----------
    plot_widget : pg.PlotItem
        PyQtGraph plot widget
    event_onsets : np.ndarray
        Event onset times in ms
    event_labels : list of str
        Event labels
    color : str
        Color for event lines
    
    Returns
    -------
    list of pg.InfiniteLine
        The created event marker lines
    """
    markers = []
    
    for onset, label in zip(event_onsets, event_labels):
        line = pg.InfiniteLine(
            pos=onset,
            angle=90,
            pen=pg.mkPen(color=color, width=1, style=_DashLine),
            movable=False,
            label=label,
            labelOpts={'position': 0.95, 'color': color, 'rotateAxis': (1, 0)}  # Rotate 90 degrees
        )
        plot_widget.addItem(line)
        markers.append(line)
    
    return markers


def add_mask_regions(
    plot_widget,
    time: np.ndarray,
    mask: np.ndarray,
    color: str = '#FFA500',  # Orange
    alpha: float = 0.2,
    max_regions: int = 100  # Strict limit for performance
) -> List[pg.LinearRegionItem]:
    """Add shaded regions for masked data intervals.
    
    Parameters
    ----------
    plot_widget : pg.PlotItem
        PyQtGraph plot widget
    time : np.ndarray
        Time vector in ms
    mask : np.ndarray
        Boolean mask array
    color : str
        Color for shaded regions
    alpha : float
        Transparency (0-1), default 0.2
    max_regions : int
        Maximum number of regions to create (for performance)
    
    Returns
    -------
    list of pg.LinearRegionItem
        The created region items
    """
    regions = []
    
    # Find contiguous masked regions
    mask_diff = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts = np.where(mask_diff == 1)[0]
    ends = np.where(mask_diff == -1)[0]
    
    # Limit number of regions for performance
    if len(starts) > max_regions:
        # Sample evenly across the recording
        indices = np.linspace(0, len(starts)-1, max_regions, dtype=int)
        starts = starts[indices]
        ends = ends[indices]
    
    for start_idx, end_idx in zip(starts, ends):
        if start_idx < len(time) and end_idx <= len(time):
            start_time = time[start_idx]
            end_time = time[end_idx - 1] if end_idx > 0 else time[-1]
            
            # Create brush with proper alpha using QColor
            from pyqtgraph.Qt import QtGui
            qcolor = QtGui.QColor(color)
            qcolor.setAlpha(int(alpha * 255))
            
            region = pg.LinearRegionItem(
                values=[start_time, end_time],
                movable=False,
                brush=pg.mkBrush(qcolor),
                swapMode='none'  # Disable swap behavior for performance
            )
            # Remove border lines by making them invisible
            region.lines[0].setPen(None)
            region.lines[1].setPen(None)
            
            # Disable all interactivity for performance
            region.lines[0].setMovable(False)
            region.lines[1].setMovable(False)
            region.setAcceptHoverEvents(False)
            
            # Send to back so it appears behind curves
            region.setZValue(-100)
            
            plot_widget.addItem(region)
            regions.append(region)
    
    return regions


def setup_plot_appearance(plot_widget, title: str, ylabel: str, show_x_axis: bool = True):
    """Configure plot appearance with labels and styling.
    
    Parameters
    ----------
    plot_widget : pg.PlotItem
        PyQtGraph plot widget
    title : str
        Plot title
    ylabel : str
        Y-axis label
    show_x_axis : bool
        Whether to show x-axis labels and ticks
    """
    # Set title and labels with dark text for light background
    plot_widget.setTitle(title, color='k', size='12pt')
    plot_widget.setLabel('left', ylabel, color='k')
    
    if show_x_axis:
        axis = plot_widget.getAxis('bottom')
        plot_widget.setLabel('bottom', 'Time (s)', color='k')  # Don't use units parameter
        axis.setPen('k')
        axis.setTextPen('k')
        axis.enableAutoSIPrefix(False)  # Disable SI prefixes to prevent ks, mmin, etc.
        axis.autoSIPrefixScale = 1.0  # Force the SI prefix scale to 1.0
        plot_widget.showAxis('bottom')
    else:
        # Hide x-axis labels for upper plots
        plot_widget.getAxis('bottom').setStyle(showValues=False)
        plot_widget.setLabel('bottom', '')
    
    # No grid
    plot_widget.showGrid(x=False, y=False)
    
    # Set y-axis colors
    plot_widget.getAxis('left').setPen('k')
    plot_widget.getAxis('left').setTextPen('k')
    
    plot_widget.setMouseEnabled(x=True, y=True)
    plot_widget.enableAutoRange()


def get_plot_label(modality: str) -> str:
    """Get human-readable label for a data modality.
    
    Parameters
    ----------
    modality : str
        Data modality name
    
    Returns
    -------
    str
        Human-readable label
    """
    labels = {
        'left_pupil': 'Left Pupil Size',
        'right_pupil': 'Right Pupil Size',
        'left_x': 'Left Gaze X',
        'left_y': 'Left Gaze Y',
        'right_x': 'Right Gaze X',
        'right_y': 'Right Gaze Y',
    }
    return labels.get(modality, modality.replace('_', ' ').title())

