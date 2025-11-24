"""Visualization helper functions for the interactive viewer."""

import numpy as np
import numpy.ma as ma
import pyqtgraph as pg
from typing import Optional, Dict, List, Tuple


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
    connect_mode: str = 'finite'
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
    
    Returns
    -------
    pg.PlotDataItem
        The created plot item
    """
    color = get_color_for_modality(modality)
    
    # Handle masked arrays
    if ma.is_masked(data):
        if connect_mode == 'finite':
            # Replace masked values with NaN for gaps
            plot_data = data.filled(np.nan)
        else:
            # Use only unmasked data
            plot_data = data.compressed()
            time = time[~data.mask]
    else:
        plot_data = data
    
    # Create plot with performance optimizations
    pen = pg.mkPen(color=color, width=2)  # Thicker lines for better visibility
    curve = plot_widget.plot(
        time, plot_data,
        pen=pen,
        name=modality,
        connect=connect_mode
    )
    
    # Enable downsampling for performance
    curve.setDownsampling(ds=True, auto=True, method='peak')
    curve.setClipToView(True)
    
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
            pen=pg.mkPen(color=color, width=1, style=pg.QtCore.Qt.DashLine),
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
    alpha: float = 0.8
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
    
    for start_idx, end_idx in zip(starts, ends):
        if start_idx < len(time) and end_idx <= len(time):
            start_time = time[start_idx]
            end_time = time[end_idx - 1] if end_idx > 0 else time[-1]
            
            region = pg.LinearRegionItem(
                values=[start_time, end_time],
                movable=False,
                brush=pg.mkBrush(color=color, alpha=int(alpha * 255)),
                pen=None  # No border
            )
            # Send to back so it appears behind curves
            region.setZValue(-10)
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
        plot_widget.setLabel('bottom', 'Time', units='s', color='k')
        plot_widget.getAxis('bottom').setPen('k')
        plot_widget.getAxis('bottom').setTextPen('k')
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

