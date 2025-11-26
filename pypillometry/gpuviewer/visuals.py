"""GPU visual elements for the viewer (lines, masks, events)."""

import numpy as np
from vispy import scene
from vispy.color import Color
from typing import List, Optional
import numpy.ma as ma


def add_line_visual(
    viewbox: scene.ViewBox,
    time: np.ndarray,
    data: np.ndarray,
    color: str = '#0000FF',
    width: float = 1.5
) -> scene.Line:
    """Add a line visual to a viewbox.
    
    Parameters
    ----------
    viewbox : scene.ViewBox
        The viewbox to add the line to
    time : np.ndarray
        Time array in seconds
    data : np.ndarray
        Data values (can be masked array)
    color : str
        Line color as hex string
    width : float
        Line width in pixels
    
    Returns
    -------
    scene.Line
        The created line visual
    """
    # Handle masked arrays - get underlying data
    if ma.is_masked(data):
        plot_data = data.data.copy()
        # Set masked values to NaN so lines break there
        plot_data[data.mask] = np.nan
    else:
        plot_data = np.asarray(data, dtype=np.float64)
    
    # Create position array (N x 2)
    pos = np.column_stack([time, plot_data]).astype(np.float32)
    
    # Create line visual
    line = scene.Line(
        pos=pos,
        color=color,
        width=width,
        connect='strip',  # Connect adjacent points, break at NaN
        antialias=False,  # Faster rendering
        parent=viewbox.scene
    )
    
    return line


def add_mask_regions(
    viewbox: scene.ViewBox,
    time: np.ndarray,
    mask: np.ndarray,
    color: str = '#FFA500',
    alpha: float = 0.3
) -> List[scene.visuals.Rectangle]:
    """Add semi-transparent rectangles for masked data regions.
    
    Parameters
    ----------
    viewbox : scene.ViewBox
        The viewbox to add regions to
    time : np.ndarray
        Time array in seconds
    mask : np.ndarray
        Boolean mask array (True = masked)
    color : str
        Fill color as hex string
    alpha : float
        Transparency (0-1)
    
    Returns
    -------
    list
        List of created rectangle visuals
    """
    # Find contiguous masked regions
    mask_arr = np.asarray(mask, dtype=bool)
    mask_diff = np.diff(np.concatenate([[False], mask_arr, [False]]).astype(int))
    starts = np.where(mask_diff == 1)[0]
    ends = np.where(mask_diff == -1)[0]
    
    # Limit number of regions for performance
    max_regions = 500
    if len(starts) > max_regions:
        # Sample evenly across the recording
        indices = np.linspace(0, len(starts)-1, max_regions, dtype=int)
        starts = starts[indices]
        ends = ends[indices]
    
    # Parse color with alpha
    c = Color(color)
    c.alpha = alpha
    
    regions = []
    
    # Get approximate y-range for rectangle height
    # We'll use a very large height that extends beyond any realistic data range
    y_min = -1e6
    y_max = 1e6
    height = y_max - y_min
    
    for start_idx, end_idx in zip(starts, ends):
        if start_idx < len(time) and end_idx <= len(time):
            start_time = time[start_idx]
            end_time = time[end_idx - 1] if end_idx > 0 else time[-1]
            
            width = end_time - start_time
            center_x = (start_time + end_time) / 2
            center_y = (y_min + y_max) / 2
            
            rect = scene.visuals.Rectangle(
                center=(center_x, center_y),
                width=width,
                height=height,
                color=c,
                border_color=None,
                parent=viewbox.scene
            )
            
            # Send to back so it appears behind curves
            rect.order = -10
            
            regions.append(rect)
    
    return regions


def add_event_markers(
    viewbox: scene.ViewBox,
    event_times: np.ndarray,
    event_labels: List[str],
    color: str = '#888888',
    alpha: float = 0.5,
    width: float = 2.0
) -> List:
    """Add vertical lines and labels for event markers.
    
    Parameters
    ----------
    viewbox : scene.ViewBox
        The viewbox to add markers to
    event_times : np.ndarray
        Event onset times in seconds
    event_labels : list of str
        Labels for each event
    color : str
        Line color as hex string
    alpha : float
        Transparency (0-1)
    width : float
        Line width in pixels
    
    Returns
    -------
    list
        List of created visual elements (lines and text)
    """
    # Parse color with alpha
    c = Color(color)
    c.alpha = alpha
    
    markers = []
    
    # Get y-range for vertical lines
    y_min = -1e6
    y_max = 1e6
    
    # Limit number of event markers for performance
    max_events = 1000
    if len(event_times) > max_events:
        # Sample evenly
        indices = np.linspace(0, len(event_times)-1, max_events, dtype=int)
        event_times = event_times[indices]
        event_labels = [event_labels[i] for i in indices]
    
    for t, label in zip(event_times, event_labels):
        # Create vertical line as two-point line
        pos = np.array([[t, y_min], [t, y_max]], dtype=np.float32)
        
        line = scene.Line(
            pos=pos,
            color=c,
            width=width,
            connect='strip',
            parent=viewbox.scene
        )
        line.order = -5  # Behind curves but in front of mask regions
        markers.append(line)
        
        # Add text label at top
        text = scene.Text(
            text=str(label),
            pos=(t, 0),  # Y position will be adjusted by camera
            color=color,
            font_size=8,
            anchor_x='left',
            anchor_y='bottom',
            parent=viewbox.scene
        )
        text.order = 10  # In front of everything
        markers.append(text)
    
    return markers

