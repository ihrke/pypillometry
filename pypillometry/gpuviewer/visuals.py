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
        plot_data = data.data.copy().astype(np.float32)
        # Set masked values to NaN so lines break there
        plot_data[data.mask] = np.nan
    else:
        plot_data = np.asarray(data, dtype=np.float32)
    
    # Create position array (N x 2)
    pos = np.column_stack([time.astype(np.float32), plot_data])
    
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
) -> Optional[scene.Mesh]:
    """Add semi-transparent mesh for all masked data regions (single draw call).
    
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
    scene.Mesh or None
        Single mesh visual for all regions, or None if no masked regions
    """
    # Find contiguous masked regions
    mask_arr = np.asarray(mask, dtype=bool)
    if not np.any(mask_arr):
        return None
    
    mask_diff = np.diff(np.concatenate([[False], mask_arr, [False]]).astype(int))
    starts = np.where(mask_diff == 1)[0]
    ends = np.where(mask_diff == -1)[0]
    
    if len(starts) == 0:
        return None
    
    # Limit number of regions for performance
    max_regions = 200
    if len(starts) > max_regions:
        # Sample evenly across the recording
        indices = np.linspace(0, len(starts)-1, max_regions, dtype=int)
        starts = starts[indices]
        ends = ends[indices]
    
    # Build single mesh with all rectangles as triangles
    # Each rectangle = 4 vertices, 2 triangles (6 indices)
    n_rects = len(starts)
    vertices = np.zeros((n_rects * 4, 2), dtype=np.float32)
    faces = np.zeros((n_rects * 2, 3), dtype=np.uint32)
    
    # Large y-range to cover any data
    y_min, y_max = -1e6, 1e6
    
    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        if start_idx >= len(time) or end_idx > len(time):
            continue
        
        x0 = time[start_idx]
        x1 = time[min(end_idx, len(time)-1)]
        
        # 4 vertices per rectangle
        base = i * 4
        vertices[base] = [x0, y_min]      # bottom-left
        vertices[base+1] = [x1, y_min]    # bottom-right
        vertices[base+2] = [x1, y_max]    # top-right
        vertices[base+3] = [x0, y_max]    # top-left
        
        # 2 triangles per rectangle
        face_base = i * 2
        faces[face_base] = [base, base+1, base+2]
        faces[face_base+1] = [base, base+2, base+3]
    
    # Parse color with alpha
    c = Color(color)
    rgba = list(c.rgba)
    rgba[3] = alpha
    
    # Create single mesh for all rectangles
    mesh = scene.Mesh(
        vertices=vertices,
        faces=faces,
        color=rgba,
        parent=viewbox.scene
    )
    mesh.order = -10  # Behind curves
    
    return mesh


def add_event_markers(
    viewbox: scene.ViewBox,
    event_times: np.ndarray,
    event_labels: List[str],
    color: str = '#888888',
    alpha: float = 0.5,
    width: float = 2.0
) -> List:
    """Add vertical lines for event markers using a single Line visual with NaN breaks.
    
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
        List containing the line visual and text visuals
    """
    if len(event_times) == 0:
        return []
    
    # Limit number of events for performance
    max_events = 500
    if len(event_times) > max_events:
        indices = np.linspace(0, len(event_times)-1, max_events, dtype=int)
        event_times = event_times[indices]
        event_labels = [event_labels[i] for i in indices]
    
    # Parse color with alpha
    c = Color(color)
    rgba = list(c.rgba)
    rgba[3] = alpha
    
    # Build single line array with NaN breaks between segments
    # Each vertical line: 2 points + 1 NaN break = 3 points
    y_min, y_max = -1e6, 1e6
    n_events = len(event_times)
    
    # Create positions: for each event, add bottom point, top point, then NaN
    pos = np.zeros((n_events * 3, 2), dtype=np.float32)
    for i, t in enumerate(event_times):
        base = i * 3
        pos[base] = [t, y_min]
        pos[base + 1] = [t, y_max]
        pos[base + 2] = [np.nan, np.nan]  # Break line
    
    markers = []
    
    # Single line visual for all event markers
    line = scene.Line(
        pos=pos,
        color=rgba,
        width=width,
        connect='strip',
        antialias=False,
        parent=viewbox.scene
    )
    line.order = -5  # Behind curves but in front of mask
    markers.append(line)
    
    # Add text labels (limit to avoid too many)
    max_labels = 100
    label_step = max(1, len(event_times) // max_labels)
    
    for i in range(0, len(event_times), label_step):
        t = event_times[i]
        label = event_labels[i]
        
        text = scene.Text(
            text=str(label)[:20],  # Truncate long labels
            pos=(t, 0),
            color=color,
            font_size=8,
            anchor_x='left',
            anchor_y='bottom',
            parent=viewbox.scene
        )
        text.order = 10
        markers.append(text)
    
    return markers
