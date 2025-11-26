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
    width: float = 1.0
) -> scene.Line:
    """Add a line visual to a viewbox.
    
    Uses vispy.scene.Line which handles GPU upload efficiently.
    """
    # Handle masked arrays - get underlying data
    if ma.is_masked(data):
        plot_data = data.data.astype(np.float32, copy=True)
        plot_data[data.mask] = np.nan  # NaN breaks the line
    else:
        plot_data = np.asarray(data, dtype=np.float32)
    
    # Create position array (N x 2) - use float32 for GPU efficiency
    time_f32 = time.astype(np.float32) if time.dtype != np.float32 else time
    pos = np.column_stack([time_f32, plot_data])
    
    # Create line visual with GL_LINE_STRIP mode
    line = scene.Line(
        pos=pos,
        color=color,
        width=width,
        connect='strip',
        antialias=False,
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
    """Add semi-transparent mesh for masked regions using vectorized numpy."""
    mask_arr = np.asarray(mask, dtype=bool)
    if not np.any(mask_arr):
        return None
    
    # Find region boundaries using numpy diff
    padded = np.concatenate([[False], mask_arr, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return None
    
    # Limit regions
    max_regions = 100
    if len(starts) > max_regions:
        idx = np.linspace(0, len(starts)-1, max_regions, dtype=np.intp)
        starts, ends = starts[idx], ends[idx]
    
    n = len(starts)
    
    # Vectorized vertex creation
    y_min, y_max = -1e6, 1e6
    
    # Clip indices to valid range
    starts = np.clip(starts, 0, len(time)-1)
    ends = np.clip(ends, 1, len(time)) - 1
    
    x0 = time[starts].astype(np.float32)
    x1 = time[ends].astype(np.float32)
    
    # Build vertices: 4 per rectangle [bl, br, tr, tl]
    vertices = np.zeros((n * 4, 2), dtype=np.float32)
    vertices[0::4, 0] = x0  # bottom-left x
    vertices[0::4, 1] = y_min
    vertices[1::4, 0] = x1  # bottom-right x
    vertices[1::4, 1] = y_min
    vertices[2::4, 0] = x1  # top-right x
    vertices[2::4, 1] = y_max
    vertices[3::4, 0] = x0  # top-left x
    vertices[3::4, 1] = y_max
    
    # Build faces: 2 triangles per rectangle
    base_idx = np.arange(n, dtype=np.uint32) * 4
    faces = np.zeros((n * 2, 3), dtype=np.uint32)
    faces[0::2, 0] = base_idx
    faces[0::2, 1] = base_idx + 1
    faces[0::2, 2] = base_idx + 2
    faces[1::2, 0] = base_idx
    faces[1::2, 1] = base_idx + 2
    faces[1::2, 2] = base_idx + 3
    
    # Color with alpha
    c = Color(color)
    rgba = list(c.rgba)
    rgba[3] = alpha
    
    mesh = scene.Mesh(
        vertices=vertices,
        faces=faces,
        color=rgba,
        parent=viewbox.scene
    )
    mesh.order = -10
    
    return mesh


def add_event_markers(
    viewbox: scene.ViewBox,
    event_times: np.ndarray,
    event_labels: List[str],
    color: str = '#888888',
    alpha: float = 0.5,
    width: float = 1.0
) -> List:
    """Add vertical lines for event markers - NO text labels for performance."""
    if len(event_times) == 0:
        return []
    
    # Limit events
    max_events = 200
    if len(event_times) > max_events:
        idx = np.linspace(0, len(event_times)-1, max_events, dtype=np.intp)
        event_times = event_times[idx]
    
    n = len(event_times)
    y_min, y_max = -1e6, 1e6
    
    # Vectorized position array: [bottom, top, nan] for each event
    pos = np.empty((n * 3, 2), dtype=np.float32)
    pos[0::3, 0] = event_times  # bottom x
    pos[0::3, 1] = y_min        # bottom y
    pos[1::3, 0] = event_times  # top x
    pos[1::3, 1] = y_max        # top y
    pos[2::3, 0] = np.nan       # break x
    pos[2::3, 1] = np.nan       # break y
    
    c = Color(color)
    rgba = list(c.rgba)
    rgba[3] = alpha
    
    line = scene.Line(
        pos=pos,
        color=rgba,
        width=width,
        connect='strip',
        antialias=False,
        parent=viewbox.scene
    )
    line.order = -5
    
    return [line]
