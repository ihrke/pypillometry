"""GPU visual elements for the viewer (lines, masks, events) with LOD support."""

import numpy as np
from vispy import scene
from vispy.color import Color
from typing import List, Optional, Tuple, Dict
import numpy.ma as ma


class LODLine:
    """Line visual with LOD - normal color + lighter color in masked regions."""
    
    def __init__(
        self,
        viewbox: scene.ViewBox,
        time: np.ndarray,
        data: np.ndarray,
        color: str = '#0000FF',
        mask: np.ndarray = None,
        width: float = 2.0,
        lod_factors: Tuple[int, ...] = (1, 10, 100, 1000)
    ):
        self.viewbox = viewbox
        self.time_full = time.astype(np.float32)
        self.width = width
        self.lod_factors = lod_factors
        self.n_points = len(time)
        self.color = color
        
        # Create lighter version of color for masked regions
        c = Color(color)
        # Blend with white to make lighter
        light_rgba = (
            min(1.0, c.rgba[0] * 0.5 + 0.5),
            min(1.0, c.rgba[1] * 0.5 + 0.5),
            min(1.0, c.rgba[2] * 0.5 + 0.5),
            1.0
        )
        self.light_color = Color(light_rgba)
        
        # Get raw data and mask
        if ma.is_masked(data):
            self.data_full = data.data.copy().astype(np.float32)
            self.mask = np.asarray(data.mask, dtype=bool)
        else:
            self.data_full = np.asarray(data, dtype=np.float32)
            self.mask = np.zeros(len(data), dtype=bool)
        
        # Override with explicit mask if provided
        if mask is not None:
            self.mask = np.asarray(mask, dtype=bool)
        
        # Pre-compute LOD levels for both normal and masked
        self.lod_normal: Dict[int, np.ndarray] = {}
        self.lod_masked: Dict[int, np.ndarray] = {}
        self._precompute_lods()
        
        # Create lines
        self.current_lod = max(lod_factors)
        self.line_normal = None
        self.line_masked = None
        self._create_lines()
    
    def _precompute_lods(self):
        """Pre-compute downsampled versions for normal and masked."""
        for factor in self.lod_factors:
            if factor == 1:
                # Full resolution
                data_normal = self.data_full.copy()
                data_normal[self.mask] = np.nan
                
                data_masked = self.data_full.copy()
                data_masked[~self.mask] = np.nan
                
                self.lod_normal[1] = np.column_stack([self.time_full, data_normal]).astype(np.float32)
                self.lod_masked[1] = np.column_stack([self.time_full, data_masked]).astype(np.float32)
            else:
                n_out = max(1, self.n_points // factor)
                time_ds = np.zeros(n_out * 2, dtype=np.float32)
                data_normal_ds = np.zeros(n_out * 2, dtype=np.float32)
                data_masked_ds = np.zeros(n_out * 2, dtype=np.float32)
                
                for i in range(n_out):
                    start = i * factor
                    end = min(start + factor, self.n_points)
                    chunk = self.data_full[start:end]
                    chunk_mask = self.mask[start:end]
                    
                    valid = np.isfinite(chunk)
                    if not np.any(valid):
                        time_ds[i*2] = self.time_full[start]
                        time_ds[i*2+1] = self.time_full[end-1]
                        data_normal_ds[i*2] = np.nan
                        data_normal_ds[i*2+1] = np.nan
                        data_masked_ds[i*2] = np.nan
                        data_masked_ds[i*2+1] = np.nan
                    else:
                        valid_chunk = np.where(valid, chunk, np.inf)
                        min_idx = np.argmin(valid_chunk)
                        valid_chunk = np.where(valid, chunk, -np.inf)
                        max_idx = np.argmax(valid_chunk)
                        
                        if min_idx <= max_idx:
                            time_ds[i*2] = self.time_full[start + min_idx]
                            time_ds[i*2+1] = self.time_full[start + max_idx]
                            v1, v2 = chunk[min_idx], chunk[max_idx]
                            m1, m2 = chunk_mask[min_idx], chunk_mask[max_idx]
                        else:
                            time_ds[i*2] = self.time_full[start + max_idx]
                            time_ds[i*2+1] = self.time_full[start + min_idx]
                            v1, v2 = chunk[max_idx], chunk[min_idx]
                            m1, m2 = chunk_mask[max_idx], chunk_mask[min_idx]
                        
                        # Normal: NaN where masked
                        data_normal_ds[i*2] = np.nan if m1 else v1
                        data_normal_ds[i*2+1] = np.nan if m2 else v2
                        # Masked: NaN where NOT masked
                        data_masked_ds[i*2] = v1 if m1 else np.nan
                        data_masked_ds[i*2+1] = v2 if m2 else np.nan
                
                self.lod_normal[factor] = np.column_stack([time_ds, data_normal_ds]).astype(np.float32)
                self.lod_masked[factor] = np.column_stack([time_ds, data_masked_ds]).astype(np.float32)
    
    def _create_lines(self):
        """Create two line visuals - normal and masked (lighter)."""
        # Normal line (full color)
        self.line_normal = scene.Line(
            pos=self.lod_normal[self.current_lod],
            color=self.color,
            width=self.width,
            connect='strip',
            antialias=False,
            parent=self.viewbox.scene
        )
        self.line_normal.order = 1000
        
        # Masked line (lighter color)
        self.line_masked = scene.Line(
            pos=self.lod_masked[self.current_lod],
            color=self.light_color.rgba,
            width=self.width,
            connect='strip',
            antialias=False,
            parent=self.viewbox.scene
        )
        self.line_masked.order = 1000
    
    def update_for_view(self, x_min: float, x_max: float):
        """Update LOD based on current view range."""
        view_span = x_max - x_min
        total_span = self.time_full[-1] - self.time_full[0]
        zoom_ratio = total_span / max(view_span, 1e-6)
        
        if zoom_ratio > 50:
            target_lod = 1
        elif zoom_ratio > 10:
            target_lod = 10
        elif zoom_ratio > 2:
            target_lod = 100
        else:
            target_lod = 1000
        
        available = [f for f in self.lod_factors if f <= target_lod]
        new_lod = max(available) if available else min(self.lod_factors)
        
        if new_lod != self.current_lod:
            self.current_lod = new_lod
            self.line_normal.set_data(pos=self.lod_normal[self.current_lod])
            self.line_masked.set_data(pos=self.lod_masked[self.current_lod])
    
    def set_mask_visible(self, visible: bool):
        """Show/hide the masked (lighter) line."""
        if self.line_masked is not None:
            self.line_masked.visible = visible


class DynamicMaskRegions:
    """Mask regions that update based on visible range."""
    
    def __init__(
        self,
        viewbox: scene.ViewBox,
        time: np.ndarray,
        mask: np.ndarray,
        color: str = '#FFA500',
        alpha: float = 0.25
    ):
        self.viewbox = viewbox
        self.time = time.astype(np.float32)
        self.color = color
        self.alpha = alpha
        self.visible = True
        
        mask_arr = np.asarray(mask, dtype=bool)
        mask_diff = np.diff(np.concatenate([[False], mask_arr, [False]]).astype(int))
        self.starts = np.where(mask_diff == 1)[0]
        self.ends = np.where(mask_diff == -1)[0]
        
        self.start_times = self.time[np.clip(self.starts, 0, len(self.time)-1)]
        self.end_times = self.time[np.clip(self.ends - 1, 0, len(self.time)-1)]
        
        self.mesh = None
        self._create_mesh_for_range(self.time[0], self.time[-1])
    
    def _create_mesh_for_range(self, x_min: float, x_max: float, max_regions: int = 500):
        if not self.visible:
            if self.mesh is not None:
                self.mesh.visible = False
            return
        
        visible = (self.end_times >= x_min) & (self.start_times <= x_max)
        vis_starts = self.start_times[visible]
        vis_ends = self.end_times[visible]
        
        if len(vis_starts) == 0:
            if self.mesh is not None:
                self.mesh.visible = False
            return
        
        if len(vis_starts) > max_regions:
            indices = np.linspace(0, len(vis_starts)-1, max_regions, dtype=int)
            vis_starts = vis_starts[indices]
            vis_ends = vis_ends[indices]
        
        n_rects = len(vis_starts)
        y_min, y_max = -1e9, 1e9
        
        vertices = np.zeros((n_rects * 4, 2), dtype=np.float32)
        vertices[0::4, 0] = vis_starts
        vertices[0::4, 1] = y_min
        vertices[1::4, 0] = vis_ends
        vertices[1::4, 1] = y_min
        vertices[2::4, 0] = vis_ends
        vertices[2::4, 1] = y_max
        vertices[3::4, 0] = vis_starts
        vertices[3::4, 1] = y_max
        
        base_indices = np.arange(n_rects, dtype=np.uint32) * 4
        faces = np.zeros((n_rects * 2, 3), dtype=np.uint32)
        faces[0::2, 0] = base_indices
        faces[0::2, 1] = base_indices + 1
        faces[0::2, 2] = base_indices + 2
        faces[1::2, 0] = base_indices
        faces[1::2, 1] = base_indices + 2
        faces[1::2, 2] = base_indices + 3
        
        c = Color(self.color)
        rgba = list(c.rgba)
        rgba[3] = self.alpha
        
        if self.mesh is None:
            from vispy.visuals.filters import Alpha
            self.mesh = scene.Mesh(
                vertices=vertices, faces=faces, color=rgba,
                parent=self.viewbox.scene
            )
            self.mesh.order = -1000  # Way behind
            # Enable proper alpha blending
            self.mesh.set_gl_state('translucent', depth_test=False)
        else:
            self.mesh.set_data(vertices=vertices, faces=faces, color=rgba)
            self.mesh.visible = True
    
    def update_for_view(self, x_min: float, x_max: float):
        self._create_mesh_for_range(x_min, x_max)
    
    def set_visible(self, visible: bool):
        self.visible = visible
        if self.mesh is not None:
            self.mesh.visible = visible


class DynamicEventMarkers:
    """Event markers with rotated labels when zoomed in."""
    
    MAX_LABELS = 10
    
    def __init__(
        self,
        viewbox: scene.ViewBox,
        event_times: np.ndarray,
        event_labels: List[str],
        color: str = '#666666',
        alpha: float = 0.7,
        show_labels: bool = False
    ):
        self.viewbox = viewbox
        self.event_times = np.asarray(event_times, dtype=np.float32)
        self.event_labels = list(event_labels)
        self.color = color
        self.alpha = alpha
        self.show_labels = show_labels
        self.visible = True
        self._current_y_bottom = 0
        
        c = Color(self.color)
        self.rgba = list(c.rgba)
        self.rgba[3] = self.alpha
        
        self.line = None
        self.text_visuals: List[scene.Text] = []
        self._last_update_hash = None
        
        if self.show_labels:
            for _ in range(self.MAX_LABELS):
                text = scene.Text(
                    text='',
                    pos=(0, 0),
                    color='black',
                    font_size=12,
                    anchor_x='left',
                    anchor_y='bottom',
                    rotation=-90,
                    parent=self.viewbox.scene
                )
                text.order = 10
                text.visible = False
                self.text_visuals.append(text)
        
        self._create_line_for_all()
    
    def _create_line_for_all(self):
        if len(self.event_times) == 0:
            return
        
        y_min, y_max = -1e9, 1e9
        n_events = len(self.event_times)
        
        pos = np.zeros((n_events * 3, 2), dtype=np.float32)
        pos[0::3, 0] = self.event_times
        pos[0::3, 1] = y_min
        pos[1::3, 0] = self.event_times
        pos[1::3, 1] = y_max
        pos[2::3, 0] = np.nan
        pos[2::3, 1] = np.nan
        
        self.line = scene.Line(
            pos=pos, color=self.rgba, width=1.0,
            connect='strip', antialias=False,
            parent=self.viewbox.scene
        )
        self.line.order = -5
    
    def update_for_view(self, x_min: float, x_max: float, y_bottom: float = None):
        if not self.show_labels:
            return
        
        if y_bottom is None:
            try:
                rect = self.viewbox.camera.rect
                y_bottom = rect.bottom
            except:
                y_bottom = 0
        self._current_y_bottom = y_bottom
        
        visible_mask = (self.event_times >= x_min) & (self.event_times <= x_max)
        vis_indices = np.where(visible_mask)[0]
        n_visible = len(vis_indices)
        
        update_hash = (round(x_min, 1), round(x_max, 1), n_visible, round(y_bottom, 0))
        if update_hash == self._last_update_hash:
            return
        self._last_update_hash = update_hash
        
        for text in self.text_visuals:
            text.visible = False
        
        if not self.visible or n_visible == 0 or n_visible > self.MAX_LABELS:
            return
        
        for i, event_idx in enumerate(vis_indices[:self.MAX_LABELS]):
            t = self.event_times[event_idx]
            label = self.event_labels[event_idx]
            
            text = self.text_visuals[i]
            text.text = str(label)[:25]
            text.pos = (t, y_bottom)
            text.visible = True
    
    def set_visible(self, visible: bool):
        self.visible = visible
        if self.line is not None:
            self.line.visible = visible
        
        if not visible:
            for text in self.text_visuals:
                text.visible = False
        else:
            self._last_update_hash = None


class SelectionRegion:
    """Visual for user-selected regions (green semi-transparent rectangles)."""
    
    def __init__(self, viewbox: scene.ViewBox, color: str = '#00CC00', alpha: float = 0.3):
        """
        Initialize selection region visual.
        
        Parameters
        ----------
        viewbox : scene.ViewBox
            The viewbox to draw selections in
        color : str
            Color for selection regions (default green)
        alpha : float
            Transparency (0-1)
        """
        self.viewbox = viewbox
        self.color = color
        self.alpha = alpha
        self.intervals: List[Tuple[float, float]] = []  # (start, end) in seconds
        self.mesh = None
    
    def add_interval(self, start: float, end: float):
        """Add a new selection interval."""
        if start > end:
            start, end = end, start
        self.intervals.append((start, end))
        self._update_mesh()
    
    def remove_last(self) -> bool:
        """Remove the last added interval. Returns True if removed, False if empty."""
        if self.intervals:
            self.intervals.pop()
            self._update_mesh()
            return True
        return False
    
    def clear(self):
        """Remove all intervals."""
        self.intervals.clear()
        self._update_mesh()
    
    def _update_mesh(self):
        """Recreate mesh from current intervals."""
        if not self.intervals:
            if self.mesh is not None:
                self.mesh.visible = False
            return
        
        n_rects = len(self.intervals)
        y_min, y_max = -1e9, 1e9
        
        vertices = np.zeros((n_rects * 4, 2), dtype=np.float32)
        for i, (start, end) in enumerate(self.intervals):
            vertices[i*4 + 0] = [start, y_min]
            vertices[i*4 + 1] = [end, y_min]
            vertices[i*4 + 2] = [end, y_max]
            vertices[i*4 + 3] = [start, y_max]
        
        base_indices = np.arange(n_rects, dtype=np.uint32) * 4
        faces = np.zeros((n_rects * 2, 3), dtype=np.uint32)
        faces[0::2, 0] = base_indices
        faces[0::2, 1] = base_indices + 1
        faces[0::2, 2] = base_indices + 2
        faces[1::2, 0] = base_indices
        faces[1::2, 1] = base_indices + 2
        faces[1::2, 2] = base_indices + 3
        
        c = Color(self.color)
        rgba = list(c.rgba)
        rgba[3] = self.alpha
        
        if self.mesh is None:
            self.mesh = scene.Mesh(
                vertices=vertices, faces=faces, color=rgba,
                parent=self.viewbox.scene
            )
            self.mesh.order = 500  # Above mask regions, below signal lines
            self.mesh.set_gl_state('translucent', depth_test=False)
        else:
            self.mesh.set_data(vertices=vertices, faces=faces, color=rgba)
            self.mesh.visible = True
    
    def get_intervals(self) -> List[Tuple[float, float]]:
        """Return list of (start, end) intervals in seconds."""
        return list(self.intervals)


class HighlightRegion:
    """Visual for pre-defined highlight regions (semi-transparent rectangles)."""
    
    def __init__(self, viewbox: scene.ViewBox, intervals: List[Tuple[float, float]], 
                 color: str = 'lightblue', alpha: float = 0.3):
        """
        Initialize highlight region visual with pre-defined intervals.
        
        Parameters
        ----------
        viewbox : scene.ViewBox
            The viewbox to draw highlights in
        intervals : list of tuples
            List of (start, end) tuples in seconds
        color : str
            Color for highlight regions (default 'lightblue')
        alpha : float
            Transparency (0-1)
        """
        self.viewbox = viewbox
        self.color = color
        self.alpha = alpha
        self.intervals = [(min(s, e), max(s, e)) for s, e in intervals]
        self.mesh = None
        self._create_mesh()
    
    def _create_mesh(self):
        """Create mesh from intervals."""
        if not self.intervals:
            return
        
        n_rects = len(self.intervals)
        y_min, y_max = -1e9, 1e9
        
        vertices = np.zeros((n_rects * 4, 2), dtype=np.float32)
        for i, (start, end) in enumerate(self.intervals):
            vertices[i*4 + 0] = [start, y_min]
            vertices[i*4 + 1] = [end, y_min]
            vertices[i*4 + 2] = [end, y_max]
            vertices[i*4 + 3] = [start, y_max]
        
        base_indices = np.arange(n_rects, dtype=np.uint32) * 4
        faces = np.zeros((n_rects * 2, 3), dtype=np.uint32)
        faces[0::2, 0] = base_indices
        faces[0::2, 1] = base_indices + 1
        faces[0::2, 2] = base_indices + 2
        faces[1::2, 0] = base_indices
        faces[1::2, 1] = base_indices + 2
        faces[1::2, 2] = base_indices + 3
        
        c = Color(self.color)
        rgba = list(c.rgba)
        rgba[3] = self.alpha
        
        self.mesh = scene.Mesh(
            vertices=vertices, faces=faces, color=rgba,
            parent=self.viewbox.scene
        )
        self.mesh.order = 400  # Below selection regions, above mask regions
        self.mesh.set_gl_state('translucent', depth_test=False)
    
    def set_visible(self, visible: bool):
        """Show/hide the highlight regions."""
        if self.mesh is not None:
            self.mesh.visible = visible
