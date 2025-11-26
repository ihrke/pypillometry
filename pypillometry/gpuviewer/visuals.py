"""GPU visual elements for the viewer (lines, masks, events) with LOD support."""

import numpy as np
from vispy import scene
from vispy.color import Color
from typing import List, Optional, Tuple, Dict
import numpy.ma as ma


class LODLine:
    """Line visual with Level of Detail support and per-vertex coloring for masked data."""
    
    def __init__(
        self,
        viewbox: scene.ViewBox,
        time: np.ndarray,
        data: np.ndarray,
        color: str = '#0000FF',
        masked_color: str = '#FF00FF',
        mask: np.ndarray = None,
        width: float = 1.0,
        lod_factors: Tuple[int, ...] = (1, 10, 100, 1000)
    ):
        self.viewbox = viewbox
        self.time_full = time.astype(np.float32)
        self.width = width
        self.lod_factors = lod_factors
        self.n_points = len(time)
        
        # Parse colors to RGBA
        self.color_rgba = np.array(Color(color).rgba, dtype=np.float32)
        self.masked_color_rgba = np.array(Color(masked_color).rgba, dtype=np.float32)
        
        # Get raw data and mask
        if ma.is_masked(data):
            self.data_full = data.data.copy().astype(np.float32)
            raw_mask = mask if mask is not None else data.mask
            self.mask = np.asarray(raw_mask, dtype=bool)
        else:
            self.data_full = np.asarray(data, dtype=np.float32)
            self.mask = np.asarray(mask, dtype=bool) if mask is not None else np.zeros(len(data), dtype=bool)
        
        # Pre-compute LOD levels
        self.lod_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._precompute_lods()
        
        # Create single line with per-vertex colors
        self.current_lod = max(lod_factors)
        self.line = None
        self.masked_line_visible = True
        self._create_line()
    
    def _precompute_lods(self):
        """Pre-compute downsampled versions."""
        for factor in self.lod_factors:
            if factor == 1:
                self.lod_data[1] = (self.time_full.copy(), self.data_full.copy(), self.mask.copy())
            else:
                n_out = max(1, self.n_points // factor)
                time_ds = np.zeros(n_out * 2, dtype=np.float32)
                data_ds = np.zeros(n_out * 2, dtype=np.float32)
                mask_ds = np.zeros(n_out * 2, dtype=bool)
                
                for i in range(n_out):
                    start = i * factor
                    end = min(start + factor, self.n_points)
                    chunk = self.data_full[start:end]
                    chunk_mask = self.mask[start:end]
                    
                    valid = np.isfinite(chunk)
                    if not np.any(valid):
                        time_ds[i*2] = self.time_full[start]
                        time_ds[i*2+1] = self.time_full[end-1]
                        data_ds[i*2] = np.nan
                        data_ds[i*2+1] = np.nan
                        mask_ds[i*2] = True
                        mask_ds[i*2+1] = True
                    else:
                        valid_chunk = np.where(valid, chunk, np.inf)
                        min_idx = np.argmin(valid_chunk)
                        valid_chunk = np.where(valid, chunk, -np.inf)
                        max_idx = np.argmax(valid_chunk)
                        
                        if min_idx <= max_idx:
                            time_ds[i*2] = self.time_full[start + min_idx]
                            time_ds[i*2+1] = self.time_full[start + max_idx]
                            data_ds[i*2] = chunk[min_idx]
                            data_ds[i*2+1] = chunk[max_idx]
                            mask_ds[i*2] = chunk_mask[min_idx]
                            mask_ds[i*2+1] = chunk_mask[max_idx]
                        else:
                            time_ds[i*2] = self.time_full[start + max_idx]
                            time_ds[i*2+1] = self.time_full[start + min_idx]
                            data_ds[i*2] = chunk[max_idx]
                            data_ds[i*2+1] = chunk[min_idx]
                            mask_ds[i*2] = chunk_mask[max_idx]
                            mask_ds[i*2+1] = chunk_mask[min_idx]
                
                self.lod_data[factor] = (time_ds, data_ds, mask_ds)
    
    def _create_line(self):
        """Create line with per-vertex colors."""
        time_lod, data_lod, mask_lod = self.lod_data[self.current_lod]
        
        pos = np.column_stack([time_lod, data_lod]).astype(np.float32)
        
        # Create per-vertex colors
        n_points = len(time_lod)
        colors = np.zeros((n_points, 4), dtype=np.float32)
        
        # Normal points get normal color, masked points get masked color
        colors[~mask_lod] = self.color_rgba
        if self.masked_line_visible:
            colors[mask_lod] = self.masked_color_rgba
        else:
            # Make masked points transparent
            colors[mask_lod] = [0, 0, 0, 0]
        
        self.line = scene.Line(
            pos=pos,
            color=colors,
            width=self.width,
            connect='strip',
            antialias=False,
            parent=self.viewbox.scene
        )
        self.line.order = 5
    
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
            self._update_line_data()
    
    def _update_line_data(self):
        """Update line with current LOD data."""
        time_lod, data_lod, mask_lod = self.lod_data[self.current_lod]
        
        pos = np.column_stack([time_lod, data_lod]).astype(np.float32)
        
        n_points = len(time_lod)
        colors = np.zeros((n_points, 4), dtype=np.float32)
        colors[~mask_lod] = self.color_rgba
        if self.masked_line_visible:
            colors[mask_lod] = self.masked_color_rgba
        else:
            colors[mask_lod] = [0, 0, 0, 0]
        
        self.line.set_data(pos=pos, color=colors)
    
    def set_mask_visible(self, visible: bool):
        """Show/hide the masked portion of the line."""
        self.masked_line_visible = visible
        self._update_line_data()


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
            self.mesh = scene.Mesh(
                vertices=vertices, faces=faces, color=rgba,
                parent=self.viewbox.scene
            )
            self.mesh.order = -10
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
