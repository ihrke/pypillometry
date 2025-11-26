"""GPU visual elements for the viewer (lines, masks, events) with LOD support."""

import numpy as np
from vispy import scene
from vispy.color import Color
from typing import List, Optional, Tuple, Dict
import numpy.ma as ma


class LODLine:
    """Line visual with Level of Detail support.
    
    Pre-computes multiple downsampled versions of the data and switches
    between them based on the current view range.
    """
    
    def __init__(
        self,
        viewbox: scene.ViewBox,
        time: np.ndarray,
        data: np.ndarray,
        color: str = '#0000FF',
        width: float = 1.0,
        lod_factors: Tuple[int, ...] = (1, 10, 100, 1000)
    ):
        self.viewbox = viewbox
        self.time_full = time.astype(np.float32)
        self.color = color
        self.width = width
        self.lod_factors = lod_factors
        self.n_points = len(time)
        
        # Handle masked arrays
        if ma.is_masked(data):
            self.data_full = data.data.copy().astype(np.float32)
            self.data_full[data.mask] = np.nan
        else:
            self.data_full = np.asarray(data, dtype=np.float32)
        
        # Pre-compute LOD levels
        self.lod_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._precompute_lods()
        
        # Create line visual (start with most downsampled)
        self.current_lod = max(lod_factors)
        time_lod, data_lod = self.lod_data[self.current_lod]
        pos = np.column_stack([time_lod, data_lod])
        
        self.line = scene.Line(
            pos=pos,
            color=color,
            width=width,
            connect='strip',
            antialias=False,
            parent=viewbox.scene
        )
    
    def _precompute_lods(self):
        """Pre-compute downsampled versions using min-max decimation."""
        for factor in self.lod_factors:
            if factor == 1:
                # Full resolution
                self.lod_data[1] = (self.time_full, self.data_full)
            else:
                # Downsample using min-max to preserve peaks
                n_out = max(1, self.n_points // factor)
                time_ds = np.zeros(n_out * 2, dtype=np.float32)
                data_ds = np.zeros(n_out * 2, dtype=np.float32)
                
                for i in range(n_out):
                    start = i * factor
                    end = min(start + factor, self.n_points)
                    chunk = self.data_full[start:end]
                    
                    # Handle NaN chunks
                    if np.all(np.isnan(chunk)):
                        time_ds[i*2] = self.time_full[start]
                        time_ds[i*2+1] = self.time_full[end-1] if end > start else self.time_full[start]
                        data_ds[i*2] = np.nan
                        data_ds[i*2+1] = np.nan
                    else:
                        min_idx = np.nanargmin(chunk)
                        max_idx = np.nanargmax(chunk)
                        
                        # Order by time (min first if it comes first)
                        if min_idx <= max_idx:
                            time_ds[i*2] = self.time_full[start + min_idx]
                            time_ds[i*2+1] = self.time_full[start + max_idx]
                            data_ds[i*2] = chunk[min_idx]
                            data_ds[i*2+1] = chunk[max_idx]
                        else:
                            time_ds[i*2] = self.time_full[start + max_idx]
                            time_ds[i*2+1] = self.time_full[start + min_idx]
                            data_ds[i*2] = chunk[max_idx]
                            data_ds[i*2+1] = chunk[min_idx]
                
                self.lod_data[factor] = (time_ds, data_ds)
    
    def update_for_view(self, x_min: float, x_max: float):
        """Update LOD based on current view range."""
        view_span = x_max - x_min
        total_span = self.time_full[-1] - self.time_full[0]
        
        # Choose LOD based on how much of the data is visible
        # More zoomed in = lower LOD factor (more detail)
        zoom_ratio = total_span / max(view_span, 1e-6)
        
        # Select appropriate LOD
        # If zoom_ratio > 100, we're very zoomed in -> use full res
        # If zoom_ratio < 2, we're zoomed out -> use heavy downsampling
        if zoom_ratio > 50:
            target_lod = 1
        elif zoom_ratio > 10:
            target_lod = 10
        elif zoom_ratio > 2:
            target_lod = 100
        else:
            target_lod = 1000
        
        # Find closest available LOD
        available = [f for f in self.lod_factors if f <= target_lod]
        new_lod = max(available) if available else min(self.lod_factors)
        
        if new_lod != self.current_lod:
            self.current_lod = new_lod
            time_lod, data_lod = self.lod_data[new_lod]
            pos = np.column_stack([time_lod, data_lod])
            self.line.set_data(pos=pos)


class DynamicMaskRegions:
    """Mask regions that update based on visible range."""
    
    def __init__(
        self,
        viewbox: scene.ViewBox,
        time: np.ndarray,
        mask: np.ndarray,
        color: str = '#FFA500',
        alpha: float = 0.3
    ):
        self.viewbox = viewbox
        self.time = time.astype(np.float32)
        self.color = color
        self.alpha = alpha
        
        # Pre-compute all mask intervals once
        mask_arr = np.asarray(mask, dtype=bool)
        mask_diff = np.diff(np.concatenate([[False], mask_arr, [False]]).astype(int))
        self.starts = np.where(mask_diff == 1)[0]
        self.ends = np.where(mask_diff == -1)[0]
        
        # Convert to time values
        self.start_times = self.time[np.clip(self.starts, 0, len(self.time)-1)]
        self.end_times = self.time[np.clip(self.ends - 1, 0, len(self.time)-1)]
        
        self.mesh = None
        self._create_mesh_for_range(self.time[0], self.time[-1])
    
    def _create_mesh_for_range(self, x_min: float, x_max: float, max_regions: int = 500):
        """Create mesh for regions visible in the given range."""
        # Find regions that overlap with view
        visible = (self.end_times >= x_min) & (self.start_times <= x_max)
        vis_starts = self.start_times[visible]
        vis_ends = self.end_times[visible]
        
        if len(vis_starts) == 0:
            if self.mesh is not None:
                self.mesh.visible = False
            return
        
        # Limit for performance, but prioritize regions in center of view
        if len(vis_starts) > max_regions:
            # Keep all regions, just subsample evenly
            indices = np.linspace(0, len(vis_starts)-1, max_regions, dtype=int)
            vis_starts = vis_starts[indices]
            vis_ends = vis_ends[indices]
        
        n_rects = len(vis_starts)
        
        # Build mesh vertices (vectorized)
        y_min, y_max = -1e9, 1e9
        
        vertices = np.zeros((n_rects * 4, 2), dtype=np.float32)
        vertices[0::4, 0] = vis_starts  # bottom-left x
        vertices[0::4, 1] = y_min       # bottom-left y
        vertices[1::4, 0] = vis_ends    # bottom-right x
        vertices[1::4, 1] = y_min       # bottom-right y
        vertices[2::4, 0] = vis_ends    # top-right x
        vertices[2::4, 1] = y_max       # top-right y
        vertices[3::4, 0] = vis_starts  # top-left x
        vertices[3::4, 1] = y_max       # top-left y
        
        # Build faces (vectorized)
        base_indices = np.arange(n_rects, dtype=np.uint32) * 4
        faces = np.zeros((n_rects * 2, 3), dtype=np.uint32)
        faces[0::2, 0] = base_indices
        faces[0::2, 1] = base_indices + 1
        faces[0::2, 2] = base_indices + 2
        faces[1::2, 0] = base_indices
        faces[1::2, 1] = base_indices + 2
        faces[1::2, 2] = base_indices + 3
        
        # Parse color
        c = Color(self.color)
        rgba = list(c.rgba)
        rgba[3] = self.alpha
        
        if self.mesh is None:
            self.mesh = scene.Mesh(
                vertices=vertices,
                faces=faces,
                color=rgba,
                parent=self.viewbox.scene
            )
            self.mesh.order = -10
        else:
            self.mesh.set_data(vertices=vertices, faces=faces, color=rgba)
            self.mesh.visible = True
    
    def update_for_view(self, x_min: float, x_max: float):
        """Update mask regions for current view."""
        self._create_mesh_for_range(x_min, x_max)


class DynamicEventMarkers:
    """Event markers with labels that update based on visible range."""
    
    def __init__(
        self,
        viewbox: scene.ViewBox,
        event_times: np.ndarray,
        event_labels: List[str],
        color: str = '#888888',
        alpha: float = 0.6
    ):
        self.viewbox = viewbox
        self.event_times = np.asarray(event_times, dtype=np.float32)
        self.event_labels = list(event_labels)
        self.color = color
        self.alpha = alpha
        
        # Visual elements
        self.line = None
        self.text_visuals: List[scene.Text] = []
        
        # Create initial visuals
        self._update_visuals(self.event_times[0] if len(self.event_times) > 0 else 0,
                            self.event_times[-1] if len(self.event_times) > 0 else 1)
    
    def _update_visuals(self, x_min: float, x_max: float, max_labels: int = 30):
        """Update event markers and labels for current view."""
        # Find visible events
        visible = (self.event_times >= x_min) & (self.event_times <= x_max)
        vis_times = self.event_times[visible]
        vis_labels = [self.event_labels[i] for i, v in enumerate(visible) if v]
        
        if len(vis_times) == 0:
            if self.line is not None:
                self.line.visible = False
            for t in self.text_visuals:
                t.visible = False
            return
        
        # Parse color
        c = Color(self.color)
        rgba = list(c.rgba)
        rgba[3] = self.alpha
        
        # Build line positions (vectorized)
        y_min, y_max = -1e9, 1e9
        n_events = len(vis_times)
        
        pos = np.zeros((n_events * 3, 2), dtype=np.float32)
        pos[0::3, 0] = vis_times
        pos[0::3, 1] = y_min
        pos[1::3, 0] = vis_times
        pos[1::3, 1] = y_max
        pos[2::3, 0] = np.nan
        pos[2::3, 1] = np.nan
        
        if self.line is None:
            self.line = scene.Line(
                pos=pos,
                color=rgba,
                width=1.0,
                connect='strip',
                antialias=False,
                parent=self.viewbox.scene
            )
            self.line.order = -5
        else:
            self.line.set_data(pos=pos, color=rgba)
            self.line.visible = True
        
        # Update text labels
        # Show up to max_labels, evenly distributed
        label_step = max(1, len(vis_times) // max_labels)
        n_labels_to_show = min(len(vis_times), max_labels)
        
        # Reuse or create text visuals
        for i, text in enumerate(self.text_visuals):
            text.visible = False
        
        label_idx = 0
        for i in range(0, len(vis_times), label_step):
            if label_idx >= n_labels_to_show:
                break
            
            t = vis_times[i]
            label = vis_labels[i]
            
            if label_idx < len(self.text_visuals):
                # Reuse existing
                text = self.text_visuals[label_idx]
                text.text = str(label)[:15]
                text.pos = (t, 0)
                text.visible = True
            else:
                # Create new
                text = scene.Text(
                    text=str(label)[:15],
                    pos=(t, 0),
                    color=self.color,
                    font_size=7,
                    anchor_x='left',
                    anchor_y='top',
                    parent=self.viewbox.scene
                )
                text.order = 10
                self.text_visuals.append(text)
            
            label_idx += 1
    
    def update_for_view(self, x_min: float, x_max: float):
        """Update event markers for current view."""
        self._update_visuals(x_min, x_max)
