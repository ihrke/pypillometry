"""Interactive parameter tweaking viewer using VisPy and Qt.

This module provides a viewer for interactively tweaking function parameters
while seeing the result overlaid on the original data.
"""

import numpy as np
from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.color import Color
from typing import Dict, Callable, Any, Optional, List, Union, Tuple

from .visuals import LODLine, DynamicMaskRegions, HighlightRegion
from .navigation import NavigationHandler

# Colors for multiple overlay lines
OVERLAY_COLORS = [
    '#00DD00',  # Green
    '#DD00DD',  # Magenta
    '#00DDDD',  # Cyan
    '#DDDD00',  # Yellow
    '#FF6600',  # Orange
    '#6600FF',  # Violet
]

# Colors for interval highlights
INTERVAL_COLORS = [
    '#87CEEB',  # Light blue
    '#90EE90',  # Light green
    '#FFB6C1',  # Light pink
    '#FFFACD',  # Lemon chiffon
    '#E6E6FA',  # Lavender
    '#FFDAB9',  # Peach puff
]


class TweakCanvas(SceneCanvas):
    """GPU-accelerated canvas for parameter tweaking with original data + overlay."""
    
    def __init__(self, time_seconds: np.ndarray, original_data: np.ndarray,
                 original_mask: np.ndarray = None,
                 overlay_data: Union[np.ndarray, Dict[str, np.ndarray]] = None,
                 overlay_mask: np.ndarray = None,
                 overlay_intervals: Dict[str, List[Tuple[float, float]]] = None,
                 sampling_rate: float = None,
                 title: str = 'Tweak Viewer'):
        """Initialize the tweak canvas.
        
        Parameters
        ----------
        time_seconds : ndarray
            Time vector in seconds.
        original_data : ndarray
            Original data to display (1D array).
        original_mask : ndarray, optional
            Boolean mask for original data (True = masked/invalid).
        overlay_data : ndarray or dict, optional
            Initial overlay data from function. Can be:
            - Single 1D array (same length as original)
            - Dict mapping names to 1D arrays (all same length)
        overlay_mask : ndarray, optional
            Boolean mask for overlay data (True = masked/invalid).
        overlay_intervals : dict, optional
            Dict mapping names to lists of (start, end) tuples in seconds.
            These are displayed as highlighted regions.
        sampling_rate : float, optional
            Sampling rate in Hz (used for Intervals conversion).
        title : str
            Window title.
        """
        super().__init__(
            keys='interactive',
            size=(1200, 600),
            bgcolor='white',
            title=title
        )
        
        self.unfreeze()
        
        self.time_seconds = time_seconds.astype(np.float32)
        self.original_data = original_data.astype(np.float32)
        self.original_mask = original_mask
        self.overlay_mask = overlay_mask
        self.sampling_rate = sampling_rate
        self.data_min = float(self.time_seconds[0])
        self.data_max = float(self.time_seconds[-1])
        
        # Store overlay data (can be dict or single array)
        self._overlay_data_dict: Dict[str, np.ndarray] = {}
        if overlay_data is not None:
            if isinstance(overlay_data, dict):
                self._overlay_data_dict = {
                    k: np.asarray(v, dtype=np.float32) for k, v in overlay_data.items()
                }
            else:
                self._overlay_data_dict = {'tweaked': np.asarray(overlay_data, dtype=np.float32)}
        
        # Store overlay intervals
        self._overlay_intervals_dict: Dict[str, List[Tuple[float, float]]] = overlay_intervals or {}
        
        # LOD factors based on data length
        n_points = len(time_seconds)
        if n_points > 1000000:
            self.lod_factors = (1, 10, 100, 1000)
        elif n_points > 100000:
            self.lod_factors = (1, 10, 100)
        elif n_points > 10000:
            self.lod_factors = (1, 10)
        else:
            self.lod_factors = (1,)
        
        # Storage for visuals
        self.original_line: LODLine = None
        self.overlay_lines: Dict[str, LODLine] = {}
        self.mask_regions: List[DynamicMaskRegions] = []
        self.interval_highlights: Dict[str, HighlightRegion] = {}
        
        # Visibility state
        self.masks_visible = True
        self.intervals_visible = True
        
        # Reference to parent viewer for closing
        self._viewer = None
        
        # Y-axis zoom state
        self.manual_y_range: Optional[tuple] = None
        self._mouse_pos = None
        
        # Create layout
        self.grid = self.central_widget.add_grid(spacing=0)
        self._create_subplot()
        
        # Navigation handler
        self.navigation = NavigationHandler(
            [self.viewbox],
            data_min=self.data_min,
            data_max=self.data_max
        )
        
        # Create mask regions first (behind everything)
        self._create_mask_regions()
        
        # Create interval highlights (above masks, below lines)
        self._create_interval_highlights()
        
        # Plot data - original first, then overlays on top
        self._plot_original()
        self._plot_overlays()
        
        self._create_legend()
        self._set_initial_view()
        self._last_x_range = (self.data_min, self.data_max)
        
        self.freeze()
    
    def _create_subplot(self):
        """Create single subplot with y-axis and viewbox."""
        # Y-axis
        y_axis = scene.AxisWidget(
            orientation='left',
            axis_font_size=8,
            axis_label='Value',
            axis_label_margin=50,
            tick_label_margin=5,
            text_color='black',
            axis_color='black',
            tick_color='black',
        )
        y_axis.stretch = (0.08, 1)
        self.grid.add_widget(y_axis, row=0, col=0)
        
        # Viewbox
        self.viewbox = self.grid.add_view(row=0, col=1, border_color='#cccccc')
        self.viewbox.stretch = (1, 1)
        self.viewbox.height_min = 50
        
        camera = scene.PanZoomCamera(aspect=None)
        camera.interactive = False
        self.viewbox.camera = camera
        
        y_axis.link_view(self.viewbox)
        
        # X-axis
        is_sample_index = (
            len(self.time_seconds) > 1 and
            np.allclose(self.time_seconds, np.arange(len(self.time_seconds)))
        )
        x_label = 'Sample' if is_sample_index else 'Time (s)'
        
        x_axis = scene.AxisWidget(
            orientation='bottom',
            axis_label=x_label,
            axis_font_size=8,
            axis_label_margin=30,
            tick_label_margin=5,
            text_color='black',
            axis_color='black',
            tick_color='black',
        )
        x_axis.height_min = 50
        x_axis.height_max = 50
        x_axis.stretch = (1, 0.0001)
        self.grid.add_widget(x_axis, row=1, col=1)
        x_axis.link_view(self.viewbox)
    
    def _create_mask_regions(self):
        """Create orange highlight regions for masked data."""
        # Clear existing mask regions
        for region in self.mask_regions:
            if region.mesh is not None:
                region.mesh.parent = None
        self.mask_regions = []
        
        # Create mask region for original data
        if self.original_mask is not None and np.any(self.original_mask):
            mask_vis = DynamicMaskRegions(
                self.viewbox, self.time_seconds, self.original_mask,
                color='#FFA500', alpha=0.25
            )
            self.mask_regions.append(mask_vis)
        
        # Create mask region for overlay data (if different from original)
        if self.overlay_mask is not None and np.any(self.overlay_mask):
            # Only add if different from original mask
            if self.original_mask is None or not np.array_equal(self.original_mask, self.overlay_mask):
                # Use a slightly different color for overlay mask
                mask_vis = DynamicMaskRegions(
                    self.viewbox, self.time_seconds, self.overlay_mask,
                    color='#FF6600', alpha=0.2
                )
                self.mask_regions.append(mask_vis)
    
    def _create_interval_highlights(self):
        """Create highlight regions for overlay intervals."""
        # Clear existing interval highlights
        for region in self.interval_highlights.values():
            if region.mesh is not None:
                region.mesh.parent = None
        self.interval_highlights = {}
        
        if not self._overlay_intervals_dict:
            return
        
        for idx, (name, intervals) in enumerate(self._overlay_intervals_dict.items()):
            if not intervals:
                continue
            color = INTERVAL_COLORS[idx % len(INTERVAL_COLORS)]
            highlight = HighlightRegion(
                self.viewbox, intervals, color=color, alpha=0.35
            )
            self.interval_highlights[name] = highlight
    
    def _plot_original(self):
        """Plot the original data line."""
        # Determine mask for the line (NaN-based)
        mask = None
        if self.original_mask is not None:
            mask = self.original_mask
        
        self.original_line = LODLine(
            self.viewbox,
            self.time_seconds,
            self.original_data,
            color='#0066CC',  # Blue
            mask=mask,
            width=2.0,
            lod_factors=self.lod_factors
        )
        # Set order so original is drawn above masks but below overlays
        if self.original_line.line_normal:
            self.original_line.line_normal.order = 1000
        if self.original_line.line_masked:
            self.original_line.line_masked.order = 1000
    
    def _plot_overlays(self):
        """Plot all overlay lines."""
        # Clear existing overlay lines
        for line in self.overlay_lines.values():
            if line.line_normal is not None:
                line.line_normal.parent = None
            if line.line_masked is not None:
                line.line_masked.parent = None
        self.overlay_lines = {}
        
        if not self._overlay_data_dict:
            return
        
        # Plot each overlay with different colors
        for idx, (name, data) in enumerate(self._overlay_data_dict.items()):
            color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
            
            # Determine mask for overlay
            mask = None
            if self.overlay_mask is not None:
                mask = self.overlay_mask
            
            line = LODLine(
                self.viewbox,
                self.time_seconds,
                data,
                color=color,
                mask=mask,
                width=2.5,
                lod_factors=self.lod_factors
            )
            # Set order so overlays are on top
            if line.line_normal:
                line.line_normal.order = 2000 + idx
            if line.line_masked:
                line.line_masked.order = 2000 + idx
            
            self.overlay_lines[name] = line
        
        # Update LOD for current view
        if hasattr(self, '_last_x_range'):
            x_min, x_max = self._last_x_range
            for line in self.overlay_lines.values():
                line.update_for_view(x_min, x_max)
    
    def update_overlay(self, data: Union[np.ndarray, Dict[str, np.ndarray]], 
                       mask: np.ndarray = None):
        """Update the overlay with new data.
        
        Parameters
        ----------
        data : ndarray or dict
            New overlay data. Can be single array or dict of arrays.
        mask : ndarray, optional
            New mask for overlay data.
        """
        self.unfreeze()
        
        # Update overlay data dict
        if isinstance(data, dict):
            self._overlay_data_dict = {
                k: np.asarray(v, dtype=np.float32) for k, v in data.items()
            }
        else:
            self._overlay_data_dict = {'tweaked': np.asarray(data, dtype=np.float32)}
        
        # Update overlay mask
        if mask is not None:
            self.overlay_mask = mask
            self._create_mask_regions()
        
        self._plot_overlays()
        self._update_legend()
        self._update_y_range()
        self.update()
        self.freeze()
    
    def update_intervals(self, intervals_dict: Dict[str, List[Tuple[float, float]]]):
        """Update interval highlights with new data.
        
        Parameters
        ----------
        intervals_dict : dict
            Dict mapping names to lists of (start, end) tuples in seconds.
        """
        self.unfreeze()
        self._overlay_intervals_dict = intervals_dict
        self._create_interval_highlights()
        self._update_legend()
        self.update()
        self.freeze()
    
    def _create_legend(self):
        """Create a legend showing original vs tweaked."""
        legend_row = 2
        self.legend_view = self.grid.add_view(row=legend_row, col=0, col_span=2, border_color=None)
        self.legend_view.camera = scene.PanZoomCamera(aspect=None)
        self.legend_view.camera.interactive = False
        self.legend_view.height_min = 30
        self.legend_view.height_max = 30
        self.legend_view.stretch = (1, 0.0001)
        
        self._update_legend()
    
    def _update_legend(self):
        """Update legend to reflect current overlay names."""
        # Clear existing legend items
        for child in list(self.legend_view.scene.children):
            child.parent = None
        
        # Build legend items: (label, color, is_interval)
        legend_items = [('Original', '#0066CC', False)]
        
        # Add array overlays (lines)
        for idx, name in enumerate(self._overlay_data_dict.keys()):
            color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
            legend_items.append((name, color, False))
        
        # Add interval overlays (rectangles)
        for idx, name in enumerate(self._overlay_intervals_dict.keys()):
            color = INTERVAL_COLORS[idx % len(INTERVAL_COLORS)]
            legend_items.append((name, color, True))
        
        total_width = sum(len(label) * 0.012 + 0.06 for label, _, _ in legend_items)
        x_start = 0.5 - total_width / 2
        x_pos = x_start
        
        for label, color, is_interval in legend_items:
            if is_interval:
                # Draw a filled rectangle for intervals
                rect_verts = np.array([
                    [x_pos - 0.012, 0.3],
                    [x_pos + 0.012, 0.3],
                    [x_pos + 0.012, 0.7],
                    [x_pos - 0.012, 0.7],
                ], dtype=np.float32)
                c = Color(color)
                rgba = list(c.rgba)
                rgba[3] = 0.5
                scene.Mesh(
                    vertices=rect_verts,
                    faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
                    color=rgba,
                    parent=self.legend_view.scene
                )
            else:
                # Draw a line for arrays
                line_pos = np.array([
                    [x_pos - 0.015, 0.5],
                    [x_pos + 0.015, 0.5]
                ], dtype=np.float32)
                scene.Line(pos=line_pos, color=color, width=4, parent=self.legend_view.scene)
            
            scene.Text(
                text=label,
                pos=(x_pos + 0.02, 0.5),
                color='black',
                font_size=8,
                anchor_x='left',
                anchor_y='center',
                parent=self.legend_view.scene
            )
            
            x_pos += len(label) * 0.012 + 0.06
    
    def _set_initial_view(self):
        """Set initial view range."""
        x_min = self.data_min
        x_max = self.data_max
        self.navigation.set_view(x_min, x_max)
        self._set_view_range(x_min, x_max)
    
    def _set_view_range(self, x_min: float, x_max: float):
        """Set the view range for the plot."""
        if self.manual_y_range:
            y_min, y_max = self.manual_y_range
        else:
            y_min, y_max = self._get_y_range(x_min, x_max)
        self._safe_set_camera_range(self.viewbox.camera, x_min, x_max, y_min, y_max)
        self._update_lod_visuals(x_min, x_max)
        self._last_x_range = (x_min, x_max)
    
    def _update_y_range(self):
        """Update Y range based on current X range."""
        if hasattr(self, '_last_x_range'):
            x_min, x_max = self._last_x_range
            if self.manual_y_range:
                y_min, y_max = self.manual_y_range
            else:
                y_min, y_max = self._get_y_range(x_min, x_max)
            self._safe_set_camera_range(self.viewbox.camera, x_min, x_max, y_min, y_max)
    
    def _safe_set_camera_range(self, camera, x_min: float, x_max: float,
                                y_min: float, y_max: float):
        """Set camera range with validation."""
        x_range = x_max - x_min
        if x_range < 1e-6:
            x_center = (x_min + x_max) / 2
            x_min = x_center - 0.5
            x_max = x_center + 0.5
        
        y_range = y_max - y_min
        if y_range < 1e-6:
            y_center = (y_min + y_max) / 2
            y_min = y_center - 0.5
            y_max = y_center + 0.5
        
        try:
            rect = camera._viewbox.rect
            if rect is None or rect.width <= 0 or rect.height <= 0:
                return
        except (AttributeError, TypeError):
            pass
        
        try:
            camera.set_range(x=(x_min, x_max), y=(y_min, y_max))
        except (np.linalg.LinAlgError, Exception):
            pass
    
    def _get_y_range(self, x_min: float, x_max: float) -> tuple:
        """Get Y range for visible data."""
        time = self.time_seconds
        start_idx = np.searchsorted(time, x_min, side='left')
        end_idx = np.searchsorted(time, x_max, side='right')
        
        y_min, y_max = float('inf'), float('-inf')
        
        # Original data
        visible = self.original_data[start_idx:end_idx]
        valid = np.isfinite(visible)
        if np.any(valid):
            y_min = min(y_min, float(np.nanmin(visible[valid])))
            y_max = max(y_max, float(np.nanmax(visible[valid])))
        
        # All overlay data
        for data in self._overlay_data_dict.values():
            overlay_visible = data[start_idx:end_idx]
            valid = np.isfinite(overlay_visible)
            if np.any(valid):
                y_min = min(y_min, float(np.nanmin(overlay_visible[valid])))
                y_max = max(y_max, float(np.nanmax(overlay_visible[valid])))
        
        if y_min == float('inf'):
            return (0, 1)
        
        data_range = y_max - y_min
        if data_range < 1e-10:
            center = (y_min + y_max) / 2
            return (center - 0.5, center + 0.5)
        
        padding = data_range * 0.1
        return (y_min - padding, y_max + padding)
    
    def _update_lod_visuals(self, x_min: float, x_max: float):
        """Update LOD for all lines and mask regions."""
        if self.original_line is not None:
            self.original_line.update_for_view(x_min, x_max)
        for line in self.overlay_lines.values():
            line.update_for_view(x_min, x_max)
        for mask_vis in self.mask_regions:
            mask_vis.update_for_view(x_min, x_max)
    
    def _toggle_masks(self):
        """Toggle visibility of mask regions."""
        self.masks_visible = not self.masks_visible
        for mask_vis in self.mask_regions:
            mask_vis.set_visible(self.masks_visible)
        self.update()
    
    def _toggle_intervals(self):
        """Toggle visibility of interval highlight regions."""
        self.intervals_visible = not self.intervals_visible
        for highlight in self.interval_highlights.values():
            highlight.set_visible(self.intervals_visible)
        self.update()
    
    def _zoom_y_axis(self, factor: float, center_y: Optional[float] = None):
        """Zoom Y-axis by factor, centered on center_y."""
        try:
            camera_rect = self.viewbox.camera.rect
            y_min = camera_rect.bottom
            y_max = camera_rect.top
            y_span = y_max - y_min
            
            if center_y is None:
                center_y = (y_min + y_max) / 2
            
            new_span = y_span * factor
            
            # Calculate new range centered on center_y
            rel_pos = (center_y - y_min) / y_span if y_span > 0 else 0.5
            new_y_min = center_y - new_span * rel_pos
            new_y_max = center_y + new_span * (1 - rel_pos)
            
            self.manual_y_range = (new_y_min, new_y_max)
            
            x_min, x_max = self._last_x_range
            self._safe_set_camera_range(self.viewbox.camera, x_min, x_max, new_y_min, new_y_max)
            self.update()
        except Exception:
            pass
    
    def _reset_y_axis(self):
        """Reset Y-axis to auto-fit mode."""
        self.manual_y_range = None
        self._update_y_range()
        self.update()
    
    def on_mouse_move(self, event):
        """Track mouse position for axis zoom."""
        self._mouse_pos = event.pos
    
    def _get_mouse_y_in_data_coords(self) -> Optional[float]:
        """Get the Y coordinate of mouse in data coordinates."""
        if self._mouse_pos is None:
            return None
        
        canvas_y = self._mouse_pos[1]
        canvas_height = self.size[1]
        
        try:
            # Calculate plot area dimensions (account for x-axis ~50px and legend ~30px)
            plot_area_top = 0
            plot_area_height = canvas_height - 80
            
            # Relative Y position within the plot area (0=top, 1=bottom)
            rel_y = (canvas_y - plot_area_top) / plot_area_height
            rel_y = max(0.0, min(1.0, rel_y))
            
            # Map to data coordinates (invert because canvas Y grows down)
            camera_rect = self.viewbox.camera.rect
            y_data = camera_rect.top - rel_y * camera_rect.height
            return float(y_data)
        except Exception:
            return None
    
    def _get_mouse_x_in_data_coords(self) -> Optional[float]:
        """Get the X coordinate of mouse in data coordinates."""
        if self._mouse_pos is None:
            return None
        
        canvas_x = self._mouse_pos[0]
        canvas_width = self.size[0]
        
        try:
            # Account for Y-axis label area on left (~60px)
            plot_left = 60
            plot_width = canvas_width - plot_left
            
            # Relative X position within plot area (0=left, 1=right)
            rel_x = (canvas_x - plot_left) / plot_width
            rel_x = max(0.0, min(1.0, rel_x))
            
            # Map to data coordinates
            camera_rect = self.viewbox.camera.rect
            x_data = camera_rect.left + rel_x * camera_rect.width
            return float(x_data)
        except Exception:
            return None
    
    def _zoom_x_axis(self, factor: float, center_x: Optional[float] = None):
        """Zoom X-axis by factor, centered on center_x.
        
        Parameters
        ----------
        factor : float
            Zoom factor. <1 zooms in, >1 zooms out.
        center_x : float, optional
            X coordinate to center zoom on. If None, uses current center.
        """
        try:
            # Get current X range from navigation
            x_min, x_max = self.navigation.current_x_min, self.navigation.current_x_max
            x_span = x_max - x_min
            
            if center_x is None:
                center_x = (x_min + x_max) / 2
            
            # Calculate new span
            new_span = x_span * factor
            
            # Clamp to data range
            if new_span >= (self.data_max - self.data_min) * 0.99:
                # Snap to full view
                new_x_min = self.data_min
                new_x_max = self.data_max
            else:
                # Calculate new range preserving relative position of center_x
                rel_pos = (center_x - x_min) / x_span if x_span > 0 else 0.5
                new_x_min = center_x - rel_pos * new_span
                new_x_max = center_x + (1 - rel_pos) * new_span
                
                # Clamp to data bounds
                if new_x_min < self.data_min:
                    new_x_min = self.data_min
                    new_x_max = new_x_min + new_span
                if new_x_max > self.data_max:
                    new_x_max = self.data_max
                    new_x_min = new_x_max - new_span
                    if new_x_min < self.data_min:
                        new_x_min = self.data_min
            
            # Update navigation state
            self.navigation.set_view(new_x_min, new_x_max)
            
            # Apply the new range
            self._set_view_range(new_x_min, new_x_max)
            self.update()
        except Exception:
            pass
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for axis zooming.
        
        - Scroll wheel (no modifier): X-axis zoom centered on mouse X position
        - Shift + scroll wheel: Y-axis zoom centered on mouse Y position
        """
        event.handled = True
        
        # Get wheel delta
        try:
            if hasattr(event.delta, '__len__') and len(event.delta) >= 2:
                delta = event.delta[1]
            else:
                delta = float(event.delta)
        except (TypeError, IndexError):
            delta = 0
        
        if delta == 0:
            return
        
        # Determine zoom factor based on scroll direction
        if delta > 0:
            factor = 0.8  # Zoom in
        else:
            factor = 1.25  # Zoom out
        
        # Check for Shift modifier
        modifiers = event.modifiers
        shift_held = 'Shift' in modifiers if modifiers else False
        
        if shift_held:
            # Y-axis zoom
            center_y = self._get_mouse_y_in_data_coords()
            self._zoom_y_axis(factor, center_y)
        else:
            # X-axis zoom
            center_x = self._get_mouse_x_in_data_coords()
            self._zoom_x_axis(factor, center_x)
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        key = event.key.name if hasattr(event.key, 'name') else str(event.key)
        modifiers = event.modifiers if hasattr(event, 'modifiers') else []
        shift_held = 'Shift' in modifiers if modifiers else False
        
        # Close window with Q or Escape
        if key in ('Q', 'Escape'):
            if self._viewer is not None:
                self._viewer.close()
            else:
                self.close()
            return
        
        # Toggle masks with 'M'
        if key == 'M':
            self._toggle_masks()
            return
        
        # Toggle intervals with 'I'
        if key == 'I':
            self._toggle_intervals()
            return
        
        # Y-axis zoom with Shift
        if shift_held:
            if key == 'Up':
                self._zoom_y_axis(0.8)  # Zoom in
                return
            elif key == 'Down':
                self._zoom_y_axis(1.25)  # Zoom out
                return
            elif key == 'Space':
                self._reset_y_axis()
                return
        
        # Use NavigationHandler for X-axis navigation
        new_range = self.navigation.handle_key_press(event)
        
        if new_range:
            self._set_view_range(*new_range)
            self.update()


class TweakViewer:
    """Main window combining plot canvas with embedded parameter controls."""
    
    def __init__(self, canvas: TweakCanvas, params: Dict[str, Any], 
                 on_change: Callable[[Dict[str, Any]], None],
                 title: str = 'Tweak Viewer'):
        """Initialize the tweak viewer with embedded parameter panel.
        
        Parameters
        ----------
        canvas : TweakCanvas
            The VisPy canvas for plotting.
        params : dict
            Initial parameter values (int or float).
        on_change : callable
            Callback function called with updated params dict when values change.
        title : str
            Window title.
        """
        self.canvas = canvas
        self.canvas._viewer = self  # Link canvas to viewer for closing
        self.params = dict(params)
        self.initial_params = dict(params)
        self.on_change = on_change
        self.spinboxes: Dict[str, Any] = {}
        self.auto_update = True
        self._pending_update = False
        self._computing = False
        
        # Import Qt
        try:
            from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                                         QHBoxLayout, QSplitter, QLabel, 
                                         QDoubleSpinBox, QSpinBox, QPushButton,
                                         QCheckBox, QFrame, QApplication)
            from PyQt6.QtCore import Qt
            self.Qt = Qt
            self.QDoubleSpinBox = QDoubleSpinBox
            self.QSpinBox = QSpinBox
            self.QApplication = QApplication
        except ImportError:
            from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                                         QHBoxLayout, QSplitter, QLabel, 
                                         QDoubleSpinBox, QSpinBox, QPushButton,
                                         QCheckBox, QFrame, QApplication)
            from PyQt5.QtCore import Qt
            self.Qt = Qt
            self.QDoubleSpinBox = QDoubleSpinBox
            self.QSpinBox = QSpinBox
            self.QApplication = QApplication
        
        # Create main window
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle(title)
        self.main_window.resize(1400, 600)
        
        # Create central widget with horizontal layout
        central_widget = QWidget()
        self.main_window.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for resizable panels
        splitter = QSplitter(self.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Add canvas to splitter (left side)
        canvas_widget = self.canvas.native
        splitter.addWidget(canvas_widget)
        
        # Create parameter panel (right side)
        param_panel = self._create_param_panel(params, QPushButton, QCheckBox, 
                                                QLabel, QVBoxLayout, QHBoxLayout, 
                                                QFrame, QWidget)
        splitter.addWidget(param_panel)
        
        # Set initial splitter sizes (85% canvas, 15% params)
        splitter.setSizes([1200, 200])
        
        # Set minimum width for param panel
        param_panel.setMinimumWidth(180)
        param_panel.setMaximumWidth(300)
    
    def _create_param_panel(self, params, QPushButton, QCheckBox, QLabel, 
                            QVBoxLayout, QHBoxLayout, QFrame, QWidget):
        """Create the parameter control panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Title
        title_label = QLabel("<b>Parameters</b>")
        layout.addWidget(title_label)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Create spinbox for each parameter
        for name, value in params.items():
            row = QHBoxLayout()
            
            label = QLabel(f"{name}:")
            label.setMinimumWidth(80)
            row.addWidget(label)
            
            if isinstance(value, int):
                spinbox = self.QSpinBox()
                spinbox.setRange(-999999999, 999999999)
                spinbox.setValue(value)
                spinbox.valueChanged.connect(lambda v, n=name: self._on_value_changed(n, v))
            else:
                spinbox = self.QDoubleSpinBox()
                spinbox.setRange(-1e9, 1e9)
                spinbox.setDecimals(4)
                spinbox.setSingleStep(0.1)
                spinbox.setValue(float(value))
                spinbox.valueChanged.connect(lambda v, n=name: self._on_value_changed(n, v))
            
            spinbox.setMinimumWidth(80)
            row.addWidget(spinbox)
            row.addStretch()
            
            self.spinboxes[name] = spinbox
            layout.addLayout(row)
        
        # Separator
        layout.addSpacing(10)
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line2)
        
        # Auto-update checkbox
        self.auto_update_checkbox = QCheckBox("Auto Update")
        self.auto_update_checkbox.setChecked(True)
        self.auto_update_checkbox.stateChanged.connect(self._on_auto_update_changed)
        layout.addWidget(self.auto_update_checkbox)
        
        # Update button (disabled when auto-update is on)
        self.update_btn = QPushButton("Update")
        self.update_btn.clicked.connect(self._on_update_clicked)
        self.update_btn.setEnabled(False)
        layout.addWidget(self.update_btn)
        
        # Reset button
        reset_btn = QPushButton("Reset to Initial")
        reset_btn.clicked.connect(self._reset_params)
        layout.addWidget(reset_btn)
        
        # Stretch at bottom
        layout.addStretch()
        
        return panel
    
    def _on_auto_update_changed(self, state):
        """Handle auto-update checkbox state change."""
        self.auto_update = bool(state)
        self.update_btn.setEnabled(not self.auto_update)
        
        if self.auto_update and self._pending_update:
            self._trigger_update()
    
    def _on_update_clicked(self):
        """Handle update button click."""
        self._trigger_update()
    
    def _trigger_update(self):
        """Trigger the update callback."""
        self._pending_update = False
        self._computing = True
        self.update_btn.setEnabled(False)
        self.update_btn.setText("Computing...")
        
        self.QApplication.processEvents()
        
        try:
            self.on_change(self.params)
        finally:
            self._computing = False
            self.update_btn.setText("Update")
            if not self.auto_update:
                self.update_btn.setEnabled(True)
    
    def _on_value_changed(self, name: str, value: Any):
        """Handle spinbox value change."""
        self.params[name] = value
        
        if self.auto_update:
            self._trigger_update()
        else:
            self._pending_update = True
    
    def _reset_params(self):
        """Reset all parameters to initial values."""
        for name, value in self.initial_params.items():
            self.params[name] = value
            spinbox = self.spinboxes[name]
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)
        
        if self.auto_update:
            self._trigger_update()
        else:
            self._pending_update = True
    
    def show(self):
        """Show the main window."""
        self.main_window.show()
    
    def close(self):
        """Close the main window."""
        self.main_window.close()
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return dict(self.params)
