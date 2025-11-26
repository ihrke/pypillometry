"""Main VisPy canvas with subplot grid for GPU viewer with LOD support."""

import numpy as np
from vispy import app, scene
from vispy.scene import SceneCanvas
from typing import List, Dict

from .visuals import LODLine, DynamicMaskRegions, DynamicEventMarkers
from .navigation import NavigationHandler


# Color scheme - signal drawn on top of orange mask regions
MODALITY_COLORS = {
    'left_pupil': '#0066CC',   # Blue (left eye)
    'right_pupil': '#CC0000',  # Red (right eye)
    'left_x': '#0066CC',
    'left_y': '#0066CC',
    'right_x': '#CC0000',
    'right_y': '#CC0000',
}


class GPUViewerCanvas(SceneCanvas):
    """GPU-accelerated viewer canvas with Level of Detail support."""
    
    def __init__(self, eyedata, overlays=None):
        super().__init__(
            keys='interactive',
            size=(1400, 800),
            bgcolor='white',
            title=f'GPU Viewer - {getattr(eyedata, "name", "Unknown")}'
        )
        
        self.unfreeze()
        self.eyedata = eyedata
        self.overlays = overlays or {}
        
        self.time_seconds = eyedata.tx.astype(np.float32) * 0.001
        self.data_min = float(self.time_seconds[0])
        self.data_max = float(self.time_seconds[-1])
        
        self.available_modalities = self._detect_modalities()
        
        self.has_events = (
            hasattr(eyedata, 'event_onsets') and 
            eyedata.event_onsets is not None and 
            len(eyedata.event_onsets) > 0
        )
        
        # Visibility states
        self.events_visible = True
        self.masks_visible = True
        
        # Storage for visuals
        self.lod_lines: List[LODLine] = []
        self.overlay_lines: List[LODLine] = []
        self.mask_regions: List[DynamicMaskRegions] = []
        self.event_markers: List[DynamicEventMarkers] = []
        
        self.grid = self.central_widget.add_grid(spacing=0)
        
        self.viewboxes: List[scene.ViewBox] = []
        self.view_types: List[str] = []
        self._create_subplots()
        
        self.navigation = NavigationHandler(
            self.viewboxes, 
            data_min=self.data_min,
            data_max=self.data_max
        )
        
        self._plot_all_data()
        self._plot_overlays()
        self._set_initial_view()
        
        self._last_x_range = (self.data_min, self.data_max)
        
        self.freeze()
    
    def _detect_modalities(self) -> Dict[str, List[str]]:
        available_data = {}
        for modality in ['left_pupil', 'right_pupil', 'left_x', 'right_x', 'left_y', 'right_y']:
            try:
                data = self.eyedata[modality]
                if data is not None and len(data) > 0:
                    available_data[modality] = True
            except (KeyError, AttributeError):
                pass
        
        grouped = {'pupil': [], 'x': [], 'y': []}
        for modality in available_data.keys():
            if 'pupil' in modality:
                grouped['pupil'].append(modality)
            elif '_x' in modality:
                grouped['x'].append(modality)
            elif '_y' in modality:
                grouped['y'].append(modality)
        
        return {k: v for k, v in grouped.items() if v}
    
    def _create_subplots(self):
        row = 0
        plot_labels = {'pupil': 'Pupil', 'x': 'Gaze X', 'y': 'Gaze Y'}
        
        for var_type in ['pupil', 'x', 'y']:
            if var_type not in self.available_modalities:
                continue
            
            y_axis = scene.AxisWidget(
                orientation='left',
                axis_font_size=7,
                axis_label=plot_labels[var_type],
                axis_label_margin=50,
                tick_label_margin=5,
                text_color='black',
                axis_color='black',
                tick_color='black',
            )
            y_axis.stretch = (0.08, 1)
            self.grid.add_widget(y_axis, row=row, col=0)
            
            viewbox = self.grid.add_view(row=row, col=1, border_color='#cccccc')
            camera = scene.PanZoomCamera(aspect=None)
            camera.interactive = False
            viewbox.camera = camera
            
            y_axis.link_view(viewbox)
            
            self.viewboxes.append(viewbox)
            self.view_types.append(var_type)
            row += 1
        
        if self.viewboxes:
            x_axis = scene.AxisWidget(
                orientation='bottom',
                axis_label='Time (s)',
                axis_font_size=8,
                axis_label_margin=35,
                tick_label_margin=5,
                text_color='black',
                axis_color='black',
                tick_color='black',
            )
            x_axis.stretch = (1, 0.12)
            self.grid.add_widget(x_axis, row=row, col=1)
            x_axis.link_view(self.viewboxes[-1])
    
    def _plot_all_data(self):
        time = self.time_seconds
        
        n_points = len(time)
        if n_points > 1000000:
            lod_factors = (1, 10, 100, 1000)
        elif n_points > 100000:
            lod_factors = (1, 10, 100)
        elif n_points > 10000:
            lod_factors = (1, 10)
        else:
            lod_factors = (1,)
        
        event_times = None
        event_labels = None
        if self.has_events:
            event_times = self.eyedata.event_onsets.astype(np.float32) * 0.001
            event_labels = list(self.eyedata.event_labels)
        
        for idx, (viewbox, var_type) in enumerate(zip(self.viewboxes, self.view_types)):
            modalities = self.available_modalities.get(var_type, [])
            
            # Add mask regions
            for modality in modalities:
                try:
                    mask = self.eyedata.data.mask.get(modality)
                    if mask is not None and np.any(mask):
                        mask_vis = DynamicMaskRegions(viewbox, time, mask)
                        self.mask_regions.append(mask_vis)
                        break
                except (AttributeError, KeyError):
                    pass
            
            # Add event markers (labels only in first plot)
            if event_times is not None:
                show_labels = (idx == 0)  # Only first viewbox shows labels
                event_vis = DynamicEventMarkers(
                    viewbox, event_times, event_labels, 
                    show_labels=show_labels
                )
                self.event_markers.append(event_vis)
            
            # Add LOD lines
            for modality in modalities:
                data = self.eyedata[modality]
                color = MODALITY_COLORS.get(modality, '#666666')
                
                # Get mask for this modality
                mask = None
                try:
                    mask = self.eyedata.data.mask.get(modality)
                except (AttributeError, KeyError):
                    pass
                
                lod_line = LODLine(
                    viewbox, time, data, color,
                    mask=mask,
                    lod_factors=lod_factors
                )
                self.lod_lines.append(lod_line)
    
    def _plot_overlays(self):
        """Plot overlay timeseries on top of existing data."""
        if not self.overlays:
            return
        
        time = self.time_seconds
        n_points = len(time)
        
        # Determine LOD factors based on data length
        if n_points > 1000000:
            lod_factors = (1, 10, 100, 1000)
        elif n_points > 100000:
            lod_factors = (1, 10, 100)
        elif n_points > 10000:
            lod_factors = (1, 10)
        else:
            lod_factors = (1,)
        
        # Colors for overlays (distinct from main signal colors)
        overlay_colors = [
            '#00AA00',  # Green
            '#AA00AA',  # Purple  
            '#00AAAA',  # Cyan
            '#AAAA00',  # Yellow
            '#FF6600',  # Orange
            '#6600FF',  # Violet
            '#FF0066',  # Pink
            '#006666',  # Teal
        ]
        
        for viewbox, var_type in zip(self.viewboxes, self.view_types):
            overlay_dict = self.overlays.get(var_type, {})
            if not overlay_dict:
                continue
            
            color_idx = 0
            for label, data_spec in overlay_dict.items():
                # Get the data
                if isinstance(data_spec, str):
                    # It's a key in eyedata.data
                    try:
                        data = self.eyedata.data[data_spec]
                    except (KeyError, AttributeError):
                        print(f"Warning: overlay '{label}' key '{data_spec}' not found in data")
                        continue
                else:
                    # It's an array
                    data = np.asarray(data_spec)
                
                if len(data) != n_points:
                    print(f"Warning: overlay '{label}' has {len(data)} points, expected {n_points}")
                    continue
                
                color = overlay_colors[color_idx % len(overlay_colors)]
                color_idx += 1
                
                # Create overlay line (no mask, different width/style)
                overlay_line = LODLine(
                    viewbox, time, data, color,
                    mask=None,
                    width=1.0,  # Thinner than main signal
                    lod_factors=lod_factors
                )
                self.overlay_lines.append(overlay_line)
                
                # Add a label for this overlay
                self._add_overlay_label(viewbox, label, color)
    
    def _add_overlay_label(self, viewbox, label, color):
        """Add a small label for an overlay in the corner of the viewbox."""
        # Count existing labels in this viewbox to offset position
        n_existing = sum(1 for vb in self.viewboxes[:self.viewboxes.index(viewbox)+1] 
                        for _ in self.overlays.get(self.view_types[self.viewboxes.index(vb)], {}))
        
        text = scene.Text(
            text=f"― {label}",
            pos=(10, 20 + (n_existing - 1) * 15),
            color=color,
            font_size=9,
            anchor_x='left',
            anchor_y='top',
            parent=viewbox.scene
        )
        text.order = 2000  # On top of everything
    
    def _set_initial_view(self):
        total_duration = self.data_max - self.data_min
        initial_window = min(30.0, total_duration * 0.05)
        if total_duration > 10.0:
            initial_window = max(initial_window, 10.0)
        
        x_min = self.data_min
        x_max = x_min + initial_window
        
        self.navigation.set_view(x_min, x_max)
        self._set_view_range(x_min, x_max)
    
    def _set_view_range(self, x_min: float, x_max: float):
        for viewbox, var_type in zip(self.viewboxes, self.view_types):
            y_min, y_max = self._get_y_range(var_type, x_min, x_max)
            viewbox.camera.set_range(x=(x_min, x_max), y=(y_min, y_max))
        
        self._update_lod_visuals(x_min, x_max)
        self._last_x_range = (x_min, x_max)
    
    def _get_y_range(self, var_type: str, x_min: float, x_max: float) -> tuple:
        time = self.time_seconds
        start_idx = np.searchsorted(time, x_min, side='left')
        end_idx = np.searchsorted(time, x_max, side='right')
        
        y_min, y_max = float('inf'), float('-inf')
        modalities = self.available_modalities.get(var_type, [])
        
        for modality in modalities:
            data = self.eyedata[modality]
            if hasattr(data, 'data'):
                visible_data = data.data[start_idx:end_idx]
            else:
                visible_data = data[start_idx:end_idx]
            
            valid = np.isfinite(visible_data)
            if np.any(valid):
                y_min = min(y_min, np.nanmin(visible_data[valid]))
                y_max = max(y_max, np.nanmax(visible_data[valid]))
        
        # Include overlay data in y-range
        overlay_dict = self.overlays.get(var_type, {})
        for label, data_spec in overlay_dict.items():
            if isinstance(data_spec, str):
                try:
                    data = self.eyedata.data[data_spec]
                except (KeyError, AttributeError):
                    continue
            else:
                data = data_spec
            
            # Convert to numpy array, handling masked arrays
            if hasattr(data, 'data'):
                data = np.asarray(data.data)
            else:
                data = np.asarray(data)
            
            if len(data) == len(time):
                visible_data = data[start_idx:end_idx]
                valid = np.isfinite(visible_data)
                if np.any(valid):
                    valid_data = visible_data[valid]
                    y_min = min(y_min, float(np.nanmin(valid_data)))
                    y_max = max(y_max, float(np.nanmax(valid_data)))
        
        if y_min == float('inf'):
            return (0, 1)
        
        padding = (y_max - y_min) * 0.1
        return (y_min - padding, y_max + padding)
    
    def _update_lod_visuals(self, x_min: float, x_max: float):
        for lod_line in self.lod_lines:
            lod_line.update_for_view(x_min, x_max)
        
        for overlay_line in self.overlay_lines:
            overlay_line.update_for_view(x_min, x_max)
        
        for mask_vis in self.mask_regions:
            mask_vis.update_for_view(x_min, x_max)
        
        # Get y_bottom from first viewbox for label positioning
        y_bottom = None
        if self.viewboxes:
            try:
                rect = self.viewboxes[0].camera.rect
                y_bottom = rect.bottom
            except:
                pass
        
        for event_vis in self.event_markers:
            event_vis.update_for_view(x_min, x_max, y_bottom=y_bottom)
    
    def _toggle_events(self):
        """Toggle event markers visibility."""
        self.events_visible = not self.events_visible
        for event_vis in self.event_markers:
            event_vis.set_visible(self.events_visible)
        
        # Force update to refresh labels if events turned back on
        if self.events_visible:
            x_min, x_max = self._last_x_range
            self._update_lod_visuals(x_min, x_max)
        
        self.update()
    
    def _toggle_masks(self):
        """Toggle mask regions and masked signal visibility."""
        self.masks_visible = not self.masks_visible
        for mask_vis in self.mask_regions:
            mask_vis.set_visible(self.masks_visible)
        for lod_line in self.lod_lines:
            lod_line.set_mask_visible(self.masks_visible)
        self.update()
    
    def _show_help(self):
        """Show help dialog with keybindings."""
        try:
            from PyQt6.QtWidgets import QMessageBox
            from PyQt6.QtCore import Qt
        except ImportError:
            try:
                from PyQt5.QtWidgets import QMessageBox
                from PyQt5.QtCore import Qt
            except ImportError:
                print("Help: arrows=navigate, +/-=zoom, m=masks, o=events, q=quit")
                return
        
        help_text = """
<h2>GPU Viewer - Keyboard Controls</h2>

<h3>Navigation</h3>
<table>
<tr><td><b>←  →</b></td><td>Pan left / right (10%)</td></tr>
<tr><td><b>PgUp  PgDn</b></td><td>Pan left / right (50%)</td></tr>
<tr><td><b>Home  End</b></td><td>Jump to start / end</td></tr>
<tr><td><b>Space</b></td><td>Show full signal</td></tr>
</table>

<h3>Zoom</h3>
<table>
<tr><td><b>↑  or  +</b></td><td>Zoom in</td></tr>
<tr><td><b>↓  or  -</b></td><td>Zoom out</td></tr>
</table>

<h3>Display</h3>
<table>
<tr><td><b>M</b></td><td>Toggle mask regions</td></tr>
<tr><td><b>O</b></td><td>Toggle event markers</td></tr>
</table>

<h3>Other</h3>
<table>
<tr><td><b>H  or  ?</b></td><td>Show this help</td></tr>
<tr><td><b>Q  or  Esc</b></td><td>Close viewer</td></tr>
</table>
"""
        
        msg = QMessageBox(self.native)
        msg.setWindowTitle("GPU Viewer Help")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        key = event.key
        
        # Help with 'h' or '?'
        if key in ['h', 'H', '?']:
            self._show_help()
            return
        
        # Toggle events with 'o'
        if key in ['o', 'O']:
            self._toggle_events()
            return
        
        # Toggle masks with 'm'
        if key in ['m', 'M']:
            self._toggle_masks()
            return
        
        # Navigation keys
        result = self.navigation.handle_key_press(event)
        
        if result is not None:
            x_min, x_max = result
            
            for viewbox, var_type in zip(self.viewboxes, self.view_types):
                y_min, y_max = self._get_y_range(var_type, x_min, x_max)
                viewbox.camera.set_range(x=(x_min, x_max), y=(y_min, y_max))
            
            self._update_lod_visuals(x_min, x_max)
            self._last_x_range = (x_min, x_max)
