"""Main VisPy canvas with subplot grid for GPU viewer with LOD support."""

import numpy as np
from vispy import app, scene
from vispy.scene import SceneCanvas
from typing import List, Dict, Optional

from .visuals import LODLine, DynamicMaskRegions, DynamicEventMarkers
from .navigation import NavigationHandler


# Color scheme for different data modalities
MODALITY_COLORS = {
    'left_pupil': '#0066CC',    # Blue (left eye)
    'right_pupil': '#CC0000',   # Red (right eye)
    'left_x': '#0066CC',
    'left_y': '#0066CC',
    'right_x': '#CC0000',
    'right_y': '#CC0000',
}


class GPUViewerCanvas(SceneCanvas):
    """GPU-accelerated viewer canvas with Level of Detail support.
    
    Features:
    - LOD for lines: switches between resolutions based on zoom
    - Dynamic masks: shows all mask regions when zoomed in
    - Dynamic events: shows labels when zoomed in
    """
    
    def __init__(self, eyedata):
        super().__init__(
            keys='interactive',
            size=(1400, 800),
            bgcolor='white',
            title=f'GPU Viewer - {getattr(eyedata, "name", "Unknown")}'
        )
        
        self.unfreeze()
        self.eyedata = eyedata
        
        # Convert time to seconds
        self.time_seconds = eyedata.tx.astype(np.float32) * 0.001
        self.data_min = float(self.time_seconds[0])
        self.data_max = float(self.time_seconds[-1])
        
        # Detect available modalities
        self.available_modalities = self._detect_modalities()
        
        # Check for events
        self.has_events = (
            hasattr(eyedata, 'event_onsets') and 
            eyedata.event_onsets is not None and 
            len(eyedata.event_onsets) > 0
        )
        
        # Storage for LOD visuals
        self.lod_lines: List[LODLine] = []
        self.mask_regions: List[DynamicMaskRegions] = []
        self.event_markers: List[DynamicEventMarkers] = []
        
        # Create grid layout
        self.grid = self.central_widget.add_grid(spacing=0)
        
        # Create viewboxes
        self.viewboxes: List[scene.ViewBox] = []
        self.view_types: List[str] = []
        self._create_subplots()
        
        # Set up navigation handler
        self.navigation = NavigationHandler(
            self.viewboxes, 
            data_min=self.data_min,
            data_max=self.data_max
        )
        
        # Plot data with LOD
        self._plot_all_data()
        
        # Set initial view
        self._set_initial_view()
        
        # Connect to view change events for LOD updates
        for viewbox in self.viewboxes:
            viewbox.camera.rect_changed.connect(self._on_view_changed)
        
        self.freeze()
    
    def _detect_modalities(self) -> Dict[str, List[str]]:
        """Detect available data modalities grouped by type."""
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
        """Create subplot viewboxes."""
        row = 0
        
        for var_type in ['pupil', 'x', 'y']:
            if var_type not in self.available_modalities:
                continue
            
            viewbox = self.grid.add_view(row=row, col=0, border_color='#cccccc')
            viewbox.camera = scene.PanZoomCamera(aspect=None)
            viewbox.camera.interactive = True
            
            self.viewboxes.append(viewbox)
            self.view_types.append(var_type)
            row += 1
    
    def _plot_all_data(self):
        """Plot all data with LOD support."""
        time = self.time_seconds
        
        # Determine LOD factors based on data size
        n_points = len(time)
        if n_points > 1000000:
            lod_factors = (1, 10, 100, 1000)
        elif n_points > 100000:
            lod_factors = (1, 10, 100)
        elif n_points > 10000:
            lod_factors = (1, 10)
        else:
            lod_factors = (1,)
        
        for viewbox, var_type in zip(self.viewboxes, self.view_types):
            modalities = self.available_modalities.get(var_type, [])
            
            # Add mask regions (dynamic)
            for modality in modalities:
                try:
                    mask = self.eyedata.data.mask.get(modality)
                    if mask is not None and np.any(mask):
                        mask_vis = DynamicMaskRegions(viewbox, time, mask)
                        self.mask_regions.append(mask_vis)
                        break  # One mask per plot
                except (AttributeError, KeyError):
                    pass
            
            # Add event markers (dynamic)
            if self.has_events:
                event_times = self.eyedata.event_onsets.astype(np.float32) * 0.001
                event_labels = list(self.eyedata.event_labels)
                event_vis = DynamicEventMarkers(viewbox, event_times, event_labels)
                self.event_markers.append(event_vis)
            
            # Add LOD lines
            for modality in modalities:
                data = self.eyedata[modality]
                color = MODALITY_COLORS.get(modality, '#666666')
                lod_line = LODLine(viewbox, time, data, color, lod_factors=lod_factors)
                self.lod_lines.append(lod_line)
    
    def _set_initial_view(self):
        """Set initial view range."""
        total_duration = self.data_max - self.data_min
        initial_window = min(30.0, total_duration * 0.05)
        if total_duration > 10.0:
            initial_window = max(initial_window, 10.0)
        
        x_min = self.data_min
        x_max = x_min + initial_window
        self._set_view_range(x_min, x_max)
    
    def _set_view_range(self, x_min: float, x_max: float):
        """Set view range for all viewboxes and update LOD."""
        for viewbox, var_type in zip(self.viewboxes, self.view_types):
            # Calculate y-range from data
            y_min, y_max = self._get_y_range(var_type, x_min, x_max)
            viewbox.camera.set_range(x=(x_min, x_max), y=(y_min, y_max))
        
        # Update LOD visuals
        self._update_lod_visuals(x_min, x_max)
    
    def _get_y_range(self, var_type: str, x_min: float, x_max: float) -> tuple:
        """Get y-range for data in the given x-range."""
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
        
        if y_min == float('inf'):
            return (0, 1)
        
        padding = (y_max - y_min) * 0.1
        return (y_min - padding, y_max + padding)
    
    def _update_lod_visuals(self, x_min: float, x_max: float):
        """Update all LOD visuals for current view."""
        for lod_line in self.lod_lines:
            lod_line.update_for_view(x_min, x_max)
        
        for mask_vis in self.mask_regions:
            mask_vis.update_for_view(x_min, x_max)
        
        for event_vis in self.event_markers:
            event_vis.update_for_view(x_min, x_max)
    
    def _on_view_changed(self, event=None):
        """Called when view changes - update LOD."""
        if self.viewboxes:
            rect = self.viewboxes[0].camera.rect
            x_min, x_max = rect.left, rect.right
            self._update_lod_visuals(x_min, x_max)
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if self.navigation.handle_key_press(event):
            if self.viewboxes:
                rect = self.viewboxes[0].camera.rect
                self._set_view_range(rect.left, rect.right)
