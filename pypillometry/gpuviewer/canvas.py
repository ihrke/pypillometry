"""Main VisPy canvas with subplot grid for GPU viewer."""

import numpy as np
from vispy import app, scene
from vispy.scene import SceneCanvas, Grid
from typing import List, Dict, Optional, Any

from .visuals import add_line_visual, add_mask_regions, add_event_markers
from .navigation import NavigationHandler


# Color scheme for different data modalities (consistent left=blue, right=red)
MODALITY_COLORS = {
    'left_pupil': '#0000FF',    # Blue (left eye)
    'right_pupil': '#FF0000',   # Red (right eye)
    'left_x': '#0000FF',        # Blue (left eye)
    'left_y': '#0000FF',        # Blue (left eye)
    'right_x': '#FF0000',       # Red (right eye)
    'right_y': '#FF0000',       # Red (right eye)
}


class GPUViewerCanvas(SceneCanvas):
    """GPU-accelerated viewer canvas for eye-tracking data.
    
    Uses VisPy's scene graph with a grid layout for multiple subplots,
    each showing a different variable type (pupil, x, y) with curves
    for left and right eye data.
    
    Parameters
    ----------
    eyedata : EyeData-like object
        Eye-tracking data with tx, data dictionary, events, and masks
    """
    
    def __init__(self, eyedata):
        super().__init__(
            keys='interactive',
            size=(1400, 800),
            bgcolor='white',
            title=f'GPU Viewer - {getattr(eyedata, "name", "Unknown")}'
        )
        
        self.unfreeze()  # Allow adding attributes (must be before setting any)
        self.eyedata = eyedata
        
        # Convert time to seconds
        self.time_seconds = eyedata.tx * 0.001
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
        
        # Create grid layout
        self.grid = self.central_widget.add_grid(spacing=0)
        
        # Create viewboxes for each variable type
        self.viewboxes: List[scene.ViewBox] = []
        self.view_types: List[str] = []  # Track which variable type each viewbox shows
        self._create_subplots()
        
        # Set up navigation handler
        self.navigation = NavigationHandler(
            self.viewboxes, 
            data_min=self.data_min,
            data_max=self.data_max
        )
        
        # Plot data
        self._plot_all_data()
        
        # Set initial view to first 30 seconds or 5% of data
        self._set_initial_view()
        
        self.freeze()  # Freeze again
    
    def _detect_modalities(self) -> Dict[str, List[str]]:
        """Detect available data modalities grouped by type.
        
        Returns
        -------
        dict
            Maps plot types ('pupil', 'x', 'y') to lists of available modalities
        """
        available_data = {}
        for modality in ['left_pupil', 'right_pupil', 'left_x', 'right_x', 'left_y', 'right_y']:
            try:
                data = self.eyedata[modality]
                if data is not None and len(data) > 0:
                    available_data[modality] = True
            except (KeyError, AttributeError):
                pass
        
        # Group by plot type
        grouped = {
            'pupil': [],
            'x': [],
            'y': []
        }
        
        for modality in available_data.keys():
            if 'pupil' in modality:
                grouped['pupil'].append(modality)
            elif '_x' in modality:
                grouped['x'].append(modality)
            elif '_y' in modality:
                grouped['y'].append(modality)
        
        # Remove empty groups
        return {k: v for k, v in grouped.items() if v}
    
    def _create_subplots(self):
        """Create subplot viewboxes with linked x-axes."""
        row = 0
        labels = {'pupil': 'Pupil Size', 'x': 'Gaze X', 'y': 'Gaze Y'}
        
        for var_type in ['pupil', 'x', 'y']:
            if var_type not in self.available_modalities:
                continue
            
            # Create viewbox
            viewbox = self.grid.add_view(row=row, col=0, border_color='black')
            viewbox.camera = scene.PanZoomCamera(aspect=None)
            viewbox.camera.interactive = True
            
            # Add axes
            axis_left = scene.AxisWidget(
                orientation='left',
                text_color='black',
                axis_color='black',
                tick_color='black'
            )
            axis_left.stretch = (0.1, 1)
            self.grid.add_widget(axis_left, row=row, col=1)
            axis_left.link_view(viewbox)
            
            # Add title/label as text
            title = scene.Label(labels[var_type], color='black', font_size=10)
            title.stretch = (0.1, 0.05)
            self.grid.add_widget(title, row=row, col=2)
            
            self.viewboxes.append(viewbox)
            self.view_types.append(var_type)
            row += 1
        
        # Add bottom x-axis to the last plot
        if self.viewboxes:
            axis_bottom = scene.AxisWidget(
                orientation='bottom',
                text_color='black',
                axis_color='black',
                tick_color='black'
            )
            axis_bottom.stretch = (1, 0.1)
            self.grid.add_widget(axis_bottom, row=row, col=0)
            axis_bottom.link_view(self.viewboxes[-1])
            
            # Add time label
            time_label = scene.Label('Time (s)', color='black', font_size=10)
            time_label.stretch = (1, 0.05)
            self.grid.add_widget(time_label, row=row+1, col=0)
    
    def _plot_all_data(self):
        """Plot all data modalities in their respective subplots."""
        time = self.time_seconds
        
        for viewbox, var_type in zip(self.viewboxes, self.view_types):
            modalities = self.available_modalities.get(var_type, [])
            
            # First, add mask regions (behind curves)
            for modality in modalities:
                try:
                    mask = self.eyedata.data.mask.get(modality)
                    if mask is not None and np.any(mask):
                        add_mask_regions(viewbox, time, mask)
                        break  # Only add mask once per plot
                except (AttributeError, KeyError):
                    pass
            
            # Add event markers if available
            if self.has_events:
                event_times = self.eyedata.event_onsets * 0.001
                event_labels = self.eyedata.event_labels
                add_event_markers(viewbox, event_times, event_labels)
            
            # Then add line curves (on top)
            for modality in modalities:
                data = self.eyedata[modality]
                color = MODALITY_COLORS.get(modality, '#7f7f7f')
                add_line_visual(viewbox, time, data, color)
    
    def _set_initial_view(self):
        """Set initial zoomed view for fast startup."""
        total_duration = self.data_max - self.data_min
        
        # Show first 30 seconds or 5% of data, whichever is smaller
        initial_window = min(30.0, total_duration * 0.05)
        
        # But show at least 10 seconds if data is longer
        if total_duration > 10.0:
            initial_window = max(initial_window, 10.0)
        
        x_min = self.data_min
        x_max = x_min + initial_window
        
        # Set the view range for the primary viewbox (others will follow via linking)
        if self.viewboxes:
            # Calculate y-range from visible data
            self._set_view_range(x_min, x_max)
    
    def _set_view_range(self, x_min: float, x_max: float):
        """Set the x-range for all viewboxes and auto-scale y."""
        for viewbox, var_type in zip(self.viewboxes, self.view_types):
            # Find data indices in range
            time = self.time_seconds
            start_idx = np.searchsorted(time, x_min, side='left')
            end_idx = np.searchsorted(time, x_max, side='right')
            
            # Get y-range from visible data
            y_min, y_max = float('inf'), float('-inf')
            modalities = self.available_modalities.get(var_type, [])
            
            for modality in modalities:
                data = self.eyedata[modality]
                if hasattr(data, 'data'):
                    # Masked array
                    visible_data = data.data[start_idx:end_idx]
                else:
                    visible_data = data[start_idx:end_idx]
                
                valid = np.isfinite(visible_data)
                if np.any(valid):
                    y_min = min(y_min, np.nanmin(visible_data[valid]))
                    y_max = max(y_max, np.nanmax(visible_data[valid]))
            
            # Add padding
            if y_min != float('inf') and y_max != float('-inf'):
                padding_y = (y_max - y_min) * 0.1
                y_min -= padding_y
                y_max += padding_y
            else:
                y_min, y_max = 0, 1
            
            # Set camera range
            viewbox.camera.set_range(x=(x_min, x_max), y=(y_min, y_max))
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if self.navigation.handle_key_press(event):
            # Update view range with auto y-scaling
            if self.viewboxes:
                x_range = self.viewboxes[0].camera.get_state()['rect']
                self._set_view_range(x_range.left, x_range.right)

