"""Main VisPy canvas with subplot grid for GPU viewer with LOD support."""

import numpy as np
from vispy import app, scene
from vispy.scene import SceneCanvas
from typing import List, Dict, Union, Optional

from .visuals import LODLine, DynamicMaskRegions, DynamicEventMarkers, SelectionRegion, HighlightRegion
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


class ViewerCanvas(SceneCanvas):
    """GPU-accelerated viewer canvas with Level of Detail support."""
    
    def __init__(self, data, mode='eyedata', time_seconds=None, variables=None,
                 overlays=None, extra_plots=None, highlight=None, highlight_color='lightblue'):
        """Initialize the viewer canvas.
        
        Parameters
        ----------
        data : EyeData or dict
            In 'eyedata' mode: EyeData object with tx and data attributes.
            In 'arrays' mode: dict mapping plot labels to lists of arrays.
        mode : str
            'eyedata' for EyeData objects, 'arrays' for raw numpy arrays.
        time_seconds : ndarray
            Time vector in seconds.
        variables : list of str, optional
            For EyeData mode: filter modalities to display ('pupil', 'x', 'y').
        overlays : dict, optional
            Overlays for EyeData mode.
        extra_plots : dict, optional
            For EyeData mode: additional arrays to plot as separate subplots.
            Dict mapping label to list of arrays.
        highlight : Intervals or dict, optional
            Intervals to highlight.
        highlight_color : str
            Color for highlighted regions.
        """
        self.mode = mode
        
        if mode == 'eyedata':
            title = f'Viewer - {getattr(data, "name", "Unknown")}'
            self.eyedata = data
        else:
            title = 'Viewer - Arrays'
            self.eyedata = None
            self.plot_spec = data  # dict of {label: [arrays]}
        
        super().__init__(
            keys='interactive',
            size=(1400, 800),
            bgcolor='white',
            title=title
        )
        
        self.unfreeze()
        self.overlays = overlays or {}
        self.extra_plots = extra_plots or {}  # {label: [arrays]} for extra subplots
        self.highlight = highlight
        self.highlight_color = highlight_color
        self.variables_filter = variables  # Store the filter for later use
        
        self.time_seconds = time_seconds
        self.data_min = float(self.time_seconds[0])
        self.data_max = float(self.time_seconds[-1])
        
        if mode == 'eyedata':
            self.available_modalities = self._detect_modalities()
            # Apply variables filter
            if variables is not None:
                self.available_modalities = {
                    k: v for k, v in self.available_modalities.items() 
                    if k in variables
                }
        else:
            # For array mode, plot_spec keys become the "modalities"
            self.available_modalities = None
        
        self.has_events = (
            mode == 'eyedata' and
            hasattr(data, 'event_onsets') and 
            data.event_onsets is not None and 
            len(data.event_onsets) > 0
        )
        
        # Visibility states
        self.events_visible = True
        self.masks_visible = True
        
        # Selection mode state
        self.selection_mode = False
        self.selection_drag_start = None  # (viewbox_idx, x_time) or None
        self.selection_preview = None  # Temporary preview rectangle
        self.selection_regions: Dict[str, SelectionRegion] = {}  # var_type -> SelectionRegion
        
        # Storage for visuals
        self.lod_lines: List[LODLine] = []
        self.overlay_lines: List[LODLine] = []
        self.overlay_info: List[tuple] = []  # [(label, color, var_type), ...]
        self.mask_regions: List[DynamicMaskRegions] = []
        self.event_markers: List[DynamicEventMarkers] = []
        self.highlight_regions: Dict[str, HighlightRegion] = {}  # var_type -> HighlightRegion
        
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
        self._create_selection_regions()
        self._create_highlight_regions()
        self._create_legend()
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
        
        if self.mode == 'eyedata':
            # EyeData mode: use standard modality labels
            plot_labels = {'pupil': 'Pupil', 'x': 'Gaze X', 'y': 'Gaze Y'}
            plot_order = ['pupil', 'x', 'y']
            
            for var_type in plot_order:
                if var_type not in self.available_modalities:
                    continue
                
                self._add_subplot(row, plot_labels[var_type], var_type)
                row += 1
            
            # Add extra_plots subplots (additional arrays below EyeData)
            for label in self.extra_plots.keys():
                # Use 'extra_<label>' as var_type to distinguish from EyeData modalities
                self._add_subplot(row, label, f'extra_{label}')
                row += 1
        else:
            # Array mode: use dict keys as labels
            for label in self.plot_spec.keys():
                self._add_subplot(row, label, label)
                row += 1
        
        if self.viewboxes:
            # Determine x-axis label based on whether time was provided
            # If time is just sample indices (0, 1, 2, ...), label as "Sample"
            # Otherwise label as "Time (s)"
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
            x_axis.height_min = 50  # Fixed minimum height in pixels
            x_axis.height_max = 50  # Fixed maximum height
            x_axis.stretch = (1, 0.0001)  # Near-zero vertical stretch
            self.grid.add_widget(x_axis, row=row, col=1)
            x_axis.link_view(self.viewboxes[-1])
            self._x_axis_row = row
    
    def _add_subplot(self, row: int, label: str, var_type: str):
        """Add a single subplot with y-axis and viewbox."""
        y_axis = scene.AxisWidget(
            orientation='left',
            axis_font_size=7,
            axis_label=label,
            axis_label_margin=50,
            tick_label_margin=5,
            text_color='black',
            axis_color='black',
            tick_color='black',
        )
        y_axis.stretch = (0.08, 1)
        self.grid.add_widget(y_axis, row=row, col=0)
        
        viewbox = self.grid.add_view(row=row, col=1, border_color='#cccccc')
        viewbox.stretch = (1, 1)  # Equal vertical stretch for all plot viewboxes
        viewbox.height_min = 50   # Minimum height to prevent zero-size during layout
        
        camera = scene.PanZoomCamera(aspect=None)
        camera.interactive = False
        viewbox.camera = camera
        
        y_axis.link_view(viewbox)
        y_axis.stretch = (0.08, 1)  # Fixed width, equal vertical stretch
        y_axis.height_min = 50     # Match viewbox min height
        
        self.viewboxes.append(viewbox)
        self.view_types.append(var_type)
    
    def _plot_all_data(self):
        time = self.time_seconds
        
        n_points = len(time)
        lod_factors = self._get_lod_factors(n_points)
        
        if self.mode == 'eyedata':
            self._plot_eyedata(time, lod_factors)
        else:
            self._plot_arrays(time, lod_factors)
    
    def _get_lod_factors(self, n_points: int) -> tuple:
        """Determine LOD factors based on data length."""
        if n_points > 1000000:
            return (1, 10, 100, 1000)
        elif n_points > 100000:
            return (1, 10, 100)
        elif n_points > 10000:
            return (1, 10)
        else:
            return (1,)
    
    def _plot_eyedata(self, time: np.ndarray, lod_factors: tuple):
        """Plot EyeData modalities with masks and events."""
        event_times = None
        event_labels = None
        if self.has_events:
            event_times = self.eyedata.event_onsets.astype(np.float32) * 0.001
            event_labels = list(self.eyedata.event_labels)
        
        # Colors for extra_plots arrays
        array_colors = [
            '#0066CC', '#CC0000', '#00AA00', '#AA00AA',
            '#00AAAA', '#AAAA00', '#FF6600', '#6600FF',
        ]
        
        for idx, (viewbox, var_type) in enumerate(zip(self.viewboxes, self.view_types)):
            # Check if this is an extra_plots subplot
            if var_type.startswith('extra_'):
                # Extract the label from 'extra_<label>'
                label = var_type[6:]  # Remove 'extra_' prefix
                arrays = self.extra_plots.get(label, [])
                
                # Add mask regions (orange highlights) for the first masked array
                mask_added = False
                for arr in arrays:
                    if isinstance(arr, np.ma.MaskedArray) and arr.mask is not np.ma.nomask:
                        mask = arr.mask
                        if np.any(mask):
                            mask_vis = DynamicMaskRegions(viewbox, time, mask)
                            self.mask_regions.append(mask_vis)
                            mask_added = True
                            break  # Only add one mask region per subplot
                
                # Add event markers to extra plots too
                if event_times is not None:
                    event_vis = DynamicEventMarkers(
                        viewbox, event_times, event_labels, 
                        show_labels=False  # No labels on extra plots
                    )
                    self.event_markers.append(event_vis)
                
                # Plot the arrays
                for i, arr in enumerate(arrays):
                    color = array_colors[i % len(array_colors)]
                    
                    # Handle masked arrays
                    mask = None
                    if isinstance(arr, np.ma.MaskedArray):
                        mask = arr.mask.astype(int) if arr.mask is not np.ma.nomask else None
                        arr = arr.data
                    
                    # Convert to float32 for GPU rendering
                    arr = np.asarray(arr, dtype=np.float32)
                    
                    lod_line = LODLine(
                        viewbox, time, arr, color,
                        mask=mask,
                        lod_factors=lod_factors
                    )
                    self.lod_lines.append(lod_line)
                continue
            
            # Standard EyeData modality plotting
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
    
    def _plot_arrays(self, time: np.ndarray, lod_factors: tuple):
        """Plot raw numpy arrays."""
        # Colors for array lines
        array_colors = [
            '#0066CC',  # Blue
            '#CC0000',  # Red
            '#00AA00',  # Green
            '#AA00AA',  # Purple  
            '#00AAAA',  # Cyan
            '#AAAA00',  # Yellow
            '#FF6600',  # Orange
            '#6600FF',  # Violet
        ]
        
        for viewbox, var_type in zip(self.viewboxes, self.view_types):
            arrays = self.plot_spec.get(var_type, [])
            
            # Add mask regions (orange highlights) for the first masked array
            for arr in arrays:
                if isinstance(arr, np.ma.MaskedArray) and arr.mask is not np.ma.nomask:
                    mask = arr.mask
                    if np.any(mask):
                        mask_vis = DynamicMaskRegions(viewbox, time, mask)
                        self.mask_regions.append(mask_vis)
                        break  # Only add one mask region per subplot
            
            for i, arr in enumerate(arrays):
                color = array_colors[i % len(array_colors)]
                
                # Handle masked arrays
                mask = None
                if isinstance(arr, np.ma.MaskedArray):
                    mask = arr.mask.astype(int) if arr.mask is not np.ma.nomask else None
                    arr = arr.data
                
                # Convert to float32 for GPU rendering
                arr = np.asarray(arr, dtype=np.float32)
                
                lod_line = LODLine(
                    viewbox, time, arr, color,
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
                
                # Create overlay line (no mask, slightly thinner than main signal)
                overlay_line = LODLine(
                    viewbox, time, data, color,
                    mask=None,
                    width=1.5,
                    lod_factors=lod_factors
                )
                self.overlay_lines.append(overlay_line)
                
                # Store info for legend
                self.overlay_info.append((label, color, var_type))
    
    def _create_legend(self):
        """Create a fixed legend bar below the plots with eye colors and overlays."""
        # Get the row after x-axis
        legend_row = getattr(self, '_x_axis_row', len(self.view_types)) + 1
        
        # Create a viewbox for the legend (no camera interaction)
        legend_view = self.grid.add_view(row=legend_row, col=0, col_span=2, border_color=None)
        legend_view.camera = scene.PanZoomCamera(aspect=None)
        legend_view.camera.interactive = False
        legend_view.height_min = 30  # Fixed height in pixels
        legend_view.height_max = 30  # Same as min to prevent stretching
        legend_view.stretch = (1, 0.0001)  # Near-zero vertical stretch
        
        # Build legend items based on mode
        legend_items = []
        
        if self.mode == 'eyedata':
            # Add eye color entries
            if any('left' in m for mods in self.available_modalities.values() for m in mods):
                legend_items.append(('Left eye', '#0066CC', None))
            if any('right' in m for mods in self.available_modalities.values() for m in mods):
                legend_items.append(('Right eye', '#CC0000', None))
            
            # Add overlay entries
            for label, color, var_type in self.overlay_info:
                plot_label = {'pupil': 'P', 'x': 'X', 'y': 'Y'}.get(var_type, '')
                legend_items.append((f"{label} [{plot_label}]", color, var_type))
            
            # Add extra_plots entries
            array_colors = [
                '#0066CC', '#CC0000', '#00AA00', '#AA00AA',
                '#00AAAA', '#AAAA00', '#FF6600', '#6600FF',
            ]
            for plot_label, arrays in self.extra_plots.items():
                if len(arrays) > 1:
                    # Multiple arrays in this extra plot - show numbered legend
                    for i, _ in enumerate(arrays):
                        color = array_colors[i % len(array_colors)]
                        legend_items.append((f"{plot_label} #{i+1}", color, f'extra_{plot_label}'))
                else:
                    # Single array - just show the label
                    legend_items.append((plot_label, array_colors[0], f'extra_{plot_label}'))
        else:
            # Array mode: show colors for each array in each plot
            array_colors = [
                '#0066CC', '#CC0000', '#00AA00', '#AA00AA',  
                '#00AAAA', '#AAAA00', '#FF6600', '#6600FF',
            ]
            
            for plot_label, arrays in self.plot_spec.items():
                if len(arrays) > 1:
                    # Multiple arrays in this plot - show numbered legend
                    for i, _ in enumerate(arrays):
                        color = array_colors[i % len(array_colors)]
                        legend_items.append((f"{plot_label} #{i+1}", color, plot_label))
        
        if not legend_items:
            return
        
        # Calculate positions for horizontal layout
        n_items = len(legend_items)
        spacing = 1.0 / (n_items + 1)
        
        for i, (label, color, var_type) in enumerate(legend_items):
            x_pos = spacing * (i + 1)
            
            # Add colored line segment
            line_pos = np.array([
                [x_pos - 0.015, 0.5],
                [x_pos + 0.015, 0.5]
            ], dtype=np.float32)
            line = scene.Line(pos=line_pos, color=color, width=4, parent=legend_view.scene)
            
            # Add label text
            text = scene.Text(
                text=label,
                pos=(x_pos + 0.02, 0.5),
                color='black',
                font_size=8,
                anchor_x='left',
                anchor_y='center',
                parent=legend_view.scene
            )
    
    def _set_initial_view(self):
        total_duration = self.data_max - self.data_min
        
        if self.mode == 'arrays':
            # For array mode, show the full data initially
            # (arrays are typically shorter than eye-tracking recordings)
            x_min = self.data_min
            x_max = self.data_max
        else:
            # For EyeData mode, show a reasonable window for long recordings
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
            self._safe_set_camera_range(viewbox.camera, x_min, x_max, y_min, y_max)
        
        self._update_lod_visuals(x_min, x_max)
        self._last_x_range = (x_min, x_max)
    
    def _safe_set_camera_range(self, camera, x_min: float, x_max: float, 
                                y_min: float, y_max: float):
        """Set camera range with validation to avoid singular matrix errors.
        
        VisPy's PanZoomCamera can fail with singular matrix errors when:
        - Range is zero or near-zero
        - Viewbox has zero dimensions (during layout)
        """
        # Ensure minimum x range
        x_range = x_max - x_min
        if x_range < 1e-6:
            x_center = (x_min + x_max) / 2
            x_min = x_center - 0.5
            x_max = x_center + 0.5
        
        # Ensure minimum y range  
        y_range = y_max - y_min
        if y_range < 1e-6:
            y_center = (y_min + y_max) / 2
            y_min = y_center - 0.5
            y_max = y_center + 0.5
        
        # Check if viewbox has valid dimensions - if not, skip
        # (VisPy will call resize events later when dimensions are valid)
        try:
            rect = camera._viewbox.rect
            if rect is None or rect.width <= 0 or rect.height <= 0:
                return  # Skip - viewbox not ready yet
        except (AttributeError, TypeError):
            pass  # No viewbox rect available, try anyway
        
        try:
            camera.set_range(x=(x_min, x_max), y=(y_min, y_max))
        except (np.linalg.LinAlgError, Exception):
            # Camera in bad state, just skip this update
            pass
    
    def _get_y_range(self, var_type: str, x_min: float, x_max: float) -> tuple:
        time = self.time_seconds
        start_idx = np.searchsorted(time, x_min, side='left')
        end_idx = np.searchsorted(time, x_max, side='right')
        
        y_min, y_max = float('inf'), float('-inf')
        
        if self.mode == 'eyedata':
            # Check if this is an extra_plots subplot
            if var_type.startswith('extra_'):
                label = var_type[6:]  # Remove 'extra_' prefix
                arrays = self.extra_plots.get(label, [])
                y_min, y_max = self._get_y_range_from_arrays(arrays, start_idx, end_idx, y_min, y_max)
            else:
                y_min, y_max = self._get_y_range_eyedata(var_type, start_idx, end_idx, y_min, y_max)
        else:
            y_min, y_max = self._get_y_range_arrays(var_type, start_idx, end_idx, y_min, y_max)
        
        if y_min == float('inf'):
            return (0, 1)
        
        # Ensure non-zero range to avoid singular matrix in camera transform
        data_range = y_max - y_min
        if data_range < 1e-10:
            # For constant or near-constant data, use a small fixed range
            center = (y_min + y_max) / 2
            return (center - 0.5, center + 0.5)
        
        padding = data_range * 0.1
        return (y_min - padding, y_max + padding)
    
    def _get_y_range_eyedata(self, var_type: str, start_idx: int, end_idx: int, 
                              y_min: float, y_max: float) -> tuple:
        """Get y-range for EyeData mode."""
        time = self.time_seconds
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
        
        return y_min, y_max
    
    def _get_y_range_arrays(self, var_type: str, start_idx: int, end_idx: int,
                            y_min: float, y_max: float) -> tuple:
        """Get y-range for array mode."""
        arrays = self.plot_spec.get(var_type, [])
        return self._get_y_range_from_arrays(arrays, start_idx, end_idx, y_min, y_max)
    
    def _get_y_range_from_arrays(self, arrays: list, start_idx: int, end_idx: int,
                                  y_min: float, y_max: float) -> tuple:
        """Get y-range from a list of arrays."""
        for arr in arrays:
            if hasattr(arr, 'data'):
                visible_data = arr.data[start_idx:end_idx]
            else:
                visible_data = arr[start_idx:end_idx]
            
            # Convert to float for proper handling
            visible_data = np.asarray(visible_data, dtype=np.float64)
            
            if len(visible_data) == 0:
                continue
                
            valid_mask = np.isfinite(visible_data)
            if np.any(valid_mask):
                valid_data = visible_data[valid_mask]
                y_min = min(y_min, float(np.min(valid_data)))
                y_max = max(y_max, float(np.max(valid_data)))
        
        return y_min, y_max
    
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
    
    def _create_selection_regions(self):
        """Create SelectionRegion visual for each viewbox."""
        for viewbox, var_type in zip(self.viewboxes, self.view_types):
            self.selection_regions[var_type] = SelectionRegion(viewbox)
    
    def _create_highlight_regions(self):
        """Create highlight regions from the highlight parameter."""
        if self.highlight is None:
            return
        
        # Import Intervals here to check type
        from ..intervals import Intervals
        
        # Normalize highlight to a dict
        if isinstance(self.highlight, Intervals):
            # Apply to all plots
            highlight_dict = {var_type: self.highlight for var_type in self.view_types}
        elif isinstance(self.highlight, dict):
            highlight_dict = self.highlight
        else:
            return
        
        # Create highlight regions for each viewbox
        for viewbox, var_type in zip(self.viewboxes, self.view_types):
            intervals = highlight_dict.get(var_type)
            if intervals is None:
                continue
            
            # Get the intervals list and convert to seconds if needed
            if hasattr(intervals, 'intervals') and hasattr(intervals, 'to_units'):
                # Use Intervals.to_units() for proper conversion
                units = getattr(intervals, 'units', None)
                if units != 'sec':
                    # Set sampling_rate if not already set (needed for index conversion)
                    if intervals.sampling_rate is None and self.mode == 'eyedata':
                        intervals.sampling_rate = self.eyedata.fs
                    intervals = intervals.to_units('sec')
                interval_list = intervals.intervals
            elif hasattr(intervals, 'intervals'):
                interval_list = intervals.intervals
            else:
                # Assume it's already a list of tuples in seconds
                interval_list = list(intervals)
            
            if interval_list:
                highlight_region = HighlightRegion(
                    viewbox, interval_list, color=self.highlight_color
                )
                self.highlight_regions[var_type] = highlight_region
    
    def _get_viewbox_at_pos(self, canvas_pos):
        """Find which viewbox contains the given canvas position and return x in data coords."""
        canvas_x = canvas_pos[0]
        canvas_y = canvas_pos[1]
        canvas_width = self.size[0]
        canvas_height = self.size[1]
        
        # Calculate which viewbox based on y position
        n_plots = len(self.viewboxes)
        if n_plots == 0:
            return None, None, None
        
        # Account for x-axis and legend at bottom (roughly 75px total)
        plot_area_height = canvas_height - 75
        plot_height = plot_area_height / n_plots
        
        # Determine which plot index based on y position
        vb_idx = int(canvas_y / plot_height)
        vb_idx = max(0, min(vb_idx, n_plots - 1))
        
        viewbox = self.viewboxes[vb_idx]
        
        try:
            # Get viewbox width from its rect (local coordinates)
            vb_rect = viewbox.rect
            if vb_rect is not None:
                plot_width = vb_rect.width
                # Y-axis takes up the remaining space on the left
                plot_left = canvas_width - plot_width
            else:
                # Fallback estimate
                plot_left = canvas_width * 0.08
                plot_width = canvas_width - plot_left
            
            # Calculate relative x position within the plot area (0 to 1)
            rel_x = (canvas_x - plot_left) / plot_width
            rel_x = max(0.0, min(1.0, rel_x))
            
            # Map to data coordinates using camera's visible range
            camera_rect = viewbox.camera.rect
            x_data = camera_rect.left + rel_x * camera_rect.width
            return vb_idx, viewbox, float(x_data)
        except Exception:
            return None, None, None
    
    def _start_selection_mode(self):
        """Enter selection mode - cursor becomes crosshair."""
        self.selection_mode = True
        self.selection_drag_start = None
        # Change cursor to crosshair
        try:
            from PyQt6.QtCore import Qt
        except ImportError:
            from PyQt5.QtCore import Qt
        self.native.setCursor(Qt.CursorShape.CrossCursor)
        name = getattr(self.eyedata, "name", "Arrays") if self.eyedata else "Arrays"
        self.title = f'Viewer - {name} [SELECTION MODE - drag to select region]'
    
    def _exit_selection_mode(self):
        """Exit selection mode - restore normal cursor."""
        self.selection_mode = False
        self.selection_drag_start = None
        self._clear_preview()
        # Restore normal cursor
        try:
            from PyQt6.QtCore import Qt
        except ImportError:
            from PyQt5.QtCore import Qt
        self.native.setCursor(Qt.CursorShape.ArrowCursor)
        name = getattr(self.eyedata, "name", "Arrays") if self.eyedata else "Arrays"
        self.title = f'Viewer - {name}'
    
    def _clear_preview(self):
        """Remove the selection preview rectangle."""
        if self.selection_preview is not None:
            self.selection_preview.parent = None
            self.selection_preview = None
    
    def _remove_last_selection(self):
        """Remove the last selection from the most recently modified viewbox."""
        # Try to remove from the viewbox that had the last selection
        for var_type in reversed(self.view_types):
            region = self.selection_regions.get(var_type)
            if region and region.intervals:
                region.remove_last()
                self.update()
                return True
        return False
    
    def get_selections(self) -> Dict[str, List[tuple]]:
        """Get all selections as dict of var_type -> list of (start, end) tuples in seconds."""
        result = {}
        for var_type, region in self.selection_regions.items():
            if region.intervals:
                result[var_type] = region.get_intervals()
        return result
    
    def on_mouse_press(self, event):
        """Handle mouse press events for selection mode - start drag."""
        if not self.selection_mode:
            return
        
        if event.button != 1:  # Only left click
            return
        
        # Get canvas position
        canvas_pos = event.pos
        
        # Find which viewbox was clicked and get x coordinate in data space
        vb_idx, viewbox, x_time = self._get_viewbox_at_pos(canvas_pos)
        
        if vb_idx is None:
            return
        
        # Start drag
        self.selection_drag_start = (vb_idx, x_time, viewbox)
        name = getattr(self.eyedata, "name", "Arrays") if self.eyedata else "Arrays"
        self.title = f'Viewer - {name} [Drag to select...]'
    
    def on_mouse_move(self, event):
        """Handle mouse move events for selection preview during drag."""
        if not self.selection_mode or self.selection_drag_start is None:
            return
        
        # Get current position
        canvas_pos = event.pos
        start_vb_idx, start_x, start_viewbox = self.selection_drag_start
        
        # Get current x in data space (use the same viewbox as start)
        vb_idx, viewbox, current_x = self._get_viewbox_at_pos(canvas_pos)
        
        if current_x is None:
            return
        
        # Update preview rectangle
        self._update_preview(start_viewbox, start_x, current_x)
        self.update()
    
    def on_mouse_release(self, event):
        """Handle mouse release events for selection mode - complete drag."""
        if not self.selection_mode or self.selection_drag_start is None:
            return
        
        if event.button != 1:  # Only left click
            return
        
        # Get end position
        canvas_pos = event.pos
        start_vb_idx, start_x, start_viewbox = self.selection_drag_start
        vb_idx, viewbox, end_x = self._get_viewbox_at_pos(canvas_pos)
        
        # Clear preview
        self._clear_preview()
        
        name = getattr(self.eyedata, "name", "Arrays") if self.eyedata else "Arrays"
        
        if end_x is None:
            self.selection_drag_start = None
            self.title = f'Viewer - {name} [SELECTION MODE - drag to select region]'
            return
        
        # Add selection - any drag counts, no minimum
        var_type = self.view_types[start_vb_idx]
        region = self.selection_regions.get(var_type)
        if region and start_x != end_x:
            region.add_interval(start_x, end_x)
        
        # Reset for next selection
        self.selection_drag_start = None
        self.title = f'Viewer - {name} [SELECTION MODE - drag to select region]'
        self.update()
    
    def _update_preview(self, viewbox, x_start, x_end):
        """Update the selection preview rectangle."""
        from vispy import scene
        from vispy.color import Color
        
        if x_start > x_end:
            x_start, x_end = x_end, x_start
        
        y_min, y_max = -1e9, 1e9
        
        # Create vertices for rectangle
        vertices = np.array([
            [x_start, y_min],
            [x_end, y_min],
            [x_end, y_max],
            [x_start, y_max]
        ], dtype=np.float32)
        
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        
        c = Color('#00CC00')
        rgba = list(c.rgba)
        rgba[3] = 0.2  # More transparent for preview
        
        if self.selection_preview is None:
            self.selection_preview = scene.Mesh(
                vertices=vertices, faces=faces, color=rgba,
                parent=viewbox.scene
            )
            self.selection_preview.order = 400
            self.selection_preview.set_gl_state('translucent', depth_test=False)
        else:
            self.selection_preview.set_data(vertices=vertices, faces=faces, color=rgba)
            self.selection_preview.parent = viewbox.scene
    
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
<h2>Viewer - Keyboard Controls</h2>

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

<h3>Selection</h3>
<table>
<tr><td><b>S</b></td><td>Enter/exit selection mode (drag to select)</td></tr>
<tr><td><b>D  or  Backspace</b></td><td>Remove last selection</td></tr>
</table>

<h3>Other</h3>
<table>
<tr><td><b>H  or  ?</b></td><td>Show this help</td></tr>
<tr><td><b>Q  or  Esc</b></td><td>Close viewer</td></tr>
</table>
"""
        
        msg = QMessageBox(self.native)
        msg.setWindowTitle("Viewer Help")
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
        
        # Selection mode with 's'
        if key in ['s', 'S']:
            if self.selection_mode:
                self._exit_selection_mode()
            else:
                self._start_selection_mode()
            return
        
        # Remove last selection with 'd' or Backspace
        if key in ['d', 'D', 'Backspace']:
            self._remove_last_selection()
            return
        
        # Escape exits selection mode (or closes if not in selection mode)
        if key == 'Escape':
            if self.selection_mode:
                self._exit_selection_mode()
                return
        
        # Navigation keys
        result = self.navigation.handle_key_press(event)
        
        if result is not None:
            x_min, x_max = result
            
            for viewbox, var_type in zip(self.viewboxes, self.view_types):
                y_min, y_max = self._get_y_range(var_type, x_min, x_max)
                self._safe_set_camera_range(viewbox.camera, x_min, x_max, y_min, y_max)
            
            self._update_lod_visuals(x_min, x_max)
            self._last_x_range = (x_min, x_max)
