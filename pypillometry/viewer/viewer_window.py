"""Main viewer window for interactive eye-tracking data visualization."""

import numpy as np
import numpy.ma as ma
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from typing import Optional, Callable, Dict, List

from .controls import ControlPanel
from .navigation import NavigationHandler
from .region_selector import RegionSelector
from .visualization import (
    plot_timeseries, add_event_markers, add_mask_regions,
    setup_plot_appearance, get_plot_label
)
from ..intervals import Intervals


class ViewerWindow(QtWidgets.QMainWindow):
    """Main window for interactive eye-tracking data viewer.
    
    Parameters
    ----------
    eyedata : EyeData-like object
        Eye-tracking data object with time, data, events, etc.
    separate_plots : bool, default True
        If True, create separate plot for each modality. If False, plot all in one.
    callback : callable, optional
        Callback function to call when regions change (non-blocking mode)
    """
    
    def __init__(
        self,
        eyedata,
        separate_plots: bool = True,
        callback: Optional[Callable] = None
    ):
        super().__init__()
        
        self.eyedata = eyedata
        self.separate_plots = separate_plots
        self.callback = callback
        self.selected_intervals: Optional[Intervals] = None
        self.accepted = False
        
        # Storage for plot items
        self.plot_curves: Dict[str, pg.PlotDataItem] = {}
        self.event_markers: List = []
        self.mask_regions: List = []
        
        # Timer for throttling auto y-axis updates
        self.auto_y_update_timer = QtCore.QTimer()
        self.auto_y_update_timer.setSingleShot(True)
        self.auto_y_update_timer.timeout.connect(self._update_auto_y_axes)
        self.auto_y_throttle_ms = 300  # Update only after 300ms of no interaction for smooth panning
        
        # Cache time array in seconds for performance
        self._time_seconds_cache = eyedata.tx * 0.001
        
        # Detect available data modalities (grouped by type)
        self.available_modalities = self._detect_modalities()
        
        # Map plot types to modalities for easy lookup
        self.plot_type_map = {}  # Maps modality name to plot type
        for plot_type, modalities in self.available_modalities.items():
            for modality in modalities:
                self.plot_type_map[modality] = plot_type
        
        # Check for events
        self.has_events = (
            hasattr(eyedata, 'event_onsets') and 
            eyedata.event_onsets is not None and 
            len(eyedata.event_onsets) > 0
        )
        
        try:
            self._setup_ui()
        except Exception as e:
            print(f"Error in _setup_ui: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        try:
            self._plot_data()
        except Exception as e:
            print(f"Error in _plot_data: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Set window title
        name = getattr(eyedata, 'name', 'Unknown')
        self.setWindowTitle(f'Eye Tracking Viewer - {name}')
        
        # Start with an initial zoomed view for fast rendering
        # Show first 30 seconds or 5% of data, whichever is smaller
        try:
            self._set_initial_view()
        except Exception as e:
            print(f"Error in _set_initial_view: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise - initial view is not critical
    
    def _detect_modalities(self) -> Dict[str, List[str]]:
        """Detect available data modalities in eyedata, grouped by type.
        
        Returns
        -------
        dict
            Dictionary mapping plot types ('pupil', 'x', 'y') to lists of available modalities
        """
        # Check what data is available
        available_data = {}
        for modality in ['left_pupil', 'right_pupil', 'left_x', 'right_x', 'left_y', 'right_y']:
            try:
                data = self.eyedata[modality]
                if data is not None and len(data) > 0:
                    available_data[modality] = True
            except (KeyError, AttributeError):
                pass
        
        # Group by plot type (pupil, x, y)
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
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Central widget with horizontal layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Create graphics layout for plots with light background
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground('w')  # White background
        
        # Note: OpenGL viewport doesn't accelerate 2D line plots in PyQtGraph
        # Performance comes from downsampling and clipToView instead
        self._opengl_enabled = False
        
        main_layout.addWidget(self.graphics_layout)
        
        # Create plots (max 3: pupil, x, y)
        self.plot_widgets = []
        self.plot_types = []  # Track which plot type each widget represents
        
        # Track current time unit and axis scale
        self.time_unit = 's'  # Default to seconds
        self.time_scale = 1.0  # Default axis scale (data is in seconds)
        
        if self.separate_plots:
            # Create one plot per data type (pupil, x, y)
            row = 0
            for plot_type in ['pupil', 'x', 'y']:
                if plot_type in self.available_modalities:
                    plot = self.graphics_layout.addPlot(row=row, col=0)
                    
                    # Set plot appearance with dark pen for light background
                    plot_label = {'pupil': 'Pupil Size', 'x': 'Gaze X', 'y': 'Gaze Y'}[plot_type]
                    setup_plot_appearance(plot, plot_label, plot_type, show_x_axis=(row == len([p for p in ['pupil', 'x', 'y'] if p in self.available_modalities]) - 1))
                    
                    # Add legend for combined left/right
                    plot.addLegend()
                    
                    # Link x-axes
                    if row > 0:
                        plot.setXLink(self.plot_widgets[0])
                    
                    # Optimize ViewBox for better performance with large datasets
                    vb = plot.getViewBox()
                    vb.setLimits(minXRange=0.001)  # Prevent zooming in too far
                    vb.disableAutoRange()  # Disable auto-ranging during mouse interaction
                    
                    self.plot_widgets.append(plot)
                    self.plot_types.append(plot_type)
                    row += 1
        else:
            # Single plot for all modalities
            plot = self.graphics_layout.addPlot(row=0, col=0)
            setup_plot_appearance(plot, "Eye Tracking Data", "Value", show_x_axis=True)
            plot.addLegend()
            
            # Optimize ViewBox for better performance with large datasets
            vb = plot.getViewBox()
            vb.setLimits(minXRange=0.001)  # Prevent zooming in too far
            vb.disableAutoRange()  # Disable auto-ranging during mouse interaction
            
            self.plot_widgets.append(plot)
            self.plot_types.append('all')
        
        # Disable SI prefixes on time axis to prevent ks, mmin, etc.
        for i, plot_widget in enumerate(self.plot_widgets):
            if i == len(self.plot_widgets) - 1:  # Only the bottom plot has visible x-axis
                axis = plot_widget.getAxis('bottom')
                axis.enableAutoSIPrefix(False)
                axis.autoSIPrefixScale = 1.0  # Force the SI prefix scale to 1.0
        
        # Set up navigation handler with time in seconds
        time_in_seconds = self.eyedata.tx * 0.001
        self.navigation = NavigationHandler(self.plot_widgets, time_data=time_in_seconds)
        
        # Set view limits to data bounds for all plots (in seconds)
        for plot_widget in self.plot_widgets:
            vb = plot_widget.getViewBox()
            # Set limits so view cannot go beyond data
            vb.setLimits(xMin=time_in_seconds[0], xMax=time_in_seconds[-1])
        
        # Set up region selector on primary plot
        self.region_selector = RegionSelector(
            self.plot_widgets[0],
            callback=self._on_regions_changed
        )
        
        # Connect x-range changed signal for auto y-axis with throttling
        for i, plot_widget in enumerate(self.plot_widgets):
            # Throttle updates to prevent excessive computation during fast panning
            plot_widget.getViewBox().sigRangeChangedManually.connect(
                lambda: self.auto_y_update_timer.start(self.auto_y_throttle_ms)
            )
        
        # Create control panel (pass flattened list of all modalities)
        all_modalities = []
        for modalities in self.available_modalities.values():
            all_modalities.extend(modalities)
        
        self.control_panel = ControlPanel(
            all_modalities,
            has_events=self.has_events,
            plot_types=self.plot_types
        )
        
        # Connect control panel signals
        self.control_panel.modality_toggled.connect(self._on_modality_toggled)
        if self.has_events:
            self.control_panel.events_toggled.connect(self._on_events_toggled)
        self.control_panel.mask_display_toggled.connect(self._on_mask_display_toggled)
        self.control_panel.region_added.connect(self._on_add_region)
        self.control_panel.regions_cleared.connect(self._on_clear_regions)
        self.control_panel.accept_clicked.connect(self._on_accept)
        self.control_panel.cancel_clicked.connect(self._on_cancel)
        self.control_panel.auto_y_toggled.connect(self._on_auto_y_toggled)
        self.control_panel.time_unit_changed.connect(self._on_time_unit_changed)
        self.control_panel.show_whole_signal.connect(self._on_show_whole_signal)
        
        # Track auto y-axis state (default to enabled)
        self.auto_y_enabled = {i: True for i in range(len(self.plot_widgets))}
        
        # Add control panel to layout
        main_layout.addWidget(self.control_panel)
        
        # Set window size
        self.resize(1400, 800)
    
    def _plot_data(self):
        """Plot the eye-tracking data."""
        # Always plot in seconds (time_scale is applied via axis scaling)
        time = self.eyedata.tx * 0.001  # Convert ms to seconds
        mask_display_enabled = self.control_panel.get_mask_display_enabled()
        # Always show masked data values (no gaps)
        # Mask shading optionally indicates which values are masked
        connect_mode = 'finite'  # Always use finite to prevent line interpolation
        show_masked_data = True  # Always plot masked data values
        
        if self.separate_plots:
            # Plot modalities grouped by type (pupil, x, y)
            for plot_idx, (plot_widget, plot_type) in enumerate(zip(self.plot_widgets, self.plot_types)):
                # Get modalities for this plot type
                modalities = self.available_modalities.get(plot_type, [])
                
                # Add ALL mask regions FIRST (before any curves) if mask display is enabled
                if mask_display_enabled:
                    for modality in modalities:
                        data = self.eyedata[modality]
                        if ma.is_masked(data):
                            regions = add_mask_regions(plot_widget, time, data.mask, alpha=0.2)
                            self.mask_regions.extend(regions)
                            break  # Only need to add mask once per plot (mask is same for all modalities)
                
                # Then plot ALL curves (so they appear on top)
                for modality in modalities:
                    data = self.eyedata[modality]
                    curve = plot_timeseries(
                        plot_widget, time, data, modality, connect_mode, show_masked_data
                    )
                    self.plot_curves[modality] = curve
        else:
            # Plot all modalities in single plot
            plot_widget = self.plot_widgets[0]
            
            # Flatten all modalities
            all_modalities = []
            for modalities in self.available_modalities.values():
                all_modalities.extend(modalities)
            
            # Add mask regions first if mask display is enabled
            if mask_display_enabled and all_modalities:
                first_modality = all_modalities[0]
                data = self.eyedata[first_modality]
                if ma.is_masked(data):
                    regions = add_mask_regions(plot_widget, time, data.mask, alpha=0.2)
                    self.mask_regions.extend(regions)
            
            # Then plot curves
            for modality in all_modalities:
                data = self.eyedata[modality]
                curve = plot_timeseries(
                    plot_widget, time, data, modality, connect_mode, show_masked_data
                )
                self.plot_curves[modality] = curve
        
        # Add event markers to all plots (on top, but hidden by default)
        if self.has_events:
            # Convert event onsets to seconds (axis scaling will handle display units)
            event_onsets = self.eyedata.event_onsets * 0.001
            event_labels = self.eyedata.event_labels
            
            for plot_widget in self.plot_widgets:
                markers = add_event_markers(plot_widget, event_onsets, event_labels)
                # Hide markers by default
                for marker in markers:
                    marker.setVisible(False)
                self.event_markers.extend(markers)
        
        # Ensure all plots auto-range to show the data
        for plot_widget in self.plot_widgets:
            plot_widget.enableAutoRange(axis='y')
            plot_widget.autoRange()
        
        # Trigger auto y-axis update for all plots
        self._update_auto_y_axes()
    
    def _clear_plots(self):
        """Clear all plots."""
        for plot_widget in self.plot_widgets:
            plot_widget.clear()
        
        self.plot_curves.clear()
        self.event_markers.clear()
        self.mask_regions.clear()
    
    def _on_modality_toggled(self, modality: str, visible: bool):
        """Handle modality checkbox toggle."""
        if modality in self.plot_curves:
            curve = self.plot_curves[modality]
            curve.setVisible(visible)
    
    def _on_events_toggled(self, visible: bool):
        """Handle events checkbox toggle."""
        for marker in self.event_markers:
            marker.setVisible(visible)
    
    def _on_mask_display_toggled(self, enabled: bool):
        """Handle mask display toggle."""
        # Redraw plots with new mask display state
        self._clear_plots()
        self._plot_data()
        
        # Restore region selections
        if self.region_selector.regions:
            old_intervals = self.region_selector.get_intervals()
            self.region_selector.clear_regions()
            if old_intervals:
                self.region_selector.set_intervals(old_intervals)
    
    def _on_add_region(self):
        """Handle add region button click."""
        self.region_selector.add_region()
    
    def _on_clear_regions(self):
        """Handle clear regions button click."""
        self.region_selector.clear_regions()
    
    def _on_regions_changed(self, intervals: Optional[Intervals]):
        """Handle region selection changes."""
        # Store current intervals
        self.selected_intervals = intervals
        
        # Call user callback if in non-blocking mode
        if self.callback is not None and intervals is not None:
            self.callback(intervals)
    
    def _on_accept(self):
        """Handle accept button click."""
        self.accepted = True
        self.selected_intervals = self.region_selector.get_intervals()
        self.close()
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self.accepted = False
        self.selected_intervals = None
        self.close()
    
    def _on_auto_y_toggled(self, plot_idx: int, enabled: bool):
        """Handle auto y-axis toggle."""
        self.auto_y_enabled[plot_idx] = enabled
        if enabled:
            self._update_auto_y_axis(plot_idx)
    
    def _update_auto_y_axis(self, plot_idx: int):
        """Update y-axis range for a specific plot based on visible data."""
        if plot_idx >= len(self.plot_widgets):
            return
        
        plot_widget = self.plot_widgets[plot_idx]
        # Get current x-axis range (displayed units, need to convert to seconds)
        x_range = plot_widget.viewRange()[0]
        # Convert displayed range back to seconds (data coordinates)
        x_range_seconds = [x_range[0] / self.time_scale, x_range[1] / self.time_scale]
        
        # Find y-data bounds in visible x range
        plot_type = self.plot_types[plot_idx] if plot_idx < len(self.plot_types) else None
        if plot_type and plot_type in self.available_modalities:
            y_min, y_max = float('inf'), float('-inf')
            
            # Use cached time array for performance
            time = self._time_seconds_cache
            
            # Use searchsorted for fast index lookup (much faster than boolean masking)
            start_idx = np.searchsorted(time, x_range_seconds[0], side='left')
            end_idx = np.searchsorted(time, x_range_seconds[1], side='right')
            
            for modality in self.available_modalities[plot_type]:
                if modality in self.plot_curves and self.plot_curves[modality].isVisible():
                    data = self.eyedata[modality]
                    # Slice data using indices (much faster than boolean mask)
                    if ma.is_masked(data):
                        visible_data = data[start_idx:end_idx].compressed()
                    else:
                        visible_data = data[start_idx:end_idx]
                    
                    if len(visible_data) > 0:
                        y_min = min(y_min, np.nanmin(visible_data))
                        y_max = max(y_max, np.nanmax(visible_data))
            
            if y_min != float('inf') and y_max != float('-inf'):
                # Add 10% padding
                padding = (y_max - y_min) * 0.1
                plot_widget.setYRange(y_min - padding, y_max + padding, padding=0)
    
    def _update_auto_y_axes(self):
        """Update all plots with auto y-axis enabled."""
        for plot_idx, enabled in self.auto_y_enabled.items():
            if enabled:
                self._update_auto_y_axis(plot_idx)
    
    def _on_time_unit_changed(self, unit: str):
        """Handle time unit change by rescaling the axis."""
        # Map unit to scale factor (from seconds)
        unit_map = {
            'ms': (1000.0, 'ms'),      # 1 second = 1000 ms
            's': (1.0, 's'),            # 1 second = 1 second
            'min': (1.0 / 60.0, 'min')  # 1 second = 1/60 minutes
        }
        
        if not self.plot_widgets:
            return
        
        # Update scale and unit
        axis_scale, self.time_unit = unit_map[unit]
        self.time_scale = axis_scale
        
        # Update x-axis scaling for all plots
        for i, plot_widget in enumerate(self.plot_widgets):
            axis = plot_widget.getAxis('bottom')
            
            # Set the axis scale to convert seconds to the target unit
            axis.setScale(axis_scale)
            
            # Ensure SI prefix is disabled
            axis.enableAutoSIPrefix(False)
            axis.autoSIPrefixScale = 1.0
            
            if i == len(self.plot_widgets) - 1:
                # Update label for bottom plot only
                plot_widget.setLabel('bottom', f'Time ({self.time_unit})', color='k')
        
        # Force a complete update of all widgets
        for plot_widget in self.plot_widgets:
            plot_widget.update()
            plot_widget.getViewBox().update()
    
    def _set_initial_view(self):
        """Set initial zoomed view for fast startup."""
        if not self.navigation:
            return
        
        # Time is in seconds
        time_in_seconds = self.eyedata.tx * 0.001
        total_duration = time_in_seconds[-1] - time_in_seconds[0]
        
        # Show first 30 seconds or 5% of data, whichever is smaller
        initial_window = min(30.0, total_duration * 0.05)
        
        # But show at least 10 seconds if data is longer
        if total_duration > 10.0:
            initial_window = max(initial_window, 10.0)
        
        # Set the view to show the initial window
        x_min = time_in_seconds[0]
        x_max = x_min + initial_window
        
        if self.plot_widgets:
            self.plot_widgets[0].setXRange(x_min, x_max, padding=0)
            # Update auto y-axis for the initial view
            self._update_auto_y_axes()
    
    def _on_show_whole_signal(self):
        """Show the entire signal."""
        if self.navigation:
            self.navigation.reset_view()
            # Update auto y-axis for the full view
            self._update_auto_y_axes()
    
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Handle keyboard events."""
        if self.navigation.handle_key_press(event):
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def get_intervals(self) -> Optional[Intervals]:
        """Get current selected intervals.
        
        Returns
        -------
        Intervals or None
            Current selected intervals, or None if no selections
        """
        return self.region_selector.get_intervals()

