"""Interactive parameter tweaking viewer using VisPy and Qt.

This module provides a viewer for interactively tweaking function parameters
while seeing the result overlaid on the original data.
"""

import numpy as np
from vispy import app, scene
from vispy.scene import SceneCanvas
from typing import Dict, Callable, Any, Optional, List

from .visuals import LODLine
from .navigation import NavigationHandler


class TweakCanvas(SceneCanvas):
    """GPU-accelerated canvas for parameter tweaking with original data + overlay."""
    
    def __init__(self, time_seconds: np.ndarray, original_data: np.ndarray,
                 overlay_data: np.ndarray = None, title: str = 'Tweak Viewer'):
        """Initialize the tweak canvas.
        
        Parameters
        ----------
        time_seconds : ndarray
            Time vector in seconds.
        original_data : ndarray
            Original data to display (1D array).
        overlay_data : ndarray, optional
            Initial overlay data from function (same length as original).
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
        self.data_min = float(self.time_seconds[0])
        self.data_max = float(self.time_seconds[-1])
        
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
        
        # Storage
        self.original_line: LODLine = None
        self.overlay_line: LODLine = None
        
        # Create layout
        self.grid = self.central_widget.add_grid(spacing=0)
        self._create_subplot()
        
        # Navigation handler
        self.navigation = NavigationHandler(
            [self.viewbox],
            data_min=self.data_min,
            data_max=self.data_max
        )
        
        # Plot data
        self._plot_original()
        if overlay_data is not None:
            self._plot_overlay(overlay_data)
        
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
    
    def _plot_original(self):
        """Plot the original data line."""
        self.original_line = LODLine(
            self.viewbox,
            self.time_seconds,
            self.original_data,
            color='#0066CC',  # Blue
            mask=None,
            width=2.0,
            lod_factors=self.lod_factors
        )
    
    def _plot_overlay(self, data: np.ndarray):
        """Plot or update the overlay line."""
        data = np.asarray(data, dtype=np.float32)
        
        # Remove existing overlay if present
        if self.overlay_line is not None:
            if self.overlay_line.line_normal is not None:
                self.overlay_line.line_normal.parent = None
            if self.overlay_line.line_masked is not None:
                self.overlay_line.line_masked.parent = None
        
        # Create new overlay line
        self.overlay_line = LODLine(
            self.viewbox,
            self.time_seconds,
            data,
            color='#CC0000',  # Red
            mask=None,
            width=2.0,
            lod_factors=self.lod_factors
        )
        
        # Update LOD for current view
        if hasattr(self, '_last_x_range'):
            x_min, x_max = self._last_x_range
            self.overlay_line.update_for_view(x_min, x_max)
    
    def update_overlay(self, data: np.ndarray):
        """Update the overlay with new data.
        
        Parameters
        ----------
        data : ndarray
            New overlay data (same length as original).
        """
        self.unfreeze()
        self._plot_overlay(data)
        self._update_y_range()
        self.update()
        self.freeze()
    
    def _create_legend(self):
        """Create a legend showing original vs tweaked."""
        legend_row = 2
        legend_view = self.grid.add_view(row=legend_row, col=0, col_span=2, border_color=None)
        legend_view.camera = scene.PanZoomCamera(aspect=None)
        legend_view.camera.interactive = False
        legend_view.height_min = 30
        legend_view.height_max = 30
        legend_view.stretch = (1, 0.0001)
        
        legend_items = [
            ('Original', '#0066CC'),
            ('Tweaked', '#CC0000'),
        ]
        
        total_width = sum(len(label) * 0.012 + 0.06 for label, _ in legend_items)
        x_start = 0.5 - total_width / 2
        x_pos = x_start
        
        for label, color in legend_items:
            line_pos = np.array([
                [x_pos - 0.015, 0.5],
                [x_pos + 0.015, 0.5]
            ], dtype=np.float32)
            scene.Line(pos=line_pos, color=color, width=4, parent=legend_view.scene)
            
            scene.Text(
                text=label,
                pos=(x_pos + 0.02, 0.5),
                color='black',
                font_size=8,
                anchor_x='left',
                anchor_y='center',
                parent=legend_view.scene
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
        y_min, y_max = self._get_y_range(x_min, x_max)
        self._safe_set_camera_range(self.viewbox.camera, x_min, x_max, y_min, y_max)
        self._update_lod_visuals(x_min, x_max)
        self._last_x_range = (x_min, x_max)
    
    def _update_y_range(self):
        """Update Y range based on current X range."""
        if hasattr(self, '_last_x_range'):
            x_min, x_max = self._last_x_range
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
        
        # Overlay data
        if self.overlay_line is not None:
            overlay_visible = self.overlay_line.data_full[start_idx:end_idx]
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
        """Update LOD for all lines."""
        if self.original_line is not None:
            self.original_line.update_for_view(x_min, x_max)
        if self.overlay_line is not None:
            self.overlay_line.update_for_view(x_min, x_max)
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        key = event.key.name if hasattr(event.key, 'name') else str(event.key)
        
        if key in ('Q', 'Escape'):
            self.close()
            return
        
        # Navigation keys
        new_range = None
        if key == 'Left':
            new_range = self.navigation.pan_left()
        elif key == 'Right':
            new_range = self.navigation.pan_right()
        elif key == 'Up' or key == '+' or key == '=':
            new_range = self.navigation.zoom_in()
        elif key == 'Down' or key == '-':
            new_range = self.navigation.zoom_out()
        elif key == 'Page_Up':
            new_range = self.navigation.pan_left(0.5)
        elif key == 'Page_Down':
            new_range = self.navigation.pan_right(0.5)
        elif key == 'Home':
            new_range = self.navigation.jump_to_start()
        elif key == 'End':
            new_range = self.navigation.jump_to_end()
        elif key == 'Space':
            new_range = self.navigation.show_all()
        
        if new_range:
            self._set_view_range(*new_range)
            self.update()


class ParameterWindow:
    """Floating window with parameter spinboxes for interactive tweaking."""
    
    def __init__(self, params: Dict[str, Any], on_change: Callable[[Dict[str, Any]], None],
                 title: str = 'Parameters'):
        """Initialize parameter window.
        
        Parameters
        ----------
        params : dict
            Initial parameter values (int or float).
        on_change : callable
            Callback function called with updated params dict when values change.
        title : str
            Window title.
        """
        self.params = dict(params)
        self.initial_params = dict(params)
        self.on_change = on_change
        self.spinboxes: Dict[str, Any] = {}
        
        # Import Qt
        try:
            from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                                         QLabel, QDoubleSpinBox, QSpinBox, QPushButton)
            from PyQt6.QtCore import Qt
            self.Qt = Qt
            self.QDoubleSpinBox = QDoubleSpinBox
            self.QSpinBox = QSpinBox
        except ImportError:
            from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                         QLabel, QDoubleSpinBox, QSpinBox, QPushButton)
            from PyQt5.QtCore import Qt
            self.Qt = Qt
            self.QDoubleSpinBox = QDoubleSpinBox
            self.QSpinBox = QSpinBox
        
        # Create window
        self.window = QWidget()
        self.window.setWindowTitle(title)
        self.window.setMinimumWidth(250)
        
        layout = QVBoxLayout()
        self.window.setLayout(layout)
        
        # Create spinbox for each parameter
        for name, value in params.items():
            row = QHBoxLayout()
            
            label = QLabel(f"{name}:")
            label.setMinimumWidth(100)
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
            
            spinbox.setMinimumWidth(100)
            row.addWidget(spinbox)
            
            self.spinboxes[name] = spinbox
            layout.addLayout(row)
        
        # Reset button
        reset_btn = QPushButton("Reset to Initial")
        reset_btn.clicked.connect(self._reset_params)
        layout.addWidget(reset_btn)
        
        # Stretch at bottom
        layout.addStretch()
    
    def _on_value_changed(self, name: str, value: Any):
        """Handle spinbox value change."""
        self.params[name] = value
        self.on_change(self.params)
    
    def _reset_params(self):
        """Reset all parameters to initial values."""
        for name, value in self.initial_params.items():
            self.params[name] = value
            spinbox = self.spinboxes[name]
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)
        self.on_change(self.params)
    
    def show(self):
        """Show the parameter window."""
        self.window.show()
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return dict(self.params)

