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
        self.overlay_data = overlay_data.astype(np.float32) if overlay_data is not None else None
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
        
        # Reference to parameter window for coordinated close
        self.param_window = None
        
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
        
        # Plot data - overlay first (below), then original (on top)
        if self.overlay_data is not None:
            self._plot_overlay(self.overlay_data)
        self._plot_original()
        
        self._create_legend()
        self._set_initial_view()
        self._last_x_range = (self.data_min, self.data_max)
        
        self.freeze()
    
    def set_param_window(self, param_window):
        """Set reference to parameter window for coordinated closing."""
        self.param_window = param_window
    
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
        # Set order so original is drawn on top
        if self.original_line.line_normal:
            self.original_line.line_normal.order = 1001
        if self.original_line.line_masked:
            self.original_line.line_masked.order = 1001
    
    def _plot_overlay(self, data: np.ndarray):
        """Plot or update the overlay line."""
        data = np.asarray(data, dtype=np.float32)
        self.overlay_data = data
        
        # Remove existing overlay if present
        if self.overlay_line is not None:
            if self.overlay_line.line_normal is not None:
                self.overlay_line.line_normal.parent = None
            if self.overlay_line.line_masked is not None:
                self.overlay_line.line_masked.parent = None
        
        # Create new overlay line with bright green color for visibility
        self.overlay_line = LODLine(
            self.viewbox,
            self.time_seconds,
            data,
            color='#00DD00',  # Bright green for high visibility
            mask=None,
            width=3.0,  # Thicker line to be clearly visible
            lod_factors=self.lod_factors
        )
        # Set order so overlay is drawn ON TOP of original (higher order = drawn later)
        if self.overlay_line.line_normal:
            self.overlay_line.line_normal.order = 2000
        if self.overlay_line.line_masked:
            self.overlay_line.line_masked.order = 2000
        
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
            ('Tweaked', '#00DD00'),  # Bright green to match overlay
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
        
        # Overlay data
        if self.overlay_data is not None:
            overlay_visible = self.overlay_data[start_idx:end_idx]
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
    
    def on_close(self, event):
        """Handle close event - also close parameter window."""
        if self.param_window is not None:
            try:
                self.param_window.close()
            except Exception:
                pass
        super().on_close(event)
    
    def on_mouse_move(self, event):
        """Track mouse position for Y-axis zoom."""
        self._mouse_pos = event.pos
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for Y-axis zoom with Shift."""
        event.handled = True
        
        modifiers = event.modifiers
        shift_held = 'Shift' in modifiers if modifiers else False
        
        if not shift_held:
            return
        
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
        
        # Zoom in/out based on scroll direction
        if delta > 0:
            factor = 0.8  # Zoom in
        else:
            factor = 1.25  # Zoom out
        
        self._zoom_y_axis(factor)
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        key = event.key.name if hasattr(event.key, 'name') else str(event.key)
        modifiers = event.modifiers if hasattr(event, 'modifiers') else []
        shift_held = 'Shift' in modifiers if modifiers else False
        
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


class ParameterWindow:
    """Floating window with parameter spinboxes for interactive tweaking."""
    
    def __init__(self, params: Dict[str, Any], on_change: Callable[[Dict[str, Any]], None],
                 title: str = 'Parameters', parent=None):
        """Initialize parameter window.
        
        Parameters
        ----------
        params : dict
            Initial parameter values (int or float).
        on_change : callable
            Callback function called with updated params dict when values change.
        title : str
            Window title.
        parent : QWidget, optional
            Parent widget for window management.
        """
        self.params = dict(params)
        self.initial_params = dict(params)
        self.on_change = on_change
        self.spinboxes: Dict[str, Any] = {}
        self.canvas = None  # Reference to canvas for coordinated close
        
        # Import Qt
        try:
            from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                                         QLabel, QDoubleSpinBox, QSpinBox, QPushButton)
            from PyQt6.QtCore import Qt
            self.Qt = Qt
            self.QDoubleSpinBox = QDoubleSpinBox
            self.QSpinBox = QSpinBox
            self.QWidget = QWidget
        except ImportError:
            from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                         QLabel, QDoubleSpinBox, QSpinBox, QPushButton)
            from PyQt5.QtCore import Qt
            self.Qt = Qt
            self.QDoubleSpinBox = QDoubleSpinBox
            self.QSpinBox = QSpinBox
            self.QWidget = QWidget
        
        # Create a custom widget class to handle close event
        class ParamWidget(QWidget):
            def __init__(self, param_window, parent=None):
                super().__init__(parent)
                self.param_window = param_window
            
            def closeEvent(self, event):
                # Close the canvas when parameter window is closed
                if self.param_window.canvas is not None:
                    try:
                        self.param_window.canvas.close()
                    except Exception:
                        pass
                event.accept()
        
        # Create window
        self.window = ParamWidget(self, parent)
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
    
    def set_canvas(self, canvas):
        """Set reference to canvas for coordinated closing."""
        self.canvas = canvas
    
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
    
    def close(self):
        """Close the parameter window."""
        self.window.close()
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return dict(self.params)
