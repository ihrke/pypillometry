"""Navigation and keyboard/mouse interaction handling for the viewer."""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from typing import List

# PyQt5/PyQt6 compatibility for key constants
try:
    # PyQt6 style enums
    _Key_Left = QtCore.Qt.Key.Key_Left
    _Key_Right = QtCore.Qt.Key.Key_Right
    _Key_PageUp = QtCore.Qt.Key.Key_PageUp
    _Key_PageDown = QtCore.Qt.Key.Key_PageDown
    _Key_Up = QtCore.Qt.Key.Key_Up
    _Key_Down = QtCore.Qt.Key.Key_Down
    _Key_Home = QtCore.Qt.Key.Key_Home
    _Key_End = QtCore.Qt.Key.Key_End
    _Key_Plus = QtCore.Qt.Key.Key_Plus
    _Key_Equal = QtCore.Qt.Key.Key_Equal
    _Key_Minus = QtCore.Qt.Key.Key_Minus
    _Key_Space = QtCore.Qt.Key.Key_Space
    _Key_H = QtCore.Qt.Key.Key_H
    _RichText = QtCore.Qt.TextFormat.RichText
except AttributeError:
    # PyQt5 style enums
    _Key_Left = _Key_Left
    _Key_Right = _Key_Right
    _Key_PageUp = _Key_PageUp
    _Key_PageDown = _Key_PageDown
    _Key_Up = _Key_Up
    _Key_Down = _Key_Down
    _Key_Home = _Key_Home
    _Key_End = _Key_End
    _Key_Plus = _Key_Plus
    _Key_Equal = _Key_Equal
    _Key_Minus = _Key_Minus
    _Key_Space = _Key_Space
    _Key_H = _Key_H
    _RichText = _RichText


class NavigationHandler:
    """Handles keyboard and mouse navigation for the viewer.
    
    Attributes
    ----------
    plot_widgets : list of pg.PlotItem
        List of plot widgets to navigate
    data_min : float
        Minimum time value in data
    data_max : float
        Maximum time value in data
    """
    
    def __init__(self, plot_widgets: List, time_data=None):
        """Initialize navigation handler.
        
        Parameters
        ----------
        plot_widgets : list of pg.PlotItem
            Plot widgets to control
        time_data : array-like, optional
            Time data array to get exact bounds
        """
        self.plot_widgets = plot_widgets if isinstance(plot_widgets, list) else [plot_widgets]
        self.primary_plot = self.plot_widgets[0] if self.plot_widgets else None
        
        # Store exact data bounds if provided
        if time_data is not None:
            self.data_min = float(time_data[0])
            self.data_max = float(time_data[-1])
        else:
            self.data_min = None
            self.data_max = None
        
        # Configure mouse interactions for all plots
        for plot in self.plot_widgets:
            self._setup_mouse_interaction(plot)
    
    def _setup_mouse_interaction(self, plot_widget):
        """Configure mouse interactions for a plot.
        
        Parameters
        ----------
        plot_widget : pg.PlotItem
            Plot widget to configure
        """
        # Enable mouse interactions
        plot_widget.setMouseEnabled(x=True, y=True)
        
        # Enable menu for right-click
        plot_widget.setMenuEnabled(True)
        
        # Get the ViewBox and set limits
        vb = plot_widget.getViewBox()
        
        # Disable auto-range buttons to prevent zoom out beyond data
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        
        # Override wheel event to respect data bounds
        original_wheelEvent = vb.wheelEvent
        
        def bounded_wheelEvent(ev, axis=None):
            # Get data bounds (use stored values if available)
            if self.data_min is not None and self.data_max is not None:
                data_min, data_max = self.data_min, self.data_max
            else:
                data_bounds = vb.childrenBounds()[0]
                data_min, data_max = data_bounds[0], data_bounds[1]
            
            # Get current range
            current_range = vb.viewRange()[0]
            current_span = current_range[1] - current_range[0]
            data_span = data_max - data_min
            
            # Check if zooming out
            if ev.delta() < 0:  # Scrolling down = zoom out
                # If we're already at full view, don't zoom out more
                if current_span >= data_span * 0.95:  # If within 95% of full view
                    # Snap to exactly full view
                    vb.setXRange(data_min, data_max, padding=0)
                    ev.ignore()
                    return
            
            # Call original wheel event
            original_wheelEvent(ev, axis)
            
            # After zoom, check if we're close to full view
            new_range = vb.viewRange()[0]
            new_min, new_max = new_range[0], new_range[1]
            new_span = new_max - new_min
            
            # If zoomed out to within 95% of full view, snap to exactly full view
            if new_span >= data_span * 0.95:
                vb.setXRange(data_min, data_max, padding=0)
        
        vb.wheelEvent = bounded_wheelEvent
    
    def handle_key_press(self, event: QtGui.QKeyEvent):
        """Handle keyboard shortcuts.
        
        Parameters
        ----------
        event : QtGui.QKeyEvent
            Keyboard event
        
        Returns
        -------
        bool
            True if event was handled
        """
        key = event.key()
        modifiers = event.modifiers()
        
        # Get current view range of primary plot
        if not self.primary_plot:
            return False
        
        view_range = self.primary_plot.viewRange()
        x_range = view_range[0]
        y_range = view_range[1]
        x_center = (x_range[0] + x_range[1]) / 2
        x_span = x_range[1] - x_range[0]
        
        # Get data bounds (use stored values if available, otherwise from plot)
        if self.data_min is not None and self.data_max is not None:
            data_min, data_max = self.data_min, self.data_max
        else:
            data_bounds = self.primary_plot.vb.childrenBounds()[0]
            data_min, data_max = data_bounds[0], data_bounds[1]
        data_span = data_max - data_min
        
        handled = False
        
        # Arrow key navigation - pan left/right
        if key == _Key_Left:
            # Pan left (10% of span), stop at boundary
            shift = x_span * 0.1
            new_min = x_range[0] - shift
            new_max = x_range[1] - shift
            # Clamp to data bounds
            if new_min < data_min:
                new_min = data_min
                new_max = data_min + x_span
            self.set_x_range(new_min, new_max)
            handled = True
        
        elif key == _Key_Right:
            # Pan right (10% of span), stop at boundary
            shift = x_span * 0.1
            new_min = x_range[0] + shift
            new_max = x_range[1] + shift
            # Clamp to data bounds
            if new_max > data_max:
                new_max = data_max
                new_min = data_max - x_span
            self.set_x_range(new_min, new_max)
            handled = True
        
        elif key == _Key_PageUp:
            # Pan left faster (50% of span)
            shift = x_span * 0.5
            new_min = x_range[0] - shift
            new_max = x_range[1] - shift
            # Clamp to data bounds
            if new_min < data_min:
                new_min = data_min
                new_max = data_min + x_span
            self.set_x_range(new_min, new_max)
            handled = True
        
        elif key == _Key_PageDown:
            # Pan right faster (50% of span)
            shift = x_span * 0.5
            new_min = x_range[0] + shift
            new_max = x_range[1] + shift
            # Clamp to data bounds
            if new_max > data_max:
                new_max = data_max
                new_min = data_max - x_span
            self.set_x_range(new_min, new_max)
            handled = True
        
        elif key == _Key_Up:
            # Zoom in (reduce span by 20%)
            new_span = x_span * 0.8
            new_min = x_center - new_span/2
            new_max = x_center + new_span/2
            self.set_x_range(new_min, new_max)
            handled = True
        
        elif key == _Key_Down:
            # Zoom out (increase span by 20%), stop at full data view
            new_span = x_span * 1.2
            
            # Check if new span would exceed or be close to data bounds
            if new_span >= data_span * 0.95:  # Within 95% of full view
                # Snap to exactly full data range
                self.set_x_range(data_min, data_max)
            else:
                # Zoom out centered on current view
                new_min = x_center - new_span/2
                new_max = x_center + new_span/2
                
                # Adjust if we exceed bounds
                if new_min < data_min:
                    new_min = data_min
                    new_max = data_min + new_span
                if new_max > data_max:
                    new_max = data_max
                    new_min = data_max - new_span
                
                self.set_x_range(new_min, new_max)
            handled = True
        
        # Home/End keys
        elif key == _Key_Home:
            # Jump to start
            new_min = data_min
            new_max = min(data_min + x_span, data_max)
            self.set_x_range(new_min, new_max)
            handled = True
        
        elif key == _Key_End:
            # Jump to end
            new_max = data_max
            new_min = max(data_max - x_span, data_min)
            self.set_x_range(new_min, new_max)
            handled = True
        
        # Plus/minus zoom
        elif key in [_Key_Plus, _Key_Equal]:
            # Zoom in
            new_span = x_span * 0.8
            new_min = x_center - new_span/2
            new_max = x_center + new_span/2
            self.set_x_range(new_min, new_max)
            handled = True
        
        elif key == _Key_Minus:
            # Zoom out, stop at full data view
            new_span = x_span * 1.2
            
            # Check if new span would exceed or be close to data bounds
            if new_span >= data_span * 0.95:  # Within 95% of full view
                # Snap to exactly full data range
                self.set_x_range(data_min, data_max)
            else:
                # Zoom out centered on current view
                new_min = x_center - new_span/2
                new_max = x_center + new_span/2
                
                # Adjust if we exceed bounds
                if new_min < data_min:
                    new_min = data_min
                    new_max = data_min + new_span
                if new_max > data_max:
                    new_max = data_max
                    new_min = data_max - new_span
                
                self.set_x_range(new_min, new_max)
            handled = True
        
        # Space to reset view
        elif key == _Key_Space:
            self.reset_view()
            handled = True
        
        # H for help
        elif key == _Key_H:
            self.show_help_dialog()
            handled = True
        
        return handled
    
    def set_x_range(self, x_min: float, x_max: float):
        """Set X-axis range for all linked plots.
        
        Parameters
        ----------
        x_min : float
            Minimum x value
        x_max : float
            Maximum x value
        """
        # Set range for primary plot (others follow via linking)
        if self.primary_plot:
            self.primary_plot.setXRange(x_min, x_max, padding=0)
    
    def reset_view(self):
        """Reset view to show all data."""
        if self.primary_plot:
            # Get data bounds (use stored values if available)
            if self.data_min is not None and self.data_max is not None:
                data_min, data_max = self.data_min, self.data_max
            else:
                data_bounds = self.primary_plot.vb.childrenBounds()[0]
                data_min, data_max = data_bounds[0], data_bounds[1]
            
            # Set to exactly full data range
            self.set_x_range(data_min, data_max)
        
        # Also reset y-axis for all plots
        for plot in self.plot_widgets:
            plot.enableAutoRange(axis='y')
            plot.autoRange()
    
    def show_help_dialog(self):
        """Show keyboard shortcut help dialog."""
        help_text = """
<h3>Keyboard Shortcuts</h3>
<table>
<tr><td><b>←/→</b></td><td>Pan left/right (10% of view)</td></tr>
<tr><td><b>PgUp/PgDn</b></td><td>Pan left/right (50% of view)</td></tr>
<tr><td><b>↑/↓</b></td><td>Zoom in/out</td></tr>
<tr><td><b>Home</b></td><td>Jump to start of data</td></tr>
<tr><td><b>End</b></td><td>Jump to end of data</td></tr>
<tr><td><b>Space</b></td><td>Show whole signal</td></tr>
<tr><td><b>H</b></td><td>Show this help</td></tr>
</table>

<h3>Mouse Controls</h3>
<table>
<tr><td><b>Left drag</b></td><td>Pan view</td></tr>
<tr><td><b>Right drag</b></td><td>Zoom to rectangle</td></tr>
<tr><td><b>Wheel</b></td><td>Zoom in/out</td></tr>
<tr><td><b>Right click</b></td><td>Context menu</td></tr>
</table>

<h3>Notes</h3>
<ul>
<li>Panning stops at data boundaries</li>
<li>Zooming out stops at full data view</li>
<li>Use "Show Whole Signal" button or Space key to reset</li>
</ul>
        """
        
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Viewer Help")
        msg_box.setTextFormat(_RichText)
        msg_box.setText(help_text)
        msg_box.exec_()

