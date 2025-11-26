"""Keyboard navigation handler for GPU viewer."""

from vispy import scene
from vispy.util import keys
from typing import List, Optional


class NavigationHandler:
    """Handles keyboard navigation for the GPU viewer.
    
    Supports panning, zooming, and jumping to start/end of data.
    All viewboxes are kept synchronized (linked x-axis).
    
    Parameters
    ----------
    viewboxes : list of scene.ViewBox
        List of viewboxes to navigate
    data_min : float
        Minimum time value in data (seconds)
    data_max : float
        Maximum time value in data (seconds)
    """
    
    def __init__(
        self, 
        viewboxes: List[scene.ViewBox],
        data_min: float,
        data_max: float
    ):
        self.viewboxes = viewboxes
        self.data_min = data_min
        self.data_max = data_max
        self.data_span = data_max - data_min
    
    def handle_key_press(self, event) -> bool:
        """Handle keyboard events.
        
        Parameters
        ----------
        event : vispy key event
            Keyboard event from canvas
        
        Returns
        -------
        bool
            True if event was handled
        """
        if not self.viewboxes:
            return False
        
        # Get current view range from first viewbox
        camera = self.viewboxes[0].camera
        rect = camera.get_state()['rect']
        x_min, x_max = rect.left, rect.right
        x_center = (x_min + x_max) / 2
        x_span = x_max - x_min
        
        handled = False
        new_x_min = None
        new_x_max = None
        
        key = event.key
        
        # Left arrow: pan left 10%
        if key == keys.LEFT:
            shift = x_span * 0.1
            new_x_min = x_min - shift
            new_x_max = x_max - shift
            # Clamp to data bounds
            if new_x_min < self.data_min:
                new_x_min = self.data_min
                new_x_max = self.data_min + x_span
            handled = True
        
        # Right arrow: pan right 10%
        elif key == keys.RIGHT:
            shift = x_span * 0.1
            new_x_min = x_min + shift
            new_x_max = x_max + shift
            # Clamp to data bounds
            if new_x_max > self.data_max:
                new_x_max = self.data_max
                new_x_min = self.data_max - x_span
            handled = True
        
        # Page Up: pan left 50%
        elif key == keys.PAGEUP:
            shift = x_span * 0.5
            new_x_min = x_min - shift
            new_x_max = x_max - shift
            # Clamp to data bounds
            if new_x_min < self.data_min:
                new_x_min = self.data_min
                new_x_max = self.data_min + x_span
            handled = True
        
        # Page Down: pan right 50%
        elif key == keys.PAGEDOWN:
            shift = x_span * 0.5
            new_x_min = x_min + shift
            new_x_max = x_max + shift
            # Clamp to data bounds
            if new_x_max > self.data_max:
                new_x_max = self.data_max
                new_x_min = self.data_max - x_span
            handled = True
        
        # Up arrow: zoom in 20%
        elif key == keys.UP:
            new_span = x_span * 0.8
            new_x_min = x_center - new_span / 2
            new_x_max = x_center + new_span / 2
            handled = True
        
        # Down arrow: zoom out 20% (stop at full data view)
        elif key == keys.DOWN:
            new_span = x_span * 1.2
            if new_span >= self.data_span * 0.95:
                # Snap to full data range
                new_x_min = self.data_min
                new_x_max = self.data_max
            else:
                new_x_min = x_center - new_span / 2
                new_x_max = x_center + new_span / 2
                # Adjust if exceeds bounds
                if new_x_min < self.data_min:
                    new_x_min = self.data_min
                    new_x_max = self.data_min + new_span
                if new_x_max > self.data_max:
                    new_x_max = self.data_max
                    new_x_min = self.data_max - new_span
            handled = True
        
        # Home: jump to start
        elif key == keys.HOME:
            new_x_min = self.data_min
            new_x_max = min(self.data_min + x_span, self.data_max)
            handled = True
        
        # End: jump to end
        elif key == keys.END:
            new_x_max = self.data_max
            new_x_min = max(self.data_max - x_span, self.data_min)
            handled = True
        
        # Plus/Equal: zoom in
        elif key in ['+', '=']:
            new_span = x_span * 0.8
            new_x_min = x_center - new_span / 2
            new_x_max = x_center + new_span / 2
            handled = True
        
        # Minus: zoom out (stop at full data view)
        elif key == '-':
            new_span = x_span * 1.2
            if new_span >= self.data_span * 0.95:
                new_x_min = self.data_min
                new_x_max = self.data_max
            else:
                new_x_min = x_center - new_span / 2
                new_x_max = x_center + new_span / 2
                if new_x_min < self.data_min:
                    new_x_min = self.data_min
                    new_x_max = self.data_min + new_span
                if new_x_max > self.data_max:
                    new_x_max = self.data_max
                    new_x_min = self.data_max - new_span
            handled = True
        
        # Space: reset to full view
        elif key == keys.SPACE:
            new_x_min = self.data_min
            new_x_max = self.data_max
            handled = True
        
        # Q or Escape: close viewer
        elif key in ['Q', 'q', keys.ESCAPE]:
            # Close the canvas
            if self.viewboxes:
                canvas = self.viewboxes[0].canvas
                if canvas:
                    canvas.close()
            handled = True
        
        # Apply new range to all viewboxes
        if handled and new_x_min is not None and new_x_max is not None:
            self._set_x_range(new_x_min, new_x_max)
        
        return handled
    
    def _set_x_range(self, x_min: float, x_max: float):
        """Set x-axis range for all viewboxes.
        
        Parameters
        ----------
        x_min : float
            New minimum x value
        x_max : float
            New maximum x value
        """
        for viewbox in self.viewboxes:
            # Get current y-range
            rect = viewbox.camera.get_state()['rect']
            y_min, y_max = rect.bottom, rect.top
            
            # Set new range preserving y
            viewbox.camera.set_range(x=(x_min, x_max), y=(y_min, y_max))
    
    def reset_view(self):
        """Reset view to show all data."""
        self._set_x_range(self.data_min, self.data_max)

