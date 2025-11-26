"""Keyboard navigation handler for GPU viewer."""

from vispy import scene
from vispy.util import keys
from typing import List


class NavigationHandler:
    """Handles keyboard navigation for the GPU viewer.
    
    Keyboard-only navigation with strict data bounds.
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
        """Handle keyboard events."""
        if not self.viewboxes:
            return False
        
        # Get current view range
        camera = self.viewboxes[0].camera
        rect = camera.get_state()['rect']
        x_min, x_max = rect.left, rect.right
        x_span = x_max - x_min
        x_center = (x_min + x_max) / 2
        
        key = event.key
        handled = False
        new_x_min, new_x_max = None, None
        
        # LEFT: Pan left 10% (stop at start)
        if key == keys.LEFT:
            if x_min > self.data_min:  # Only move if not at start
                shift = x_span * 0.1
                new_x_min = max(self.data_min, x_min - shift)
                new_x_max = new_x_min + x_span
            handled = True
        
        # RIGHT: Pan right 10% (stop at end)
        elif key == keys.RIGHT:
            if x_max < self.data_max:  # Only move if not at end
                shift = x_span * 0.1
                new_x_max = min(self.data_max, x_max + shift)
                new_x_min = new_x_max - x_span
            handled = True
        
        # PAGE UP: Pan left 50% (stop at start)
        elif key == keys.PAGEUP:
            if x_min > self.data_min:
                shift = x_span * 0.5
                new_x_min = max(self.data_min, x_min - shift)
                new_x_max = new_x_min + x_span
            handled = True
        
        # PAGE DOWN: Pan right 50% (stop at end)
        elif key == keys.PAGEDOWN:
            if x_max < self.data_max:
                shift = x_span * 0.5
                new_x_max = min(self.data_max, x_max + shift)
                new_x_min = new_x_max - x_span
            handled = True
        
        # UP: Zoom in 20% (centered)
        elif key == keys.UP:
            new_span = x_span * 0.8
            new_x_min = x_center - new_span / 2
            new_x_max = x_center + new_span / 2
            
            # Keep within bounds
            if new_x_min < self.data_min:
                new_x_min = self.data_min
                new_x_max = self.data_min + new_span
            if new_x_max > self.data_max:
                new_x_max = self.data_max
                new_x_min = self.data_max - new_span
            handled = True
        
        # DOWN: Zoom out 20% (snap to full if would exceed bounds)
        elif key == keys.DOWN:
            new_span = x_span * 1.25
            
            # If new span would exceed or nearly exceed data, snap to full view
            if new_span >= self.data_span * 0.95:
                new_x_min = self.data_min
                new_x_max = self.data_max
            else:
                new_x_min = x_center - new_span / 2
                new_x_max = x_center + new_span / 2
                
                # Keep within bounds
                if new_x_min < self.data_min:
                    new_x_min = self.data_min
                    new_x_max = self.data_min + new_span
                if new_x_max > self.data_max:
                    new_x_max = self.data_max
                    new_x_min = self.data_max - new_span
            handled = True
        
        # HOME: Jump to start
        elif key == keys.HOME:
            new_x_min = self.data_min
            new_x_max = min(self.data_min + x_span, self.data_max)
            handled = True
        
        # END: Jump to end
        elif key == keys.END:
            new_x_max = self.data_max
            new_x_min = max(self.data_max - x_span, self.data_min)
            handled = True
        
        # PLUS: Zoom in
        elif key in ['+', '=']:
            new_span = x_span * 0.8
            new_x_min = x_center - new_span / 2
            new_x_max = x_center + new_span / 2
            
            if new_x_min < self.data_min:
                new_x_min = self.data_min
                new_x_max = self.data_min + new_span
            if new_x_max > self.data_max:
                new_x_max = self.data_max
                new_x_min = self.data_max - new_span
            handled = True
        
        # MINUS: Zoom out
        elif key == '-':
            new_span = x_span * 1.25
            
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
        
        # SPACE: Reset to full view
        elif key == keys.SPACE:
            new_x_min = self.data_min
            new_x_max = self.data_max
            handled = True
        
        # Q/ESC: Close
        elif key in ['Q', 'q', keys.ESCAPE]:
            if self.viewboxes:
                canvas = self.viewboxes[0].canvas
                if canvas:
                    canvas.close()
            handled = True
        
        # Apply new range
        if new_x_min is not None and new_x_max is not None:
            self._set_x_range(new_x_min, new_x_max)
        
        return handled
    
    def _set_x_range(self, x_min: float, x_max: float):
        """Set x-axis range for all viewboxes (y will be updated by canvas)."""
        for viewbox in self.viewboxes:
            rect = viewbox.camera.get_state()['rect']
            viewbox.camera.set_range(x=(x_min, x_max), y=(rect.bottom, rect.top))
