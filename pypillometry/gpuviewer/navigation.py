"""Keyboard navigation handler for GPU viewer."""

from vispy.util import keys
from typing import List, Optional, Tuple


class NavigationHandler:
    """Handles keyboard navigation for the GPU viewer.
    
    Returns new x-range values; does NOT set them directly.
    """
    
    def __init__(
        self, 
        viewboxes: List,
        data_min: float,
        data_max: float
    ):
        self.viewboxes = viewboxes
        self.data_min = data_min
        self.data_max = data_max
        self.data_span = data_max - data_min
        
        # Current view state (managed by this class)
        self.current_x_min = data_min
        self.current_x_max = data_max
    
    def set_view(self, x_min: float, x_max: float):
        """Set current view state."""
        self.current_x_min = x_min
        self.current_x_max = x_max
    
    def handle_key_press(self, event) -> Optional[Tuple[float, float]]:
        """Handle keyboard events.
        
        Returns
        -------
        tuple or None
            (new_x_min, new_x_max) if view should change, None otherwise
        """
        key = event.key
        
        x_min = self.current_x_min
        x_max = self.current_x_max
        x_span = x_max - x_min
        x_center = (x_min + x_max) / 2
        
        new_x_min, new_x_max = None, None
        
        # LEFT: Pan left 10%
        if key == keys.LEFT:
            if x_min > self.data_min + 0.001:  # Not at start
                shift = x_span * 0.1
                new_x_min = x_min - shift
                if new_x_min < self.data_min:
                    new_x_min = self.data_min
                new_x_max = new_x_min + x_span
        
        # RIGHT: Pan right 10%
        elif key == keys.RIGHT:
            if x_max < self.data_max - 0.001:  # Not at end
                shift = x_span * 0.1
                new_x_max = x_max + shift
                if new_x_max > self.data_max:
                    new_x_max = self.data_max
                new_x_min = new_x_max - x_span
        
        # PAGE UP: Pan left 50%
        elif key == keys.PAGEUP:
            if x_min > self.data_min + 0.001:
                shift = x_span * 0.5
                new_x_min = x_min - shift
                if new_x_min < self.data_min:
                    new_x_min = self.data_min
                new_x_max = new_x_min + x_span
        
        # PAGE DOWN: Pan right 50%
        elif key == keys.PAGEDOWN:
            if x_max < self.data_max - 0.001:
                shift = x_span * 0.5
                new_x_max = x_max + shift
                if new_x_max > self.data_max:
                    new_x_max = self.data_max
                new_x_min = new_x_max - x_span
        
        # UP: Zoom in 20%
        elif key == keys.UP:
            new_span = x_span * 0.8
            new_x_min = x_center - new_span / 2
            new_x_max = x_center + new_span / 2
            # Clamp
            if new_x_min < self.data_min:
                new_x_min = self.data_min
                new_x_max = self.data_min + new_span
            if new_x_max > self.data_max:
                new_x_max = self.data_max
                new_x_min = self.data_max - new_span
        
        # DOWN: Zoom out 25%
        elif key == keys.DOWN:
            new_span = x_span * 1.25
            if new_span >= self.data_span * 0.95:
                # Snap to full
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
        
        # HOME: Jump to start
        elif key == keys.HOME:
            new_x_min = self.data_min
            new_x_max = min(self.data_min + x_span, self.data_max)
        
        # END: Jump to end
        elif key == keys.END:
            new_x_max = self.data_max
            new_x_min = max(self.data_max - x_span, self.data_min)
        
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
        
        # SPACE: Full view
        elif key == keys.SPACE:
            new_x_min = self.data_min
            new_x_max = self.data_max
        
        # Q/ESC: Close
        elif key in ['Q', 'q', keys.ESCAPE]:
            if self.viewboxes:
                canvas = self.viewboxes[0].canvas
                if canvas:
                    canvas.close()
            return None
        
        # Update internal state and return
        if new_x_min is not None and new_x_max is not None:
            self.current_x_min = new_x_min
            self.current_x_max = new_x_max
            return (new_x_min, new_x_max)
        
        return None
