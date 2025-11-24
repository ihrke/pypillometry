"""Interactive viewer for eye-tracking data.

This module provides a fast, interactive viewer for eye-tracking data using PyQtGraph.

Usage
-----
Simple API (blocking mode, waits for user to accept/cancel):

>>> import pypillometry as pp
>>> data = pp.EyeData.from_eyelink('data.edf')  # doctest: +SKIP
>>> intervals = pp.view(data)  # doctest: +SKIP

With callback for non-blocking mode:

>>> def on_regions_changed(intervals):  # doctest: +SKIP
...     print(f"Selected {len(intervals)} regions")  # doctest: +SKIP
>>> viewer = pp.view(data, callback=on_regions_changed)  # doctest: +SKIP
>>> selected = viewer.get_intervals()  # doctest: +SKIP

Single plot mode:

>>> intervals = pp.view(data, separate_plots=False)  # doctest: +SKIP

Features
--------
- Fast rendering with automatic downsampling for large datasets
- Mouse navigation: wheel zoom, drag pan, right-click zoom box
- Keyboard shortcuts: arrows for pan/zoom, Home/End, Space to reset
- Toggle data modalities on/off
- Toggle event markers
- Mask visualization (shaded regions or gaps)
- Interactive region selection returning Intervals objects
- Works from Jupyter notebooks (opens separate window)
"""

import sys
from typing import Optional, Callable
from pyqtgraph.Qt import QtWidgets

from .viewer_window import ViewerWindow
from ..intervals import Intervals

__all__ = ['view', 'ViewerWindow']


def view(
    eyedata,
    separate_plots: bool = True,
    callback: Optional[Callable[[Intervals], None]] = None
) -> Optional[Intervals]:
    """View eye-tracking data interactively.
    
    Opens an interactive viewer window for exploring eye-tracking data.
    Supports fast navigation, data toggling, and region selection.
    
    In blocking mode (default, no callback), the function waits until the user
    clicks "Accept" or "Cancel", then returns the selected Intervals object.
    
    In non-blocking mode (with callback), the window stays open and the callback
    is called whenever the region selection changes. The window object is returned
    for programmatic access.
    
    Parameters
    ----------
    eyedata : EyeData-like object
        Eye-tracking data object. Must have:
        - `tx` attribute: time vector in ms
        - Dictionary-like access to data: eyedata['left_pupil'], etc.
        - Optional: `event_onsets`, `event_labels` for event markers
    separate_plots : bool, default True
        If True, create separate vertically-stacked plots for each data modality
        with linked x-axes. If False, plot all modalities in a single plot.
    callback : callable, optional
        Callback function with signature `callback(intervals)` that is called
        whenever region selection changes. If provided, enables non-blocking mode.
    
    Returns
    -------
    Intervals or None (blocking mode)
        Selected time intervals when user clicks "Accept", or None if "Cancel".
    ViewerWindow (non-blocking mode)
        The viewer window object. Use `viewer.get_intervals()` to get selections.
    
    Notes
    -----
    The viewer requires PyQtGraph and PyQt5/PySide2 to be installed.
    
    Mouse controls:
    - Left drag: Pan
    - Right drag: Zoom to rectangle
    - Wheel: Zoom in/out
    - Right click: Context menu
    
    Keyboard shortcuts:
    - Arrow keys: Pan (left/right), zoom (up/down)
    - Home/End: Jump to start/end
    - +/-: Zoom in/out
    - Space: Reset view
    - H: Show help
    
    Examples
    --------
    Simple blocking usage:
    
    >>> import pypillometry as pp
    >>> data = pp.EyeData.from_eyelink('data.edf')  # doctest: +SKIP
    >>> intervals = pp.view(data)  # Opens window, waits for user  # doctest: +SKIP
    >>> if intervals:  # doctest: +SKIP
    ...     print(f"Selected {len(intervals)} regions")  # doctest: +SKIP
    
    Non-blocking with callback:
    
    >>> def on_change(intervals):  # doctest: +SKIP
    ...     if intervals:  # doctest: +SKIP
    ...         print(f"Current: {len(intervals)} regions")  # doctest: +SKIP
    >>> viewer = pp.view(data, callback=on_change)  # doctest: +SKIP
    >>> # Window stays open, callback is called on changes  # doctest: +SKIP
    
    Single plot mode:
    
    >>> intervals = pp.view(data, separate_plots=False)  # doctest: +SKIP
    """
    # Get or create QApplication
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create viewer window
    viewer = ViewerWindow(eyedata, separate_plots=separate_plots, callback=callback)
    viewer.show()
    
    # Blocking mode: wait for window to close
    if callback is None:
        app.exec_()
        
        # Return selected intervals if accepted
        if viewer.accepted:
            return viewer.selected_intervals
        else:
            return None
    
    # Non-blocking mode: return viewer object
    else:
        return viewer

