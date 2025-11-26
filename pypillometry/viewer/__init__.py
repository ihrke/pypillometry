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
import os
import locale
from typing import Optional, Callable

# Suppress Qt platform warnings before importing Qt
os.environ.setdefault('QT_LOGGING_RULES', 'qt.qpa.*=false;qt.glx=false')

from pyqtgraph.Qt import QtWidgets, QtGui
import pyqtgraph as pg

# Try to configure Qt for better OpenGL support on Wayland/XWayland
try:
    fmt = QtGui.QSurfaceFormat()
    fmt.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
    fmt.setSwapBehavior(QtGui.QSurfaceFormat.DoubleBuffer)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
except:
    pass

# Configure PyQtGraph for performance
# Note: useOpenGL only affects 3D plots, not 2D line plots
# 2D performance comes from downsampling and clipToView
try:
    pg.setConfigOptions(
        antialias=False,    # Disable antialiasing for speed
        enableExperimental=False
    )
except Exception:
    pass  # Ignore configuration errors

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
    # Save LC_TIME locale before Qt initialization (Qt may change it)
    # This prevents issues with date parsing in EDF files after viewer is opened
    try:
        saved_lc_time = locale.getlocale(locale.LC_TIME)
    except:
        saved_lc_time = None
    
    # Get or create QApplication
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Restore LC_TIME locale after Qt initialization
    # Qt can change the locale (e.g., to system locale like nb_NO), which breaks
    # date parsing in eyelinkio that expects English month/day abbreviations
    if saved_lc_time is not None:
        try:
            locale.setlocale(locale.LC_TIME, saved_lc_time)
        except:
            # If restoring fails, force English locale for date parsing
            try:
                locale.setlocale(locale.LC_TIME, 'C')
            except:
                pass  # If all else fails, continue with whatever Qt set
    
    # Create viewer window
    try:
        viewer = ViewerWindow(eyedata, separate_plots=separate_plots, callback=callback)
        viewer.show()
    except Exception as e:
        import traceback
        print("Error creating viewer window:")
        traceback.print_exc()
        raise
    
    # Check if running in Jupyter/IPython
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        in_jupyter = ipython is not None
        
        # Enable Qt event loop integration in Jupyter automatically
        if in_jupyter and ipython is not None:
            try:
                current_loop = getattr(ipython, 'active_eventloop', None)
                
                if current_loop not in ['qt', 'qt5']:
                    # Silently enable Qt event loop
                    try:
                        ipython.run_line_magic('gui', 'qt5')
                    except:
                        try:
                            ipython.run_line_magic('gui', 'qt')
                        except:
                            pass
            except:
                pass
    except:
        in_jupyter = False
    
    # Process initial events to render the window properly
    try:
        app.processEvents()
    except:
        pass
    
    # Blocking mode: wait for window to close
    # Note: In Jupyter, blocking mode may not work well - use callback instead
    if callback is None:
        if in_jupyter:
            # In Jupyter, don't block - just return viewer and let event loop run
            import warnings
            warnings.warn(
                "Running viewer in Jupyter in blocking mode. "
                "Window will open but may not block properly. "
                "For better control in Jupyter, use: viewer = pp.view(data, callback=lambda x: None)",
                UserWarning
            )
            return viewer
        else:
            app.exec_()
            
            # Return selected intervals if accepted
            if viewer.accepted:
                return viewer.selected_intervals
            else:
                return None
    
    # Non-blocking mode: return viewer object
    else:
        return viewer

