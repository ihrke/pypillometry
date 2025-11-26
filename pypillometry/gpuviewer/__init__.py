"""GPU-accelerated viewer for eye-tracking data using VisPy.

This module provides a fast, GPU-accelerated viewer for long eye-tracking
recordings using VisPy for OpenGL rendering.

Usage
-----
>>> import pypillometry as pp
>>> data = pp.EyeData.from_eyelink('data.edf')  # doctest: +SKIP
>>> pp.gpuview(data)  # doctest: +SKIP

Features
--------
- GPU-accelerated rendering for smooth navigation of large datasets
- Keyboard-only navigation (no GUI controls needed)
- Masked data highlighted with colored background
- Event markers shown as vertical grey stripes with labels
- Separate plots for each variable type (pupil, x, y)
- Different colors for left (blue) and right (red) eye data
"""

__all__ = ['gpuview']


def gpuview(eyedata, overlay_pupil=None, overlay_x=None, overlay_y=None) -> None:
    """View eye-tracking data with GPU-accelerated rendering.
    
    Opens an interactive viewer window using VisPy for fast GPU-based
    rendering. Suitable for long recordings (60+ minutes at 1000 Hz).
    
    Parameters
    ----------
    eyedata : EyeData-like object
        Eye-tracking data object with:
        - `tx` attribute: time vector in ms
        - Dictionary-like access to data: eyedata['left_pupil'], etc.
        - Optional: `event_onsets`, `event_labels` for event markers
    overlay_pupil : dict, optional
        Additional timeseries to overlay on the pupil plot.
        Keys are labels for legend, values are either:
        - str: name of data in eyedata.data (e.g., 'left_pupil_filtered')
        - array-like: timeseries of same length as eyedata
    overlay_x : dict, optional
        Additional timeseries to overlay on the gaze X plot.
    overlay_y : dict, optional
        Additional timeseries to overlay on the gaze Y plot.
    
    Notes
    -----
    Keyboard controls:
    - Left/Right arrows: Pan 10% of view
    - Up/Down arrows: Zoom in/out 20%
    - PgUp/PgDn: Pan 50% of view  
    - Home/End: Jump to start/end
    - Space: Reset to full view
    - +/-: Zoom in/out
    - M: Toggle mask regions
    - O: Toggle event markers
    - H/?: Show help
    - Q/Esc: Close viewer
    
    Examples
    --------
    >>> import pypillometry as pp
    >>> data = pp.EyeData.from_eyelink('recording.edf')  # doctest: +SKIP
    >>> pp.gpuview(data)  # doctest: +SKIP
    
    # With overlays
    >>> import numpy as np  # doctest: +SKIP
    >>> smoothed = np.convolve(data['left_pupil'], np.ones(100)/100, 'same')  # doctest: +SKIP
    >>> pp.gpuview(data, overlay_pupil={'smoothed': smoothed})  # doctest: +SKIP
    """
    import sys
    import locale
    import vispy
    
    # Save LC_TIME locale before Qt initialization (Qt may change it)
    # This prevents issues with date parsing in EDF files after viewer is opened
    try:
        saved_lc_time = locale.getlocale(locale.LC_TIME)
    except Exception:
        saved_lc_time = None
    
    # Configure vispy to use an available Qt backend
    # Detect which Qt is already imported and use that, or try in order
    if 'PyQt6' in sys.modules or 'PyQt6.QtCore' in sys.modules:
        vispy.use(app='pyqt6')
    elif 'PyQt5' in sys.modules or 'PyQt5.QtCore' in sys.modules:
        vispy.use(app='pyqt5')
    elif 'PySide6' in sys.modules:
        vispy.use(app='pyside6')
    elif 'PySide2' in sys.modules:
        vispy.use(app='pyside2')
    else:
        # Try PyQt6 first (more modern), then PyQt5
        for backend in ['pyqt6', 'pyqt5', 'pyside6', 'pyside2']:
            try:
                vispy.use(app=backend)
                break
            except RuntimeError:
                continue
    
    from vispy import app
    
    # Import here to avoid circular imports and defer vispy loading
    from .canvas import GPUViewerCanvas
    
    # Build overlays dict
    overlays = {}
    if overlay_pupil:
        overlays['pupil'] = overlay_pupil
    if overlay_x:
        overlays['x'] = overlay_x
    if overlay_y:
        overlays['y'] = overlay_y
    
    # Create the viewer
    canvas = GPUViewerCanvas(eyedata, overlays=overlays)
    
    # Show canvas
    canvas.show()
    
    # Ensure window is visible and focused
    if hasattr(canvas, 'native') and canvas.native is not None:
        canvas.native.raise_()
        canvas.native.activateWindow()
    
    # Force an initial draw
    canvas.update()
    canvas.app.process_events()
    
    # Run the Qt event loop directly (blocks until window closed)
    # vispy's app.run() doesn't block properly in Jupyter
    qt_app = canvas.native.parent() if hasattr(canvas.native, 'parent') else None
    if qt_app is None:
        # Get the QApplication instance
        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            from PyQt5.QtWidgets import QApplication
        qt_app = QApplication.instance()
    
    if qt_app is not None:
        qt_app.exec()  # PyQt6
    else:
        app.run()  # Fallback
    
    # Restore LC_TIME locale after Qt event loop ends
    # Qt can change the locale (e.g., to system locale like nb_NO), which breaks
    # date parsing in eyelinkio that expects English month/day abbreviations
    if saved_lc_time is not None:
        try:
            locale.setlocale(locale.LC_TIME, saved_lc_time)
        except Exception:
            # If restoring fails, force C locale for date parsing
            try:
                locale.setlocale(locale.LC_TIME, 'C')
            except Exception:
                pass  # If all else fails, continue with whatever Qt set

