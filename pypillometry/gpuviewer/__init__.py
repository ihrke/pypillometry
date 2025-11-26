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


def gpuview(eyedata) -> None:
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
    
    Notes
    -----
    Keyboard controls:
    - Left/Right arrows: Pan 10% of view
    - Up/Down arrows: Zoom in/out 20%
    - PgUp/PgDn: Pan 50% of view  
    - Home/End: Jump to start/end
    - Space: Reset to full view
    - +/-: Zoom in/out
    - Q/Esc: Close viewer
    
    Examples
    --------
    >>> import pypillometry as pp
    >>> data = pp.EyeData.from_eyelink('recording.edf')  # doctest: +SKIP
    >>> pp.gpuview(data)  # doctest: +SKIP
    """
    # Configure vispy to use an available Qt backend
    import sys
    import vispy
    
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
    
    # Create the viewer
    canvas = GPUViewerCanvas(eyedata)
    
    # Show and ensure window is visible
    canvas.show()
    canvas.native.raise_()  # Bring to front
    canvas.native.activateWindow()  # Activate
    
    # Process initial events
    canvas.app.process_events()
    
    # Run the event loop (blocking)
    app.run()

