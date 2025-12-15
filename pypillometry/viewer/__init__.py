"""GPU-accelerated viewer for eye-tracking data using VisPy.

This module provides a fast, GPU-accelerated viewer for long eye-tracking
recordings using VisPy for OpenGL rendering.

Usage
-----
>>> import pypillometry as pp
>>> data = pp.EyeData.from_eyelink('data.edf')  # doctest: +SKIP
>>> pp.view(data)  # doctest: +SKIP

# View raw numpy arrays
>>> import numpy as np  # doctest: +SKIP
>>> x = np.random.randn(1000)  # doctest: +SKIP
>>> pp.view(x)  # Single array  # doctest: +SKIP
>>> pp.view([x, y])  # Multiple arrays in same plot  # doctest: +SKIP
>>> pp.view({"upper": [x, y], "lower": z})  # Multiple plots  # doctest: +SKIP

Features
--------
- GPU-accelerated rendering for smooth navigation of large datasets
- Keyboard-only navigation (no GUI controls needed)
- Masked data highlighted with colored background
- Event markers shown as vertical grey stripes with labels
- Separate plots for each variable type (pupil, x, y)
- Different colors for left (blue) and right (red) eye data
- Support for viewing raw numpy arrays and masked arrays
"""

from typing import Dict, Optional, Union, List
import numpy as np

__all__ = ['view']


def _validate_array_lengths(arrays: List[np.ndarray]) -> None:
    """Validate that all arrays have the same length.
    
    Parameters
    ----------
    arrays : list of ndarray
        Arrays to validate.
        
    Raises
    ------
    ValueError
        If arrays have different lengths.
    """
    if not arrays:
        return
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"All arrays must have the same length, got: {lengths}")


def _normalize_input(data, time=None):
    """Normalize input data to a common format.
    
    Parameters
    ----------
    data : various
        Input data: EyeData object, ndarray, list of ndarrays, or dict.
    time : ndarray, optional
        Time vector for raw arrays.
        
    Returns
    -------
    tuple
        (mode, plot_spec, time_vector) where:
        - mode is 'eyedata' or 'arrays'
        - plot_spec is EyeData object or dict of {label: [arrays]}
        - time_vector is the time axis in seconds
    """
    # Check for EyeData-like object (has tx and data attributes)
    if hasattr(data, 'tx') and hasattr(data, 'data'):
        return 'eyedata', data, data.tx.astype(np.float32) * 0.001
    
    # Single array
    if isinstance(data, (np.ndarray, np.ma.MaskedArray)):
        time_vec = time if time is not None else np.arange(len(data), dtype=np.float32)
        return 'arrays', {'Signal': [np.asarray(data)]}, time_vec.astype(np.float32)
    
    # List of arrays -> one plot with multiple lines
    if isinstance(data, list):
        # Validate all items are arrays
        arrays = [np.asarray(a) for a in data]
        _validate_array_lengths(arrays)
        time_vec = time if time is not None else np.arange(len(arrays[0]), dtype=np.float32)
        return 'arrays', {'Signal': arrays}, time_vec.astype(np.float32)
    
    # Dict of arrays -> multiple plots
    if isinstance(data, dict):
        plots = {}
        all_arrays = []
        for label, arr_or_list in data.items():
            if isinstance(arr_or_list, list):
                arrays = [np.asarray(a) for a in arr_or_list]
                plots[label] = arrays
                all_arrays.extend(arrays)
            else:
                arr = np.asarray(arr_or_list)
                plots[label] = [arr]
                all_arrays.append(arr)
        
        _validate_array_lengths(all_arrays)
        first_arr = all_arrays[0]
        time_vec = time if time is not None else np.arange(len(first_arr), dtype=np.float32)
        return 'arrays', plots, time_vec.astype(np.float32)
    
    raise TypeError(
        f"Unsupported data type: {type(data)}. "
        "Expected EyeData, ndarray, list of ndarrays, or dict of ndarrays."
    )


def view(data, variables=None, time=None,
         overlay_pupil=None, overlay_x=None, overlay_y=None,
         highlight=None, highlight_color='lightblue') -> Optional[Dict[str, 'Intervals']]:
    """View eye-tracking data or numpy arrays with GPU-accelerated rendering.
    
    Opens an interactive viewer window using VisPy for fast GPU-based
    rendering. Suitable for long recordings (60+ minutes at 1000 Hz).
    
    Parameters
    ----------
    data : EyeData, ndarray, list of ndarrays, or dict
        Data to view. Supports multiple input types:
        
        - **EyeData object**: Eye-tracking data with `tx` attribute and 
          dictionary-like access to modalities (left_pupil, right_x, etc.)
        - **ndarray or MaskedArray**: Single array displayed as one plot
        - **list of arrays**: Multiple arrays overlaid in one plot with 
          different colors
        - **dict of arrays**: Keys become plot labels (y-axis), values are 
          arrays or lists of arrays. Example: ``{"upper": [x, y], "lower": z}``
          creates two subplots with x,y overlaid in "upper" and z in "lower"
    variables : list of str, optional
        For EyeData objects only. Filter which modalities to display.
        Valid values: 'pupil', 'x', 'y'. Default: show all available.
        Example: ``variables=['pupil']`` shows only pupil data.
    time : ndarray, optional
        Time vector for raw array viewing. If not provided, uses sample
        index (0, 1, 2, ...). Should have same length as data arrays.
    overlay_pupil : dict, optional
        (EyeData only) Additional timeseries to overlay on the pupil plot.
        Keys are labels for legend, values are either:
        - str: name of data in eyedata.data (e.g., 'left_pupil_filtered')
        - array-like: timeseries of same length as eyedata
    overlay_x : dict, optional
        (EyeData only) Additional timeseries to overlay on the gaze X plot.
    overlay_y : dict, optional
        (EyeData only) Additional timeseries to overlay on the gaze Y plot.
    highlight : Intervals or dict, optional
        Intervals to highlight in the plots. Can be:
        - An Intervals object (applied to all plots)
        - A dict mapping variable type ('pupil', 'x', 'y') to Intervals objects
    highlight_color : str, optional
        Color for highlighted regions (default: 'lightblue')
    
    Returns
    -------
    Optional[Dict[str, Intervals]]
        If regions were selected (using 's' key + mouse clicks), returns a dict
        mapping plot label to Intervals objects containing the selected regions
        in seconds. Returns None if no selections were made.
    
    Raises
    ------
    ValueError
        If arrays have different lengths.
    TypeError
        If data type is not supported.
    
    Notes
    -----
    Keyboard controls:
    
    - Left/Right arrows: Pan 10% of view
    - Up/Down arrows: Zoom in/out 20%
    - PgUp/PgDn: Pan 50% of view  
    - Home/End: Jump to start/end
    - Space: Reset to full view
    - +/-: Zoom in/out
    - M: Toggle mask regions (EyeData only)
    - O: Toggle event markers (EyeData only)
    - S: Enter selection mode (drag to mark a region)
    - D/Backspace: Remove last selection
    - H/?: Show help
    - Q/Esc: Close viewer
    
    Examples
    --------
    View EyeData:
    
    >>> import pypillometry as pp
    >>> data = pp.EyeData.from_eyelink('recording.edf')  # doctest: +SKIP
    >>> pp.view(data)  # doctest: +SKIP
    >>> pp.view(data, variables=['pupil'])  # Only show pupil plots  # doctest: +SKIP
    
    View numpy arrays:
    
    >>> import numpy as np  # doctest: +SKIP
    >>> x = np.random.randn(1000)  # doctest: +SKIP
    >>> y = np.random.randn(1000)  # doctest: +SKIP
    >>> pp.view(x)  # Single array  # doctest: +SKIP
    >>> pp.view([x, y])  # Multiple arrays in one plot  # doctest: +SKIP
    >>> pp.view({"Signal 1": x, "Signal 2": y})  # Two separate plots  # doctest: +SKIP
    >>> pp.view({"upper": [x, y], "lower": x + y})  # Complex layout  # doctest: +SKIP
    
    With time vector:
    
    >>> t = np.linspace(0, 10, 1000)  # 10 seconds  # doctest: +SKIP
    >>> pp.view(x, time=t)  # doctest: +SKIP
    
    EyeData with overlays:
    
    >>> smoothed = np.convolve(data['left_pupil'], np.ones(100)/100, 'same')  # doctest: +SKIP
    >>> pp.view(data, overlay_pupil={'smoothed': smoothed})  # doctest: +SKIP
    """
    import sys
    import locale
    import vispy
    
    # Normalize input to common format
    mode, plot_spec, time_seconds = _normalize_input(data, time)
    
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
    from .canvas import ViewerCanvas
    
    # Build overlays dict (only used for EyeData mode)
    overlays = {}
    if overlay_pupil:
        overlays['pupil'] = overlay_pupil
    if overlay_x:
        overlays['x'] = overlay_x
    if overlay_y:
        overlays['y'] = overlay_y
    
    # Create the viewer based on mode
    if mode == 'eyedata':
        canvas = ViewerCanvas(
            plot_spec,  # EyeData object
            mode='eyedata',
            time_seconds=time_seconds,
            variables=variables,
            overlays=overlays, 
            highlight=highlight, 
            highlight_color=highlight_color
        )
    else:
        # Array mode
        canvas = ViewerCanvas(
            plot_spec,  # dict of {label: [arrays]}
            mode='arrays',
            time_seconds=time_seconds,
            variables=None,
            overlays=None, 
            highlight=highlight, 
            highlight_color=highlight_color
        )
    
    # For array mode, set the view range before showing to avoid
    # singular matrix errors during initial resize events
    if mode == 'arrays':
        # Process pending events to let canvas initialize
        canvas.app.process_events()
        # Set view range now
        try:
            canvas._set_view_range(canvas.data_min, canvas.data_max)
        except Exception:
            pass
    
    # Show canvas
    canvas.show()
    
    # Ensure window is visible and focused
    if hasattr(canvas, 'native') and canvas.native is not None:
        canvas.native.raise_()
        canvas.native.activateWindow()
    
    # Force an initial draw
    canvas.update()
    canvas.app.process_events()
    
    # Re-set the view range now that canvas is sized properly
    if mode == 'arrays':
        try:
            canvas._set_view_range(canvas.data_min, canvas.data_max)
            canvas.update()
        except Exception:
            pass
    
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
    
    # Get selections from canvas and convert to Intervals
    selections = canvas.get_selections()
    
    if not selections:
        return None
    
    # Import Intervals here to avoid circular imports
    from ..intervals import Intervals
    
    result = {}
    for var_type, intervals_list in selections.items():
        result[var_type] = Intervals(intervals_list, units='sec', label=f'selected_{var_type}')
    
    return result
