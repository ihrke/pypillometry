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

__all__ = ['view', 'tweak']


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
    
    # Helper to convert to array while preserving MaskedArrays
    def _to_array(a):
        if isinstance(a, np.ma.MaskedArray):
            return a
        return np.asarray(a)
    
    # Single array
    if isinstance(data, (np.ndarray, np.ma.MaskedArray)):
        time_vec = time if time is not None else np.arange(len(data), dtype=np.float32)
        return 'arrays', {'Signal': [_to_array(data)]}, time_vec.astype(np.float32)
    
    # List of arrays -> one plot with multiple lines
    if isinstance(data, list):
        # Validate all items are arrays (preserve MaskedArrays)
        arrays = [_to_array(a) for a in data]
        _validate_array_lengths(arrays)
        time_vec = time if time is not None else np.arange(len(arrays[0]), dtype=np.float32)
        return 'arrays', {'Signal': arrays}, time_vec.astype(np.float32)
    
    # Dict of arrays -> multiple plots
    if isinstance(data, dict):
        plots = {}
        all_arrays = []
        for label, arr_or_list in data.items():
            if isinstance(arr_or_list, list):
                arrays = [_to_array(a) for a in arr_or_list]
                plots[label] = arrays
                all_arrays.extend(arrays)
            else:
                arr = _to_array(arr_or_list)
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
         extra_plots=None,
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
    extra_plots : dict, optional
        (EyeData only) Additional numpy arrays to display as separate subplots
        below the EyeData variables. Keys become subplot labels (y-axis), 
        values are arrays or lists of arrays. Arrays must have the same length
        as the EyeData. Example: ``extra_plots={"velocity": vel, "custom": [a, b]}``
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
    
    # Normalize and validate extra_plots (only for EyeData mode)
    normalized_extra_plots = None
    if extra_plots is not None and mode == 'eyedata':
        normalized_extra_plots = {}
        eyedata_len = len(plot_spec.tx)
        for label, arr_or_list in extra_plots.items():
            if isinstance(arr_or_list, list):
                # Preserve MaskedArrays, convert others to ndarray
                arrays = [
                    a if isinstance(a, np.ma.MaskedArray) else np.asarray(a)
                    for a in arr_or_list
                ]
            else:
                # Preserve MaskedArrays, convert others to ndarray
                if isinstance(arr_or_list, np.ma.MaskedArray):
                    arrays = [arr_or_list]
                else:
                    arrays = [np.asarray(arr_or_list)]
            # Validate lengths match EyeData
            for arr in arrays:
                if len(arr) != eyedata_len:
                    raise ValueError(
                        f"extra_plots['{label}'] has length {len(arr)}, "
                        f"but EyeData has length {eyedata_len}"
                    )
            normalized_extra_plots[label] = arrays
    
    # Create the viewer based on mode
    if mode == 'eyedata':
        canvas = ViewerCanvas(
            plot_spec,  # EyeData object
            mode='eyedata',
            time_seconds=time_seconds,
            variables=variables,
            overlays=overlays,
            extra_plots=normalized_extra_plots,
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


def tweak(data, func, params, variable=None, time=None):
    """Interactively tweak function parameters while viewing the result.
    
    Opens a viewer showing the original data with the function result overlaid.
    A separate parameter window allows adjusting numeric parameters in real-time,
    with the overlay updating to reflect the changes.
    
    Parameters
    ----------
    data : EyeData, ndarray, or list of ndarrays
        Original data to display and pass to the function.
    func : callable
        Function that takes (data, **params) and returns transformed data
        of the same type. For EyeData input, function receives the full EyeData
        object and should return an EyeData object. For arrays, receives and
        returns arrays.
    params : dict
        Initial parameter values. Only numeric types (int, float) are supported.
        These become the adjustable parameters in the GUI.
    variable : str, optional
        For EyeData objects, specifies which variable to display (e.g., 
        'left_pupil', 'right_pupil', 'left_x', 'right_y'). If not provided,
        uses the first available pupil channel.
    time : ndarray, optional
        Time vector for array data. If not provided, uses sample indices.
        For EyeData, the time vector is taken from the object.
    
    Returns
    -------
    dict
        Final parameter values after user adjustments. Returns the parameter
        dict as it was when the viewer was closed.
    
    Examples
    --------
    With EyeData - function receives full EyeData object:
    
    >>> import pypillometry as pp
    >>> 
    >>> eyedata = pp.get_example_data("rlmw_002_short")  # doctest: +SKIP
    >>> 
    >>> def smooth(d, lp=5):
    ...     return d.pupil_lowpass_filter(cutoff=lp, inplace=False)
    >>> 
    >>> # Tweak lowpass filter cutoff, display left_pupil
    >>> params = pp.tweak(eyedata, smooth, {'lp': 5}, variable='left_pupil')  # doctest: +SKIP
    
    Basic usage with numpy arrays:
    
    >>> import numpy as np
    >>> 
    >>> # Create sample data
    >>> data = np.sin(np.linspace(0, 10, 1000)) + np.random.randn(1000) * 0.1
    >>> 
    >>> # Define a smoothing function
    >>> def smooth(x, window_size=10):
    ...     kernel = np.ones(window_size) / window_size
    ...     return np.convolve(x, kernel, mode='same')
    >>> 
    >>> # Tweak the window_size parameter
    >>> final_params = pp.tweak(data, smooth, {'window_size': 10})  # doctest: +SKIP
    
    Notes
    -----
    Keyboard controls in the viewer:
    
    - Left/Right arrows: Pan view
    - Up/Down arrows: Zoom in/out
    - Space: Reset to full view
    - Q/Esc: Close viewer
    
    The parameter window stays on top and can be repositioned. Changes to
    parameters are applied immediately and the overlay updates in real-time.
    """
    import sys
    import locale
    import vispy
    
    # Validate params - only numeric types allowed
    for name, value in params.items():
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Parameter '{name}' has type {type(value).__name__}, "
                "but only int and float are supported."
            )
    
    # Determine if we have EyeData or raw arrays
    is_eyedata = hasattr(data, 'tx') and hasattr(data, 'data')
    
    if is_eyedata:
        # EyeData object
        eyedata = data
        time_vec = eyedata.tx.astype(np.float32) * 0.001  # Convert ms to seconds
        
        # Determine which variable to display
        if variable is not None:
            # User specified a variable
            try:
                data_array = np.asarray(eyedata[variable], dtype=np.float32)
            except (KeyError, AttributeError):
                raise ValueError(f"Variable '{variable}' not found in EyeData object")
            display_variable = variable
        else:
            # Find first available data channel
            data_array = None
            display_variable = None
            for key in ['left_pupil', 'right_pupil', 'left_x', 'right_x', 'left_y', 'right_y']:
                try:
                    arr = eyedata[key]
                    if arr is not None and len(arr) > 0:
                        data_array = np.asarray(arr, dtype=np.float32)
                        display_variable = key
                        break
                except (KeyError, AttributeError):
                    continue
            
            if data_array is None:
                raise ValueError("No valid data channel found in EyeData object")
        
        title = f'Tweak - {getattr(eyedata, "name", "Unknown")} [{display_variable}]'
        
        # Helper to extract display array from function result
        def extract_display_array(result):
            if hasattr(result, 'data') and hasattr(result, 'tx'):
                # Result is EyeData - extract the same variable
                try:
                    return np.asarray(result[display_variable], dtype=np.float32)
                except (KeyError, AttributeError):
                    raise ValueError(
                        f"Function result does not contain variable '{display_variable}'"
                    )
            else:
                # Result is array
                return np.asarray(result, dtype=np.float32)
    
    elif isinstance(data, list):
        # List of arrays - use the first one
        data_array = np.asarray(data[0], dtype=np.float32)
        time_vec = time if time is not None else np.arange(len(data_array), dtype=np.float32)
        title = 'Tweak Viewer'
        extract_display_array = lambda result: np.asarray(result, dtype=np.float32)
    else:
        # Single array
        data_array = np.asarray(data, dtype=np.float32)
        time_vec = time if time is not None else np.arange(len(data_array), dtype=np.float32)
        title = 'Tweak Viewer'
        extract_display_array = lambda result: np.asarray(result, dtype=np.float32)
    
    time_vec = np.asarray(time_vec, dtype=np.float32)
    
    # Save LC_TIME locale before Qt initialization
    try:
        saved_lc_time = locale.getlocale(locale.LC_TIME)
    except Exception:
        saved_lc_time = None
    
    # Configure vispy backend
    if 'PyQt6' in sys.modules or 'PyQt6.QtCore' in sys.modules:
        vispy.use(app='pyqt6')
    elif 'PyQt5' in sys.modules or 'PyQt5.QtCore' in sys.modules:
        vispy.use(app='pyqt5')
    elif 'PySide6' in sys.modules:
        vispy.use(app='pyside6')
    elif 'PySide2' in sys.modules:
        vispy.use(app='pyside2')
    else:
        for backend in ['pyqt6', 'pyqt5', 'pyside6', 'pyside2']:
            try:
                vispy.use(app=backend)
                break
            except RuntimeError:
                continue
    
    from vispy import app
    from .tweak import TweakCanvas, ParameterWindow
    
    # Compute initial overlay - pass full data to function
    current_params = dict(params)
    try:
        result = func(data, **current_params)
        initial_overlay = extract_display_array(result)
    except Exception as e:
        raise RuntimeError(f"Function failed with initial parameters: {e}")
    
    if len(initial_overlay) != len(data_array):
        raise ValueError(
            f"Function returned array of length {len(initial_overlay)}, "
            f"but input has length {len(data_array)}"
        )
    
    # Create canvas
    canvas = TweakCanvas(
        time_seconds=time_vec,
        original_data=data_array,
        overlay_data=initial_overlay,
        title=title
    )
    
    # Callback for parameter changes - pass full data to function
    def on_params_change(new_params):
        nonlocal current_params
        current_params = dict(new_params)
        try:
            result = func(data, **new_params)
            new_overlay = extract_display_array(result)
            canvas.update_overlay(new_overlay)
        except Exception as e:
            print(f"Warning: Function failed with parameters {new_params}: {e}")
    
    # Create parameter window
    func_name = getattr(func, '__name__', 'function')
    param_window = ParameterWindow(
        params=params,
        on_change=on_params_change,
        title=f'Parameters: {func_name}'
    )
    
    # Link windows so they close together
    canvas.set_param_window(param_window)
    param_window.set_canvas(canvas)
    
    # Show windows
    canvas.show()
    param_window.show()
    
    # Position parameter window to the right of the canvas
    if hasattr(canvas, 'native') and canvas.native is not None:
        canvas.native.raise_()
        canvas.native.activateWindow()
        # Position param window to the right of canvas
        canvas_geom = canvas.native.geometry()
        param_window.window.move(
            canvas_geom.x() + canvas_geom.width() + 10,
            canvas_geom.y()
        )
    
    canvas.update()
    canvas.app.process_events()
    
    # Run Qt event loop
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        from PyQt5.QtWidgets import QApplication
    
    qt_app = QApplication.instance()
    if qt_app is not None:
        qt_app.exec()
    else:
        app.run()
    
    # Restore locale
    if saved_lc_time is not None:
        try:
            locale.setlocale(locale.LC_TIME, saved_lc_time)
        except Exception:
            try:
                locale.setlocale(locale.LC_TIME, 'C')
            except Exception:
                pass
    
    return current_params
