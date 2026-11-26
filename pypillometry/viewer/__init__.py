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

from typing import Dict, Optional, List, TYPE_CHECKING
import numpy as np
from types import SimpleNamespace

from . import _qt

if TYPE_CHECKING:
    from ..intervals import Intervals

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


def view(data, variables=None, eyes=None, time=None, overlay=None,
         overlay_pupil=None, overlay_x=None, overlay_y=None,
         extra_plots=None,
         highlight=None, highlight_color='lightblue') -> Optional[Dict[str, 'Intervals']]:
    """View eye-tracking data or numpy arrays with GPU-accelerated rendering.
    
    Opens an interactive viewer window using VisPy for fast GPU-based
    rendering. Suitable for long recordings (60+ minutes at 1000 Hz).

    .. image:: /images/viewer_eyedata.png
       :alt: Viewer showing pupil size and gaze of both eyes

    See :doc:`/user_guide/viewer` for more examples.

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
    eyes : list of str, optional
        For EyeData objects only. Filter which eyes to display.
        Example: ``eyes=['left']`` shows only left eye data.
        If not provided, all available eyes are detected and displayed.
    time : ndarray, optional
        Time vector for raw array viewing. If not provided, uses sample
        index (0, 1, 2, ...). Should have same length as data arrays.
        Ignored for EyeData objects, where `data.tx` (converted to seconds)
        is used.
    overlay : EyeData or list of EyeData, optional
        (EyeData only) other versions of the data to overlay (e.g., with
        the blinks interpolated in another way): their pupil and gaze are
        drawn on the plots of the same variable, one line per eye, labelled
        with their name (``data.name``) and the eye. They must have the
        same number of samples as `data`.
    overlay_pupil : dict, optional
        (EyeData only) Additional timeseries to overlay on the pupil plot.
        Keys are labels for legend, values are either:

        - str: name of data in eyedata.data (e.g., 'left_pupil_filtered')
        - array-like: timeseries of same length as eyedata

        Overlays with an unknown key or a wrong length are skipped with a
        warning.
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

        - An :class:`~pypillometry.Intervals` object (applied to all plots)
        - A dict mapping plot key to :class:`~pypillometry.Intervals`
          objects; keys are 'pupil', 'x', 'y' and ``'extra_<label>'`` for
          `extra_plots` (EyeData), or the plot labels (arrays). Keys like
          ``'left_pupil'`` (e.g. from :meth:`~pypillometry.EyeData.get_blinks`)
          go to the plot of the variable; the intervals of both eyes are
          shown together.

        Intervals in time units are converted to seconds; Intervals in
        sample indices (``units="samples"``) are placed using the time vector.
        For raw arrays without `time`, the x-axis is in sample indices and
        only Intervals in sample indices can be shown.
    highlight_color : str, optional
        Color for highlighted regions (default: 'lightblue')
    
    Returns
    -------
    dict of Intervals or None
        If regions were selected (``S`` key, then drag with the left mouse
        button), returns a dict mapping plot key (same keys as for
        `highlight`) to :class:`~pypillometry.Intervals` objects containing
        the selected regions with ``units='sec'``; for raw arrays without
        `time` the values are sample indices (``units="samples"``). Returns None
        if no selections were made.
    
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
    - Up/Down arrows: Zoom in/out (20%/25%)
    - PgUp/PgDn: Pan 50% of view  
    - Home/End: Jump to start/end
    - Space: Reset to full view
    - +/-: Zoom in/out
    - Mouse wheel: Zoom X-axis around the mouse position
    - Shift + Up/Down, Shift + mouse wheel: Zoom Y-axis of the plot under
      the mouse
    - Shift + Space: Reset Y-axis of the plot under the mouse
    - M: Toggle mask regions (EyeData only)
    - O: Toggle event markers (EyeData only)
    - S: Enter/exit selection mode (drag to mark a region)
    - D/Backspace: Remove last selection
    - H/?: Show help
    - Q/Esc: Close viewer (Esc first leaves selection mode)
    
    Examples
    --------
    View EyeData:
    
    >>> import pypillometry as pp
    >>> data = pp.EyeData.from_eyelink('recording.edf')  # doctest: +SKIP
    >>> pp.view(data)  # doctest: +SKIP
    >>> pp.view(data, variables=['pupil'])  # Only show pupil plots  # doctest: +SKIP
    >>> pp.view(data, eyes=['left'])  # Only show left eye data  # doctest: +SKIP
    
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
    
    EyeData with overlays, e.g. two interpolations of the blinks:

    >>> mahot = data.pupil.blinks_interpolate(inplace=False)  # doctest: +SKIP
    >>> kalman = data.pupil.blinks_interpolate(method="kalman", inplace=False)  # doctest: +SKIP
    >>> pp.view(mahot, overlay=kalman, variables="pupil")  # doctest: +SKIP
    
    >>> smoothed = np.convolve(data['left_pupil'], np.ones(100)/100, 'same')  # doctest: +SKIP
    >>> pp.view(data, overlay_pupil={'smoothed': smoothed})  # doctest: +SKIP
    """
    import sys
    import locale
    import vispy
    
    # Normalize input to common format
    mode, plot_spec, time_seconds = _normalize_input(data, time)
    for name, value in (("overlay_pupil", overlay_pupil), ("overlay_x", overlay_x),
                        ("overlay_y", overlay_y)):
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"{name} must be a dict {{label: array or channel name}}, "
                            f"got {type(value).__name__}; use overlay= for EyeData")
    if overlay is not None and mode != 'eyedata':
        raise TypeError("overlay= can only be used when viewing an EyeData object")
    from_overlay = (_overlays_from_eyedata(overlay, len(plot_spec.tx), eyes)
                    if overlay is not None else {})
    
    # Save LC_TIME locale before Qt initialization (Qt may change it)
    # This prevents issues with date parsing in EDF files after viewer is opened
    try:
        saved_lc_time = locale.getlocale(locale.LC_TIME)
    except Exception:
        saved_lc_time = None
    
    # Configure vispy to use an available Qt backend
    # Detect which Qt is already imported and use that, or try in order
    _qt.use_vispy()
    
    from vispy import app
    
    # Import here to avoid circular imports and defer vispy loading
    from .canvas import ViewerCanvas
    
    # Build overlays dict (only used for EyeData mode)
    overlays = {}
    for var, given in (('pupil', overlay_pupil), ('x', overlay_x), ('y', overlay_y)):
        merged = {**from_overlay.get(var, {}), **(given or {})}
        if merged:
            overlays[var] = merged
    
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
        if eyes:
            # unknown eyes raise a KeyError (with the available channels)
            eyes = list(dict.fromkeys(e for e, _ in plot_spec.data.select(eyes)))
        canvas = ViewerCanvas(
            plot_spec,  # EyeData object
            mode='eyedata',
            time_seconds=time_seconds,
            variables=variables,
            eyes=eyes,
            overlays=overlays,
            extra_plots=normalized_extra_plots,
            highlight=highlight, 
            highlight_color=highlight_color,
            sample_index=False
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
            highlight_color=highlight_color,
            sample_index=(time is None)
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
        qt_app = _qt.qt("QtWidgets").QApplication.instance()
    
    if qt_app is not None:
        qt_app.exec()
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
    
    return _selections_to_intervals(selections, sample_index=(mode == 'arrays' and time is None))


def _intervals_to_axis(intervals, time_seconds, sample_index=False):
    """Convert Intervals to a list of (start, end) in x-axis units.

    The x-axis is in seconds, or in sample indices if `sample_index` is True
    (raw arrays without `time`). Index Intervals (``units="samples"``) are placed
    using `time_seconds` (end indices are exclusive). `intervals` is not
    modified.
    """
    if sample_index:
        if intervals.units != "samples":
            raise ValueError("Intervals in time units cannot be placed on an axis of "
                             "sample indices (no time vector); give them in samples")
        return list(intervals.intervals)
    # the time vector as data object for the conversion from sample indices
    t = np.asarray(time_seconds, dtype=float)
    fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 1.0
    base = SimpleNamespace(tx=t * 1000., fs=fs)
    return [(float(s), float(e)) for s, e in intervals.to_units('sec', base).intervals]


def _tweak_fields(method, pars):
    """the numeric fields of :func:`tweak` for the parameters `pars` of
    ``blinks_interpolate(method=...)`` (the margin as two fields), and the
    other parameters, which are passed unchanged"""
    from ..eyedata.pupil import _INTERP_DEFAULTS
    if method not in _INTERP_DEFAULTS:
        raise ValueError(f"unknown method {method!r}; use one of "
                         f"{', '.join(map(repr, _INTERP_DEFAULTS))}")
    allowed = set(_INTERP_DEFAULTS[method]) | {"blinkwindow"}
    unknown = set(pars) - allowed
    if unknown:
        raise ValueError(f"unknown parameter(s) of method={method!r}: "
                         f"{', '.join(sorted(unknown))}")
    full = dict(_INTERP_DEFAULTS[method], blinkwindow=500)
    full.update(pars)
    fields, fixed = {}, {}
    for name, value in full.items():
        if name == "margin":
            if value is None:
                raise ValueError("margin=None (exactly the stored blinks) cannot be tweaked")
            before, after = (value, value) if np.ndim(value) == 0 else value
            fields["margin_before"], fields["margin_after"] = float(before), float(after)
        elif isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            if isinstance(value, str) and name.startswith("vel_"):
                raise TypeError(f"{name}={value!r}: relative velocity thresholds cannot be "
                                "tweaked; give them in units per ms")
            fixed[name] = value
        else:
            fields[name] = float(value)
    return fields, fixed


def _tweak_interval(data, eye, duration, rng):
    """`(start, end)` (ms of the time vector) of `duration` ms around a
    random blink of `eye` that is not in an excluded period"""
    bl = data.get_blinks(eye, "pupil", units="samples")[f"{eye}_pupil"]
    excl = data._excluded_mask()
    cand = [(a, b) for a, b in bl if not excl[a:b].any()]
    if not cand:
        raise ValueError(f"no detected blinks of {eye}_pupil (outside excluded periods): "
                         "detect the blinks first (blinks_detect())")
    a, b = cand[rng.integers(len(cand))]
    tx = data.tx
    mid = (tx[a] + tx[min(b, len(tx) - 1)]) / 2
    start = max(tx[0], min(mid - duration / 2, tx[-1] - duration))
    return start, min(tx[-1], start + duration)


def tweak_blink_interpolation(data, pars=None, method=None, eye=None, interval=None,
                              duration=30, units="sec", seed=None):
    """Tweak the parameters of the interpolation of blinks interactively.

    Opens :func:`tweak` with the pupil of one eye and its interpolation by
    :meth:`~pypillometry.eyedata.pupil.PupilNamespace.blinks_interpolate`
    overlaid; the parameters can be changed in the panel (the margin as
    ``margin_before`` and ``margin_after``). "Auto Update" is off, as the
    interpolation takes a moment: press "Update" (or Enter) to see the
    result. To keep it quick, only a short part of the recording is shown:
    by default `duration` around a random blink outside the excluded
    periods (another one with another `seed`), or the given `interval`.

    Parameters
    ----------
    data : EyeData
        data with detected blinks (see
        :meth:`~pypillometry.eyedata.pupil.PupilNamespace.blinks_detect`)
    pars : dict or BlinkTuneResult, optional
        starting parameters as arguments of ``blinks_interpolate()``, e.g.
        ``result.params`` of
        :meth:`~pypillometry.eyedata.pupil.PupilNamespace.blinks_optimize_parameters`
        (or the result itself); missing ones are the defaults
    method : str, optional
        ``"mahot"`` or ``"kalman"``; default: ``pars["method"]``, or
        ``"mahot"``. The fields of the panel are the parameters of this
        method.
    eye : str, optional
        eye to show; default: the first eye with pupil data
    interval : tuple, optional
        `(start, end)` of the part to show (values of the time vector, in
        `units`); default: `duration` around a random blink
    duration : float
        length of the part around a random blink (in `units`)
    units : str
        units of `interval` and `duration` (default seconds)
    seed : int, optional
        seed for the choice of the blink

    Returns
    -------
    dict
        the final parameters, as arguments of ``blinks_interpolate()``
        (including the `method`), e.g. for
        ``d.pupil.blinks_interpolate(**result)``

    Examples
    --------
    >>> d = d.pupil.blinks_detect()  # doctest: +SKIP
    >>> res = d.pupil.blinks_optimize_parameters()  # doctest: +SKIP
    >>> pars = pp.tweak_blink_interpolation(d, pars=res)  # doctest: +SKIP
    >>> d = d.pupil.blinks_interpolate(**pars)  # doctest: +SKIP
    """
    from ..units import parse_time
    if not (hasattr(data, "tx") and hasattr(data, "data")):
        raise TypeError("tweak_blink_interpolation() needs an EyeData object")
    pars = dict(getattr(pars, "params", pars) or {})
    method = method or pars.get("method", "mahot")
    pars.pop("method", None)
    fields, fixed = _tweak_fields(method, pars)
    if eye is None:
        eye = data._eyes_with([], "pupil")[0]
    if interval is None:
        dur = float(parse_time(duration, to="ms", default=units))
        start, end = _tweak_interval(data, eye, dur, np.random.default_rng(seed))
        part = data.crop(start, end, units="ms", inplace=False)
    else:
        part = data.crop(interval[0], interval[1], units=units, inplace=False)

    def interpolate(d, **fields):
        kw = dict(fixed)
        for name, value in fields.items():
            if name not in ("margin_before", "margin_after"):
                kw[name] = value
        kw["margin"] = (fields["margin_before"], fields["margin_after"])
        if "winsize" in kw:
            kw["winsize"] = max(1.0, kw["winsize"])
        return d.pupil.blinks_interpolate(eyes=eye, method=method, inplace=False, **kw)

    final = tweak(part, interpolate, fields, variable=f"{eye}_pupil", auto_update=False)
    out = dict(method=method, **fixed)
    for name, value in final.items():
        if name not in ("margin_before", "margin_after"):
            out[name] = value
    out["margin"] = (final["margin_before"], final["margin_after"])
    return out


def _overlays_from_eyedata(overlay, n, eyes=None):
    """Overlays (``{variable: {label: array}}``) from an EyeData object or a
    list of them: their pupil, x and y for each eye (of `eyes`, if given),
    labelled with the name of the object and the eye. Raises a TypeError
    for other objects and a ValueError if they do not have `n` samples."""
    objs = list(overlay) if isinstance(overlay, (list, tuple)) else [overlay]
    if isinstance(eyes, str):
        eyes = [eyes]
    out = {}
    for i, obj in enumerate(objs):
        if not (hasattr(obj, 'tx') and hasattr(obj, 'data')):
            raise TypeError(f"overlay must be EyeData or a list of EyeData, got "
                            f"{type(obj).__name__}")
        if len(obj.tx) != n:
            raise ValueError(f"overlay {obj.name!r} has {len(obj.tx)} samples, the data {n}")
        name = obj.name
        for key in obj.data.keys():
            eye, _, var = key.partition('_')
            if var not in ('pupil', 'x', 'y') or (eyes and eye not in eyes):
                continue
            label = f"{name} {eye}"
            if label in out.get(var, {}):        # the same name twice
                label = f"{name} ({i + 1}) {eye}"
            out.setdefault(var, {})[label] = np.asarray(obj.data[key], dtype=float)
    return out


def _highlight_by_panel(highlight, panels):
    """`highlight` (Intervals or dict) as a dict panel -> Intervals.

    Intervals go to all `panels`. Dict keys are panel names or
    ``"<eye>_<var>"`` (e.g. from :meth:`~pypillometry.EyeData.get_blinks`),
    which go to the panel `var`; the intervals of several eyes for the same
    panel are combined. Keys that match no panel are ignored with a warning.
    """
    from ..intervals import Intervals
    if highlight is None:
        return {}
    if isinstance(highlight, Intervals):
        return {p: highlight for p in panels}
    if not isinstance(highlight, dict):
        raise TypeError(f"highlight must be Intervals or a dict, got {type(highlight).__name__}")
    out, unknown = {}, []
    for key, iv in highlight.items():
        panel = key
        if key not in panels and "_" in key and key.split("_", 1)[1] in panels:
            panel = key.split("_", 1)[1]
        if panel not in panels:
            # variables that exist but are not shown are fine
            if key.split("_", 1)[-1] not in ("pupil", "x", "y"):
                unknown.append(key)
            continue
        out[panel] = out[panel] + iv if panel in out else iv
    if unknown:
        from loguru import logger
        logger.warning(f"highlight: no plot for {', '.join(map(repr, unknown))}; "
                       f"the plots are {', '.join(map(repr, panels))}")
    return out


def _selections_to_intervals(selections, sample_index=False):
    """Convert selections (dict of lists of (start, end)) to Intervals.

    The selections are in seconds, or in sample indices if `sample_index`
    is True (raw arrays without `time`); these get ``units="samples"``.
    """
    # Import Intervals here to avoid circular imports
    from ..intervals import Intervals
    units = 'samples' if sample_index else 'sec'
    return {var_type: Intervals(intervals_list, units=units, label=f'selected_{var_type}')
            for var_type, intervals_list in selections.items()}


def tweak(data, func, params, variable=None, time=None, auto_update=True):
    """Interactively tweak function parameters while viewing the result.
    
    Opens a viewer showing the original data with the function result overlaid.
    A parameter panel next to the plot allows adjusting numeric parameters in
    real-time, with the overlay updating to reflect the changes.

    .. image:: /images/tweak_intervals.png
       :alt: Tweak window with detected blinks and the parameter panel

    See :doc:`/user_guide/tweak` for more examples.

    Parameters
    ----------
    data : EyeData, ndarray, or list of ndarrays
        Original data to display and pass to the function. For a list of
        arrays, only the first array is displayed (the whole list is passed
        to `func`).
    func : callable
        Function called as ``func(data, **params)``. It receives `data`
        unchanged (for EyeData input the full EyeData object) and may return:

        - an EyeData object: channel `variable` (and its mask) is overlaid
        - an array of the same length as the displayed data: overlaid as a line
        - an :class:`~pypillometry.Intervals` object: shown as highlighted
          regions
        - a dict of arrays and/or :class:`~pypillometry.Intervals` objects:
          one overlay line or highlight per entry

    params : dict
        Initial parameter values. Only numeric types (int, float) are supported.
        These become the adjustable parameters in the GUI.
    variable : str, optional
        For EyeData objects, specifies which variable to display (e.g., 
        'left_pupil', 'right_pupil', 'left_x', 'right_y'). If not provided,
        uses the first available of 'left_pupil', 'right_pupil', 'left_x',
        'right_x', 'left_y', 'right_y'.
    time : ndarray, optional
        Time vector for array data. If not provided, uses sample indices.
        For EyeData, the time vector is taken from the object.
    auto_update : bool
        whether "Auto Update" is on at the start (default); turn it off for
        slow functions
    
    Returns
    -------
    dict
        Final parameter values after user adjustments. Returns the parameter
        dict as it was when the viewer was closed.

    Raises
    ------
    TypeError
        If a value in `params` is not an int or float, or if `func` returns
        something that cannot be shown (e.g., None).
    ValueError
        If `variable` is not found in the EyeData object, or if `func`
        returns an array with a different length than the displayed data.
    RuntimeError
        If `func` fails with the initial parameters (failures with later
        parameter values only print a warning).
    
    Examples
    --------
    With EyeData - function receives full EyeData object:
    
    >>> import pypillometry as pp
    >>> 
    >>> eyedata = pp.get_example_data("rlmw_002_short")  # doctest: +SKIP
    >>> 
    >>> def smooth(d, lp=5):
    ...     return d.pupil.lowpass_filter(cutoff=lp, inplace=False)
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
    
    - Left/Right arrows, PgUp/PgDn: Pan view (10%/50%)
    - Up/Down arrows, +/-, mouse wheel: Zoom X-axis in/out
    - Home/End: Jump to start/end
    - Space: Reset to full view
    - Shift + Up/Down, Shift + mouse wheel: Zoom Y-axis
    - Shift + Space: Reset Y-axis
    - M: Toggle mask regions
    - I: Toggle interval highlights
    - H/?: Show help
    - Q/Esc: Close viewer
    
    With "Auto Update" checked (default), changes to parameters are applied
    immediately and the overlay updates in real-time; otherwise use the
    "Update" button or press Enter in a parameter field. "Reset to Initial"
    restores the initial parameter values.
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
    original_mask = None
    
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
        
        # Extract mask for original data
        try:
            original_mask = eyedata.data.mask.get(display_variable)
            if original_mask is not None:
                original_mask = np.asarray(original_mask, dtype=bool)
        except (AttributeError, KeyError):
            original_mask = None
        
        title = f'Tweak - {getattr(eyedata, "name", "Unknown")} [{display_variable}]'
        sampling_rate = eyedata.fs
    
    elif isinstance(data, list):
        # List of arrays - use the first one
        data_array = np.asarray(data[0], dtype=np.float32)
        time_vec = time if time is not None else np.arange(len(data_array), dtype=np.float32)
        title = 'Tweak Viewer'
        sampling_rate = None
        display_variable = None
    else:
        # Single array
        data_array = np.asarray(data, dtype=np.float32)
        time_vec = time if time is not None else np.arange(len(data_array), dtype=np.float32)
        title = 'Tweak Viewer'
        sampling_rate = None
        display_variable = None
    
    # Import Intervals for type checking
    from ..intervals import Intervals
    
    sample_index = time is None and display_variable is None

    def _intervals_to_seconds(intervals, default_fs):
        """Convert Intervals object to list of (start, end) tuples in x-axis units."""
        try:
            return _intervals_to_axis(intervals, time_vec, sample_index)
        except ValueError:
            return list(intervals.intervals)  # Keep original if conversion fails
    
    def extract_result(result):
        """Extract display arrays, mask, and intervals from function result.
        
        Returns (arrays_dict, mask, intervals_dict) where:
        - arrays_dict: dict mapping names to ndarrays (for line plots)
        - mask: boolean mask array or None
        - intervals_dict: dict mapping names to list of (start,end) tuples in seconds
        """
        arrays_dict = {}
        intervals_dict = {}
        mask = None
        
        if hasattr(result, 'data') and hasattr(result, 'tx'):
            # Result is EyeData - extract the same variable
            try:
                arr = np.asarray(result[display_variable], dtype=np.float32)
                arrays_dict['tweaked'] = arr
            except (KeyError, AttributeError):
                raise ValueError(
                    f"Function result does not contain variable '{display_variable}'"
                )
            # Extract mask
            try:
                mask = result.data.mask.get(display_variable)
                if mask is not None:
                    mask = np.asarray(mask, dtype=bool)
            except (AttributeError, KeyError):
                pass
        
        elif isinstance(result, Intervals):
            # Single Intervals object
            intervals_dict['result'] = _intervals_to_seconds(result, sampling_rate)
        
        elif isinstance(result, dict):
            # Dict - can contain arrays and/or Intervals
            for k, v in result.items():
                if isinstance(v, Intervals):
                    intervals_dict[k] = _intervals_to_seconds(v, sampling_rate)
                else:
                    arrays_dict[k] = np.asarray(v, dtype=np.float32)
        
        else:
            # Single array
            arrays_dict['tweaked'] = np.asarray(result, dtype=np.float32)

        for k, v in arrays_dict.items():
            if v.ndim != 1:
                hint = " (does the function return its result?)" if result is None else ""
                raise TypeError(
                    f"Function returned {type(result).__name__}{hint}; expected "
                    "EyeData, a 1-d array, Intervals or a dict of these"
                )
        return arrays_dict, mask, intervals_dict
    
    time_vec = np.asarray(time_vec, dtype=np.float32)

    # Compute initial overlay - pass full data to function
    current_params = dict(params)
    try:
        result = func(data, **current_params)
    except Exception as e:
        raise RuntimeError(f"Function failed with initial parameters: {e}") from e
    initial_arrays, overlay_mask, initial_intervals = extract_result(result)

    # Validate overlay array lengths
    for k, v in initial_arrays.items():
        if len(v) != len(data_array):
            raise ValueError(
                f"Function returned array '{k}' of length {len(v)}, "
                f"but input has length {len(data_array)}"
            )

    # Save LC_TIME locale before Qt initialization
    try:
        saved_lc_time = locale.getlocale(locale.LC_TIME)
    except Exception:
        saved_lc_time = None
    
    # Configure vispy backend
    _qt.use_vispy()
    
    from vispy import app
    from .tweak_canvas import TweakCanvas, TweakViewer
    
    # Create canvas with mask and intervals support
    canvas = TweakCanvas(
        time_seconds=time_vec,
        original_data=data_array,
        original_mask=original_mask,
        overlay_data=initial_arrays if initial_arrays else None,
        overlay_mask=overlay_mask,
        overlay_intervals=initial_intervals if initial_intervals else None,
        sampling_rate=sampling_rate,
        title=title
    )
    
    # Callback for parameter changes - pass full data to function
    def on_params_change(new_params):
        nonlocal current_params
        current_params = dict(new_params)
        try:
            result = func(data, **new_params)
            new_arrays, new_mask, new_intervals = extract_result(result)
            if new_arrays:
                canvas.update_overlay(new_arrays, new_mask)
            if new_intervals:
                canvas.update_intervals(new_intervals)
        except Exception as e:
            print(f"Warning: Function failed with parameters {new_params}: {e}")
    
    # Create viewer with embedded parameter panel
    func_name = getattr(func, '__name__', 'function')
    viewer = TweakViewer(
        canvas=canvas,
        params=params,
        on_change=on_params_change,
        title=f'Tweak: {func_name}',
        auto_update=auto_update,
    )
    
    # Show viewer
    viewer.show()
    
    canvas.update()
    canvas.app.process_events()
    
    # Run Qt event loop
    qt_app = _qt.qt("QtWidgets").QApplication.instance()
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
    
    return viewer.get_params()
