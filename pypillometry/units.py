"""Unit handling for pypillometry using Pint.

This module provides flexible unit parsing for geometric and angle parameters,
accepting three formats:
1. Plain numbers (backward compatible, issues one-time warning)
2. String format: "30 cm", "600 mm", "20 degrees", "0.5 radians"
3. Pint Quantity: 600 * ureg.mm, 20 * ureg.degree

All values are internally converted to canonical units (mm for distances, 
radians for angles) and stored as plain floats.

Examples
--------
>>> from pypillometry.units import ureg, parse_distance, parse_angle
>>> 
>>> # Three equivalent ways to specify 60 cm:
>>> d1 = parse_distance(600)           # Plain number (warning, assumes mm)
>>> d2 = parse_distance("60 cm")       # String format (recommended)
>>> d3 = parse_distance(60 * ureg.cm)  # Pint Quantity (explicit)
>>> 
>>> # All return the same value in mm:
>>> assert d1 == d2 == d3 == 600.0
>>> 
>>> # Angles work similarly:
>>> a1 = parse_angle(0.349)                    # Plain (radians, warning)
>>> a2 = parse_angle("20 degrees")             # String (recommended)
>>> a3 = parse_angle(20 * ureg.degree)         # Pint Quantity
>>> 
>>> # All return radians:
>>> import numpy as np
>>> assert np.isclose(a1, 0.349) and np.isclose(a2, a3)
"""

import inspect
import warnings
import pint

# Create application-wide unit registry
ureg = pint.UnitRegistry()

# Session-wide tracking of issued warnings (one per parameter name)
_unit_warnings_issued = set()


def _get_param_name():
    """Extract parameter name from calling context using inspect.
    
    Attempts to determine the variable name being assigned to in the
    calling context, e.g., for `camera_eye_distance = parse_distance(value)`,
    this function returns "camera_eye_distance".
    
    Returns
    -------
    str
        Parameter name if successfully extracted, otherwise "parameter"
    """
    try:
        # Go back two frames: _get_param_name -> parse_* -> caller
        frame = inspect.currentframe().f_back.f_back
        code_context = inspect.getframeinfo(frame).code_context
        
        if code_context:
            line = code_context[0].strip()
            # Simple pattern matching: "varname = parse_..."
            if '=' in line and not '==' in line.split('=')[0]:
                varname = line.split('=')[0].strip()
                return varname
    except:
        pass
    
    return "parameter"


def parse_distance(value):
    """Parse distance value with flexible unit handling.
    
    Accepts three input formats and converts to canonical units (mm):
    1. Plain float/int: assumed to be mm, issues one-time warning
    2. String: parsed with units, e.g., "60 cm", "600 mm", "0.6 m"
    3. Pint Quantity: explicit units, e.g., 60 * ureg.cm
    
    Parameters
    ----------
    value : float, int, str, pint.Quantity, or None
        Distance value with optional units
    
    Returns
    -------
    float or None
        Distance in mm (canonical units), or None if input is None
    
    Raises
    ------
    ValueError
        If string format is invalid or units are incompatible with distance
    
    Examples
    --------
    >>> parse_distance(600)          # 600.0 (mm, with warning)
    >>> parse_distance("60 cm")      # 600.0 (mm)
    >>> parse_distance("0.6 m")      # 600.0 (mm)
    >>> parse_distance(60 * ureg.cm) # 600.0 (mm)
    """
    if value is None:
        return None
    
    # Pint Quantity
    if isinstance(value, ureg.Quantity):
        return value.to(ureg.mm).magnitude
    
    # String format
    if isinstance(value, str):
        try:
            q = ureg.Quantity(value)
            return q.to(ureg.mm).magnitude
        except Exception as e:
            param_name = _get_param_name()
            raise ValueError(
                f"Invalid unit string for {param_name}: '{value}'. "
                f"Expected format like '60 cm', '600 mm', or '0.6 m'"
            ) from e
    
    # Plain number - issue warning once per parameter
    param_name = _get_param_name()
    if param_name not in _unit_warnings_issued:
        warnings.warn(
            f"{param_name}: plain number provided, assuming mm. "
            f"Use string ('{value} mm') or pint.Quantity for explicit units.",
            UserWarning,
            stacklevel=2
        )
        _unit_warnings_issued.add(param_name)
    
    return float(value)


def parse_angle(value):
    """Parse angle value with flexible unit handling.
    
    Accepts three input formats and converts to canonical units (radians):
    1. Plain float/int: assumed to be radians, issues one-time warning
    2. String: parsed with units, e.g., "20 degrees", "0.349 radians"
    3. Pint Quantity: explicit units, e.g., 20 * ureg.degree
    
    Parameters
    ----------
    value : float, int, str, pint.Quantity, or None
        Angle value with optional units
    
    Returns
    -------
    float or None
        Angle in radians (canonical units), or None if input is None
    
    Raises
    ------
    ValueError
        If string format is invalid or units are incompatible with angle
    
    Examples
    --------
    >>> import numpy as np
    >>> parse_angle(0.349)                # 0.349 (radians, with warning)
    >>> parse_angle("20 degrees")         # 0.349... (radians)
    >>> parse_angle("0.349 radians")      # 0.349 (radians)
    >>> parse_angle(20 * ureg.degree)     # 0.349... (radians)
    """
    if value is None:
        return None
    
    # Pint Quantity
    if isinstance(value, ureg.Quantity):
        return value.to(ureg.radian).magnitude
    
    # String format
    if isinstance(value, str):
        try:
            q = ureg.Quantity(value)
            return q.to(ureg.radian).magnitude
        except Exception as e:
            param_name = _get_param_name()
            raise ValueError(
                f"Invalid unit string for {param_name}: '{value}'. "
                f"Expected format like '20 degrees' or '0.349 radians'"
            ) from e
    
    # Plain number - assume radians, issue warning once per parameter
    param_name = _get_param_name()
    if param_name not in _unit_warnings_issued:
        warnings.warn(
            f"{param_name}: plain number provided, assuming radians. "
            f"Use string ('{value} radians') or pint.Quantity for explicit units.",
            UserWarning,
            stacklevel=2
        )
        _unit_warnings_issued.add(param_name)
    
    return float(value)

