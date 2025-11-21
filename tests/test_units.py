"""Tests for unit handling functionality."""

import pytest
import numpy as np
import warnings
from pypillometry.units import ureg, parse_distance, parse_angle, _unit_warnings_issued


def test_parse_distance_plain_number():
    """Test parse_distance with plain number (assumes mm, issues warning)."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_distance(600)
        
        # Should return 600.0 (mm)
        assert result == 600.0
        
        # Should issue warning
        assert len(w) == 1
        assert "assuming mm" in str(w[0].message).lower()


def test_parse_distance_string_mm():
    """Test parse_distance with string format (mm)."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_distance("600 mm")
        
        # Should return 600.0
        assert result == 600.0
        
        # Should not issue warning
        assert len(w) == 0


def test_parse_distance_string_cm():
    """Test parse_distance with string format (cm), converts to mm."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_distance("60 cm")
        
        # Should return 600.0 mm
        assert result == 600.0
        
        # Should not issue warning
        assert len(w) == 0


def test_parse_distance_string_m():
    """Test parse_distance with string format (m), converts to mm."""
    _unit_warnings_issued.clear()
    
    result = parse_distance("0.6 m")
    
    # Should return 600.0 mm
    assert np.isclose(result, 600.0)


def test_parse_distance_pint_quantity():
    """Test parse_distance with Pint Quantity."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_distance(60 * ureg.cm)
        
        # Should return 600.0 mm
        assert result == 600.0
        
        # Should not issue warning
        assert len(w) == 0


def test_parse_distance_none():
    """Test parse_distance with None returns None."""
    result = parse_distance(None)
    assert result is None


def test_parse_distance_invalid_string():
    """Test parse_distance with invalid string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid unit string"):
        parse_distance("not a distance")


def test_parse_distance_warning_once_per_parameter():
    """Test that warning is issued only once per parameter name."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # First call should warn
        camera_eye_distance = parse_distance(600)
        assert len(w) == 1
        
        # Second call with same variable name should not warn
        camera_eye_distance = parse_distance(700)
        assert len(w) == 1  # Still only 1 warning


def test_parse_angle_plain_number():
    """Test parse_angle with plain number (assumes radians, issues warning)."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_angle(0.349)
        
        # Should return 0.349 (radians)
        assert np.isclose(result, 0.349)
        
        # Should issue warning
        assert len(w) == 1
        assert "assuming radians" in str(w[0].message).lower()


def test_parse_angle_string_degrees():
    """Test parse_angle with string format (degrees), converts to radians."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_angle("20 degrees")
        
        # Should return ~0.349 radians
        assert np.isclose(result, np.radians(20))
        
        # Should not issue warning
        assert len(w) == 0


def test_parse_angle_string_radians():
    """Test parse_angle with string format (radians)."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_angle("0.349 radians")
        
        # Should return 0.349
        assert np.isclose(result, 0.349)
        
        # Should not issue warning
        assert len(w) == 0


def test_parse_angle_pint_quantity():
    """Test parse_angle with Pint Quantity."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_angle(20 * ureg.degree)
        
        # Should return ~0.349 radians
        assert np.isclose(result, np.radians(20))
        
        # Should not issue warning
        assert len(w) == 0


def test_parse_angle_negative():
    """Test parse_angle with negative angle."""
    result = parse_angle("-90 degrees")
    assert np.isclose(result, np.radians(-90))


def test_parse_angle_none():
    """Test parse_angle with None returns None."""
    result = parse_angle(None)
    assert result is None


def test_parse_angle_invalid_string():
    """Test parse_angle with invalid string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid unit string"):
        parse_angle("not an angle")


def test_parse_angle_warning_once_per_parameter():
    """Test that warning is issued only once per parameter name."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # First call should warn
        theta = parse_angle(0.349)
        assert len(w) == 1
        
        # Second call with same variable name should not warn
        theta = parse_angle(0.5)
        assert len(w) == 1  # Still only 1 warning


def test_mixed_units_distance():
    """Test various distance unit conversions."""
    _unit_warnings_issued.clear()
    
    # All should convert to the same value in mm
    d1 = parse_distance("600 mm")
    d2 = parse_distance("60 cm")
    d3 = parse_distance("0.6 m")
    d4 = parse_distance(600 * ureg.mm)
    d5 = parse_distance(60 * ureg.cm)
    
    # All should be 600.0 mm
    assert all(np.isclose(d, 600.0) for d in [d1, d2, d3, d4, d5])


def test_mixed_units_angle():
    """Test various angle unit conversions."""
    _unit_warnings_issued.clear()
    
    # All should convert to the same value in radians
    a1 = parse_angle("20 degrees")
    a2 = parse_angle(f"{np.radians(20)} radians")
    a3 = parse_angle(20 * ureg.degree)
    a4 = parse_angle(np.radians(20) * ureg.radian)
    
    # All should be ~0.349 radians
    expected = np.radians(20)
    assert all(np.isclose(a, expected) for a in [a1, a2, a3, a4])


def test_ureg_available():
    """Test that ureg is properly exported and usable."""
    # Should be able to create quantities
    distance = 600 * ureg.mm
    angle = 20 * ureg.degree
    
    # Should be able to convert
    assert distance.to(ureg.cm).magnitude == 60.0
    assert np.isclose(angle.to(ureg.radian).magnitude, np.radians(20))


def test_parse_distance_tuple():
    """Test parsing a tuple of distances (e.g., for physical_screen_size)."""
    _unit_warnings_issued.clear()
    
    screen_size = ("52 cm", "29 cm")
    parsed = tuple(parse_distance(val) for val in screen_size)
    
    # Should return (520.0, 290.0) in mm
    assert parsed == (520.0, 290.0)


def test_distance_precision():
    """Test that distance conversions maintain precision."""
    # Test with a non-round number
    result = parse_distance("12.345 cm")
    assert np.isclose(result, 123.45)


def test_angle_precision():
    """Test that angle conversions maintain precision."""
    # Test with a non-round number
    result = parse_angle("12.345 degrees")
    assert np.isclose(result, np.radians(12.345))

