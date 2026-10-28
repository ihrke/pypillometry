"""Tests for unit handling functionality."""

import pytest
import numpy as np
import warnings
from pypillometry.units import u, parse_distance, parse_angle, _unit_warnings_issued


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


def test_parse_distance_unit_quantity():
    """Test parse_distance with Pint Quantity."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_distance(60 * u.cm)
        
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


def test_parse_angle_unit_quantity():
    """Test parse_angle with Pint Quantity."""
    _unit_warnings_issued.clear()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_angle(20 * u.degree)
        
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
    d4 = parse_distance(600 * u.mm)
    d5 = parse_distance(60 * u.cm)
    
    # All should be 600.0 mm
    assert all(np.isclose(d, 600.0) for d in [d1, d2, d3, d4, d5])


def test_mixed_units_angle():
    """Test various angle unit conversions."""
    _unit_warnings_issued.clear()
    
    # All should convert to the same value in radians
    a1 = parse_angle("20 degrees")
    a2 = parse_angle(f"{np.radians(20)} radians")
    a3 = parse_angle(20 * u.degree)
    a4 = parse_angle(np.radians(20) * u.radian)
    
    # All should be ~0.349 radians
    expected = np.radians(20)
    assert all(np.isclose(a, expected) for a in [a1, a2, a3, a4])


def test_unit_namespace_available():
    """Test that the unit namespace is exported and usable."""
    # Should be able to create quantities
    distance = 600 * u.mm
    angle = 20 * u.degree
    
    # Should be able to convert
    assert distance.to(u.cm).magnitude == 60.0
    assert np.isclose(angle.to(u.radian).magnitude, np.radians(20))


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



# --- unit namespace, Quantity and parse_time ---------------------------------

from pypillometry.units import Quantity, Unit, parse_time, normalize_unit


def test_quantity_from_tuple_and_list():
    q = (3, 5) * u.min
    assert isinstance(q, Quantity)
    assert q.unit is u.min
    np.testing.assert_array_equal(q.value, [3, 5])
    q2 = [(0, 100), (200, 300)] * u.ms
    assert q2.value.shape == (2, 2)


def test_quantity_from_numpy_array():
    q = np.array([1.0, 2.0]) * u.sec
    assert isinstance(q, Quantity)
    np.testing.assert_array_equal(q.value, [1, 2])


def test_quantity_unpack_and_arithmetic():
    start, end = (3, 5) * u.min
    assert isinstance(start, Quantity) and start.unit is u.min
    assert (start + 30 * u.sec).value == 3.5
    assert (-start).value == -3
    assert (2 * start).value == 6
    assert 3 * u.min == 180 * u.sec


def test_quantity_to():
    assert (3 * u.min).to(u.sec).value == 180
    assert (3 * u.min).to("ms").value == 180000
    assert (60 * u.cm).to(u.mm).value == 600
    with pytest.raises(ValueError, match="unit of time"):
        (3 * u.min).to(u.cm)


def test_unit_aliases():
    assert u.seconds is u.sec
    assert u.s is u.sec
    assert u.degree is u.deg
    assert u.m.dimension == "length"
    with pytest.raises(AttributeError):
        u.parsec


def test_parse_time_forms():
    assert parse_time(500) == 500
    assert parse_time("2 sec") == 2000
    assert parse_time("2sec") == 2000
    assert parse_time("3 m") == 180000  # "m" is minutes for times
    assert parse_time(3 * u.min, to="sec") == 180
    np.testing.assert_allclose(parse_time((-200, 800), to="sec"), [-0.2, 0.8])
    np.testing.assert_allclose(parse_time(("1 min", 30 * u.sec)), [60000, 30000])
    np.testing.assert_allclose(parse_time((1, 2), default="sec"), [1000, 2000])
    np.testing.assert_array_equal(parse_time((-np.inf, np.inf)), [-np.inf, np.inf])
    assert parse_time(None) is None


def test_parse_time_samples():
    assert parse_time(10 * u.samples, fs=500) == 20
    assert parse_time(100, to="samples", fs=500) == 50
    with pytest.raises(ValueError, match="sampling rate"):
        parse_time(10 * u.samples)


def test_parse_time_wrong_dimension():
    with pytest.raises(ValueError, match="unit of time"):
        parse_time("3 cm")
    with pytest.raises(ValueError, match="unit of time"):
        parse_time(3 * u.deg)


def test_parse_pint_like_quantity():
    class FakePint:
        magnitude = 60
        units = "centimeter"
    assert parse_distance(FakePint()) == 600


def test_quantity_pickle():
    import pickle
    q = pickle.loads(pickle.dumps((3, 5) * u.min))
    assert q.unit is u.min


def test_normalize_unit_accepts_units():
    assert normalize_unit(u.min) == "min"
    assert normalize_unit(u.samples) is None
    with pytest.raises(ValueError):
        normalize_unit(u.cm)


# --- Intervals and Events with units ------------------------------------------

from pypillometry import Intervals, Events


def test_intervals_from_quantity():
    iv = Intervals([(0, 100), (200, 300)] * u.ms)
    assert iv.units == "ms"
    assert iv.intervals == [(0, 100), (200, 300)]
    iv = Intervals([("1 min", "2 min")])
    assert iv.units == "min"
    assert iv.intervals == [(1, 2)]


def test_intervals_quantity_converted_to_units():
    iv = Intervals([(0, 1)] * u.sec, units="ms")
    assert iv.units == "ms" and iv.intervals == [(0, 1000)]
    iv = Intervals([(0, 1)] * u.sec, units=None, sampling_rate=500)
    assert iv.units is None and iv.intervals == [(0, 500)]
    assert isinstance(iv.intervals[0][1], int)


def test_intervals_plain_numbers_need_units():
    with pytest.raises(ValueError, match="need units"):
        Intervals([(0, 1)])


def test_intervals_methods_accept_units():
    iv = Intervals([(100, 200), (300, 400)], "ms")
    assert iv.pad(left=10 * u.ms, right="0.02 sec").intervals == [(90, 220), (290, 420)]
    assert iv.merge(distance="100 ms").intervals == [(100, 400)]
    assert len(iv.filter(min_duration=0.2 * u.sec)) == 0
    assert iv.filter(time_range=(0.25, 1) * u.sec).intervals == [(300, 400)]
    idx = Intervals([(100, 200)], None, sampling_rate=500)
    assert idx.pad(left=10 * u.ms).intervals == [(95, 200)]


def test_events_from_quantity():
    ev = Events([1, 2] * u.sec, ["a", "b"])
    assert ev.units == "sec"
    np.testing.assert_array_equal(ev.onsets, [1, 2])
    ev = Events([1, 2] * u.sec, ["a", "b"], units="ms")
    np.testing.assert_array_equal(ev.onsets, [1000, 2000])
    assert Events([1, 2], ["a", "b"]).units == "ms"
    assert Events([1, 2], ["a", "b"], units=None).units is None


# --- EyeData methods with units ------------------------------------------------

@pytest.fixture(scope="module")
def eyedata():
    from pypillometry import get_rlmw_002_short
    return get_rlmw_002_short()


def test_sub_slice_with_units(eyedata):
    ref = eyedata.sub_slice(10, 20, units="sec", inplace=False)
    for start, end in [(10 * u.sec, 20 * u.sec), ("10 sec", "20 sec"), (10000, 20000)]:
        d = eyedata.sub_slice(start, end, inplace=False)
        np.testing.assert_array_equal(d.tx, ref.tx)


def test_get_intervals_with_units(eyedata):
    ref = eyedata.get_intervals("F", (-200, 800), units="ms")
    iv = eyedata.get_intervals("F", (-200, 800) * u.ms)
    assert iv.units == "ms"
    assert iv.intervals == ref.intervals
    assert iv.interval_window == (-200, 800)
    # converted to the requested units
    iv = eyedata.get_intervals("F", (-0.2, 0.8) * u.sec, units="ms")
    np.testing.assert_allclose(iv.intervals, ref.intervals)
    # ... including sampling units
    ix = eyedata.get_intervals("F", (-200, 800) * u.ms, units=None)
    n = int(eyedata.fs * 0.2)
    assert ix.intervals == eyedata.get_intervals("F", (-n, 4 * n)).intervals


def test_stat_per_event_with_units(eyedata):
    ref = eyedata.stat_per_event(interval=(-200, 0), event_select="F")
    res = eyedata.stat_per_event(interval=(-200, 0) * u.ms, event_select="F")
    for k in ref:
        np.testing.assert_allclose(res[k], ref[k])


def test_blink_parameters_with_units(eyedata):
    ref = eyedata.pupil_blinks_detect(min_duration=20, winsize=11, inplace=False)
    res = eyedata.pupil_blinks_detect(min_duration=20 * u.ms, winsize="11 ms",
                                      inplace=False)
    assert {k: v.intervals for k, v in ref.items()} == \
        {k: v.intervals for k, v in res.items()}
    d = eyedata.pupil_blinks_detect(apply_mask=True, inplace=False)
    ref = d.blinks_merge_close(distance=100)
    res = d.blinks_merge_close(distance=0.1 * u.sec)
    assert {k: v.intervals for k, v in ref.items()} == \
        {k: v.intervals for k, v in res.items()}


def test_plot_range_with_units(eyedata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eyedata.plot.pupil_plot(plot_range=(10, 20) * u.sec, units="min")
    eyedata.plot.plot_heatmap(plot_range=("10 sec", "20 sec"))
    eyedata.plot.plot_timeseries(plot_range=(10, 20) * u.sec, units="sec")
    eyedata.plot.plot_timeseries(plot_range=("10 sec", "20 sec"))
    plt.close("all")
    with pytest.raises(ValueError, match="does not overlap"):
        eyedata.plot.pupil_plot(plot_range=(10, 20) * u.min)


def test_unset_repr_and_identity():
    """UNSET shows as 'UNSET' in signatures and survives copy/pickle (bug 1.7)"""
    import copy, inspect, pickle
    from pypillometry.units import UNSET
    from pypillometry import Intervals
    assert repr(UNSET) == "UNSET"
    assert "units=UNSET" in str(inspect.signature(Intervals))
    assert copy.deepcopy(UNSET) is UNSET
    assert pickle.loads(pickle.dumps(UNSET)) is UNSET
