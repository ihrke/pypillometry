"""Velocity thresholds of blinks_detect() relative to the velocity noise ("8sd")."""
import numpy as np
import pytest

import pypillometry as pp
from tests.test_autotune import make_signal


@pytest.fixture(scope="module")
def signal():
    y, clean, _ = make_signal(n=30000, nblinks=20, seed=1)
    return y, clean


def blinks(y, **kw):
    d = pp.EyeData(left_pupil=y, sampling_rate=500, inplace=False)
    return d.pupil.blinks_detect(**kw).get_blinks(units="samples")["left_pupil"].intervals


def test_default_is_independent_of_the_units(signal):
    y, _ = signal
    # the same blinks in other units (e.g., mm instead of pixel area)
    assert blinks(y) == blinks(y / 300.0) == blinks(y * 20.0)
    # absolute thresholds depend on the units
    assert blinks(y, vel_onset=-5.0, vel_offset=5.0) != \
        blinks(y / 300.0, vel_onset=-5.0, vel_offset=5.0)


def test_default_covers_the_transients(signal):
    y, clean = signal
    art = np.isnan(y) | (np.abs(y - clean) > 1.0)
    d = pp.EyeData(left_pupil=y, sampling_rate=500, inplace=False).pupil.blinks_detect()
    m = d.get_blinks(units="samples")["left_pupil"].to_mask(d).astype(bool)
    assert m[art].mean() > 0.95 and m[~art].mean() < 0.01


def test_threshold_forms(signal):
    y, _ = signal
    assert blinks(y, vel_onset="25sd", vel_offset="25sd") == blinks(y)
    assert blinks(y, vel_onset="-25sd", vel_offset="25SD") == blinks(y)    # sign and case
    for bad in ("8 std", "eight sd", "sd", "5%"):
        with pytest.raises(ValueError, match="robust SD"):
            blinks(y, vel_onset=bad)


def test_velocity_window_rule():
    from pypillometry.eyedata.pupil import _velocity_window
    for fs, ms, expect in ((1000, 45, 45), (500, 45, 23), (250, 45, 11), (60, 45, 5),
                           (60, 11, 5), (1000, 30, 31), (500, 30, 15)):
        d = pp.EyeData(left_pupil=np.ones(100), sampling_rate=fs)
        w = _velocity_window(d, ms)
        assert w == expect and w % 2 == 1, (fs, ms, w)


def test_relative_thresholds_in_units_per_ms(signal):
    y, _ = signal
    d = pp.EyeData(left_pupil=y, sampling_rate=500, inplace=False)
    on, off = d.pupil._velocity_thresholds_ms("left", 30, "2sd", "2sd")
    assert on < 0 < off and np.isclose(-on, off)
    # proportional to k and to the units of the data
    on4, _ = d.pupil._velocity_thresholds_ms("left", 30, "4sd", "4sd")
    assert np.isclose(on4, 2 * on)
    d10 = pp.EyeData(left_pupil=10 * y, sampling_rate=500, inplace=False)
    assert np.isclose(d10.pupil._velocity_thresholds_ms("left", 30, "2sd", "2sd")[0], 10 * on)
