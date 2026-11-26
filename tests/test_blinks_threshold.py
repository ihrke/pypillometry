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
    assert blinks(y, vel_onset="8sd", vel_offset="8sd") == blinks(y)
    assert blinks(y, vel_onset="-8sd", vel_offset="8SD") == blinks(y)    # sign and case
    for bad in ("8 std", "eight sd", "sd", "5%"):
        with pytest.raises(ValueError, match="robust SD"):
            blinks(y, vel_onset=bad)
