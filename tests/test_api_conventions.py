"""The API conventions (docs/development): return values, inplace, units, eyes/variables."""
import numpy as np
import pytest

import pypillometry as pp


@pytest.fixture
def data():
    """Binocular pupil and gaze, 10 s at 100 Hz, one NaN blink in the left pupil."""
    n, fs = 1000, 100.
    rng = np.random.default_rng(1)
    chans = {f"{eye}_{var}": 500 + rng.normal(0, 1, n)
             for eye in ("left", "right") for var in ("pupil", "x", "y")}
    chans["left_pupil"][300:320] = np.nan
    with pp.nologging():
        return pp.EyeData(sampling_rate=fs, name="conv", inplace=False,
                          event_onsets=[1000., 5000.], event_labels=["a", "b"], **chans)


class TestDictReturns:
    """Rule 3: getters with per-channel results always return a dict."""

    @pytest.mark.parametrize("getter", ["get_blinks", "get_interpolated_blinks",
                                        "mask_as_intervals"])
    def test_single_channel_gives_dict(self, data, getter):
        res = getattr(data, getter)("left", "pupil")
        assert isinstance(res, dict)
        assert list(res) == ["left_pupil"]
        assert isinstance(res["left_pupil"], pp.Intervals)

    def test_nblinks_includes_zero(self, data):
        d = data.pupil_blinks_detect(strategies=["zero"])
        assert d.nblinks(variables="pupil") == {"left_pupil": 1, "right_pupil": 0}
        stats = d.blink_stats()
        assert set(stats) == {"left_pupil", "right_pupil"}
        assert stats["right_pupil"]["n"] == 0
