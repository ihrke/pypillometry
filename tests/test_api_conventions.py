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

    def test_get_layer_intervals(self, data):
        data.data.set_layer("left_x", "offscreen", np.arange(1000) // 10 == 5)
        data = data.exclude_intervals([(0, 100)])
        iv = data.get_layer_intervals("offscreen", variables="x")
        assert set(iv) == {"left_x", "right_x"}
        assert iv["left_x"].intervals == [(50, 60)] and len(iv["right_x"]) == 0
        assert data.get_layer_intervals("excluded", "left", "pupil")["left_pupil"].intervals \
            == [(0, 10)]


class TestInplace:
    """Rule 1: transforming methods take `inplace`; only set_* are always in place."""

    def test_transforming_methods_take_inplace(self):
        import inspect
        for name, f in inspect.getmembers(pp.EyeData, callable):
            if not hasattr(f, "__wrapped__") or name.startswith("_"):
                continue  # not recorded in the history: not a transformation
            params = inspect.signature(f).parameters
            if name.startswith("set_"):
                assert "inplace" not in params, name
            else:
                assert "inplace" in params, name

    def test_constructor_with_inplace_false_fills_gaps(self):
        tx = np.r_[0:50, 60:100] * 10.
        d = pp.EyeData(left_pupil=np.ones(len(tx)), time=tx, sampling_rate=100,
                       inplace=False)
        assert len(d) == 100
        e = pp.EyeData(left_pupil=np.ones(len(tx)), time=tx, sampling_rate=100,
                       fill_time_discontinuities=False, inplace=False)
        f = e.fill_time_discontinuities()
        assert len(e) == 90 and len(f) == 100

    @pytest.mark.parametrize("call", [
        lambda d: d.set_blinks(None),
        lambda d: d.set_events(d.events),
        lambda d: d.set_experimental_setup(screen_resolution=(1280, 1024))])
    def test_setters_return_the_object(self, data, call):
        assert call(data) is data
