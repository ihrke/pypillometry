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
        iv = data.get_layer_intervals("offscreen", variables="x", units="samples")
        assert set(iv) == {"left_x", "right_x"}
        assert iv["left_x"].intervals == [(50, 60)] and len(iv["right_x"]) == 0
        assert data.get_layer_intervals("excluded", "left", "pupil", units="samples")["left_pupil"].intervals \
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

    def test_baseline_correct(self, data):
        seg = data.get_segments(data.get_intervals("a", window=(-100, 200) * pp.u.ms),
                                "right_pupil")
        orig = seg.data.copy()
        bc = seg.baseline_correct((-100, 0), inplace=False)
        assert bc is not seg
        np.testing.assert_array_equal(seg.data, orig)
        assert seg.baseline_correct((-100, 0), inplace=True) is seg
        np.testing.assert_allclose(seg.data, bc.data)


class TestEyesVariables:
    """Rule 5: None or [] means all; str or list; unknown names raise."""

    @pytest.mark.parametrize("call", [
        lambda d, e: d.quality_report(eyes=e),
        lambda d, e: d.plot.pupil_plot(eyes=e),
        lambda d, e: d.plot.plot_timeseries(eyes=e),
        lambda d, e: d.plot.plot_heatmap(eyes=e),
        lambda d, e: d.plot.plot_scanpath(eyes=e),
        lambda d, e: d.pupil_lowpass_filter(5, eyes=e),
        lambda d, e: d.merge_eyes(eyes=e),
        lambda d, e: d.get_blinks(eyes=e)])
    def test_eyes(self, data, call):
        import matplotlib.pyplot as plt
        try:
            for eyes in (None, [], "left", ["left", "right"]):
                call(data, eyes)
            with pytest.raises(KeyError, match="available channels"):
                call(data, "lft")
        finally:
            plt.close("all")

    def test_drop_data_unknown_names(self, data):
        with pytest.raises(KeyError, match="lft"):
            data.drop_data(eyes=["lft"])
        with pytest.raises(KeyError, match="pupl"):
            data.drop_data(variables="pupl")
        assert set(data.drop_data(eyes=[], variables="x").variables) == {"pupil", "y"}

    def test_informative_type_errors(self, data):
        with pytest.raises(TypeError, match="event_select='a'"):
            data.stat_per_event("a", window=(0, 100))
        with pytest.raises(TypeError, match="one eye"):
            data.fit_foreshortening(["left"])

    def test_view_unknown_eye(self, data):
        pytest.importorskip("vispy")
        from unittest import mock
        with mock.patch("pypillometry.viewer.canvas.ViewerCanvas") as canvas:
            with pytest.raises(KeyError, match="lft"):
                pp.view(data, eyes=["lft"])
            canvas.assert_not_called()


class TestMs:
    """Rule 4: plain numbers (and default units) are ms."""

    def test_interval_getters_in_ms(self, data):
        d = data.pupil_blinks_detect(strategies=["zero"])
        assert d.get_blinks("left", "pupil")["left_pupil"].intervals == [(3000., 3200.)]
        assert d.blinks["left_pupil"].units == "ms"
        assert d.get_interpolated_blinks()["left_pupil"].units == "ms"
        assert d.mask_as_intervals("left", "pupil")["left_pupil"].intervals == [(3000., 3200.)]
        assert d.get_layer_intervals("blink", "left", "pupil")["left_pupil"].units == "ms"

    def test_get_intervals_window(self, data):
        ep = data.get_intervals("a", window=(-200, 800))
        assert ep.window == (-200, 800)
        iv = data.get_intervals(("a", "b"))
        assert iv.units == "ms" and iv.intervals == [(1000., 5000.)]

    def test_durations(self, data):
        assert data.get_duration() == 10000.
        assert data.get_duration("min") == pytest.approx(1 / 6)
        assert data.exclude_intervals([(0, 500)]).excluded_duration() == 500.

    def test_plot_segments_too_many(self, data):
        import matplotlib.pyplot as plt
        for f in (data.plot.plot_timeseries_segments, data.plot.pupil_plot_segments):
            with pytest.raises(ValueError, match="plain numbers are ms"):
                f(interv=1)  # 1 ms, not 1 min
        assert len(data.plot.plot_timeseries_segments(interv="5 sec")) == 2
        plt.close("all")


class TestMergedEyeName:
    def test_mean_and_name(self, data):
        assert "mean_pupil" in data.merge_eyes(variables="pupil").data
        assert "mean_pupil" in data.pupil_merge_eyes(method="offset").data
        assert "avg_pupil" in data.pupil_merge_eyes(method="offset", name="avg").data
        assert "avg_x" in data.merge_eyes(variables="x", name="avg").data


class TestConstructorEvents:
    def test_events_object(self):
        ev = pp.Events([100., 200.], ["a", "b"], meta={"cond": ["x", "y"]})
        d = pp.EyeData(left_pupil=np.ones(100), sampling_rate=100, events=ev)
        assert list(d.events.labels) == ["a", "b"]
        assert list(d.events.meta["cond"]) == ["x", "y"]
        with pytest.raises(ValueError, match="either events"):
            pp.EyeData(left_pupil=np.ones(100), sampling_rate=100, events=ev,
                       event_onsets=[100.])
