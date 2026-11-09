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

    @pytest.mark.parametrize("getter", ["get_blinks", "get_interpolated",
                                        "mask_as_intervals"])
    def test_single_channel_gives_dict(self, data, getter):
        res = getattr(data, getter)("left", "pupil")
        assert isinstance(res, dict)
        assert list(res) == ["left_pupil"]
        assert isinstance(res["left_pupil"], pp.Intervals)

    def test_nblinks_includes_zero(self, data):
        d = data.pupil.blinks_detect(strategies=["zero"])
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
        seg = data.get_segments(data.get_epochs("a", window=(-100, 200) * pp.u.ms),
                                "right_pupil")
        orig = seg["right_pupil"].copy()
        bc = seg.pupil.baseline_correct((-100, 0), inplace=False)
        assert bc is not seg
        np.testing.assert_array_equal(seg["right_pupil"], orig)
        assert seg.pupil.baseline_correct((-100, 0), inplace=True) is seg
        np.testing.assert_allclose(seg["right_pupil"], bc["right_pupil"])


class TestEyesVariables:
    """Rule 5: None or [] means all; str or list; unknown names raise."""

    @pytest.mark.parametrize("call", [
        lambda d, e: d.quality_report(eyes=e),
        lambda d, e: d.plot.pupil_plot(eyes=e),
        lambda d, e: d.plot.plot_timeseries(eyes=e),
        lambda d, e: d.plot.plot_heatmap(eyes=e),
        lambda d, e: d.plot.plot_scanpath(eyes=e),
        lambda d, e: d.pupil.lowpass_filter(5, eyes=e),
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
            data.pupil.fit_foreshortening(["left"])

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
        d = data.pupil.blinks_detect(strategies=["zero"])
        assert d.get_blinks("left", "pupil")["left_pupil"].intervals == [(3000., 3200.)]
        assert d.blinks["left_pupil"].units == "ms"
        assert d.get_interpolated()["left_pupil"].units == "ms"
        assert d.mask_as_intervals("left", "pupil")["left_pupil"].intervals == [(3000., 3200.)]
        assert d.get_layer_intervals("blink", "left", "pupil")["left_pupil"].units == "ms"

    def test_get_epochs_and_intervals_window(self, data):
        ep = data.get_epochs("a", window=(-200, 800))
        assert ep.window == (-200, 800)
        iv = data.get_intervals(start="a", end="b")
        assert iv.units == "ms" and iv.intervals == [(1000., 5000.)]

    def test_durations(self, data):
        assert data.get_duration() == 10000.
        assert data.get_duration("min") == pytest.approx(1 / 6)
        assert data.exclude_intervals([(0, 500)]).excluded_duration() == 500.

    def test_plot_segments_many_warns(self, data, monkeypatch):
        import matplotlib.pyplot as plt
        from loguru import logger
        from pypillometry.plot.generic import GenericPlots
        monkeypatch.setattr(GenericPlots, "WARN_SEGMENTS", 3)
        msgs = []
        hid = logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
        try:
            for f in (data.plot.plot_timeseries_segments, data.plot.pupil_plot_segments):
                assert len(f(interv=1, units="sec", pdf_file=None)) == 10
                plt.close("all")
        finally:
            logger.remove(hid)
        assert sum("interv gives 10 figures" in m for m in msgs) == 2
        assert len(data.plot.plot_timeseries_segments(interv="5 sec")) == 2
        plt.close("all")


class TestMergedEyeName:
    def test_mean_and_name(self, data):
        assert "mean_pupil" in data.merge_eyes(variables="pupil").data
        assert "mean_pupil" in data.pupil.merge_eyes(method="offset").data
        assert "avg_pupil" in data.pupil.merge_eyes(method="offset", name="avg").data
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


class TestNamespaceInplace:
    """Rule 1 for the methods of d.pupil / d.gaze (and their pipeline steps)."""

    # arguments for each transforming namespace method (all must be listed)
    CALLS = {
        "pupil": {"lowpass_filter": dict(cutoff=5), "smooth_window": {},
                  "merge_eyes": dict(method="offset"), "blinks_detect": {},
                  "blinks_merge_close": {}, "blinks_interpolate": {},
                  "estimate_baseline": {}, "estimate_response": {},
                  "correct_foreshortening": {}},
        "gaze": {"mask_divergences": {}, "mask_offscreen": {}},
    }

    @staticmethod
    def _snapshot(d):
        return ({k: np.array(v) for k, v in d.data.items()},
                {k: v.copy() for k, v in d.data.layers.items()}, len(d.history))

    def test_all_step_methods_listed(self):
        for ns, calls in self.CALLS.items():
            cls = getattr(pp.EyeData, ns)
            steps = {n for n in cls._methods() if getattr(getattr(cls, n), "_is_step", False)}
            assert steps == set(calls), ns

    @pytest.mark.parametrize("ns,name", [(ns, n) for ns, c in CALLS.items() for n in c])
    def test_inplace_false_leaves_original(self, data, ns, name):
        data.set_experimental_setup(screen_resolution=(1280, 1024),
                                    physical_screen_size=("30 cm", "20 cm"),
                                    eye_to_screen_center="60 cm")
        data = data.pupil.blinks_detect(strategies=["zero"], inplace=True)
        before = self._snapshot(data)
        res = getattr(getattr(data, ns), name)(**self.CALLS[ns][name], inplace=False)
        assert isinstance(res, pp.EyeData) and res is not data
        after = self._snapshot(data)
        assert before[0].keys() == after[0].keys() and before[2] == after[2]
        for k in before[0]:
            np.testing.assert_array_equal(before[0][k], after[0][k])
            np.testing.assert_array_equal(before[1][k], after[1][k])
        assert res.history[-1]["funcname"] == f"{ns}.{name}"
        # inplace=True returns the object itself
        assert getattr(getattr(data, ns), name)(**self.CALLS[ns][name], inplace=True) is data


class TestPlainNumbers:
    """Summaries and statistics hold Python numbers (they print as 500.0,
    not np.float64(500.0))."""

    @staticmethod
    def assert_plain(value):
        if isinstance(value, dict):
            for v in value.values():
                TestPlainNumbers.assert_plain(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                TestPlainNumbers.assert_plain(v)
        else:
            assert not isinstance(value, np.generic), repr(value)

    def test_summary(self, data):
        d = data.pupil.blinks_detect().scale()
        s = d.summary()
        self.assert_plain(s)
        assert isinstance(s["params"], dict) and set(s["params"]) == {"scale"}
        assert "np.float64" not in repr(d)
        assert "params            : {scale: {...}}" in repr(d)
        s["params"]["scale"]["mean"]["left_pupil"] = 0.
        assert d.params["scale"]["mean"]["left_pupil"] != 0.

    def test_intervals_stats_and_segments(self, data):
        iv = data.get_epochs("a", (-100, 200))
        self.assert_plain(iv.to_intervals().stats())
        seg = data.get_segments(iv, "left_pupil")
        self.assert_plain(seg.summary())
        assert "np.float64" not in repr(seg)
