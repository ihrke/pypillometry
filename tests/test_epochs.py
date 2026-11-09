"""Epochs: event onsets and one window."""
import numpy as np
import pytest

import pypillometry as pp
from pypillometry import u


def make_epochs():
    return pp.Epochs([1000, 3000, 5000], (-200, 800), labels=["a", "b", "a"],
                     meta={"trial": [1, 2, 3]}, label="stim")


class TestConstruction:

    def test_attributes(self):
        ep = make_epochs()
        assert len(ep) == 3
        np.testing.assert_array_equal(ep.onsets, [1000., 3000., 5000.])
        assert ep.window == (-200., 800.)
        assert ep.duration == 1000.
        assert ep.labels.tolist() == ["a", "b", "a"]
        assert ep.meta["trial"].tolist() == [1, 2, 3]
        assert repr(ep) == "Epochs 'stim' | 3 epochs, window (-200, 800) ms"

    def test_units(self):
        ep = pp.Epochs([1, 3] * u.sec, (-0.2, 0.8) * u.sec)
        np.testing.assert_array_equal(ep.onsets, [1000., 3000.])
        assert ep.window == (-200., 800.)
        assert ep.labels.tolist() == ["", ""]

    def test_immutable(self):
        ep = make_epochs()
        with pytest.raises(ValueError):
            ep.onsets[0] = 0
        with pytest.raises(TypeError):
            ep.meta["x"] = [1, 2, 3]

    @pytest.mark.parametrize("window", [(800, -200), (0, 0), (0, np.nan), (1, 2, 3)])
    def test_invalid_window(self, window):
        with pytest.raises(ValueError, match="window must be"):
            pp.Epochs([0], window)

    def test_invalid_onsets_and_labels(self):
        with pytest.raises(ValueError, match="finite"):
            pp.Epochs([0, np.nan], (0, 1))
        with pytest.raises(ValueError, match="2 labels for 3 onsets"):
            pp.Epochs([0, 1, 2], (0, 1), labels=["a", "b"])
        with pytest.raises(ValueError, match="metadata column 'x' has 2 values"):
            pp.Epochs([0, 1, 2], (0, 1), meta={"x": [1, 2]})

    def test_window_in_samples_needs_data(self):
        with pytest.raises(ValueError):
            pp.Epochs([0], (-10, 10) * u.samples)

    def test_from_events(self):
        ev = pp.Events([1, 2], ["x", "y"], units="sec", meta={"c": ["p", "q"]})
        ep = pp.Epochs.from_events(ev, (0, 500), label="all")
        np.testing.assert_array_equal(ep.onsets, [1000., 2000.])
        assert ep.labels.tolist() == ["x", "y"]
        assert ep.meta["c"].tolist() == ["p", "q"]
        assert ep.label == "all"


class TestSpans:

    def test_iterates_like_spans(self):
        ep = make_epochs()
        assert list(ep) == [(800., 1800.), (2800., 3800.), (4800., 5800.)]
        np.testing.assert_array_equal(np.asarray(ep), list(ep))

    def test_to_intervals(self):
        ep = make_epochs()
        iv = ep.to_intervals()
        assert isinstance(iv, pp.Intervals)
        assert iv.units == "ms" and iv.label == "stim"
        assert iv.intervals == list(ep)
        assert list(iv.event_labels) == ["a", "b", "a"]
        np.testing.assert_array_equal(iv.event_onsets, ep.onsets)
        assert iv.event_meta["trial"].tolist() == [1, 2, 3]
        np.testing.assert_allclose(ep.to_intervals("sec").intervals,
                                   [(0.8, 1.8), (2.8, 3.8), (4.8, 5.8)])

    def test_to_intervals_samples(self):
        d = pp.EyeData(left_pupil=np.ones(1000), sampling_rate=100)
        ep = pp.Epochs([1000, 3000], (-200, 800))
        assert ep.to_intervals("samples", d).intervals == [(80, 180), (280, 380)]
        with pytest.raises(ValueError, match="needs the data object"):
            ep.to_intervals("samples")

    def test_keep(self):
        ep = make_epochs()
        k = ep.keep(ep.labels == "a")
        np.testing.assert_array_equal(k.onsets, [1000., 5000.])
        assert k.meta["trial"].tolist() == [1, 3]
        assert k.window == ep.window and k.label == ep.label
        with pytest.raises(ValueError, match="one value per"):
            ep.keep([True, False])

    def test_equality(self):
        assert make_epochs() == make_epochs()
        assert make_epochs() != make_epochs().keep(np.array([True, True, False]))

    def test_as_pandas(self):
        pytest.importorskip("pandas")
        df = make_epochs().as_pandas()
        assert list(df.columns) == ["onset", "start", "end", "label", "trial"]
        assert df["start"].tolist() == [800., 2800., 4800.]


class TestEventsAndMetadata:

    def test_events(self):
        ep = make_epochs()
        assert isinstance(ep.events, pp.Events)
        assert ep.events.units == "ms"
        assert ep.onsets is ep.events.onsets

    def test_from_events_keeps_the_events(self):
        ev = pp.Events([1, 2], ["x", "y"], units="sec", data_time_range=(0, 3))
        ep = pp.Epochs.from_events(ev, (0, 500))
        assert ep.events.units == "ms"
        assert ep.events.data_time_range == (0., 3000.)

    def test_filter(self):
        ep = make_epochs()
        f = ep.filter("a", exact=True, trial=lambda t: t > 1)
        assert isinstance(f, pp.Epochs)
        np.testing.assert_array_equal(f.onsets, [5000.])
        assert f.window == ep.window and f.label == ep.label
        assert len(ep.filter(trial=[1, 2])) == 2
        assert len(ep.filter((0, 2000))) == 1
        with pytest.raises(KeyError, match="no metadata column"):
            ep.filter(cond="x")

    def test_metadata_methods(self):
        ep = make_epochs()
        ep2 = ep.set_meta("cond", ["p", "q", "p"])
        assert isinstance(ep2, pp.Epochs) and ep2.window == ep.window
        assert ep2.meta["cond"].tolist() == ["p", "q", "p"]
        assert "cond" not in ep.meta
        assert list(ep2.drop_meta("trial").meta) == ["cond"]
        ep3 = pp.Epochs([0, 1], (0, 1), labels=["S_1", "S_2"]).parse_labels(r"S_(?P<n>\d+)")
        assert ep3.meta["n"].tolist() == [1, 2]

    def test_state_roundtrip(self):
        ep = make_epochs().filter(trial=[1, 3])
        st = ep._to_state()
        assert set(st) == {"events", "window", "label"}
        assert ep._from_state(st) == ep


class TestMethodsAcceptEpochs:
    """The methods that take spans accept Epochs (their spans in ms)."""

    def make(self):
        n = 1000
        return pp.EyeData(left_pupil=np.arange(n) * 1., left_x=np.zeros(n),
                          left_y=np.zeros(n), sampling_rate=100,
                          event_onsets=[1000, 5000], event_labels=["a", "b"])

    def epochs(self):
        return pp.Epochs([1000, 5000], (-200, 300))

    def test_mask_intervals(self):
        d = self.make().mask_intervals(self.epochs(), eyes="left", variables="pupil",
                                       inplace=False)
        expected = np.zeros(1000, dtype=bool)
        expected[80:130] = expected[480:530] = True
        np.testing.assert_array_equal(d.data.mask["left_pupil"].astype(bool), expected)
        d = self.make().mask_intervals({"left_x": self.epochs()}, inplace=False)
        assert d.data.mask["left_x"].sum() == 100

    def test_interpolate_intervals(self):
        d = self.make().interpolate_intervals(self.epochs(), eyes="left",
                                              variables="pupil", inplace=False)
        assert d.mask("left_pupil", policy="strict").sum() == 100

    def test_exclude_and_remove(self):
        d = self.make()
        assert d.exclude_intervals(self.epochs(), inplace=False).get_excluded().intervals \
            == [(800., 1300.), (4800., 5300.)]
        assert len(d.remove_intervals(self.epochs(), inplace=False).tx) == 900

    def test_stat_per_event(self):
        d = self.make()
        st = d.stat_per_event(self.epochs(), eyes="left", variables="pupil")
        np.testing.assert_allclose(st["left_pupil"], [104.5, 504.5])
        st2 = d.stat_per_event(self.epochs().to_intervals(), eyes="left",
                               variables="pupil")
        np.testing.assert_array_equal(st["left_pupil"], st2["left_pupil"])

    def test_plot_intervals(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figs = self.make().plot.plot_intervals(self.epochs(), nrow=1, ncol=2)
        assert len(figs) == 1
        plt.close("all")

    def test_type_error(self):
        with pytest.raises(TypeError, match="must be Intervals, Epochs or a list"):
            self.make().mask_intervals({(0, 100)})


class TestSegmentsFromEpochs:

    def ramp(self, n=3000, fs=1000.):
        """Pupil = sample index."""
        return pp.EyeData(left_pupil=np.arange(n) * 1., left_x=np.zeros(n),
                          left_y=np.zeros(n), sampling_rate=fs,
                          event_onsets=[100., 1500.], event_labels=["ev", "ev"])

    def test_time_axis_from_window(self):
        # finding 2.3: the step is 1000/fs, the last sample is not at w1
        d = self.ramp(fs=50.)
        s = d.get_segments(pp.Epochs([10000.], (-200, 800)), "left_pupil")
        assert s.n_timepoints == 50
        np.testing.assert_allclose(np.diff(s.tx), 20.)
        assert s.tx[0] == -200. and s.tx[-1] == 780.

    def test_padded_not_clipped_at_start(self):
        # finding 2.2: the event stays at t=0 when the window starts before the data
        d = self.ramp()
        s = d.get_segments(pp.Epochs([100., 1500.], (-200, 300)), "left_pupil")
        assert s.n_timepoints == 500
        i0 = np.flatnonzero(s.tx == 0)[0]
        assert s["left_pupil"][i0, 0] == 100 and s["left_pupil"][i0, 1] == 1500
        mask = np.ma.getmaskarray(s["left_pupil"])
        assert mask[:100, 0].all() and not mask[100:, 0].any()
        assert not mask[:, 1].any()

    def test_padded_at_end(self):
        d = self.ramp()
        s = d.get_segments(pp.Epochs([2900.], (-100, 300)), "left_pupil")
        mask = np.ma.getmaskarray(s["left_pupil"])[:, 0]
        assert not mask[:200].any() and mask[200:].all()
        assert s["left_pupil"][199, 0] == 2999
        # the padding is NaN (not 0)
        assert np.isnan(np.ma.getdata(s["left_pupil"])[200:, 0]).all()

    def test_window_outside_the_data(self):
        d = self.ramp()
        s = d.get_segments(pp.Epochs([5000.], (0, 100)), "left_pupil")
        assert s["left_pupil"].mask.all()

    def test_keeps_the_epochs(self):
        d = self.ramp()
        ep = pp.Epochs([100., 1500.], (-200, 300), labels=["a", "b"], label="ev")
        s = d.get_segments(ep, "left_pupil")
        assert s.epochs is ep
        assert s.name == "ev"

    def test_mask_is_copied(self):
        d = self.ramp().mask_intervals(pp.Intervals([(1500, 1510)], "ms"),
                                       eyes="left", variables="pupil", inplace=False)
        s = d.get_segments(pp.Epochs([1500.], (-10, 20)), "left_pupil")
        np.testing.assert_array_equal(np.ma.getmaskarray(s["left_pupil"])[:, 0],
                                      (np.arange(30) >= 10) & (np.arange(30) < 20))


class TestGetEpochsAndIntervals:

    def make(self):
        n = 3000
        return pp.EyeData(left_pupil=np.arange(n) * 1., left_x=np.zeros(n),
                          left_y=np.zeros(n), sampling_rate=1000,
                          event_onsets=[100., 1500., 2900.], event_labels=["ev", "ev", "x"],
                          )

    def test_single_events_give_epochs(self):
        d = self.make()
        ep = d.get_epochs("ev", window=(-200, 300) * u.ms)
        assert isinstance(ep, pp.Epochs)
        np.testing.assert_array_equal(ep.onsets, [100., 1500.])
        assert ep.window == (-200., 300.) and ep.label == "ev"
        assert isinstance(d.get_epochs(d.events, window=(0, 10) * u.ms), pp.Epochs)

    def test_two_events_give_intervals(self):
        iv = self.make().get_intervals(start="ev", end="x", units="ms")
        assert isinstance(iv, pp.Intervals)
        assert iv.intervals == [(1500., 2900.)]
        assert iv.label == "ev to x"
        assert list(iv.event_labels) == ["ev to x"]

    def test_not_clipped_in_samples(self):
        # findings 2.2/2.6: windows in samples near the start are not clipped,
        # so the segment is not shifted (event at t=0 has the value 100)
        d = self.make()
        for window in [(-200, 300), (-200, 300) * u.ms]:
            ep = d.get_epochs("ev", window=window)
            assert ep.to_intervals("samples", d).intervals[0] == (-100, 400)
            s = d.get_segments(ep, "left_pupil")
            assert s["left_pupil"][s.tx == 0, 0] == 100
        ep = d.get_epochs("x", window=(-100, 300))
        assert list(ep) == [(2800., 3200.)]


class TestSegmentsFromIntervals:

    def make(self):
        n = 3000
        return pp.EyeData(left_pupil=np.arange(n) * 1., left_x=np.zeros(n),
                          left_y=np.zeros(n), sampling_rate=1000,
                          event_onsets=[100., 400., 1500., 2000.],
                          event_labels=["S_", "R_", "S_", "R_"])

    def test_different_lengths_raise(self):
        # finding 2.1: between-event intervals gave segments with 0 time points
        d = self.make()
        iv = d.get_intervals(start="S_", end="R_", units="ms")
        with pytest.raises(ValueError, match="same length, got lengths from 300 to 500 ms"):
            d.get_segments(iv, "left_pupil")

    def test_same_length(self):
        d = self.make()
        iv = pp.Intervals([(100, 400), (1500, 1800)], units="ms",
                          event_labels=["a", "b"], event_meta={"t": [1, 2]}, label="x")
        s = d.get_segments(iv, "left_pupil")
        assert s.n_timepoints == 300 and s.tx[0] == 0 and s.tx[-1] == 299
        np.testing.assert_array_equal(s["left_pupil"][0], [100, 1500])
        assert isinstance(s.epochs, pp.Epochs)
        assert s.epochs.labels.tolist() == ["a", "b"]
        assert s.epochs.meta["t"].tolist() == [1, 2]
        assert s.name == "x"
        repr(s)
        s.pupil.baseline_correct((0, 10), inplace=False)

    def test_same_length_in_samples(self):
        d = self.make()
        s = d.get_segments(pp.Intervals([(100, 400), (1500, 1800)], "samples"), "left_pupil")
        np.testing.assert_array_equal(s["left_pupil"][0], [100, 1500])
        assert s.n_timepoints == 300

    def test_empty_or_zero_length(self):
        d = self.make()
        with pytest.raises(ValueError, match="no intervals"):
            d.get_segments(pp.Intervals([], "ms"), "left_pupil")
        with pytest.raises(ValueError, match="longer than 0"):
            d.get_segments(pp.Intervals([(5, 5)], "ms"), "left_pupil")

    def test_wrong_type(self):
        with pytest.raises(TypeError, match="Epochs or Intervals"):
            self.make().get_segments([(0, 100)], "left_pupil")


class TestBaselineWindow:
    """Finding 2.4: a baseline window outside the segments raises."""

    def segments(self):
        n = 1000
        d = pp.EyeData(left_pupil=np.arange(n) * 1., sampling_rate=100,
                         event_onsets=[3000., 6000.], event_labels=["a", "a"])
        return d.get_segments(d.get_epochs("a", window=(-200, 800) * u.ms), "left_pupil")

    @pytest.mark.parametrize("window, msg", [
        ((-500, -300), r"baseline window \(-500, -300\) ms is not within the segments; "
                       r"the segments cover -200 to 790 ms"),
        ((0, 5000), "is not within the segments"),
        (-1000, "baseline time -1000 ms is outside the segments"),
        (900, "outside the segments"),
        ((0, -100), "start < end"),
    ])
    def test_outside_raises(self, window, msg):
        with pytest.raises(ValueError, match=msg):
            self.segments().pupil.baseline_correct(window, inplace=False)

    @pytest.mark.parametrize("window", [(-200, 0), (0, 800), -200, 790, 0, (-200, 800)])
    def test_within(self, window):
        s = self.segments().pupil.baseline_correct(window, inplace=False)
        assert not np.ma.getmaskarray(s["left_pupil"]).all()

    def test_values(self):
        s = self.segments()
        b = s.pupil.baseline_correct((-200, 0), inplace=False)
        # mean of the 20 samples before the event: 280 ... 299
        np.testing.assert_allclose(b["left_pupil"][20, 0], 300 - 289.5)
        np.testing.assert_allclose(s.pupil.baseline_correct(0, inplace=False)["left_pupil"][20, :], 0)


def test_plot_intervals_past_the_end():
    # finding 5.2: show_mask crashed when an interval runs past the end of the data
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 1000
    d = pp.EyeData(left_pupil=np.arange(n) * 1., left_x=np.zeros(n), left_y=np.zeros(n),
                   sampling_rate=100, event_onsets=[100., 9900.], event_labels=["a", "a"])
    for ep in [d.get_epochs("a", window=(-200, 500) * u.ms),
               d.get_epochs("a", window=(-20, 50))]:
        figs = d.plot.plot_intervals(ep, show_mask=True, nrow=1, ncol=2)
        assert len(figs) == 1
    plt.close("all")


class TestStatPerEventOutside:
    """Windows partly or entirely outside the data (not clipped in get_epochs)."""

    def make(self):
        n = 1000
        return pp.EyeData(left_pupil=np.arange(n) * 1., sampling_rate=100,
                            event_onsets=[50., 9950.], event_labels=["a", "a"])

    def test_partly_outside_uses_the_part_inside(self):
        d = self.make()
        st = d.stat_per_event(window=(-100, 100) * u.ms, event_select="a", eyes="left",
                              variables="pupil")["left_pupil"]
        # samples 0..14 and 985..999 (the last one is included)
        np.testing.assert_allclose(st, [7., 992.])

    def test_entirely_outside_gives_nan(self):
        d = self.make()
        st = d.stat_per_event(pp.Epochs([20000., -5000., 5000.], (0, 100)), eyes="left",
                              variables="pupil")["left_pupil"]
        assert np.isnan(st[0]) and np.isnan(st[1]) and st[2] == 504.5
