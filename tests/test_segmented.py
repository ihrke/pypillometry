"""
Tests for SegmentedEyeData, its namespaces (seg.pupil, seg.gaze) and
GroupLevelSegmentedData.
"""
import os
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import pypillometry as pp
from pypillometry import state
from pypillometry.intervals import Intervals
from pypillometry.segmented import SegmentedEyeData, GroupLevelSegmentedData, group_segments


def make(data, tx, channel="left_pupil", name="segments", mask=None, **kw):
    """Segments with one channel."""
    if mask is not None:
        data = ma.masked_array(data, mask=mask)
    return SegmentedEyeData(name=name, tx=tx, data={channel: data}, **kw)


def eyedata(n_samples=1000, fs=100, onsets=(1000, 3000, 5000, 7000), labels=None, **channels):
    """EyeData with events (default: a left pupil)."""
    if not channels:
        channels = {"left_pupil": np.sin(np.linspace(0, 10 * np.pi, n_samples)) * 100 + 500}
    labels = labels or [f"stim{i + 1}" for i in range(len(onsets))]
    return pp.EyeData(sampling_rate=fs, event_onsets=list(onsets), event_labels=labels,
                      **channels)


def gaze_eyedata():
    """EyeData with pupil and gaze of both eyes; the left eye looks at (100, 100)
    from 0 to 200 ms after the events "a", at (900, 900) otherwise."""
    fs, n = 100, 1000
    tx = np.arange(n) * 10.
    onsets = [1000., 3000., 5000., 7000.]
    lx, ly = np.full(n, 900.), np.full(n, 900.)
    for on in onsets[:2]:
        sel = (tx >= on) & (tx < on + 200)
        lx[sel], ly[sel] = 100., 100.
    rng = np.random.default_rng(0)
    return pp.EyeData(sampling_rate=fs, left_x=lx, left_y=ly,
                      left_pupil=rng.normal(5, 0.1, n), right_x=np.full(n, 900.),
                      right_y=np.full(n, 900.), right_pupil=rng.normal(5, 0.1, n),
                      event_onsets=onsets, event_labels=["a", "a", "b", "b"])


class TestCreation(unittest.TestCase):

    def test_creation_from_dict(self):
        tx = np.linspace(-200, 800, 100)
        data = np.random.randn(100, 10)
        mask = np.zeros((100, 10), dtype=bool)
        mask[0:5, 0] = True
        seg = make(data, tx, name="test_segments", mask=mask)
        self.assertEqual(seg.name, "test_segments")
        self.assertEqual(seg.channels, ["left_pupil"])
        self.assertEqual(seg.eyes, ["left"])
        self.assertEqual(seg.variables, ["pupil"])
        self.assertEqual(seg.n_timepoints, 100)
        self.assertEqual(seg.n_segments, 10)
        np.testing.assert_array_equal(seg.tx, tx)
        self.assertTrue(seg["left_pupil"].mask[0, 0])
        self.assertFalse(seg["left_pupil"].mask[10, 0])

    def test_nan_masked(self):
        data = np.ones((5, 3))
        data[:, 2] = np.nan
        seg = make(data, np.arange(5.), "right_x")
        self.assertTrue(np.all(seg["right_x"].mask[:, 2]))
        self.assertFalse(np.any(seg["right_x"].mask[:, 0]))

    def test_several_channels(self):
        seg = SegmentedEyeData("s", np.arange(5.), {"left_pupil": np.ones((5, 3)),
                                                    "left_x": np.zeros((5, 3))})
        self.assertEqual(seg.channels, ["left_pupil", "left_x"])
        self.assertEqual(seg.variables, ["pupil", "x"])
        self.assertIn("left_x", seg)
        self.assertNotIn("right_x", seg)

    def test_invalid(self):
        tx = np.arange(5.)
        with self.assertRaisesRegex(ValueError, "<eye>_<variable>"):
            make(np.ones((5, 2)), tx, "pupil")
        with self.assertRaisesRegex(ValueError, "at least one channel"):
            SegmentedEyeData("s", tx, {})
        with self.assertRaises(TypeError):
            SegmentedEyeData("s", tx, np.ones((5, 2)))
        with self.assertRaisesRegex(ValueError, "must have shape"):
            make(np.ones((4, 2)), tx)
        with self.assertRaisesRegex(ValueError, "segments, the others"):
            SegmentedEyeData("s", tx, {"left_x": np.ones((5, 2)), "left_y": np.ones((5, 3))})
        with self.assertRaisesRegex(ValueError, "3 epochs for 2 segments"):
            make(np.ones((5, 2)), tx, epochs=pp.Epochs([1, 2, 3], (0, 5)))

    def test_from_eyedata(self):
        d = eyedata()
        seg = d.get_segments(d.get_epochs("stim", window=(-100, 200), units="ms"), "left_pupil")
        self.assertIsInstance(seg, SegmentedEyeData)
        self.assertEqual(seg.channels, ["left_pupil"])
        self.assertEqual(seg.n_segments, 4)
        self.assertEqual(seg.sampling_rate, 100)

    def test_from_eyedata_all_channels(self):
        d = gaze_eyedata()
        ep = d.get_epochs("", window=(-100, 500), units="ms")
        seg = d.get_segments(ep)
        self.assertEqual(seg.channels, list(d.data.keys()))
        self.assertEqual(d.get_segments(ep, variables="pupil").channels,
                         ["left_pupil", "right_pupil"])
        self.assertEqual(d.get_segments(ep, eyes="left", variables=["x", "y"]).channels,
                         ["left_x", "left_y"])
        self.assertEqual(d.get_segments(ep, ["right_x", "left_x"]).channels,
                         ["right_x", "left_x"])
        # the same data as one channel at a time
        for ch in seg.channels:
            one = d.get_segments(ep, ch)
            np.testing.assert_array_equal(ma.getdata(one[ch]), ma.getdata(seg[ch]))
            np.testing.assert_array_equal(ma.getmaskarray(one[ch]), ma.getmaskarray(seg[ch]))

    def test_from_eyedata_errors(self):
        d = eyedata()
        ep = d.get_epochs("stim", window=(-50, 100), units="ms")
        with self.assertRaisesRegex(KeyError, "available channels"):
            d.get_segments(ep, "right_pupil")
        with self.assertRaisesRegex(KeyError, "available channels"):
            d.get_segments(ep, eyes="right")
        with self.assertRaisesRegex(ValueError, "not both"):
            d.get_segments(ep, "left_pupil", eyes="left")

    def test_name_from_epochs_label(self):
        d = eyedata(n_samples=500, onsets=(1000, 2000, 3000), labels=["cue"] * 3)
        seg = d.get_segments(d.get_epochs("cue", window=(-100, 200), units="ms"))
        self.assertEqual(seg.name, "cue")

    def test_epochs_kept(self):
        iv = Intervals([(0, 100), (200, 300)], units="ms", label="test_intervals",
                       event_labels=["ev1", "ev2"])
        seg = make(np.random.randn(50, 2), np.linspace(0, 100, 50), epochs=iv)
        self.assertIs(seg.epochs, iv)
        self.assertFalse(hasattr(seg, "intervals"))


class TestChannels(unittest.TestCase):

    def setUp(self):
        d = gaze_eyedata()
        self.seg = d.get_segments(d.get_epochs("", window=(-100, 500), units="ms"))

    def test_getitem_missing(self):
        with self.assertRaisesRegex(KeyError, "available channels: left_x"):
            self.seg["left_z"]

    def test_setitem(self):
        seg = self.seg
        seg["left_zero"] = np.zeros((seg.n_timepoints, seg.n_segments))
        self.assertIn("left_zero", seg.channels)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            seg["left_bad"] = np.zeros((3, seg.n_segments))
        with self.assertRaisesRegex(ValueError, "segments, the others"):
            seg["left_bad"] = np.zeros((seg.n_timepoints, 2))

    def test_select_channels(self):
        sel = self.seg.select_channels(variables="pupil")
        self.assertEqual(sel.channels, ["left_pupil", "right_pupil"])
        self.assertEqual(sel.epochs, self.seg.epochs)
        self.assertEqual(self.seg.select_channels("left_x").channels, ["left_x"])
        self.assertEqual(self.seg.select_channels(eyes="right").channels,
                         ["right_x", "right_y", "right_pupil"])
        # copies: changing the selection does not change the original
        sel["left_pupil"][:] = 0
        self.assertFalse(np.all(ma.getdata(self.seg["left_pupil"]) == 0))
        with self.assertRaisesRegex(KeyError, "available channels"):
            self.seg.select_channels(eyes="mean")
        with self.assertRaisesRegex(ValueError, "not both"):
            self.seg.select_channels("left_x", eyes="left")

    def test_drop_channels(self):
        rest = self.seg.drop_channels("left_x", "left_y")
        self.assertEqual(rest.channels, ["left_pupil", "right_x", "right_y", "right_pupil"])
        with self.assertRaises(KeyError):
            self.seg.drop_channels("left_z")
        with self.assertRaisesRegex(ValueError, "all channels"):
            self.seg.drop_channels(*self.seg.channels)


class TestTrialSelection(unittest.TestCase):

    def setUp(self):
        d = gaze_eyedata()
        ep = d.get_epochs("", window=(-100, 500), units="ms")
        ep = ep.set_meta("cond", ["easy", "hard", "easy", "hard"])
        self.seg = d.get_segments(ep)

    def test_keep(self):
        mask = np.array([True, False, True, True])
        kept = self.seg.keep(mask)
        self.assertEqual(kept.n_segments, 3)
        for ch in self.seg.channels:
            np.testing.assert_array_equal(ma.getdata(kept[ch]), ma.getdata(self.seg[ch])[:, mask])
        np.testing.assert_array_equal(kept.epochs.onsets, self.seg.epochs.onsets[mask])
        self.assertEqual(self.seg.n_segments, 4)
        with self.assertRaisesRegex(ValueError, "boolean array with one value per segment"):
            self.seg.keep([True, False])
        with self.assertRaisesRegex(ValueError, "boolean array"):
            self.seg.keep(np.array([1, 0, 1, 1]))

    def test_filter(self):
        easy = self.seg.filter(cond="easy")
        self.assertEqual(easy.n_segments, 2)
        self.assertEqual(list(easy.epochs.meta["cond"]), ["easy", "easy"])
        self.assertEqual(self.seg.filter("a", exact=True).n_segments, 2)
        self.assertEqual(self.seg.filter("b", cond="hard").n_segments, 1)
        with self.assertRaisesRegex(KeyError, "no metadata column"):
            self.seg.filter(block=1)

    def test_filter_needs_epochs(self):
        seg = make(np.ones((5, 2)), np.arange(5.))
        with self.assertRaisesRegex(ValueError, "no epochs"):
            seg.filter("a")
        self.assertEqual(seg.keep(np.array([True, False])).n_segments, 1)

    def test_keep_intervals(self):
        iv = Intervals([(0, 100), (200, 300), (400, 500)], units="ms", label="iv",
                       event_labels=["a", "b", "c"], event_onsets=[0, 200, 400],
                       event_meta={"x": [1, 2, 3]})
        seg = make(np.ones((5, 3)), np.arange(5.), epochs=iv)
        kept = seg.keep(np.array([True, False, True]))
        self.assertEqual(kept.epochs.intervals, [(0, 100), (400, 500)])
        self.assertEqual(list(kept.epochs.event_labels), ["a", "c"])
        np.testing.assert_array_equal(kept.epochs.event_meta["x"], [1, 3])


class TestDisplay(unittest.TestCase):

    def setUp(self):
        self.seg = make(np.random.randn(100, 10), np.linspace(-200, 800, 100),
                        name="test", sampling_rate=500.0)

    def test_window_property(self):
        self.assertEqual(self.seg.window, (-200, 800))

    def test_summary(self):
        summary = self.seg.summary()
        self.assertEqual(summary["name"], "test")
        self.assertEqual(summary["channels"], ["left_pupil"])
        self.assertEqual(summary["n_segments"], 10)
        self.assertEqual(summary["n_timepoints"], 100)
        self.assertEqual(summary["sampling_rate"], 500.0)
        self.assertEqual(summary["percent_masked"], {"left_pupil": 0.0})
        self.assertIn("window", summary)

    def test_percent_masked_in_summary(self):
        mask = np.zeros((100, 10), dtype=bool)
        mask[0:10, :] = True
        seg = make(np.random.randn(100, 10), np.linspace(-200, 800, 100), mask=mask)
        self.assertAlmostEqual(seg.summary()["percent_masked"]["left_pupil"], 10.0)

    def test_repr(self):
        r = repr(self.seg)
        self.assertIn("SegmentedEyeData(test)", r)
        self.assertIn("10 segments x 100 time points", r)
        self.assertIn("left_pupil", r)

    def test_repr_html(self):
        html = self.seg._repr_html_()
        for text in ("SegmentedEyeData", "test", "left_pupil", "Segments", "Time points"):
            self.assertIn(text, html)


class TestBaselineCorrection(unittest.TestCase):

    def test_single_point(self):
        tx = np.linspace(-200, 800, 100)
        data = np.zeros((100, 5))
        for i in range(5):
            data[:, i] = i * 10 + np.random.randn(100) * 0.1
        bc = make(data, tx).pupil.baseline_correct(window=0, inplace=False)
        t0 = np.argmin(np.abs(tx))
        np.testing.assert_allclose(bc["left_pupil"][t0], 0, atol=1e-12)

    def test_window(self):
        tx = np.linspace(-200, 800, 100)
        data = np.tile((np.arange(5) + 1) * 100., (100, 1))
        bc = make(data, tx).pupil.baseline_correct(window=(-200, 0), inplace=False)
        np.testing.assert_allclose(bc["left_pupil"].mean(axis=0), 0, atol=1e-9)

    def test_same_as_per_segment_loop(self):
        """vectorized correction = subtracting the mean of each segment (with masks)"""
        rng = np.random.default_rng(3)
        tx = np.arange(-100, 200, 10.)
        data = rng.normal(size=(30, 6))
        mask = rng.random((30, 6)) < 0.2
        mask[:, 4] = True  # all masked
        seg = make(data, tx, mask=mask, sampling_rate=100)
        bc = seg.pupil.baseline_correct((-50, 0), inplace=False)["left_pupil"]
        ref = seg["left_pupil"].copy()
        i0, i1 = np.argmin(np.abs(-50 - tx)), np.argmin(np.abs(0 - tx))
        for i in range(6):
            ref[:, i] -= np.mean(ref[i0:i1, i])
        np.testing.assert_array_equal(ma.getmaskarray(bc), ma.getmaskarray(ref))
        np.testing.assert_allclose(ma.getdata(bc), ma.getdata(ref), rtol=1e-12)

    def test_with_units(self):
        from pypillometry import u
        tx = np.linspace(-200, 800, 100)
        seg = make(np.random.default_rng(1).normal(size=(100, 3)), tx)

        def bc(w):
            return seg.pupil.baseline_correct(window=w, inplace=False)["left_pupil"]
        ref = bc((-200, 0))
        np.testing.assert_allclose(bc((-200, 0) * u.ms), ref)
        np.testing.assert_allclose(bc((-0.2 * u.sec, 0 * u.sec)), ref)
        np.testing.assert_allclose(bc(-100 * u.ms), bc(-100))

    def test_none(self):
        seg = make(np.random.randn(100, 5), np.linspace(-200, 800, 100))
        self.assertIs(seg.pupil.baseline_correct(window=None, inplace=False), seg)

    def test_inplace(self):
        seg = make(np.random.randn(100, 5) + 100, np.linspace(-200, 800, 100))
        orig = seg["left_pupil"].copy()
        bc = seg.pupil.baseline_correct(window=0, inplace=False)
        self.assertIsNot(bc, seg)
        np.testing.assert_array_equal(seg["left_pupil"], orig)
        self.assertIs(seg.pupil.baseline_correct(window=0, inplace=True), seg)
        np.testing.assert_allclose(seg["left_pupil"], bc["left_pupil"])

    def test_only_pupil_channels(self):
        d = gaze_eyedata()
        seg = d.get_segments(d.get_epochs("", window=(-100, 500), units="ms"))
        bc = seg.pupil.baseline_correct((-100, 0), inplace=False)
        for ch in ("left_x", "left_y", "right_x", "right_y"):
            self.assertIs(bc[ch], seg[ch])
        self.assertFalse(np.allclose(bc["left_pupil"], seg["left_pupil"]))
        left = seg.pupil.baseline_correct((-100, 0), eyes="left", inplace=False)
        self.assertIs(left["right_pupil"], seg["right_pupil"])
        np.testing.assert_allclose(left["left_pupil"], bc["left_pupil"])

    def test_no_pupil_raises(self):
        seg = make(np.ones((5, 2)), np.arange(5.), "left_x")
        with self.assertRaisesRegex(KeyError, "available channels: left_x"):
            seg.pupil.baseline_correct(0)

    def test_group(self):
        segs = [make(np.random.randn(20, 4) + i, np.arange(20.)) for i in range(2)]
        group = group_segments(segs)
        bc = group.pupil.baseline_correct((0, 5), inplace=False)
        self.assertIsInstance(bc, GroupLevelSegmentedData)
        np.testing.assert_allclose(bc["left_pupil"][:5].mean(axis=0), 0, atol=1e-12)


class TestPlotting(unittest.TestCase):

    def setUp(self):
        self.tx = np.linspace(-200, 800, 100)
        self.seg = make(np.random.randn(100, 20), self.tx, name="test_plot")

    def tearDown(self):
        plt.close("all")

    def test_plot_runs(self):
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax)
        self.seg.pupil.plot(ax=ax)

    def test_plot_standalone(self):
        plt.figure()
        ax = self.seg.pupil.plot()
        self.assertEqual(ax.get_title(), "test_plot")
        self.assertEqual(ax.get_ylabel(), "mean pupil")

    def test_plot_with_ax_no_title(self):
        fig, ax = plt.subplots()
        ax.set_title("Original Title")
        self.seg.plot(ax=ax)
        self.assertEqual(ax.get_title(), "Original Title")

    def test_plot_with_kwargs(self):
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax, color='red', linewidth=2, linestyle='--')

    def test_plot_with_custom_label(self):
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax, label="Custom Label", show_legend=True)
        self.assertIn("Custom Label", ax.get_legend_handles_labels()[1])

    def test_plot_overlay_multiple(self):
        seg2 = make(np.random.randn(100, 20), self.tx, "right_pupil", name="test_plot2")
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax, color='blue', label='Left')
        seg2.plot(ax=ax, color='red', label='Right')
        self.assertEqual(len(ax.get_legend_handles_labels()[1]), 2)

    def test_plot_several_channels(self):
        seg = SegmentedEyeData("s", self.tx, {"left_pupil": np.random.randn(100, 5),
                                              "right_pupil": np.random.randn(100, 5),
                                              "left_x": np.random.randn(100, 5)})
        fig, ax = plt.subplots()
        seg.pupil.plot(ax=ax, show_legend=True)
        self.assertEqual(ax.get_legend_handles_labels()[1], ["left_pupil", "right_pupil"])
        fig, ax = plt.subplots()
        seg.pupil.plot(ax=ax, label="cue", show_legend=True)
        self.assertEqual(ax.get_legend_handles_labels()[1], ["cue left_pupil", "cue right_pupil"])
        with self.assertRaisesRegex(ValueError, "several variables"):
            seg.plot()
        fig, ax = plt.subplots()
        seg.plot("left_x", ax=ax, show_legend=True)
        self.assertEqual(ax.get_legend_handles_labels()[1], ["s"])

    def test_plot_with_missing_overlay(self):
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax, show_missing=True, show_legend=True)
        self.assertIsNotNone(ax.get_legend())

    def test_plot_standalone_shows_missing_by_default(self):
        fig = plt.figure()
        self.seg.plot()
        self.assertEqual(len(fig.axes), 2)

    def test_plot_returns_axes(self):
        fig, ax = plt.subplots()
        self.assertIs(self.seg.plot(ax=ax), ax)
        self.assertIs(self.seg.pupil.plot(ax=ax), ax)

    def test_plot_no_zero_line(self):
        fig, ax = plt.subplots()
        self.seg.plot(ax=ax, show_zero_line=False)


class TestGaze(unittest.TestCase):

    def setUp(self):
        d = gaze_eyedata()
        self.seg = d.get_segments(d.get_epochs("", window=(-100, 500), units="ms"))
        self.roi = pp.RectangularROI((0, 0), (200, 200), name="corner")

    def tearDown(self):
        plt.close("all")

    def test_roi_hits(self):
        seg = self.seg.gaze.roi_hits(self.roi, inplace=False)
        self.assertEqual(seg.channels[-2:], ["left_corner", "right_corner"])
        self.assertNotIn("left_corner", self.seg)
        hits = seg["left_corner"]
        inside = (self.seg.tx >= 0) & (self.seg.tx < 200)
        # trials 1 and 2 ("a") look at the corner from 0 to 200 ms
        np.testing.assert_array_equal(hits[:, 0], inside.astype(float))
        np.testing.assert_array_equal(hits[:, 2], 0.)
        np.testing.assert_array_equal(seg["right_corner"], 0.)
        # the mean over trials is the share of trials in the ROI
        np.testing.assert_allclose(hits.mean(axis=1)[inside], 0.5)

    def test_roi_hits_masked_where_gaze_is(self):
        seg = self.seg
        x = seg["left_x"].copy()
        x[5:10, 1] = ma.masked
        seg["left_x"] = x
        hits = seg.gaze.roi_hits(self.roi, eyes="left", inplace=False)["left_corner"]
        self.assertTrue(hits.mask[5:10, 1].all())
        self.assertFalse(hits.mask[5:10, 0].any())

    def test_roi_hits_inplace_and_name(self):
        out = self.seg.gaze.roi_hits(self.roi, name="target", eyes="left", inplace=True)
        self.assertIs(out, self.seg)
        self.assertIn("left_target", self.seg)
        self.assertNotIn("right_target", self.seg)

    def test_roi_hits_errors(self):
        with self.assertRaisesRegex(ValueError, "no name"):
            self.seg.gaze.roi_hits(pp.CircularROI((0, 0), 10), inplace=False)
        with self.assertRaisesRegex(ValueError, "cannot be called"):
            self.seg.gaze.roi_hits(self.roi, name="x", inplace=False)
        with self.assertRaisesRegex(KeyError, "no gaze channels"):
            self.seg.gaze.roi_hits(self.roi, eyes="mean", inplace=False)
        with self.assertRaisesRegex(KeyError, "no gaze channels"):
            self.seg.select_channels(variables="pupil").gaze.roi_hits(self.roi, inplace=False)

    def test_dwell_time(self):
        table = self.seg.gaze.dwell_time(self.roi)
        np.testing.assert_allclose(table["left_corner"], [200., 200., 0., 0.])
        np.testing.assert_allclose(table["right_corner"], 0.)
        np.testing.assert_array_equal(table["onset"], self.seg.epochs.onsets)
        self.assertEqual(list(table["label"]), ["a", "a", "b", "b"])
        # only a part of the segments
        np.testing.assert_allclose(self.seg.gaze.dwell_time(self.roi, window=(100, None))
                                   ["left_corner"], [100., 100., 0., 0.])
        # from an ROI channel
        seg = self.seg.gaze.roi_hits(self.roi, inplace=False)
        np.testing.assert_allclose(seg.gaze.dwell_time("corner")["left_corner"],
                                   table["left_corner"])
        with self.assertRaisesRegex(KeyError, "roi_hits"):
            self.seg.gaze.dwell_time("corner", eyes="left")

    def test_dwell_time_all_masked_is_nan(self):
        x = self.seg["left_x"].copy()
        x[:, 3] = ma.masked
        self.seg["left_x"] = x
        table = self.seg.gaze.dwell_time(self.roi, eyes="left")
        self.assertTrue(np.isnan(table["left_corner"][3]))
        self.assertNotIn("right_corner", table)

    def test_first_entry(self):
        table = self.seg.gaze.first_entry(self.roi, eyes="left")
        np.testing.assert_allclose(table["left_corner"], [0., 0., np.nan, np.nan])
        table = self.seg.gaze.first_entry(self.roi, eyes="left", window=(50, None))
        np.testing.assert_allclose(table["left_corner"], [50., 50., np.nan, np.nan])
        table = self.seg.gaze.first_entry(self.roi, eyes="left", window=None)
        np.testing.assert_allclose(table["left_corner"], [0., 0., np.nan, np.nan])

    def test_table_has_meta(self):
        seg = self.seg
        seg.epochs = seg.epochs.set_meta("cond", ["x", "y", "x", "y"])
        self.assertEqual(list(seg.gaze.dwell_time(self.roi)["cond"]), ["x", "y", "x", "y"])

    def test_plot_roi(self):
        fig, ax = plt.subplots()
        self.seg.gaze.plot_roi(self.roi, ax=ax, show_legend=True)
        self.assertEqual(ax.get_legend_handles_labels()[1], ["left_corner", "right_corner"])
        self.assertNotIn("left_corner", self.seg)
        plt.figure()
        seg = self.seg.gaze.roi_hits(self.roi, inplace=False)
        ax = seg.gaze.plot_roi("corner", eyes="left")
        self.assertEqual(ax.get_ylabel(), "share of trials in corner")

    def test_group_fixation_curve(self):
        seg = self.seg.gaze.roi_hits(self.roi, inplace=False)
        group = group_segments({"s1": seg, "s2": seg}, channels=["left_corner"])
        inside = (seg.tx >= 0) & (seg.tx < 200)
        np.testing.assert_allclose(group["left_corner"].mean(axis=1)[inside], 0.5)
        fig, ax = plt.subplots()
        group.gaze.plot_roi("corner", ax=ax)


class TestIO(unittest.TestCase):

    def roundtrip(self, obj, ext=".pypil"):
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "seg" + ext)
            obj.write_file(fname)
            return type(obj).from_file(fname)

    def test_write_and_read(self):
        tx = np.linspace(-100, 400, 50)
        seg = SegmentedEyeData("io_test", tx, {"left_pupil": np.random.randn(50, 5),
                                               "left_x": np.random.randn(50, 5)},
                               sampling_rate=500.0)
        for ext in (".pypil", ".h5"):
            loaded = self.roundtrip(seg, ext)
            self.assertEqual(loaded.name, "io_test")
            self.assertEqual(loaded.channels, ["left_pupil", "left_x"])
            self.assertEqual(loaded.sampling_rate, 500.0)
            np.testing.assert_array_almost_equal(loaded.tx, tx)
            for ch in seg.channels:
                np.testing.assert_array_almost_equal(loaded[ch], seg[ch])

    def test_group_roundtrip(self):
        d = gaze_eyedata()
        seg = d.get_segments(d.get_epochs("", window=(-100, 500), units="ms"))
        for meanfct in (np.mean, None):
            group = group_segments({"a": seg, "b": seg}, meanfct=meanfct)
            for ext in (".pypil", ".h5"):
                loaded = self.roundtrip(group, ext)
                self.assertEqual(loaded.subjects, ["a", "b"])
                self.assertEqual(loaded.stacked, meanfct is None)
                np.testing.assert_array_equal(loaded.subject, group.subject)
                self.assertEqual(loaded.epochs, group.epochs)
                for ch in group.channels:
                    np.testing.assert_array_equal(loaded.mask_percent[ch], group.mask_percent[ch])

    def single_channel_state(self, group=False):
        """State of segments with one channel (layout of development versions of 2.2)."""
        data = ma.masked_array(np.arange(12.).reshape(4, 3), mask=np.eye(4, 3, dtype=bool))
        st = {"format_version": 3, "class": "SegmentedEyeData", "name": "old",
              "variable": "left_pupil", "tx": np.arange(4.),
              "data": {"values": ma.getdata(data), "mask": ma.getmaskarray(data)},
              "intervals": {"class": "Epochs",
                            "events": {"onsets": np.array([0., 10., 20.]),
                                       "labels": np.array(["a", "b", "c"]), "units": "ms",
                                       "data_time_range": None, "meta": {}, "needs_sync": False},
                            "window": (0., 4.), "label": "a"},
              "sampling_rate": 1000.}
        if group:
            st.update({"class": "GroupLevelSegmentedData", "n_subjects": 3,
                       "mask_percent": np.array([10., 20., 30., 40.])})
        return st

    def test_single_channel_layout(self):
        s = state.from_state(self.single_channel_state())
        self.assertEqual(s.channels, ["left_pupil"])
        self.assertEqual(s.epochs.label, "a")
        self.assertTrue(s["left_pupil"].mask[0, 0])
        self.assertNotIn("variable", state.to_state(s))

    def test_single_channel_layout_group(self):
        g = state.from_state(self.single_channel_state(group=True))
        self.assertIsInstance(g, GroupLevelSegmentedData)
        self.assertEqual(g.n_subjects, 3)
        self.assertFalse(g.stacked)
        self.assertIsNone(g.epochs)
        np.testing.assert_allclose(g.mask_percent["left_pupil"], [10., 20., 30., 40.])
        # stacked trials of unknown subjects
        st = self.single_channel_state(group=True)
        st["n_subjects"] = 2
        g = state.from_state(st)
        self.assertTrue(g.stacked)
        self.assertIsNone(g.subject)
        with self.assertRaisesRegex(ValueError, "subjects of the trials are not known"):
            g.keep(np.array([True, False, True]))


class TestTimeAxis(unittest.TestCase):

    def test_time_axis_is_relative_to_event(self):
        d = eyedata(onsets=(2000, 5000, 8000), labels=["stim"] * 3,
                    left_pupil=np.random.randn(1000) + 500)
        ep = d.get_epochs("stim", window=(-200, 800), units="ms")
        self.assertEqual(ep.window, (-200, 800))
        seg = d.get_segments(ep, "left_pupil")
        self.assertAlmostEqual(seg.tx[0], -200, delta=50)
        self.assertAlmostEqual(seg.tx[-1], 800, delta=50)
        self.assertAlmostEqual(np.argmin(np.abs(seg.tx)), int(seg.n_timepoints * 0.2), delta=5)

    def test_length_independent_of_event_timing(self):
        onsets = 2000 + np.arange(10) * 1000.7  # not on the sample grid
        labels = [f"ev{i}" for i in range(10)]
        d = pp.EyeData(sampling_rate=500, left_pupil=np.random.randn(10000) + 500,
                       event_onsets=onsets, event_labels=labels)
        lengths = {d.get_segments(d.get_epochs(lab, window=(-500, 2000), units="ms"),
                                  "left_pupil").n_timepoints for lab in labels}
        self.assertEqual(lengths, {1250})


class TestGroupLevelSegmentedData(unittest.TestCase):

    def setUp(self):
        self.n_timepoints, self.n_segments = 100, 20
        self.tx = np.linspace(-200, 800, self.n_timepoints)
        self.segments = []
        for i in range(3):
            mask = np.zeros((self.n_timepoints, self.n_segments), dtype=bool)
            mask[0:5, :] = True  # first 5 time points masked
            self.segments.append(make(np.random.randn(self.n_timepoints, self.n_segments) + i * 10,
                                      self.tx, name=f"subject_{i + 1}", mask=mask))

    def tearDown(self):
        plt.close("all")

    def test_from_segments_aggregate(self):
        group = GroupLevelSegmentedData.from_segments(self.segments, meanfct=np.mean)
        self.assertEqual(group.n_subjects, 3)
        self.assertEqual(group.subjects, ["subject_1", "subject_2", "subject_3"])
        self.assertFalse(group.stacked)
        self.assertEqual(group.n_segments, 3)
        self.assertEqual(group.channels, ["left_pupil"])
        self.assertIsNone(group.epochs)

    def test_from_segments_stack(self):
        group = GroupLevelSegmentedData.from_segments(self.segments, meanfct=None)
        self.assertEqual(group.n_subjects, 3)
        self.assertTrue(group.stacked)
        self.assertEqual(group.n_segments, self.n_segments * 3)
        np.testing.assert_array_equal(group.subject, np.repeat([0, 1, 2], self.n_segments))

    def test_mask_percent_computed(self):
        group = GroupLevelSegmentedData.from_segments(self.segments)
        mp = group.mask_percent["left_pupil"]
        self.assertEqual(mp.shape, (self.n_timepoints,))
        np.testing.assert_array_almost_equal(mp[0:5], 100.0)
        np.testing.assert_array_almost_equal(mp[5:], 0.0)
        self.assertEqual(group.subject_mask_percent["left_pupil"].shape, (self.n_timepoints, 3))

    def junk_segments(self):
        """Two subjects, value 10 everywhere except masked junk (padding 0, blink 999)."""
        segs = []
        for name in ["a", "b"]:
            d = np.full((5, 4), 10.)
            m = np.zeros((5, 4), dtype=bool)
            d[1, 0], m[1, 0] = 0., True
            d[3, 1], m[3, 1] = 999., True
            segs.append(SegmentedEyeData(name=name, tx=np.arange(5.),
                                         data={"left_pupil": ma.masked_array(d, mask=m)},
                                         sampling_rate=1000))
        return segs

    def test_stack_keeps_mask(self):
        group = GroupLevelSegmentedData.from_segments(self.junk_segments(), meanfct=None)
        data = group["left_pupil"]
        self.assertEqual(data.shape, (5, 8))
        self.assertEqual(int(data.mask.sum()), 4)
        np.testing.assert_array_equal(np.mean(data, axis=1), 10.)

    def test_aggregate_keeps_mask(self):
        segs = self.junk_segments()
        segs[1]["left_pupil"][2, :] = ma.masked
        group = GroupLevelSegmentedData.from_segments(segs, meanfct=np.mean)
        np.testing.assert_array_equal(group["left_pupil"].mask[2], [False, True])
        np.testing.assert_array_equal(np.mean(group["left_pupil"], axis=1), 10.)

    def test_warns_for_mask_ignoring_meanfct(self):
        from loguru import logger
        messages = []
        hid = logger.add(lambda m: messages.append(m), level="WARNING")
        try:
            GroupLevelSegmentedData.from_segments(self.junk_segments(), meanfct=np.ma.median)
            self.assertEqual(messages, [])
            GroupLevelSegmentedData.from_segments(self.junk_segments(), meanfct=np.median)
            self.assertEqual(len(messages), 1)
            self.assertIn("does not ignore masked", messages[0])
        finally:
            logger.remove(hid)

    def test_function_without_axis_raises_clear_error(self):
        segs = self.junk_segments()
        no_axis = lambda x: np.mean(x)
        with self.assertRaisesRegex(TypeError, "must accept an `axis`"):
            GroupLevelSegmentedData.from_segments(segs, meanfct=no_axis)
        group = GroupLevelSegmentedData.from_segments(segs)
        for obj in (segs[0], group):
            with self.assertRaisesRegex(TypeError, "meanfct.*must accept an `axis`"):
                obj.plot(meanfct=no_axis)
            with self.assertRaisesRegex(TypeError, "varfct.*must accept an `axis`"):
                obj.plot(varfct=no_axis)
            plt.close("all")
        GroupLevelSegmentedData.from_segments(segs, meanfct=lambda x, **kw: np.mean(x, **kw))

    def test_plot_warns_for_mask_ignoring_functions(self):
        from loguru import logger
        segs = self.junk_segments()
        group = GroupLevelSegmentedData.from_segments(segs, meanfct=None)
        messages = []
        hid = logger.add(lambda m: messages.append(m), level="WARNING")
        try:
            for obj in (segs[0], group):
                obj.plot()
                obj.plot(meanfct=np.ma.median, varfct=np.std)
                self.assertEqual(messages, [])
                obj.plot(meanfct=np.median, varfct=lambda a, axis: np.percentile(a, 50, axis=axis))
                self.assertEqual(len(messages), 2)
                messages.clear()
                plt.close("all")
        finally:
            logger.remove(hid)

    def test_summary(self):
        summary = GroupLevelSegmentedData.from_segments(self.segments).summary()
        self.assertEqual(summary['n_subjects'], 3)
        self.assertFalse(summary['stacked'])
        self.assertAlmostEqual(summary['percent_masked']['left_pupil'], 5.0)

    def test_repr(self):
        r = repr(GroupLevelSegmentedData.from_segments(self.segments))
        self.assertIn("GroupLevelSegmentedData", r)
        self.assertIn("3 subjects", r)

    def test_repr_html(self):
        html = GroupLevelSegmentedData.from_segments(self.segments)._repr_html_()
        self.assertIn("GroupLevelSegmentedData", html)
        self.assertIn("Subjects", html)

    def test_plot(self):
        group = GroupLevelSegmentedData.from_segments(self.segments)
        fig, ax = plt.subplots()
        group.plot(ax=ax)
        plt.figure()
        ax = group.pupil.plot()
        self.assertEqual(ax.figure.axes[1].get_ylabel(), "avg % missing")

    def test_missing_channel(self):
        bad = make(np.random.randn(self.n_timepoints, self.n_segments), self.tx, "right_pupil")
        with self.assertRaisesRegex(KeyError, "no channel.*left_pupil"):
            GroupLevelSegmentedData.from_segments([self.segments[0], bad])

    def test_channels(self):
        segs = [SegmentedEyeData(f"s{i}", self.tx, {"left_pupil": np.ones((100, 3)),
                                                    "left_x": np.ones((100, 3))})
                for i in range(2)]
        self.assertEqual(group_segments(segs).channels, ["left_pupil", "left_x"])
        group = group_segments(segs, channels="left_x")
        self.assertEqual(group.channels, ["left_x"])
        self.assertEqual(list(group.subject_mask_percent), ["left_x"])
        self.assertEqual(list(group_segments(segs).drop_channels("left_x").subject_mask_percent),
                         ["left_pupil"])

    def test_validation_timepoints_mismatch(self):
        bad = make(np.random.randn(50, self.n_segments), np.linspace(-200, 800, 50))
        with self.assertRaises(ValueError):
            GroupLevelSegmentedData.from_segments([self.segments[0], bad])

    def test_validation_empty_list(self):
        with self.assertRaises(ValueError):
            GroupLevelSegmentedData.from_segments([])

    def test_names(self):
        self.assertEqual(GroupLevelSegmentedData.from_segments(self.segments, name="my_group").name,
                         "my_group")
        self.assertEqual(GroupLevelSegmentedData.from_segments(self.segments).name,
                         "group_subject_1")
        # segments with the same name: subjects are numbered
        same = [make(np.ones((5, 2)), np.arange(5.)) for _ in range(2)]
        self.assertEqual(group_segments(same).subjects, ["1", "2"])


class TestGroupSelection(unittest.TestCase):

    def setUp(self):
        d = gaze_eyedata()
        ep = d.get_epochs("", window=(-100, 500), units="ms")
        self.seg = d.get_segments(ep.set_meta("cond", ["easy", "hard", "easy", "hard"]))
        x = self.seg["left_pupil"].copy()
        x[:, 0] = ma.masked  # trial 1 all missing
        self.seg["left_pupil"] = x

    def test_stacked_epochs_with_subject(self):
        group = group_segments({"s1": self.seg, "s2": self.seg.keep(np.array([1, 1, 0, 1], bool))},
                               meanfct=None)
        self.assertEqual(len(group.epochs), 7)
        self.assertEqual(list(group.epochs.meta["subject"]), ["s1"] * 4 + ["s2"] * 3)
        self.assertEqual(list(group.epochs.meta["cond"]), ["easy", "hard", "easy", "hard",
                                                           "easy", "hard", "hard"])

    def test_stacked_filter(self):
        group = group_segments({"s1": self.seg, "s2": self.seg}, meanfct=None)
        easy = group.filter(cond="easy")
        self.assertEqual(easy.n_segments, 4)
        self.assertEqual(easy.subjects, ["s1", "s2"])
        np.testing.assert_array_equal(easy.subject, [0, 0, 1, 1])
        # masked trials per subject are recomputed: one of two easy trials is missing
        np.testing.assert_allclose(easy.mask_percent["left_pupil"], 50.)
        np.testing.assert_allclose(group.mask_percent["left_pupil"], 25.)
        s2 = group.filter(subject="s2")
        self.assertEqual(s2.subjects, ["s2"])
        np.testing.assert_array_equal(s2.subject, 0)

    def test_aggregate_keep_selects_subjects(self):
        group = group_segments({"s1": self.seg, "s2": self.seg.filter(cond="hard")})
        np.testing.assert_allclose(group.mask_percent["left_pupil"], 12.5)
        s2 = group.keep(np.array([False, True]))
        self.assertEqual(s2.subjects, ["s2"])
        np.testing.assert_allclose(s2.mask_percent["left_pupil"], 0.)
        with self.assertRaisesRegex(ValueError, "no epochs"):
            group.filter(cond="easy")

    def test_new_channel_of_stacked_group(self):
        group = group_segments({"s1": self.seg, "s2": self.seg}, meanfct=None)
        group = group.gaze.roi_hits(pp.RectangularROI((0, 0), (200, 200), name="c"),
                                    inplace=False)
        self.assertEqual(group.subject_mask_percent["left_c"].shape, (group.n_timepoints, 2))


class TestGroupSegmentsFunction(unittest.TestCase):

    def setUp(self):
        tx = np.linspace(-100, 400, 50)
        self.segments = [make(np.random.randn(50, 10), tx, name=f"subj_{i}") for i in range(3)]

    def test_from_list(self):
        group = group_segments(self.segments)
        self.assertIsInstance(group, GroupLevelSegmentedData)
        self.assertEqual(group.n_subjects, 3)

    def test_from_dict(self):
        group = group_segments({f"S{i:02d}": seg for i, seg in enumerate(self.segments)})
        self.assertEqual(group.subjects, ["S00", "S01", "S02"])

    def test_stack_mode(self):
        self.assertEqual(group_segments(self.segments, meanfct=None).n_segments, 30)

    def test_custom_name(self):
        self.assertEqual(group_segments(self.segments, name="my_group").name, "my_group")

    def test_available_via_pp(self):
        self.assertTrue(callable(pp.group_segments))


if __name__ == '__main__':
    unittest.main()
