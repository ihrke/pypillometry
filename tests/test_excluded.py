import os
import tempfile
import unittest

import numpy as np

import pypillometry as pp
from pypillometry import u


def _make_data(n=10000, fs=500, t0=5000.0, seed=1):
    """EyeData starting at `t0` ms with a few blinks (NaN periods)."""
    rng = np.random.default_rng(seed)
    t = t0 + np.arange(n) / fs * 1000
    pupil = 1000 + 50 * np.sin(t / 1000) + rng.normal(0, 1, n)
    for start in (1000, 3000, 6000, 8000):
        pupil[start:start + 100] = np.nan
    x = 500 + rng.normal(0, 0.5, n)
    y = 400 + rng.normal(0, 0.5, n)
    return pp.EyeData(time=t, left_x=x, left_y=y, left_pupil=pupil,
                      sampling_rate=fs, name="test", inplace=False)


class TestExcluded(unittest.TestCase):
    def test_exclude_masks_and_stores(self):
        d = _make_data()
        self.assertEqual(len(d.get_excluded()), 0)
        e = d.exclude_intervals([(5000, 7000)])  # first 2 s (1000 samples)
        self.assertEqual(len(d.get_excluded()), 0)  # not in place
        np.testing.assert_array_equal(e.get_excluded().intervals, [(5000, 7000)])
        self.assertEqual(e.get_excluded("samples").intervals, [(0, 1000)])
        self.assertAlmostEqual(e.excluded_duration("sec"), 2.0)
        for key in e.data.mask:
            self.assertTrue(e.data.mask[key][:1000].all())
            self.assertFalse(e.data.mask[key][1200:1250].any())
        # the data themselves are not changed
        np.testing.assert_array_equal(e.data["left_x"], d.data["left_x"])

    def test_recording_wide_layer(self):
        e = _make_data().exclude_intervals([(5000, 7000)])
        np.testing.assert_array_equal(e.data.get_rec_layer("excluded"),
                                      np.arange(10000) < 1000)
        # variables added later are masked in the excluded periods, too
        e["left", "baseline"] = np.ones(10000)
        self.assertTrue(e.data.mask["left_baseline"][:1000].all())
        self.assertFalse(e.data.mask["left_baseline"][1000:].any())
        # interpolated samples stay masked by default if excluded
        e.data.set_layer("left_x", "interpolated", np.ones(10000, dtype=bool))
        np.testing.assert_array_equal(e.mask("left", "x"), np.arange(10000) < 1000)

    def test_merge_and_units(self):
        d = _make_data()
        d = d.exclude_intervals([(5000, 6000), (5500, 7000)])
        d = d.exclude_intervals([(10 * u.sec, 11 * u.sec)])
        np.testing.assert_array_equal(d.get_excluded().intervals,
                                      [(5000, 7000), (10000, 11000)])
        np.testing.assert_array_equal(d.get_excluded("sec").intervals, [(5, 7), (10, 11)])
        # Intervals in sample indices
        d = _make_data().exclude_intervals(pp.Intervals([(500, 1000)], "samples"))
        np.testing.assert_array_equal(d.get_excluded().intervals, [(6000, 7000)])

    def test_intervals_from_events(self):
        d = _make_data()
        d.set_events(pp.Events([7000, 12000], ["break", "break"]))
        iv = d.get_intervals("break", window=(0, 1000), units="ms")
        d = d.exclude_intervals(iv)
        np.testing.assert_array_equal(d.get_excluded().intervals,
                                      [(7000, 8000), (12000, 13000)])
        # periods between two events
        d = _make_data()
        d.set_events(pp.Events([7000, 9000], ["break_start", "break_end"]))
        d = d.exclude_intervals(d.get_intervals(("break_start", "break_end"),
                                                window=(0, 0), units="ms"))
        np.testing.assert_array_equal(d.get_excluded().intervals, [(7000, 9000)])

    def test_blinks(self):
        d = _make_data()
        n_all = d.pupil.blinks_detect().nblinks()["left_pupil"]
        self.assertEqual(n_all, 4)
        # blink at samples 3000-3100 (11000-11200 ms) is excluded
        e = d.exclude_intervals([(10900, 11500)])
        e = e.pupil.blinks_detect()
        self.assertEqual(len(e.get_blinks("left", "pupil", units="samples")["left_pupil"]), 3)
        # a blink that ends in an excluded period is cut there
        g = d.exclude_intervals(pp.Intervals([(3050, 3500)], "samples")).pupil.blinks_detect()
        blinks = g.get_blinks("left", "pupil", units="samples")["left_pupil"].intervals
        self.assertEqual(len(blinks), 4)
        self.assertEqual(blinks[1][1], 3050)
        # excluding after detection keeps the blinks, also one that only
        # touches the excluded period
        f = d.pupil.blinks_detect()
        on, off = f.get_blinks("left", "pupil", units="samples")["left_pupil"].intervals[1]
        f = f.exclude_intervals(pp.Intervals([(off, off + 50)], "samples"))
        self.assertEqual(len(f.get_blinks("left", "pupil", units="samples")["left_pupil"]), 4)
        self.assertEqual(f.nblinks()["left_pupil"], 4)

    def test_blink_next_to_excluded_keeps_its_layers(self):
        d = _make_data().pupil.blinks_detect().pupil.blinks_interpolate()
        on, off = d.get_blinks("left", "pupil", units="samples")["left_pupil"].intervals[1]
        interp = d.get_interpolated_blinks("left", "pupil", units="samples")["left_pupil"].intervals
        mask = d.data.mask["left_pupil"].copy()
        e = d.exclude_intervals(pp.Intervals([(off, off + 50)], "samples"))
        # still a blink, masked and interpolated
        self.assertEqual(len(e.get_blinks("left", "pupil", units="samples")["left_pupil"]), 4)
        self.assertEqual(len(e.blinks["left_pupil"]), 4)
        self.assertTrue(e.data.get_layer("left_pupil", "blink")[on:off].all())
        self.assertEqual(e.get_interpolated_blinks("left", "pupil", units="samples")["left_pupil"].intervals, interp)
        self.assertTrue(e.data.mask["left_pupil"][mask.astype(bool)].all())

    def test_interpolation_leaves_excluded(self):
        d = _make_data().exclude_intervals([(10900, 11500)])
        before = d.data["left_pupil"].copy()
        d = d.pupil.blinks_detect().pupil.blinks_interpolate()
        excl = d._excluded_mask()
        np.testing.assert_array_equal(d.data["left_pupil"][excl], before[excl])
        self.assertTrue(np.isnan(d.data["left_pupil"][3000:3100]).all())

    def test_time_changes(self):
        d = _make_data().exclude_intervals([(7000, 8000)])
        r = d.reset_time()
        np.testing.assert_array_equal(r.get_excluded().intervals, [(2000, 3000)])
        self.assertEqual(r.get_excluded("samples").intervals, d.get_excluded("samples").intervals)
        s = d.crop(6000, 20000, units="ms")
        np.testing.assert_array_equal(s.get_excluded().intervals, [(7000, 8000)])
        self.assertEqual(s.get_excluded("samples").intervals, [(500, 1000)])
        self.assertTrue(s.data.mask["left_x"][500:1000].all())
        # period outside of the remaining data: cut with the samples
        s2 = d.crop(9000, 20000, units="ms")
        self.assertEqual(len(s2.get_excluded()), 0)
        self.assertEqual(s2.excluded_duration("sec"), 0)
        ds = d.downsample(100)
        self.assertEqual(ds._excluded_mask().sum(), 100)
        # the runs of the layer, at the times of the new samples
        np.testing.assert_allclose(ds.get_excluded().intervals, [(7004, 8004)])
        self.assertEqual(len(d.clear_layer("excluded").get_excluded()), 0)
        c = d.copy()
        np.testing.assert_array_equal(c.get_excluded().intervals, [(7000, 8000)])

    def test_pickle_and_old_objects(self):
        d = _make_data().exclude_intervals([(7000, 8000)])
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "d.pypil")
            pp.write_pickle(d, fname)
            r = pp.read_pickle(fname)
        np.testing.assert_array_equal(r.get_excluded().intervals, [(7000, 8000)])
        # HDF5
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "d.h5")
            d.write_file(fname)
            r = pp.EyeData.from_file(fname)
        np.testing.assert_array_equal(r.get_excluded().intervals, [(7000, 8000)])
        # files saved before excluded periods existed
        r = pp.EyeData.from_file(os.path.join(os.path.dirname(__file__), "data", "state",
                                              "eyedata_v1.pypil"))
        self.assertEqual(len(r.get_excluded()), 0)
        self.assertFalse(r._excluded_mask().any())

    def test_drop_data_and_summary(self):
        d = _make_data().exclude_intervals([(7000, 8000)])
        p = d.drop_data(variables=["x", "y"], inplace=False)
        np.testing.assert_array_equal(p.get_excluded().intervals, [(7000, 8000)])
        self.assertTrue(p.data.mask["left_pupil"][1000:1500].all())
        g = d.drop_data(variables="pupil", inplace=False)
        np.testing.assert_array_equal(g.get_excluded().intervals, [(7000, 8000)])
        self.assertAlmostEqual(d.summary()["excluded_minutes"], 1 / 60)
        self.assertNotIn("excluded_minutes", _make_data().summary())
        self.assertIn("Excluded", d._repr_html_())


class TestIncludedSplitRemove(unittest.TestCase):
    def make(self):
        # 20 s at 500 Hz from 5000 ms; excluded 7-8 s and 12-13 s
        d = _make_data()
        d.set_events(pp.Events([6000, 7500, 9000, 12500, 20000],
                               ["a", "x", "b", "y", "c"]))
        d = d.pupil.blinks_detect()
        return d.exclude_intervals([(7000, 8000), (12000, 13000)])

    def test_get_included(self):
        d = self.make()
        self.assertEqual(d.get_included().intervals,
                         [(5000., 7000.), (8000., 12000.), (13000., 25000.)])
        self.assertEqual(d.get_included("samples").intervals,
                         [(0, 1000), (1500, 3500), (4000, 10000)])
        self.assertEqual(_make_data().get_included("samples").intervals, [(0, 10000)])

    def test_split_excluded(self):
        d = self.make()
        parts = d.split_excluded()
        self.assertEqual([len(p) for p in parts], [1000, 2000, 6000])
        self.assertEqual([p.tx[0] for p in parts], [5000., 8000., 13000.])
        self.assertEqual([list(p.events.labels) for p in parts], [["a"], ["b"], ["c"]])
        # the blink in the excluded period 7-8 s goes with it
        self.assertEqual(sum(len(p.get_blinks("left", "pupil", units="samples")["left_pupil"]) for p in parts),
                         len(d.get_blinks("left", "pupil", units="samples")["left_pupil"]) - 1)
        for p in parts:
            self.assertEqual(p.excluded_duration(), 0)
        self.assertEqual(len(d), 10000)  # not changed

    def test_remove_excluded(self):
        d = self.make()
        r = d.remove_intervals(d.get_excluded(), inplace=False)
        self.assertEqual(len(r), 9000)
        # time re-aligned: no gaps, starts at the same time
        self.assertEqual(r.tx[0], 5000.)
        np.testing.assert_allclose(np.diff(r.tx), 2.)
        # events shifted with the data, those in removed periods deleted
        self.assertEqual(list(r.events.labels), ["a", "b", "c"])
        np.testing.assert_allclose(r.events.onsets, [6000, 8000, 18000])
        # data and blinks at their new positions
        np.testing.assert_array_equal(r.data["left_x"][1000:3000], d.data["left_x"][1500:3500])
        bl_d = d.get_blinks("left", "pupil", units="samples")["left_pupil"].to_units("ms", d).intervals
        bl_r = r.get_blinks("left", "pupil", units="samples")["left_pupil"].to_units("ms", r).intervals
        self.assertEqual(len(bl_r), len(bl_d) - 1)  # one blink was in 7-8 s
        self.assertEqual(bl_r[-1], (bl_d[-1][0] - 2000, bl_d[-1][1] - 2000))
        # nothing excluded or masked for exclusion any more
        self.assertEqual(len(r.get_excluded()), 0)
        self.assertEqual(r.params["removed_intervals"], [(7000., 8000.), (12000., 13000.)])
        self.assertFalse(r.data.mask["left_x"].any())
        # nothing to remove: nothing changes
        e = _make_data()
        self.assertEqual(len(e.remove_intervals(e.get_excluded())), len(e))
        with self.assertRaises(ValueError):
            _make_data().remove_intervals([(0, 1e6)])

    def test_remove_intervals(self):
        d = self.make()
        # remove one period: the other excluded period moves with the data
        r = d.remove_intervals([(7 * u.sec, 8 * u.sec)], inplace=False)
        self.assertEqual(r.get_excluded().intervals, [(11000., 12000.)])
        self.assertTrue(r.data.mask["left_x"][3000:3500].all())
        # sample indices work as well
        r2 = d.remove_intervals(pp.Intervals([(1000, 1500)], "samples"), inplace=False)
        np.testing.assert_array_equal(r2.tx, r.tx)
        # removing the beginning and end does not shift: same as crop()
        e = _make_data()
        e.set_events(pp.Events([6000, 9000, 20000], ["a", "b", "c"]))
        r = e.remove_intervals([(-np.inf, 7000), (15000, np.inf)], inplace=False)
        c = e.crop(7000, 15000 - 2, inplace=False)
        np.testing.assert_array_equal(r.tx, c.tx)
        np.testing.assert_array_equal(r.data["left_pupil"], c.data["left_pupil"])
        np.testing.assert_array_equal(r.events.onsets, c.events.onsets)

    def test_sub_slice_deprecated(self):
        d = _make_data()
        with self.assertWarns(DeprecationWarning):
            s = d.sub_slice(6000, 8000)
        np.testing.assert_array_equal(s.tx, d.crop(6000, 8000).tx)

if __name__ == "__main__":
    unittest.main()


class TestKeepIntervals(unittest.TestCase):
    # _make_data(): samples at 5000, 5002, ..., 24998 ms (end 25000)

    def test_keep_one_period(self):
        d = _make_data().keep_intervals([(7000, 9000)])
        np.testing.assert_array_equal(d.get_excluded().intervals,
                                      [(5000, 7000), (9000, 25000)])
        np.testing.assert_array_equal(d.get_included().intervals, [(7000, 9000)])
        self.assertEqual(d.get_included("samples").intervals, [(1000, 2000)])
        for key in d.data.mask:
            self.assertTrue(d.data.mask[key][:1000].all())
            self.assertTrue(d.data.mask[key][2000:].all())

    def test_keep_blocks_from_events(self):
        d = _make_data()
        d.set_events(pp.Events([7000, 9000, 12000, 15000],
                               ["Start Block", "End Block"] * 2))
        blocks = d.get_intervals(("Start Block", "End Block"), window=(0, 0))
        d = d.keep_intervals(blocks)  # in sample indices
        np.testing.assert_array_equal(d.get_included().intervals,
                                      [(7000, 9000), (12000, 15000)])

    def test_same_as_exclude_complement(self):
        d = _make_data()
        iv = pp.Intervals([(6 * u.sec, 8 * u.sec)])
        a = d.keep_intervals(iv)
        b = d.exclude_intervals(iv.complement(d))
        np.testing.assert_array_equal(a.get_excluded().intervals, b.get_excluded().intervals)

    def test_keeps_existing_exclusions(self):
        d = _make_data().exclude_intervals([(7500, 8000)])
        d = d.keep_intervals([(7000, 9000)])
        np.testing.assert_array_equal(d.get_included().intervals,
                                      [(7000, 7500), (8000, 9000)])

    def test_keep_everything(self):
        d = _make_data().keep_intervals([(0, 30000)])
        self.assertEqual(len(d.get_excluded()), 0)

    def test_history(self):
        d = _make_data().keep_intervals([(7000, 9000)])
        names = [h["funcname"] for h in d.history]
        self.assertEqual(names[-1], "keep_intervals")
        self.assertNotIn("exclude_intervals", names)
