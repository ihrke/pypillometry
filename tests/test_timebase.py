"""Time vector, sampling rate and the operations that change the samples."""
import unittest

import numpy as np

import pypillometry as pp


def make(n=1000, fs=1000.):
    """Pupil-only EyeData with `n` samples at `fs` Hz (sample i at i*1000/fs ms)."""
    tx = np.arange(n) * 1000. / fs
    return pp.EyeData(left_pupil=np.full(n, 10.), time=tx, sampling_rate=fs)


def actual_fs(d):
    return 1000. / np.median(np.diff(d.tx))


class TestConstructorTime(unittest.TestCase):

    def test_default_time_vector(self):
        # without `time`, sample i is at i*1000/fs ms
        d = pp.EyeData(left_pupil=np.ones(5), sampling_rate=1000)
        np.testing.assert_array_equal(d.tx, [0., 1., 2., 3., 4.])
        d = pp.EyeData(left_pupil=np.ones(4), sampling_rate=300)
        np.testing.assert_allclose(d.tx, np.arange(4) * 1000. / 300)

    def test_rate_must_match_time_vector(self):
        tx = np.arange(100) * 3.  # 333.3 Hz
        with self.assertRaisesRegex(ValueError, "does not match the time vector"):
            pp.EyeData(left_pupil=np.ones(100), time=tx, sampling_rate=300)
        with self.assertRaisesRegex(ValueError, "positive number"):
            pp.EyeData(left_pupil=np.ones(100), time=tx, sampling_rate=0)
        d = pp.EyeData(left_pupil=np.ones(100), time=tx, sampling_rate=1000. / 3)
        self.assertAlmostEqual(d.fs, 1000. / 3)
        # without a sampling rate, it is taken from the time vector
        self.assertEqual(pp.EyeData(left_pupil=np.ones(100), time=tx).fs, 333.)

    def test_tolerated_time_vectors(self):
        # 300 Hz with time stamps rounded to whole ms (steps of 3 and 4 ms)
        tx = np.floor(np.arange(300) * 1000. / 300)
        pp.EyeData(left_pupil=np.ones(300), time=tx, sampling_rate=300)
        # a gap in the time vector
        tx = np.r_[0:500, 800:1000] * 2.
        pp.EyeData(left_pupil=np.ones(700), time=tx, sampling_rate=500,
                     fill_time_discontinuities=False)
        # a single sample
        pp.EyeData(left_pupil=np.ones(1), time=[0.], sampling_rate=500)

    def test_setting_the_time_vector_is_checked(self):
        d = make(n=10)
        d["time_sec"] = np.arange(10) / 1000. + 5
        np.testing.assert_allclose(d.tx, np.arange(10) + 5000.)
        with self.assertRaisesRegex(ValueError, "does not match the time vector"):
            d["time"] = np.arange(10) * 2.


class TestDownsampleRate(unittest.TestCase):

    def test_integer_factor(self):
        x = make().downsample(250, inplace=False)
        self.assertEqual(x.fs, 250.)
        self.assertEqual(len(x), 250)

    def test_non_integer_ratio_stores_actual_rate(self):
        # 1000/300 -> factor 3 -> 333.3 Hz, not 300
        x = make().downsample(300, inplace=False)
        self.assertAlmostEqual(x.fs, 1000. / 3)
        self.assertAlmostEqual(x.fs, actual_fs(x))
        # 1000/400 -> factor 2 -> 500 Hz
        x = make().downsample(400, inplace=False)
        self.assertEqual(x.fs, 500.)
        self.assertAlmostEqual(x.fs, actual_fs(x))

    def test_factor(self):
        x = make().downsample(4, dsfac=True, inplace=False)
        self.assertEqual(x.fs, 250.)
        self.assertAlmostEqual(x.fs, actual_fs(x))


class TestDownsampleArguments(unittest.TestCase):

    def test_bad_rates(self):
        d = make()
        for fsd in (0, -10, 2000):
            with self.assertRaisesRegex(ValueError, "sampling rate in Hz"):
                d.downsample(fsd, inplace=False)
        with self.assertRaisesRegex(TypeError, "fsd must be a finite number"):
            d.downsample("100 Hz", inplace=False)
        for fsd in (np.inf, np.nan):
            with self.assertRaisesRegex(TypeError, "fsd must be a finite number"):
                d.downsample(fsd, inplace=False)

    def test_bad_factors(self):
        d = make()
        for fac in (0, -2, 2.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                d.downsample(fac, dsfac=True, inplace=False)

    def test_same_rate_is_a_copy(self):
        d = make()
        x = d.downsample(1000, inplace=False)
        self.assertEqual(x.fs, 1000.)
        np.testing.assert_array_equal(x.tx, d.tx)

    def test_float_noise_in_ratio(self):
        # 1000/(1000/15) is 14.999999999999998 in floating point
        d = make()
        x = d.downsample(1000. / 15, inplace=False)
        self.assertEqual(len(x), int(np.ceil(len(d) / 15)))
        self.assertAlmostEqual(x.fs, 1000. / 15)


class TestResetTime(unittest.TestCase):

    def make(self, units):
        from pypillometry.intervals import Intervals
        d = make(n=2000)
        d["time"] = d.tx + 1000.
        iv = Intervals([(300, 320)], units="samples").to_units(units, d)
        d.set_blinks(iv, eyes=["left"], variables=["pupil"], apply_mask=True)
        return d

    def test_blinks_follow_the_clock(self):
        for units in ("samples", "ms", "sec"):
            with self.subTest(units=units):
                d = self.make(units).reset_time(inplace=False)
                self.assertEqual(d.tx[0], 0.)
                b = d.get_blinks("left", "pupil")
                self.assertEqual(list(b.intervals), [(300, 320)])
                # blinks and mask agree
                np.testing.assert_array_equal(np.flatnonzero(b.to_mask(d)),
                                              np.flatnonzero(d.data.mask["left_pupil"]))


class TestFakeEyeDataGroundTruth(unittest.TestCase):

    def make(self):
        from pypillometry.signal.fake import generate_foreshortening_data
        d = generate_foreshortening_data(duration=4, fs=500, seed=1)
        # the measured pupil is the true pupil plus noise: keep the true one
        # in the data to check the alignment
        d.data["left_pupil"] = d.sim_data["left_A0"].copy()
        return d

    def check(self, d):
        for k in d.sim_data.keys():
            self.assertEqual(len(d.sim_data[k]), len(d))
        np.testing.assert_array_equal(d.sim_data["left_A0"], d.data["left_pupil"])

    def test_follows_crop_remove_downsample(self):
        d = self.make()
        x = d.crop(500, 2500, inplace=False)
        self.assertEqual(len(x), 1001)
        self.check(x)
        self.check(d.remove_intervals([(1000, 1500)], inplace=False))
        self.check(d.downsample(50, inplace=False))
        self.assertEqual(len(d.sim_data["left_A0"]), 2000)  # original unchanged

    def test_follows_fill_time_discontinuities(self):
        d = self.make()
        keep = np.ones(len(d), dtype=bool)
        keep[500:600] = False
        d._keep_samples(keep)  # time gap of 200 ms
        d.fill_time_discontinuities()
        self.assertEqual(len(d), 2000)
        self.check(d)
        self.assertTrue(np.isnan(d.sim_data["left_A0"][550]))


class TestRemovedIntervalsClock(unittest.TestCase):
    """params["removed_intervals"] is in the original time after several calls."""

    def make(self):
        # 100 Hz, 10 s; the pupil holds the original time of each sample
        d = make(n=1000, fs=100.)
        d.data["left_pupil"] = d.tx.copy()
        return d

    def check(self, d):
        # the original time of the samples is what the data say
        np.testing.assert_allclose(d._original_time(d.tx), d.data["left_pupil"])
        # the removed periods are the original times without data
        orig = np.arange(1000) * 10.
        kept = np.isin(orig, d.data["left_pupil"])
        removed = np.zeros(1000, dtype=bool)
        for a, b in d.params["removed_intervals"]:
            removed |= (orig >= a) & (orig < b)
        np.testing.assert_array_equal(removed, ~kept)

    def test_one_call(self):
        d = self.make().remove_intervals([(2000, 3000)], inplace=False)
        self.assertEqual(d.params["removed_intervals"], [(2000., 3000.)])
        self.assertEqual(d.params["removed_shifts"], [(0., 0.), (2000., 1000.)])
        self.check(d)

    def test_second_call_in_original_time(self):
        d = self.make().remove_intervals([(2000, 3000)], inplace=False)
        d.remove_intervals([(5000, 6000)], inplace=True)
        self.assertEqual(d.params["removed_intervals"], [(2000., 3000.), (6000., 7000.)])
        self.check(d)

    def test_period_across_a_join(self):
        d = self.make().remove_intervals([(2000, 3000)], inplace=False)
        d.remove_intervals([(1500, 2500)], inplace=True)
        self.assertEqual(d.params["removed_intervals"], [(1500., 3500.)])
        self.check(d)

    def test_start_removed_after_a_join(self):
        d = self.make().remove_intervals([(2000, 3000)], inplace=False)
        d.remove_intervals([(0, 2500)], inplace=True)
        self.assertEqual(d.tx[0], 2500.)  # the sample at original time 3500
        d.remove_intervals([(4000, 4500)], inplace=True)
        self.assertEqual(d.params["removed_intervals"], [(0., 3500.), (5000., 5500.)])
        self.check(d)

    def test_off_grid_periods(self):
        d = self.make().remove_intervals([(2005, 3005)], inplace=False)
        d.remove_intervals([(1995, 2003), (7001, 8001)], inplace=True)
        self.check(d)

    def test_no_shift_without_a_gap(self):
        d = self.make().remove_intervals([(0, 2000), (9000, np.inf)], inplace=False)
        self.assertNotIn("removed_shifts", d.params)
        self.check(d)

    def test_reset_time_between_calls(self):
        d = self.make()
        d["time"] = d.tx + 1000.
        d.data["left_pupil"] = d.tx - 1000.  # original time after reset_time
        d.remove_intervals([(3000, 4000)], inplace=True)
        d.reset_time(1000., inplace=True)
        self.assertEqual(d.params["removed_intervals"], [(2000., 3000.)])
        d.remove_intervals([(5000, 6000)], inplace=True)
        self.assertEqual(d.params["removed_intervals"], [(2000., 3000.), (6000., 7000.)])
        self.check(d)


class TestChains(unittest.TestCase):
    """Data, masks, blinks, excluded periods and events stay aligned after
    chains of crop, remove_intervals, fill_time_discontinuities, reset_time
    and downsample."""

    BLINKS = [(2100., 2300.), (8000., 8200.), (12000., 12100.)]
    EXCLUDED = [(14000., 15000.)]
    EVENTS = [500., 2500., 7000., 9000., 13000., 16000., 18500.]

    def make(self):
        from pypillometry.intervals import Intervals
        # 500 Hz, 20 s with a time gap at 10000-10400 ms; the pupil holds
        # the original time of each sample
        tx = np.arange(10000) * 2.
        tx = tx[(tx < 10000) | (tx >= 10400)]
        d = pp.EyeData(left_pupil=tx.copy(), right_pupil=tx.copy(), time=tx,
                         sampling_rate=500, fill_time_discontinuities=False,
                         event_onsets=self.EVENTS,
                         event_labels=[str(t) for t in self.EVENTS])
        blinks = Intervals(self.BLINKS, units="ms")
        d.set_blinks(blinks.to_units("samples", d), eyes="left", apply_mask=False)
        d.set_blinks(blinks, eyes="right", apply_mask=False)
        d.exclude_intervals(self.EXCLUDED, inplace=True)
        return d

    @staticmethod
    def inside(t, periods):
        return np.any([(t >= a) & (t < b) for a, b in periods], axis=0)

    def check(self, d, t0=0.):
        # original time (as the data store it): the original time of the
        # object plus `t0`, the sum of the reset_time shifts
        orig = d._original_time(d.tx) + t0
        for k in d.data.keys():
            self.assertEqual(len(d.data[k]), len(d))
        self.assertAlmostEqual(d.fs, 1000. / np.median(np.diff(d.tx)))
        # data (outside of filled-in gaps) at the original time
        valid = ~np.isnan(d.data["left_pupil"])
        np.testing.assert_allclose(d.data["left_pupil"][valid], orig[valid])
        # events at their original time
        np.testing.assert_allclose(d._original_time(d.events.onsets) + t0,
                                   [float(l) for l in d.events.labels])
        # blinks (stored in samples and in ms) and excluded periods
        for eye in ("left", "right"):
            np.testing.assert_array_equal(
                d.get_blinks(eye, "pupil").to_mask(d).astype(bool), self.inside(orig, self.BLINKS))
        np.testing.assert_array_equal(d.get_excluded().to_mask(d).astype(bool),
                                      self.inside(orig, self.EXCLUDED))
        np.testing.assert_array_equal(d.data.mask["left_pupil"].astype(bool),
                                      self.inside(orig, self.EXCLUDED) | ~valid)

    def test_chain(self):
        d = self.make()
        self.check(d)
        steps = [("fill_time_discontinuities", (), {}),
                 ("crop", (1000, 19000), {}),
                 ("remove_intervals", ([(5000, 6000)],), {}),
                 ("reset_time", (), {}),
                 ("remove_intervals", ([(2400, 2600), (10500, 11000)],), {}),
                 ("crop", (500, 17000), {}),
                 ("reset_time", (-250.,), {})]
        t0 = 0.
        for name, args, kw in steps:
            with self.subTest(step=name, args=args):
                if name == "reset_time":
                    t0 += args[0] if args else d.tx[0]
                getattr(d, name)(*args, **kw)
                self.check(d, t0)

    def test_downsample_at_the_end(self):
        d = self.make().fill_time_discontinuities()
        d.remove_intervals([(5000, 6000)], inplace=True)
        d.reset_time(500., inplace=True)
        x = d.downsample(250, inplace=False)
        self.assertEqual(x.fs, 250.)
        # each new sample holds the mean original time of its window
        orig = d._original_time(d.tx)
        n = len(orig) // 2 * 2
        means = orig[:n].reshape(-1, 2).mean(axis=1)
        same_shift = np.diff(orig[:n].reshape(-1, 2) - d.tx[:n].reshape(-1, 2)).ravel() == 0
        np.testing.assert_allclose(x._original_time(x.tx[:n // 2])[same_shift], means[same_shift])
        np.testing.assert_allclose(x._original_time(x.events.onsets) + 500.,
                                   [float(l) for l in x.events.labels])
        # blinks still cover the blinks
        for eye in ("left", "right"):
            ds = d.get_blinks(eye, "pupil").to_mask(d).astype(bool)
            xs = x.get_blinks(eye, "pupil").to_mask(x).astype(bool)
            self.assertTrue(np.all(xs[np.flatnonzero(ds) // 2]))


if __name__ == "__main__":
    unittest.main()
