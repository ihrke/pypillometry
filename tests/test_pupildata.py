import unittest
import sys
import numpy as np
sys.path.insert(0,"..")
import pypillometry as pp
import pytest

class TestPupilData(unittest.TestCase):
    def setUp(self):
        #self.d=pp.PupilData.from_file("data/test.pd")
        pass
    def test_from_file(self):
        d = pp.get_example_data("rlmw_002_short")
        print(d)
    def test_scale(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        d=d.scale("pupil")
        self.assertEqual(d.data["left","pupil"].mean(), 0)
        self.assertEqual(d.data["right","pupil"].mean(), 0)
        self.assertEqual(d.params["scale"]["mean"]["left"]["pupil"], 1.5)
        self.assertEqual(d.params["scale"]["mean"]["right"]["pupil"], 3.5)
    def test_scale_ignores_masked(self):
        # masked (non-NaN) and NaN samples do not enter mean/sd
        d = pp.PupilData(sampling_rate=10, left_pupil=[1., 3., 1., 3., 999., np.nan])
        d.data.mask["left_pupil"][4] = 1
        d = d.scale("pupil", inplace=False)
        self.assertEqual(d.params["scale"]["mean"]["left"]["pupil"], 2.)
        self.assertEqual(d.params["scale"]["sd"]["left"]["pupil"], 1.)
    def test_scale_masked_samples(self):
        # masked samples are scaled as well and stay masked
        d = pp.PupilData(sampling_rate=10, left_pupil=[1., 3., 1., 3., 5.])
        d.data.mask["left_pupil"][4] = 1
        s = d.scale("pupil", inplace=False)
        np.testing.assert_allclose(s.data["left", "pupil"], [-1, 1, -1, 1, 3])
        self.assertEqual(s.data.mask["left_pupil"][4], 1)
        u = s.unscale("pupil", inplace=False)
        np.testing.assert_allclose(u.data["left", "pupil"], [1, 3, 1, 3, 5])
        self.assertEqual(u.data.mask["left_pupil"][4], 1)
    def test_scale_fixed(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        scalepars = {
            "left": {"pupil": 1},
            "right": {"pupil": 3}
        }
        sdpars = {
            "left": {"pupil": 0.5},
            "right": {"pupil": 0.5}
        }
        d=d.scale("pupil", mean=scalepars, sd=sdpars)
        self.assertEqual(d.data["left","pupil"].mean(), 1)
        self.assertEqual(d.data["right","pupil"].mean(), 1)
    def test_unscale(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        d=d.scale("pupil")
        d=d.unscale("pupil")
        self.assertEqual(d.data["left","pupil"].mean(), 1.5)
        self.assertEqual(d.data["right","pupil"].mean(), 3.5)
    def test_scale_unscale_numbers(self):
        # scale() and unscale() accept ints and floats
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        for mean, sd in [(1, 2), (1.0, 2.0), (np.float32(1), np.int64(2))]:
            s = d.scale("pupil", mean=mean, sd=sd, inplace=False)
            self.assertEqual(s.data["left","pupil"].mean(), 0.25)
            u = s.unscale("pupil", mean=mean, sd=sd, inplace=False)
            np.testing.assert_allclose(u.data["left","pupil"], [1,2,1,2])

    def test_constructor_history_empty(self):
        # filling time gaps in the constructor is not a user operation
        d = pp.PupilData(sampling_rate=10, left_pupil=[1., 2., 3.])
        self.assertEqual(d.history, [])
        d.fill_time_discontinuities()
        self.assertEqual([h["funcname"] for h in d.history],
                         ["fill_time_discontinuities"])

    def test_fill_time_discontinuities(self):
        # three segments sampled at 2 ms, separated by gaps of 500 ms
        segs = [np.arange(0, 1000, 2), np.arange(1500, 2500, 2), np.arange(3000, 4000, 2)]
        t = np.concatenate(segs).astype(float)
        x = np.concatenate([np.full(len(s), i+1.) for i, s in enumerate(segs)])
        d = pp.PupilData(left_pupil=x, time=t)
        self.assertEqual(len(d.tx), 2000)
        self.assertEqual(len(d.data["left_pupil"]), len(d.tx))
        np.testing.assert_allclose(np.diff(d.tx), 2.)
        # original samples are kept at their time points, gaps are filled
        # with NaN (masked)
        for i, s in enumerate(segs):
            ix = np.isin(d.tx, s)
            self.assertEqual(ix.sum(), len(s))
            np.testing.assert_array_equal(d.data["left_pupil"][ix], i+1.)
        self.assertEqual(np.isnan(d.data["left_pupil"]).sum(), 2000-len(t))
        self.assertEqual(d.data.mask["left_pupil"].sum(), 2000-len(t))


    def test_estimate_baseline_nan_raises_clear_error(self):
        d = pp.get_example_data("rlmw_002_short").sub_slice(0, 10, units="sec")
        with self.assertRaisesRegex(ValueError, "pupil_blinks_interpolate"):
            d.pupil_estimate_baseline(eyes="left", inplace=False)

    def test_estimate_baseline_after_interpolation(self):
        d = pp.get_example_data("rlmw_002_short").sub_slice(0, 10, units="sec")
        d = d.pupil_blinks_detect(eyes="left", apply_mask=True, inplace=False)
        d = d.pupil_blinks_interpolate(eyes="left", inplace=False)
        d = d.pupil_estimate_baseline(eyes="left", inplace=False)
        self.assertFalse(np.any(np.isnan(d.data["left", "baseline"])))

    def test_estimate_baseline_seed(self):
        """same seed gives the same baseline"""
        d = pp.get_example_data("rlmw_002_short").sub_slice(0, 10, units="sec")
        d = d.pupil_blinks_detect(eyes="left", apply_mask=True, inplace=False)
        d = d.pupil_blinks_interpolate(eyes="left", inplace=False)
        b1 = d.pupil_estimate_baseline(eyes="left", inplace=False, seed=1)
        b2 = d.pupil_estimate_baseline(eyes="left", inplace=False, seed=1)
        np.testing.assert_array_equal(b1.data["left", "baseline"], b2.data["left", "baseline"])

    def test_stan_model_cached(self):
        from unittest import mock
        from pypillometry.signal import baseline
        with mock.patch.dict(baseline._stan_models, clear=True), \
             mock.patch("cmdstanpy.CmdStanModel") as model:
            m1 = baseline._get_stan_model("baseline_model_asym_laplac")
            m2 = baseline._get_stan_model("baseline_model_asym_laplac")
        self.assertIs(m1, m2)
        self.assertEqual(model.call_count, 1)

    def test_stan_model_read_only_install(self):
        """without writable package dir and executable, compile in the cache"""
        import os, shutil, tempfile, pathlib
        from unittest import mock
        from pypillometry.signal import baseline
        name = "baseline_model_asym_laplac"
        with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as pkg:
            # package dir with the .stan file but without executable
            shutil.copy(baseline.files("pypillometry.stan").joinpath(name + ".stan"), pkg)
            with mock.patch.dict(baseline._stan_models, clear=True), \
                 mock.patch.dict(os.environ, {"XDG_CACHE_HOME": cache}), \
                 mock.patch.object(baseline, "files", return_value=pathlib.Path(pkg)), \
                 mock.patch.object(baseline.os, "access", return_value=False), \
                 mock.patch("cmdstanpy.CmdStanModel") as model:
                baseline._get_stan_model(name)
            stan_file = model.call_args.kwargs["stan_file"]
            self.assertTrue(stan_file.startswith(cache))
            self.assertTrue(os.path.isfile(stan_file))

    def test_blinks_interpolate_close_blinks(self):
        """close blinks are merged, so no NaN anchor points are used"""
        t = np.arange(6000)
        p = 5000 + 200*np.sin(t/800)
        for a, b in [(2000, 2200), (2300, 2400)]:
            p[a-30:a] = np.linspace(p[a-30], 1000, 30)
            p[b:b+30] = np.linspace(1000, p[b+30], 30)
            p[a:b] = np.nan
        d = pp.PupilData(left_pupil=p, sampling_rate=1000)
        d = d.pupil_blinks_detect(apply_mask=True, inplace=False)
        self.assertEqual(len(d.get_blinks("left", "pupil")), 2)
        d = d.pupil_blinks_interpolate(inplace=False)
        self.assertEqual(np.isnan(d.data["left", "pupil"]).sum(), 0)
        self.assertEqual(len(d.get_interpolated_blinks("left", "pupil")), 1)

if __name__ == '__main__':
    unittest.main()

class TestMergeBlinkWindows(unittest.TestCase):
    def merge(self, blinks, n=10000):
        from pypillometry.eyedata.pupildata import _merge_overlapping_blink_windows
        return _merge_overlapping_blink_windows(blinks, n)

    def test_separate_blinks(self):
        self.assertEqual(self.merge([(100, 110), (200, 210)]),
                         [(90, 100, 110, 120), (190, 200, 210, 220)])

    def test_overlapping_windows_merge(self):
        self.assertEqual(self.merge([(100, 110), (115, 125)]),
                         [(90, 100, 125, 135)])

    def test_no_cascade(self):
        # each blink's window only reaches its neighbour if blinks are close;
        # the merged blink must not widen the window and swallow the rest
        blinks = [(100, 200), (290, 390)] + [(1000 + 300*i, 1050 + 300*i)
                                             for i in range(20)]
        merged = self.merge(blinks)
        self.assertEqual(merged[0], (0, 100, 390, 490))
        self.assertEqual(len(merged), 21)


def test_blinks_interpolate_without_onset_crossing():
    """all blinks of the example data are interpolated, no NaN left (bug 1.1)"""
    import pypillometry as pp
    d = pp.get_example_data("rlmw_002_short")
    d = d.pupil_blinks_detect(apply_mask=True).pupil_blinks_interpolate()
    for eye in d.eyes:
        assert not np.isnan(d.data[eye, "pupil"]).any()
    d.pupil_estimate_baseline()
