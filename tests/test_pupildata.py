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
        d = pp.EyeData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        d=d.scale("pupil")
        self.assertEqual(d.data["left","pupil"].mean(), 0)
        self.assertEqual(d.data["right","pupil"].mean(), 0)
        self.assertEqual(d.params["scale"]["center"]["left_pupil"], 1.5)
        self.assertEqual(d.params["scale"]["center"]["right_pupil"], 3.5)
    def test_scale_ignores_masked(self):
        # masked (non-NaN) and NaN samples do not enter mean/sd
        d = pp.EyeData(sampling_rate=10, left_pupil=[1., 3., 1., 3., 999., np.nan])
        d.data.add_mask("left_pupil", np.s_[4])
        d = d.scale("pupil", inplace=False)
        self.assertEqual(d.params["scale"]["center"]["left_pupil"], 2.)
        self.assertEqual(d.params["scale"]["scale"]["left_pupil"], 1.)
    def test_scale_masked_samples(self):
        # masked samples are scaled as well and stay masked
        d = pp.EyeData(sampling_rate=10, left_pupil=[1., 3., 1., 3., 5.])
        d.data.add_mask("left_pupil", np.s_[4])
        s = d.scale("pupil", inplace=False)
        np.testing.assert_allclose(s.data["left", "pupil"], [-1, 1, -1, 1, 3])
        self.assertEqual(s.data.mask["left_pupil"][4], 1)
        u = s.unscale("pupil", inplace=False)
        np.testing.assert_allclose(u.data["left", "pupil"], [1, 3, 1, 3, 5])
        self.assertEqual(u.data.mask["left_pupil"][4], 1)
    def test_scale_fixed(self):
        d = pp.EyeData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        scalepars = {"left_pupil": 1, "right_pupil": 3}
        sdpars = {"left_pupil": 0.5, "right_pupil": 0.5}
        d=d.scale("pupil", center=scalepars, scale=sdpars)
        self.assertEqual(d.data["left","pupil"].mean(), 1)
        self.assertEqual(d.data["right","pupil"].mean(), 1)
        self.assertEqual(d.params["scale"], {"center": scalepars, "scale": sdpars,
                                             "method": {"left_pupil": "z", "right_pupil": "z"}})
    def test_scale_dict_by_channel(self):
        d = pp.EyeData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        with self.assertRaisesRegex(ValueError, "keyed by channel"):
            d.scale("pupil", center={"left": {"pupil": 1}, "right": {"pupil": 3}}, scale=1)
        with self.assertRaisesRegex(ValueError, "center has no value for right_pupil"):
            d.scale("pupil", center={"left_pupil": 1}, scale=1)
    def test_unscale(self):
        d = pp.EyeData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        d=d.scale("pupil")
        d=d.unscale("pupil")
        self.assertEqual(d.data["left","pupil"].mean(), 1.5)
        self.assertEqual(d.data["right","pupil"].mean(), 3.5)
    def test_scale_unscale_numbers(self):
        # scale() and unscale() accept ints and floats
        d = pp.EyeData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        for mean, sd in [(1, 2), (1.0, 2.0), (np.float32(1), np.int64(2))]:
            s = d.scale("pupil", center=mean, scale=sd, inplace=False)
            self.assertEqual(s.data["left","pupil"].mean(), 0.25)
            u = s.unscale("pupil", center=mean, scale=sd, inplace=False)
            np.testing.assert_allclose(u.data["left","pupil"], [1,2,1,2])

    def test_scale_methods(self):
        x = np.array([1., 2., 3., 4., 100.])
        d = pp.EyeData(sampling_rate=10, left_pupil=x)
        r = d.scale("pupil", method="robust", inplace=False)
        # median 3, raw MAD 1 (not multiplied by 1.4826)
        np.testing.assert_allclose(r.data["left_pupil"], x - 3)
        self.assertEqual(r.params["scale"]["method"], {"left_pupil": "robust"})
        m = d.scale("pupil", method="minmax", inplace=False)
        np.testing.assert_allclose(m.data["left_pupil"], (x - 1) / 99)
        pc = d.scale("pupil", method="percent", inplace=False)
        np.testing.assert_allclose(pc.data["left_pupil"], (x - 22) / 22 * 100)
        for s in (r, m, pc):
            np.testing.assert_allclose(s.unscale("pupil", inplace=False).data["left_pupil"], x)
    def test_scale_methods_use_policy(self):
        # the masked outlier does not enter min and max
        d = pp.EyeData(sampling_rate=10, left_pupil=[1., 2., 3., 100.])
        d.data.add_mask("left_pupil", np.s_[3])
        m = d.scale("pupil", method="minmax", inplace=False)
        np.testing.assert_allclose(m.data["left_pupil"], [0, 0.5, 1, 49.5])
    def test_scale_method_combined(self):
        d = pp.EyeData(sampling_rate=10, left_pupil=[1., 2., 3., 4.])
        d.scale("pupil").scale("pupil", method="minmax")
        self.assertEqual(d.params["scale"]["method"], {"left_pupil": "z, minmax"})
        np.testing.assert_allclose(d.unscale("pupil").data["left_pupil"], [1, 2, 3, 4])
    def test_scale_errors(self):
        d = pp.EyeData(sampling_rate=10, left_pupil=[1., 1., 1., 5.])
        with self.assertRaisesRegex(ValueError, "unknown method 'mad'"):
            d.scale(method="mad")
        with self.assertRaisesRegex(ValueError, "the scale is 0.0"):
            d.scale(method="robust")   # MAD 0
        with self.assertRaisesRegex(ValueError, "the scale is 0"):
            d.scale(scale=0)

    def test_constructor_history_empty(self):
        # filling time gaps in the constructor is not a user operation
        d = pp.EyeData(sampling_rate=10, left_pupil=[1., 2., 3.])
        self.assertEqual(d.history, [])
        d.fill_time_discontinuities()
        self.assertEqual([h["funcname"] for h in d.history],
                         ["fill_time_discontinuities"])

    def test_fill_time_discontinuities(self):
        # three segments sampled at 2 ms, separated by gaps of 500 ms
        segs = [np.arange(0, 1000, 2), np.arange(1500, 2500, 2), np.arange(3000, 4000, 2)]
        t = np.concatenate(segs).astype(float)
        x = np.concatenate([np.full(len(s), i+1.) for i, s in enumerate(segs)])
        d = pp.EyeData(left_pupil=x, time=t)
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


    def test_estimate_baseline_bridges_unusable_samples(self):
        """NaN and masked samples are bridged linearly (their values are not used)"""
        d = pp.get_example_data("rlmw_002_short").crop(0, 10, units="sec")
        x = d.data["left_pupil"].copy()
        bridged = d.copy()
        bridged.data["left_pupil"] = np.interp(np.arange(len(x)), np.flatnonzero(~np.isnan(x)),
                                               x[~np.isnan(x)])
        # zeros where masked do not matter
        zeros = d.copy()
        zeros.data["left_pupil"] = np.where(np.isnan(x), 0., x)
        zeros.mask_intervals(pp.Intervals.from_mask(np.isnan(x)), inplace=True)
        b = [o.pupil.estimate_baseline(eyes="left", inplace=False, seed=1).data["left_baseline"]
             for o in (d, bridged, zeros)]
        self.assertFalse(np.any(np.isnan(b[0])))
        np.testing.assert_allclose(b[0], b[1])
        np.testing.assert_allclose(b[0], b[2])

    def test_estimate_baseline_no_usable_samples(self):
        d = pp.EyeData(left_pupil=np.full(1000, np.nan), sampling_rate=100)
        with self.assertRaisesRegex(ValueError, "no usable samples"):
            d.pupil.estimate_baseline(eyes="left", inplace=False)

    def test_estimate_baseline_after_interpolation(self):
        d = pp.get_example_data("rlmw_002_short").crop(0, 10, units="sec")
        d = d.pupil.blinks_detect(eyes="left", inplace=False)
        d = d.pupil.blinks_interpolate(eyes="left", inplace=False)
        d = d.pupil.estimate_baseline(eyes="left", inplace=False)
        self.assertFalse(np.any(np.isnan(d.data["left", "baseline"])))

    def test_estimate_baseline_seed(self):
        """same seed gives the same baseline"""
        d = pp.get_example_data("rlmw_002_short").crop(0, 10, units="sec")
        d = d.pupil.blinks_detect(eyes="left", inplace=False)
        d = d.pupil.blinks_interpolate(eyes="left", inplace=False)
        b1 = d.pupil.estimate_baseline(eyes="left", inplace=False, seed=1)
        b2 = d.pupil.estimate_baseline(eyes="left", inplace=False, seed=1)
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
        d = pp.EyeData(left_pupil=p, sampling_rate=1000)
        d = d.pupil.blinks_detect(inplace=False)
        self.assertEqual(len(d.get_blinks("left", "pupil")["left_pupil"]), 2)
        d = d.pupil.blinks_interpolate(inplace=False)
        self.assertEqual(np.isnan(d.data["left", "pupil"]).sum(), 0)
        self.assertEqual(len(d.get_interpolated("left", "pupil")["left_pupil"]), 1)

if __name__ == '__main__':
    unittest.main()

class TestMergeBlinkWindows(unittest.TestCase):
    def merge(self, blinks, n=10000):
        from pypillometry.eyedata.pupil import _merge_overlapping_blink_windows
        return _merge_overlapping_blink_windows(blinks, n)

    def test_separate_blinks(self):
        self.assertEqual(self.merge([(100, 110), (200, 210)]),
                         [(90, 100, 110, 120), (190, 200, 210, 220)])

    def test_overlapping_blinks_merge(self):
        # anchor points from the windows of the first and the last blink
        self.assertEqual(self.merge([(100, 110), (105, 125)]),
                         [(90, 100, 125, 145)])
        self.assertEqual(self.merge([(100, 110), (110, 125)]),   # touching
                         [(90, 100, 125, 140)])

    def test_close_blinks_keep_their_curves(self):
        # the windows overlap: the anchor points move to the neighbour
        self.assertEqual(self.merge([(100, 110), (115, 125)]),
                         [(90, 100, 110, 115), (110, 115, 125, 135)])

    def test_no_cascade(self):
        # close blinks are not merged, so nothing can swallow the rest
        blinks = [(100, 200), (290, 390)] + [(1000 + 300*i, 1050 + 300*i)
                                             for i in range(20)]
        merged = self.merge(blinks)
        self.assertEqual(merged[:2], [(0, 100, 200, 290), (200, 290, 390, 490)])
        self.assertEqual(len(merged), 22)
        self.assertTrue(all(t1 < t2 < t3 < t4 for t1, t2, t3, t4 in merged))


def test_onset_search_stops_at_neighbour():
    """the onset of a blink is not taken from the transient of the previous one"""
    from pypillometry.signal.preproc import blink_onsets_mahot
    y = np.full(2000, 1000.0)
    y[480:500] -= np.linspace(0, 500, 20)        # blink 1: drop, missing, rise
    y[500:560] = np.nan
    y[560:600] -= np.linspace(500, 0, 40)
    y[700:720] = np.nan                          # blink 2: no transients
    on = blink_onsets_mahot(y, [(500, 560), (700, 720)], 5, -5, 5, (0, 0), 400)
    assert on[0, 0] < 500 and on[1, 0] >= 560    # not the drop of blink 1
    # the same in the other order of the input
    on2 = blink_onsets_mahot(y, [(700, 720), (500, 560)], 5, -5, 5, (0, 0), 400)
    np.testing.assert_array_equal(on2, on[::-1])
    # margins do not reach into the neighbour either
    on = blink_onsets_mahot(y, [(500, 560), (700, 720)], 5, -5, 5, (300, 0), 400)
    assert on[1, 0] >= 560


def test_close_anchor_points_no_overshoot():
    """a blink with an anchor point close to it gets a monotone curve"""
    from pypillometry.eyedata.pupil import _anchors_close
    assert _anchors_close(0, 100, 200, 220) and not _anchors_close(0, 100, 200, 300)
    y = np.full(4000, 1000.0)
    y[1000:1400] = np.nan                        # a long blink
    y[1400:1470] = 1000 + np.linspace(0, 300, 70)  # still rising when
    y[1470:] += 300
    y[1470:1490] = np.nan                        # a short one comes
    d = pp.EyeData(left_pupil=y, sampling_rate=1000)
    d = d.pupil.blinks_detect(strategies=["zero"], min_duration=10)
    r = d.pupil.blinks_interpolate(inplace=False).data["left_pupil"]
    assert not np.isnan(r[1000:1400]).any()
    # a cubic through these anchor points would dip to about 750
    assert np.nanmin(r[1000:1400]) >= 999 and np.nanmax(r[1000:1400]) <= 1301


def test_blinks_interpolate_without_onset_crossing():
    """all blinks of the example data are interpolated, no NaN left (bug 1.1)"""
    import pypillometry as pp
    d = pp.get_example_data("rlmw_002_short")
    d = d.pupil.blinks_detect().pupil.blinks_interpolate()
    for eye in d.eyes:
        assert not np.isnan(d.data[eye, "pupil"]).any()
    d.pupil.estimate_baseline()


class TestBlinksInterpolateParams(unittest.TestCase):
    """parameters of the methods of blinks_interpolate() as keyword arguments"""
    @classmethod
    def setUpClass(cls):
        cls.d = pp.get_example_data("rlmw_002_short").crop(0, 20, units="sec")
        cls.d = cls.d.pupil.blinks_detect(eyes="left", inplace=False)

    def interp(self, **kw):
        return self.d.pupil.blinks_interpolate(eyes="left", inplace=False, **kw)

    def test_method_parameters(self):
        """method parameters are keyword arguments; defaults as documented"""
        default = self.interp()
        same = self.interp(margin=(10, 30), winsize=25)
        np.testing.assert_array_equal(default.data["left", "pupil"], same.data["left", "pupil"])
        wider = self.interp(margin=(50, 100))
        self.assertGreater(wider.data.get_layer("left_pupil", "interpolated").sum(),
                           default.data.get_layer("left_pupil", "interpolated").sum())

    def test_unknown_parameter(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'margins'"):
            self.interp(margins=(10, 30))

    def test_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "unknown method"):
            self.interp(method="spline")

    def test_pipeline_step(self):
        d = self.d | pp.pupil.blinks_interpolate(eyes="left", margin=(50, 100))
        np.testing.assert_array_equal(d.data["left", "pupil"],
                                      self.interp(margin=(50, 100)).data["left", "pupil"])

    def test_other_method_parameter_warns(self):
        """a parameter of another method is ignored with a warning"""
        msgs = []
        hid = pp.logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
        try:
            d = self.interp(method="kalman", margin=(10, 30))
        finally:
            pp.logger.remove(hid)
        self.assertTrue(any("'margin' is only used by method='mahot'" in m for m in msgs))
        msgs.clear()
        hid = pp.logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
        try:
            self.interp(optimize=True)
        finally:
            pp.logger.remove(hid)
        self.assertTrue(any("'optimize' is only used by method='kalman'" in m for m in msgs))
        np.testing.assert_array_equal(d.data["left", "pupil"],
                                      self.interp(method="kalman").data["left", "pupil"])


class TestBlinksInterpolateKalman(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = pp.get_example_data("rlmw_002_short").pupil.blinks_detect(inplace=False)
        cls.k = cls.d.pupil.blinks_interpolate(method="kalman", inplace=False)

    def test_fills_all_blinks(self):
        for eye in self.d.eyes:
            blinks = self.d.get_blinks(eye, "pupil")[f"{eye}_pupil"].to_mask(self.d).astype(bool)
            interp = self.k.data.get_layer(f"{eye}_pupil", "interpolated")
            np.testing.assert_array_equal(interp, blinks)
            self.assertFalse(np.isnan(self.k.data[eye, "pupil"]).any())
            np.testing.assert_array_equal(self.k.data[eye, "pupil"][~blinks],
                                          self.d.data[eye, "pupil"][~blinks])

    def test_store_as(self):
        d = self.d.pupil.blinks_interpolate(eyes="left", method="kalman",
                                            store_as="pupilinterp", inplace=False)
        np.testing.assert_array_equal(d.data["left", "pupilinterp"], self.k.data["left", "pupil"])
        np.testing.assert_array_equal(d.data["left", "pupil"], self.d.data["left", "pupil"])
        self.assertTrue(d.data.get_layer("left_pupilinterp", "interpolated").any())

    def test_better_than_mahot_on_smooth_signal(self):
        """gaps in a smooth random signal are reconstructed better than by Mahot"""
        from pypillometry.signal import gaussmarkov as gm
        model = gm.GaussMarkovModel([gm.Matern(100., 300., nu=1.5)], noise=2.)
        y = gm.simulate(model, 20000, fs=500, rng=2) + 3000
        p = y.copy()
        gaps = [(s, s + 75) for s in range(1000, 19000, 1000)]
        for a, b in gaps:
            p[a:b] = np.nan
        d = pp.EyeData(left_pupil=p, sampling_rate=500).pupil.blinks_detect(
            strategies=["zero"], inplace=False)
        rmse = {}
        for method in ("mahot", "kalman"):
            r = d.pupil.blinks_interpolate(method=method, inplace=False).data["left", "pupil"]
            rmse[method] = np.sqrt(np.mean(np.concatenate([r[a:b] - y[a:b] for a, b in gaps])**2))
        self.assertLess(rmse["kalman"], rmse["mahot"])
