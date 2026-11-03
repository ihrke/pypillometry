"""
Tests for the response estimation with excluded periods and missing samples.
"""
import unittest
import numpy as np
import pypillometry as pp
from pypillometry.signal.pupil import (pupil_response, pupil_response_coverage,
                                       pupil_build_design_matrix, pupil_kernel)


def fake_pupil(duration=60, fs=100, shift_at=30, shift=500., seed=1):
    """
    Pupil signal (ms time vector) with a slow baseline that jumps by `shift`
    at `shift_at` seconds and a response to an event every 3 s.
    """
    rng = np.random.default_rng(seed)
    tx = np.arange(0, duration * 1000, 1000. / fs)
    base = 3000 + 100 * np.sin(2 * np.pi * tx / 20000.)
    base[tx >= shift_at * 1000] += shift
    onsets = np.arange(1500., duration * 1000 - 3000, 3000.)
    x1 = pupil_build_design_matrix(tx, onsets, fs, 10, 900)
    resp = np.dot(x1.T, rng.uniform(50, 150, len(onsets)))
    sy = base + resp + rng.normal(0, 5, len(tx))
    d = pp.EyeData(time=tx, left_pupil=sy, sampling_rate=fs,
                     event_onsets=onsets, event_labels=["stim"] * len(onsets),
                     inplace=False)
    return d


class TestPupilResponseValid(unittest.TestCase):

    def setUp(self):
        self.d = fake_pupil()
        self.valid = np.ones(len(self.d.tx), dtype=bool)
        self.valid[2000:3000] = False

    def test_invalid_samples_not_used(self):
        d = self.d
        sy = d.data["left", "pupil"] - 3000
        garbage = sy.copy()
        garbage[~self.valid] = 1e6
        a = pupil_response(d.tx, sy, d.events.onsets, d.fs, npar=10, tmax=900,
                           valid=self.valid)
        b = pupil_response(d.tx, garbage, d.events.onsets, d.fs, npar=10, tmax=900,
                           valid=self.valid)
        np.testing.assert_allclose(a[1], b[1])
        # the prediction covers all samples
        self.assertEqual(len(b[0]), len(d.tx))

    def test_nan_is_invalid(self):
        d = self.d
        sy = d.data["left", "pupil"] - 3000
        withnan = sy.copy()
        withnan[~self.valid] = np.nan
        a = pupil_response(d.tx, sy, d.events.onsets, d.fs, npar=10, tmax=900,
                           valid=self.valid)
        b = pupil_response(d.tx, withnan, d.events.onsets, d.fs, npar=10, tmax=900)
        np.testing.assert_allclose(a[1], b[1])

    def test_free_parameters_ignore_invalid(self):
        d = self.d
        sy = d.data["left", "pupil"] - 3000
        garbage = sy.copy()
        garbage[~self.valid] = 1e6
        tmax = [pupil_response(d.tx, y, d.events.onsets, d.fs, npar=10, tmax="free",
                               valid=self.valid, display_progress=False)[3]
                for y in (sy, garbage)]
        self.assertAlmostEqual(tmax[0], tmax[1])

    def test_min_valid(self):
        d = self.d
        sy = d.data["left", "pupil"] - 3000
        cov = pupil_response_coverage(
            pupil_build_design_matrix(d.tx, d.events.onsets, d.fs, 10, 900), self.valid)
        pred, coef, _, _, _ = pupil_response(d.tx, sy, d.events.onsets, d.fs, npar=10,
                                             tmax=900, valid=self.valid, min_valid=0.5)
        np.testing.assert_array_equal(np.isnan(coef), cov < 0.5)
        self.assertTrue(np.isnan(coef).any())
        self.assertFalse(np.isnan(pred).any())
        # default: all events
        coef0 = pupil_response(d.tx, sy, d.events.onsets, d.fs, npar=10, tmax=900,
                               valid=self.valid)[1]
        self.assertFalse(np.isnan(coef0).any())

    def test_no_valid_samples(self):
        d = self.d
        with self.assertRaises(ValueError):
            pupil_response(d.tx, d.data["left", "pupil"], d.events.onsets, d.fs,
                           npar=10, tmax=900, valid=np.zeros(len(d.tx), dtype=bool))

    def test_coverage(self):
        x1 = np.array([[0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 6, 2, 1, 0]],
                      dtype=float)
        valid = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
        # weighted by the response: the peak counts more
        np.testing.assert_array_equal(pupil_response_coverage(x1, valid),
                                      [0.75, np.nan, 0.4])


class TestResponseExcluded(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        d = fake_pupil()
        # break from 27 to 33 s with garbage in it, bridged linearly
        d.data["left_pupil"][2700:3300] = 0.
        d = d.exclude_intervals([(27000, 33000)])
        d = d.interpolate_intervals(d.get_excluded(), variables="pupil")
        d = d.pupil_estimate_baseline(eyes="left", seed=1)
        cls.d = d
        cls.est = d.pupil_estimate_response(eyes="left", npar=10, tmax=900, inplace=False)

    def test_bridged_linearly(self):
        y = self.d.data["left", "pupil"]
        np.testing.assert_allclose(np.diff(y[2700:3300], 2), 0, atol=1e-6)
        self.assertTrue(self.d.data.mask["left_pupil"][2700:3300].all())

    def test_events_in_break_left_out(self):
        par = self.est.params["response"]["left"]
        np.testing.assert_array_equal(np.isnan(par["coef"]), par["valid"] < 0.5)
        self.assertTrue(np.isnan(par["coef"][self.est.events.onsets == 28500][0]))
        self.assertEqual(par["min_valid"], 0.5)
        # no huge values in the predicted response
        self.assertLess(np.max(self.est.data["left", "response"]), 1000)
        est0 = self.d.pupil_estimate_response(eyes="left", npar=10, tmax=900,
                                              min_valid=0, inplace=False)
        self.assertFalse(np.isnan(est0.params["response"]["left"]["coef"]).any())

    def test_shares(self):
        par = self.est.params["response"]["left"]
        onsets = self.est.events.onsets
        self.assertEqual(len(par["valid"]), len(onsets))
        np.testing.assert_allclose(par["valid"][onsets < 20000], 1)
        # the response of the event at 28.5 s is in the break
        self.assertEqual(par["valid"][onsets == 28500][0], 0)
        # the event at 25.5 s reaches into the break
        self.assertTrue(0 < par["valid"][onsets == 25500][0] < 1)
        np.testing.assert_allclose(par["observed"], par["valid"])

    def test_excluded_not_used(self):
        d = self.d.copy()
        d.data["left_pupil"][2700:3300] += 1e5
        est = d.pupil_estimate_response(eyes="left", npar=10, tmax=900, inplace=False)
        np.testing.assert_allclose(est.params["response"]["left"]["coef"],
                                   self.est.params["response"]["left"]["coef"])

    def test_observed_lower_with_blinks(self):
        d = self.d.copy()
        # an interpolated blink at 1.6 s
        d.interpolate_intervals(pp.Intervals([(160, 200)], "samples"), inplace=True)
        est = d.pupil_estimate_response(eyes="left", npar=10, tmax=900, inplace=False)
        par = est.params["response"]["left"]
        self.assertLess(par["observed"][0], 1)
        self.assertAlmostEqual(par["valid"][0], 1)

    def test_masked_samples_not_used(self):
        """masked values (here zeros) are not used for the fit (D2)"""
        d = self.d.copy()
        d.data["left_pupil"][160:200] = 0.
        d.mask_intervals(pp.Intervals([(160, 200)], "samples"), inplace=True)
        est = d.pupil_estimate_response(eyes="left", npar=10, tmax=900, inplace=False)
        par = est.params["response"]["left"]
        self.assertLess(par["valid"][0], 0.99)
        ref = self.d.pupil_estimate_response(eyes="left", npar=10, tmax=900, inplace=False)
        np.testing.assert_allclose(par["coef"], ref.params["response"]["left"]["coef"],
                                   rtol=0.05)


if __name__ == "__main__":
    unittest.main()
