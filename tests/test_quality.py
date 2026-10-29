import unittest

import numpy as np

import pypillometry as pp
from pypillometry.quality import (DEFAULT_QUALITY_THRESHOLDS, _label, _run_lengths,
                                  _diff_noise)


def _make_data(n=10000, fs=500, missing=None, seed=1):
    """EyeData with smooth pupil, fixation-like gaze and optional NaN periods."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs * 1000
    pupil = 1000 + 50 * np.sin(t / 1000) + rng.normal(0, 1, n)
    x = 500 + rng.normal(0, 0.5, n)
    y = 400 + rng.normal(0, 0.5, n)
    lp, rp = pupil.copy(), pupil.copy() + rng.normal(0, 1, n)
    for start, end in (missing or []):
        lp[start:end] = np.nan
        x[start:end] = np.nan
    return pp.EyeData(time=t, left_x=x, left_y=y, left_pupil=lp,
                      right_x=x + 2, right_y=y.copy(), right_pupil=rp,
                      sampling_rate=fs, name="test")


class TestHelpers(unittest.TestCase):
    def test_label_lower_is_better(self):
        b = (5, 15, 30)
        self.assertEqual(_label(1, b), "good")
        self.assertEqual(_label(5, b), "good")
        self.assertEqual(_label(10, b), "ok")
        self.assertEqual(_label(20, b), "poor")
        self.assertEqual(_label(50, b), "bad")
        self.assertIsNone(_label(np.nan, b))
        self.assertIsNone(_label(10, None))

    def test_label_higher_is_better(self):
        b = (0.9, 0.8, 0.6)
        self.assertEqual(_label(0.95, b), "good")
        self.assertEqual(_label(0.85, b), "ok")
        self.assertEqual(_label(0.7, b), "poor")
        self.assertEqual(_label(0.1, b), "bad")

    def test_run_lengths(self):
        m = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
        np.testing.assert_array_equal(_run_lengths(m), [2, 1, 3])
        self.assertEqual(len(_run_lengths(np.zeros(5, bool))), 0)

    def test_diff_noise(self):
        rng = np.random.default_rng(0)
        v = np.cumsum(rng.normal(0, 0.01, 20000)) + rng.normal(0, 2, 20000)
        self.assertAlmostEqual(_diff_noise(v), 2, delta=0.1)


class TestQualityReport(unittest.TestCase):
    def test_raw_measures(self):
        d = _make_data(missing=[(1000, 1500), (5000, 5100)])
        r = d.quality_report()
        self.assertIsInstance(r, pp.QualityReport)
        self.assertIn("pupil", r)
        self.assertIn("gaze", r)
        self.assertIn("binocular", r)
        for section in ("blinks", "artifacts", "calibration"):
            self.assertNotIn(section, r)
        m = r["pupil"]["left"]["missing_pct"]
        self.assertAlmostEqual(m.value, 6.0)
        self.assertEqual(m.label, "ok")
        self.assertAlmostEqual(r["pupil"]["left"]["longest_dropout"].value, 1.0)
        self.assertAlmostEqual(r["pupil"]["left"]["dropout_rate"].value, 2 / (20 / 60))
        self.assertEqual(r["pupil"]["right"]["missing_pct"].value, 0)
        self.assertEqual(r["pupil"]["right"]["missing_pct"].label, "good")
        self.assertEqual(r["binocular"]["left/right"]["pupil_correlation"].label, "good")
        # no screen size/distance: precision in pixels and without label
        self.assertIn("precision_px", r["gaze"]["left"])
        self.assertIsNone(r["gaze"]["left"]["precision_px"].label)
        # gaze dropouts only reported for eyes without pupil
        self.assertNotIn("dropout_rate", r["gaze"]["left"])
        self.assertEqual(r.overall, "ok")

    def test_setup_measures(self):
        d = _make_data()
        d.set_experimental_setup(screen_resolution=(1000, 800),
                                 physical_screen_size=("40 cm", "30 cm"),
                                 eye_to_screen_center="60 cm")
        r = d.quality_report()
        self.assertEqual(r["gaze"]["left"]["offscreen_pct"].value, 0)
        self.assertIn("precision_deg", r["gaze"]["left"])
        self.assertIsNotNone(r["gaze"]["left"]["precision_deg"].label)
        self.assertIn("gaze_disparity_deg", r["binocular"]["left/right"])

    def test_preprocessing_measures(self):
        d = _make_data(missing=[(1000, 1100), (5000, 5100)])
        d.pupil_blinks_detect(apply_mask=True)
        r = d.quality_report()
        self.assertIn("blinks", r)
        self.assertEqual(r["blinks"]["left"]["n"].value, len(d.get_blinks("left", "pupil")))
        self.assertIn("artifacts", r)
        self.assertGreaterEqual(r["artifacts"]["left"]["pupil_masked_pct"].value, 2.0)
        d.pupil_blinks_interpolate()
        r = d.quality_report()
        self.assertIn("interpolated_pct", r["artifacts"]["left"])

    def test_calibration(self):
        d = _make_data()
        errors = np.c_[np.full(9, 0.7), np.zeros((9, 2))]
        d.calibration = {"left": pp.SpatialCalibration("left", "HV9", np.zeros((9, 2)),
                                                       np.zeros((9, 2)), errors)}
        r = d.quality_report()
        self.assertEqual(r["calibration"]["left"]["mean_error"].label, "ok")
        self.assertEqual(r["calibration"]["left"]["model"].value, "HV9")

    def test_pupildata_and_gazedata(self):
        d = _make_data()
        r = d.get_pupildata("left").quality_report()
        self.assertIn("pupil", r)
        self.assertNotIn("gaze", r)
        self.assertNotIn("binocular", r)
        g = pp.GazeData(time=d.tx, left_x=d.data["left_x"], left_y=d.data["left_y"],
                        sampling_rate=d.fs)
        r = g.quality_report()
        self.assertNotIn("pupil", r)
        self.assertIn("dropout_rate", r["gaze"]["left"])

    def test_eyes(self):
        r = _make_data().quality_report(eyes="right")
        self.assertEqual(list(r["pupil"].keys()), ["right"])
        self.assertNotIn("binocular", r)

    def test_thresholds(self):
        d = _make_data(missing=[(1000, 1500)])
        r = d.quality_report(thresholds={"pupil.missing_pct": (0.5, 1, 2)})
        self.assertEqual(r["pupil"]["left"]["missing_pct"].label, "bad")
        self.assertEqual(r.thresholds["pupil.missing_pct"], (0.5, 1, 2))
        r = d.quality_report(thresholds={"pupil.missing_pct": None})
        self.assertIsNone(r["pupil"]["left"]["missing_pct"].label)
        # defaults are not changed
        self.assertEqual(DEFAULT_QUALITY_THRESHOLDS["pupil.missing_pct"], (5, 15, 30))
        with self.assertRaises(ValueError):
            d.quality_report(thresholds={"pupil.nonsense": (1, 2, 3)})
        with self.assertRaises(ValueError):
            d.quality_report(thresholds={"pupil.missing_pct": (1, 2)})
        with self.assertRaises(ValueError):
            d.quality_report(thresholds={"pupil.missing_pct": (1, 5, 2)})

    def test_representations(self):
        d = _make_data(missing=[(1000, 4000)])
        r = d.quality_report()
        txt = r.to_text(color=False)
        self.assertNotIn("\033[", txt)
        self.assertIn("missing data", txt)
        self.assertIn("Issues", txt)
        self.assertIn("\033[", r.to_text(color=True))
        html = r._repr_html_()
        self.assertIn("<table", html)
        self.assertIn("Quality report", html)
        nested = r.to_dict()
        self.assertEqual(nested["overall"], r.overall)
        self.assertEqual(nested["pupil"]["left"]["missing_pct"]["label"], "poor")
        flat = r.to_dict(flat=True)
        self.assertAlmostEqual(flat["pupil.left.missing_pct"], 30.0)
        self.assertEqual(flat["pupil.left.missing_pct.label"], "poor")
        self.assertIsInstance(flat["info.all.n_samples"], int)
        issues = r.issues()
        self.assertTrue(all(m.label in ("poor", "bad") for _, _, m in issues))
        self.assertIn("pupil", r.labels())


    def test_summary(self):
        from pypillometry.quality import _summary_text, _summary_html
        good = _make_data().quality_report()
        bad = _make_data(missing=[(1000, 5000)]).quality_report()
        txt = _summary_text(["a", "b"], [good, bad], color=False)
        lines = txt.splitlines()
        self.assertEqual(lines[0], "Summary of 2 files")
        self.assertIn("Pupil", lines[1])
        self.assertNotIn("Blinks", lines[1])  # no report has blinks
        self.assertTrue(lines[3].startswith("b "))
        self.assertIn("bad", lines[3])
        html = _summary_html(["a", "b"], [good, bad], anchors=["r0", "r1"],
                             tooltips=["dir/a", "dir/b"])
        self.assertIn('<a href="#r1">b</a>', html)
        self.assertIn('title="dir/b"', html)


if __name__ == "__main__":
    unittest.main()
