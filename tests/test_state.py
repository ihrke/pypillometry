"""Tests for the versioned state of eyedata and segmented objects (pickle)."""
import copy
import os
import pickle
import tempfile
import unittest

import numpy as np

import pypillometry as pp
from pypillometry import state
from pypillometry.eyedata.spatial_calibration import SpatialCalibration
from pypillometry.signal.fake import generate_foreshortening_data

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "state")


def assert_state_equal(test, a, b, path="state"):
    """Compare two states (NaN-aware for arrays, numbers by value)."""
    if isinstance(a, dict):
        test.assertIsInstance(b, dict, path)
        test.assertEqual(list(a), list(b), path)
        for k in a:
            assert_state_equal(test, a[k], b[k], f"{path}.{k}")
    elif isinstance(a, np.ndarray):
        test.assertIsInstance(b, np.ndarray, path)
        test.assertEqual(a.dtype, b.dtype, path)
        test.assertEqual(a.shape, b.shape, path)
        if a.dtype.kind in "fc":
            np.testing.assert_array_equal(a, b, err_msg=path)
        else:
            test.assertTrue(np.array_equal(a, b), path)
    elif isinstance(a, (list, tuple)):
        test.assertEqual(type(a), type(b), path)
        test.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_state_equal(test, x, y, f"{path}[{i}]")
    elif isinstance(a, float) and np.isnan(a):
        test.assertTrue(np.isnan(b), path)
    else:
        test.assertEqual(a, b, path)


def make_eyedata(keep_orig=True):
    """Small EyeData with events, blinks, setup, calibration, info and params."""
    rng = np.random.default_rng(1)
    n = 3000
    pupil = 1000 + rng.normal(0, 5, n)
    pupil[500:560] = np.nan
    pupil[1800:1850] = np.nan
    pts = np.array([[640., 512.], [640., 87.], [640., 936.]])
    err = np.array([[0.3, -13.3, -1.6], [0.3, 13.2, -0.2], [0.42, 16.1, 9.2]])
    d = pp.EyeData(
        time=np.arange(n) * 2.0, sampling_rate=500,
        left_x=rng.normal(640, 10, n), left_y=rng.normal(512, 10, n),
        left_pupil=pupil, right_x=rng.normal(640, 10, n),
        right_y=rng.normal(512, 10, n), right_pupil=pupil + 10,
        event_onsets=np.array([1000., 3000., 5000.]),
        event_labels=np.array(["F_1", "F_2", "S_1"]),
        experimental_setup=pp.ExperimentalSetup(
            screen_resolution=(1280, 1024), physical_screen_size=("30 cm", "20 cm"),
            eye_to_screen_center="60 cm",
            camera_spherical=("20 deg", "-90 deg", "60 cm")),
        calibration={"left": SpatialCalibration("left", "HV3", pts, pts + err[:, 1:],
                                                err, (1280, 1024))},
        name="test", keep_orig=keep_orig,
        info={"subject": "s1", "nested": {"a": [1, 2], "b": (1.5, "x")}})
    d = d.pupil_blinks_detect(apply_mask=True, inplace=False)
    d = d.pupil_blinks_interpolate().scale()
    return d


class TestPickleRoundTrip(unittest.TestCase):

    def roundtrip(self, obj):
        st = state.to_state(obj)
        obj2 = pickle.loads(pickle.dumps(obj))
        self.assertIs(type(obj2), type(obj))
        assert_state_equal(self, st, state.to_state(obj2))
        return obj2

    def test_eyedata(self):
        d = make_eyedata()
        d2 = self.roundtrip(d)
        self.assertEqual(d2.nblinks(), d.nblinks())
        self.assertIsInstance(d2.original, pp.EyeData)
        self.assertIsInstance(d2.calibration["left"], SpatialCalibration)
        self.assertAlmostEqual(d2.experimental_setup.d, d.experimental_setup.d)
        self.assertEqual(d2.history, d.history)
        np.testing.assert_array_equal(d2["left_pupil"].mask, d["left_pupil"].mask)

    def test_pupildata_and_gazedata(self):
        d = make_eyedata(keep_orig=False)
        self.roundtrip(d.to_pupildata())
        self.roundtrip(d.to_gazedata())

    def test_fakeeyedata(self):
        f = generate_foreshortening_data(duration=1, fs=100, seed=1)
        f2 = self.roundtrip(f)
        self.assertIs(f2.sim_fct, generate_foreshortening_data)
        self.assertEqual(list(f2.sim_data.keys()), list(f.sim_data.keys()))

    def test_segmented(self):
        d = make_eyedata(keep_orig=False)
        iv = d.get_intervals("F_", interval=(-200, 800), units="ms")
        seg = d.get_segments(iv, "left_pupil")
        seg2 = self.roundtrip(seg)
        np.testing.assert_array_equal(seg2.data.mask, seg.data.mask)
        self.assertEqual(seg2.intervals.intervals, seg.intervals.intervals)
        self.roundtrip(pp.group_segments([seg, seg]))

    def test_write_file_from_file(self):
        d = make_eyedata()
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "d.pypil")
            d.write_file(fname)
            d2 = pp.EyeData.from_file(fname)
        assert_state_equal(self, state.to_state(d), state.to_state(d2))

    def test_copy(self):
        d = make_eyedata()
        c = d.copy()
        self.assertEqual(c.name, "test_copy")
        self.assertIsNot(c.original, d.original)
        np.testing.assert_array_equal(c.tx, d.tx)
        c2 = copy.deepcopy(d)
        assert_state_equal(self, state.to_state(d), state.to_state(c2))

    def test_state_contents(self):
        st = state.to_state(make_eyedata())
        self.assertEqual(st["format_version"], state.FORMAT_VERSION)
        self.assertEqual(st["class"], "EyeData")
        self.assertEqual(st["original"]["class"], "EyeData")
        self.assertEqual(st["blinks"]["left_pupil"]["intervals"].shape[1], 2)
        self.assertIsNone(st["blinks"]["left_x"])


class TestFormatVersion0(unittest.TestCase):
    """Pickles written before the state was versioned (pypillometry 2.0)."""

    def load(self, name):
        return pp.read_pickle(os.path.join(DATA, name))

    def test_eyedata(self):
        d = self.load("eyedata_v0.pkl")
        self.assertIsInstance(d, pp.EyeData)
        self.assertEqual(d.name, "fixture_eyedata")
        self.assertEqual(d.nblinks(), {"left_pupil": 4, "right_pupil": 3})
        self.assertEqual(len(d.event_onsets), 4)
        self.assertEqual(d.tx[0], 4500.0)
        self.assertIsInstance(d.original, pp.EyeData)
        self.assertEqual(d.calibration["left"].model, "HV3")
        self.assertAlmostEqual(d.experimental_setup.d, 600.0)
        self.assertEqual(d.info["eyelink_info"]["meas_date"].year, 2024)
        self.assertEqual(d.history[-1]["funcname"], "pupil_lowpass_filter")
        self.assertIn("scale", d.params)
        # the restored object works and saves in the current version
        d.pupil_blinks_interpolate(inplace=False)
        st = state.to_state(pickle.loads(pickle.dumps(d)))
        self.assertEqual(st["format_version"], state.FORMAT_VERSION)

    def test_pupildata_and_gazedata(self):
        p = self.load("pupildata_v0.pkl")
        self.assertIsInstance(p, pp.PupilData)
        self.assertEqual(p.eyes, ["left", "right"])
        g = self.load("gazedata_v0.pkl")
        self.assertIsInstance(g, pp.GazeData)
        self.assertEqual(g.calibration["left"].eye, "left")

    def test_fakeeyedata(self):
        f = self.load("fakeeyedata_v0.pkl")
        self.assertIs(f.sim_fct, generate_foreshortening_data)
        self.assertIn("left_A0", f.sim_data)

    def test_segmented(self):
        s = self.load("segmented_v0.pkl")
        self.assertIsInstance(s, pp.SegmentedEyeData)
        self.assertEqual(s.variable, "left_pupil")
        self.assertEqual(s.intervals.label, "F_")
        g = self.load("group_segmented_v0.pkl")
        self.assertIsInstance(g, pp.GroupLevelSegmentedData)
        self.assertEqual(g.n_subjects, 2)


class TestMigrate(unittest.TestCase):

    def test_newer_version_raises(self):
        st = state.to_state(make_eyedata(keep_orig=False))
        st["format_version"] = state.FORMAT_VERSION + 1
        with self.assertRaisesRegex(ValueError, "newer version"):
            state.from_state(st)

    def test_v0_history_arguments_dropped(self):
        d = make_eyedata(keep_orig=False)
        raw = dict(vars(d))
        raw["history"] = [{"funcstring": "scale()", "funcname": "scale",
                           "args": (), "kwargs": {"eyes": [object()]}}]
        st = state.migrate(raw, "EyeData")
        self.assertEqual(st["history"], [{"funcstring": "scale()",
                                          "funcname": "scale"}])

    def test_current_version_unchanged(self):
        st = state.to_state(make_eyedata(keep_orig=False))
        self.assertIs(state.migrate(st, "EyeData"), st)

    def test_from_state_finds_class(self):
        st = state.to_state(make_eyedata(keep_orig=False).to_pupildata())
        self.assertIsInstance(state.from_state(st), pp.PupilData)
        st["class"] = "NoSuchClass"
        with self.assertRaisesRegex(ValueError, "unknown class"):
            state.from_state(st)

    def test_renamed_classes(self):
        d = make_eyedata(keep_orig=False)
        data = pickle.dumps(d).replace(b"pypillometry.eyedata.eyedata",
                                       b"pypillometry.eyedata.olddata")
        with self.assertRaises(ModuleNotFoundError):
            pickle.loads(data)
        old = state._RENAMED_CLASSES.copy()
        state._RENAMED_CLASSES[("pypillometry.eyedata.olddata", "EyeData")] = (
            "pypillometry.eyedata.eyedata", "EyeData")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fname = os.path.join(tmp, "d.pkl")
                with open(fname, "wb") as f:
                    f.write(data)
                self.assertIsInstance(pp.read_pickle(fname), pp.EyeData)
        finally:
            state._RENAMED_CLASSES.clear()
            state._RENAMED_CLASSES.update(old)


if __name__ == "__main__":
    unittest.main()
