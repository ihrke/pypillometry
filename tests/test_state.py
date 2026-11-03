"""Tests for the versioned state of eyedata and segmented objects (pickle)."""
import copy
import inspect
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace

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
    # event metadata with int, float (missing), string and bool columns
    d.set_events(d.events.parse_labels(r"_(?P<trial>\d+)")
                 .set_meta("rt", [450.5, np.nan], select="F_")
                 .set_meta("kind", ["fix", "fix", "stim"])
                 .set_meta("ok", [True, False, True]))
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
        iv = d.get_intervals("F_", window=(-200, 800), units="ms")
        seg = d.get_segments(iv, "left_pupil")
        seg2 = self.roundtrip(seg)
        np.testing.assert_array_equal(seg2.data.mask, seg.data.mask)
        self.assertEqual(seg2.intervals, seg.intervals)
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
        # blinks are a mask layer of the data (no separate store)
        self.assertNotIn("blinks", st)
        self.assertIn("blink", st["data"]["layer_names"])
        self.assertEqual(st["data"]["layers"]["left_pupil"].dtype, np.uint16)


class TestStateCoverage(unittest.TestCase):
    """Every attribute is part of the state (or explicitly excluded)."""

    def objects(self):
        d = make_eyedata()
        d = d.crop(500, 5000, units="ms", inplace=False).reset_time()
        seg = d.get_segments(d.get_intervals("F_", window=(-200, 800), units="ms"),
                             "left_pupil")
        return [
            d, d.to_pupildata(), d.to_gazedata(),
            pp.PupilData(left_pupil=np.ones(10), sampling_rate=10),
            pp.GazeData(left_x=np.ones(10), left_y=np.ones(10), sampling_rate=10),
            generate_foreshortening_data(duration=1, fs=100, seed=1),
            seg, pp.group_segments([seg, seg]),
        ]

    def stateful_objects(self, obj, seen):
        """`obj` and all Stateful objects reachable from its attributes."""
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, state.Stateful):
            yield obj
            children = vars(obj).values()
        elif isinstance(obj, dict):
            children = obj.values()
        else:
            return
        for child in children:
            yield from self.stateful_objects(child, seen)

    def test_all_attributes_in_state(self):
        classes = set()
        seen = set()
        for top in self.objects():
            for obj in self.stateful_objects(top, seen):
                cls = type(obj)
                classes.add(cls)
                fields = {f.attr for f in cls._state_fields()}
                attrs = set(vars(obj)) - cls._state_exclude()
                # PupilData made from EyeData keeps the gaze attributes
                # (not saved, see CHANGELOG)
                if cls is pp.PupilData:
                    attrs -= {"experimental_setup", "calibration"}
                with self.subTest(cls=cls.__name__):
                    self.assertEqual(attrs - fields, set(), "attributes not in the state")
                    self.assertEqual(fields - attrs, set(), "fields without attribute")
        # every class with a state was checked
        todo, defined = [state.Stateful], set()
        while todo:
            c = todo.pop()
            todo.extend(c.__subclasses__())
            if vars(c).get("_STATE_FIELDS") and not inspect.isabstract(c):
                defined.add(c)
        self.assertEqual({c.__name__ for c in defined - classes}, set())


class TestFormatVersion0(unittest.TestCase):
    """Pickles written before the state was versioned (pypillometry 2.0)."""

    def load(self, name):
        return pp.read_pickle(os.path.join(DATA, name))

    def test_eyedata(self):
        d = self.load("eyedata_v0.pkl")
        self.assertIsInstance(d, pp.EyeData)
        self.assertEqual(d.name, "fixture_eyedata")
        self.assertEqual(d.nblinks(), {"left_pupil": 4, "right_pupil": 3})
        self.assertEqual(len(d.events), 4)
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
        # version 0 had one mask dict per EyeDataDict
        raw["data"] = SimpleNamespace(data=d.data.data, mask=dict(d.data.mask))
        # version 0 had the events as arrays
        ev = raw.pop("_events")
        raw.update(event_onsets=ev.onsets, event_labels=ev.labels)
        raw["history"] = [{"funcstring": "scale()", "funcname": "scale",
                           "args": (), "kwargs": {"eyes": [object()]}}]
        st = state.migrate(raw, "EyeData")
        self.assertEqual(st["history"], [{"funcstring": "scale()",
                                          "funcname": "scale"}])

    def test_v1_intervals(self):
        """version 1 -> 2: sample indices become "samples", fs/offset dropped"""
        for name, cls in [("eyedata_v1.pypil", pp.EyeData), ("eyedata_v1.h5", pp.EyeData),
                          ("segmented_v1.h5", pp.SegmentedEyeData)]:
            obj = cls.from_file(os.path.join(DATA, name))
            # segments: event-locked intervals are loaded as Epochs (in ms);
            # eyedata: blinks are loaded into the blink layer
            ivs = [obj.intervals.to_intervals()] if cls is pp.SegmentedEyeData else \
                [iv for iv in obj.get_blinks().values() if len(iv) > 0]
            self.assertTrue(ivs)
            for iv in ivs:
                self.assertIn(iv.units, ("samples", "ms", "sec", "min", "h"))
                self.assertFalse(hasattr(iv, "sampling_rate"))
                self.assertFalse(hasattr(iv, "time_offset"))
        st = state.migrate({"format_version": 1, "class": "PupilData",
                            "blinks": {"left_pupil": {"units": None, "sampling_rate": 10.,
                                                      "time_offset": 0., "label": "b"}},
                            "interpolated_blinks": {"left_pupil": None}}, "PupilData")
        self.assertEqual(st["blinks"]["left_pupil"],
                         {"units": "samples", "label": "b", "event_meta": None})
        self.assertEqual(st["event_meta"], {})
        self.assertIsNone(st["excluded"]["event_meta"])

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
