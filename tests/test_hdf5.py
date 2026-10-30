"""Tests for storing the state of objects in HDF5 files."""
import datetime
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

import pypillometry as pp
from pypillometry import hdf5, state
from pypillometry.signal.fake import generate_foreshortening_data

from tests.test_state import DATA, assert_state_equal, make_eyedata


class HDF5TestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def fname(self, name="d.h5"):
        return os.path.join(self.tmp, name)

    def roundtrip_state(self, st, compress=True):
        hdf5.write_state(st, self.fname(), compress=compress)
        return hdf5.read_state(self.fname())


class TestObjects(HDF5TestCase):

    def roundtrip(self, obj):
        obj.write_file(self.fname())
        obj2 = type(obj).from_file(self.fname())
        self.assertIs(type(obj2), type(obj))
        assert_state_equal(self, state.to_state(obj), state.to_state(obj2))
        return obj2

    def test_eyedata(self):
        d = make_eyedata()
        d2 = self.roundtrip(d)
        self.assertEqual(d2.nblinks(), d.nblinks())
        self.assertEqual(d2.event_labels.dtype, d.event_labels.dtype)
        self.assertEqual(d2.experimental_setup.screen_resolution, (1280, 1024))
        self.assertIsInstance(d2.original, pp.EyeData)
        self.assertEqual(d2.info["nested"]["b"], (1.5, "x"))
        self.assertEqual(list(d2.data.keys()), list(d.data.keys()))
        # the restored object can be processed further
        d2.pupil_lowpass_filter(cutoff=5, inplace=False)

    def test_pupildata_gazedata(self):
        d = make_eyedata(keep_orig=False)
        self.roundtrip(d.to_pupildata())
        self.roundtrip(d.to_gazedata())

    def test_fakeeyedata(self):
        f2 = self.roundtrip(generate_foreshortening_data(duration=1, fs=100, seed=1))
        self.assertIs(f2.sim_fct, generate_foreshortening_data)

    def test_segmented(self):
        d = make_eyedata(keep_orig=False)
        seg = d.get_segments(d.get_intervals("F_", interval=(-200, 800), units="ms"),
                             "left_pupil")
        self.roundtrip(seg)
        self.roundtrip(pp.group_segments([seg, seg]))

    def test_compress(self):
        d = make_eyedata()
        d.write_file(self.fname("c.h5"))
        d.write_file(self.fname("u.h5"), compress=False)
        self.assertLess(os.path.getsize(self.fname("c.h5")),
                        os.path.getsize(self.fname("u.h5")))
        with h5py.File(self.fname("c.h5")) as f:
            self.assertEqual(f["tx"].compression, "gzip")
            self.assertEqual(f.attrs["format_version"], state.FORMAT_VERSION)
            self.assertEqual(f.attrs["class"], "EyeData")

    def test_read_pickle_detects_hdf5(self):
        d = make_eyedata(keep_orig=False)
        fname = self.fname("d.hdf5")
        pp.write_pickle(d, fname)
        with open(fname, "rb") as f:
            self.assertEqual(f.read(4), b"\x89HDF")
        shutil.copy(fname, self.fname("other_extension.pypil"))
        self.assertIsInstance(pp.read_pickle(self.fname("other_extension.pypil")),
                              pp.EyeData)

    def test_no_state(self):
        with self.assertRaises(TypeError):
            pp.write_pickle({"a": 1}, self.fname())

    def test_newer_version(self):
        st = state.to_state(make_eyedata(keep_orig=False))
        st["format_version"] = state.FORMAT_VERSION + 1
        hdf5.write_state(st, self.fname())
        with self.assertRaisesRegex(ValueError, "newer version"):
            pp.read_pickle(self.fname())


class TestFixtures(HDF5TestCase):
    """Files written by earlier versions keep loading."""

    def test_v0_pickles_to_hdf5(self):
        for name in ["eyedata", "pupildata", "gazedata", "fakeeyedata",
                     "segmented", "group_segmented"]:
            with self.subTest(name=name):
                obj = pp.read_pickle(os.path.join(DATA, f"{name}_v0.pkl"))
                st = state.to_state(obj)
                assert_state_equal(self, st, self.roundtrip_state(st))

    def test_v1_files(self):
        d = pp.read_pickle(os.path.join(DATA, "eyedata_v0.pkl"))
        for name in ["eyedata_v1.pypil", "eyedata_v1.h5"]:
            with self.subTest(name=name):
                d1 = pp.read_pickle(os.path.join(DATA, name))
                assert_state_equal(self, state.to_state(d), state.to_state(d1))
        s = pp.read_pickle(os.path.join(DATA, "segmented_v1.h5"))
        self.assertIsInstance(s, pp.SegmentedEyeData)
        self.assertEqual(s.intervals.label, "F_")


class TestGenericWriter(HDF5TestCase):
    """Mapping of the value types to the file."""

    def test_types(self):
        st = {
            "none": None, "true": True, "npbool": np.bool_(False), "int": 3,
            "float": 1.5, "npfloat": np.float64(2.5), "str": "äbc",
            "complex": 1 + 2j,
            "when": datetime.datetime(2024, 5, 3, 10, 20, 30),
            "day": datetime.date(2024, 5, 3),
            "ints": [1, 2, 3], "floats": (1.0, 2.5), "strs": ["a", "bb"],
            "bools": [True, False], "mixed": [1, "a", None, {"x": [1.5]}],
            "empty_list": [], "empty_tuple": (), "empty_dict": {},
            "nested": {"a": {"b": {"c": np.arange(3)}}},
            "arr0": np.zeros((0,)), "arr2d": np.arange(6).reshape(2, 3),
            "big": np.arange(5000, dtype=float), "scalar_arr": np.array(4.0),
            "ustr": np.array(["F_1", "longer label"]),
            "ustr_empty": np.array([], dtype="<U5"),
            "ostr": np.array(["a", "b"], dtype=object),
            "struct": np.array([(1.0, 2)], dtype=[("x", "f8"), ("n", "i4")]),
            "bytes": np.array([b"ab", b"c"]),
            "long_text": "x" * 100000,
        }
        out = self.roundtrip_state(st)
        assert_state_equal(self, st, out)
        self.assertIs(type(out["true"]), bool)
        self.assertIs(type(out["npbool"]), bool)
        self.assertIs(type(out["int"]), int)
        self.assertIs(type(out["npfloat"]), float)
        self.assertIs(type(out["floats"]), tuple)
        self.assertEqual(out["ustr"].dtype, st["ustr"].dtype)
        self.assertEqual(out["ostr"].dtype, object)

    def test_key_order(self):
        st = {"b": np.arange(2), "a": 1, "c": {"z": None, "y": np.arange(2), "x": 2}}
        out = self.roundtrip_state(st)
        self.assertEqual(list(out), ["b", "a", "c"])
        self.assertEqual(list(out["c"]), ["z", "y", "x"])

    def test_unsupported_values_as_text(self):
        class Thing:
            def __repr__(self):
                return "Thing()"
        st = {"obj": Thing(), "intkeys": {1: "a"}, "slash": {"a/b": 1},
              "objarr": np.array([1, "a"], dtype=object),
              "masked": np.ma.masked_array([1, 2], mask=[0, 1])}
        with self.collect_warnings() as logs:
            out = self.roundtrip_state(st)
        self.assertEqual(out["obj"], "Thing()")
        self.assertEqual(out["intkeys"], "{1: 'a'}")
        self.assertEqual(out["slash"], "{'a/b': 1}")
        self.assertIsInstance(out["objarr"], str)
        self.assertIsInstance(out["masked"], str)
        self.assertEqual(len(logs), 5)

    def collect_warnings(self):
        """Context manager that collects the warnings logged by loguru."""
        from loguru import logger

        class Ctx:
            def __enter__(self):
                self.messages = []
                self.id = logger.add(self.messages.append, level="WARNING")
                return self.messages

            def __exit__(self, *exc):
                logger.remove(self.id)
                return False
        return Ctx()


if __name__ == "__main__":
    unittest.main()
