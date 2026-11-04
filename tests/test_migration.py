"""
Migration of saved states from format 2 to 3 (`state._migrate_2_to_3`).

The fixture files in ``tests/data/format2`` were written by pypillometry
before the 2.2 refactor; ``tests/test_format2_fixtures.py`` checks the loaded
objects against their manifest. The tests here check the migrated states.
"""
import contextlib
import json
import os
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pytest
from loguru import logger

import pypillometry as pp
from pypillometry import hdf5, state

from tests.test_format2_fixtures import assert_matches, facts, load_manifest
from tests.test_state import DATA as STATE_DIR, assert_state_equal

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "format2")


def format2_state(name):
    """The state in a format-2 HDF5 fixture file (not migrated)."""
    st = hdf5.read_state(os.path.join(FIXTURE_DIR, name))
    assert st["format_version"] == 2
    return st


@contextlib.contextmanager
def warnings_logged():
    """Collect the warnings logged by loguru."""
    messages = []
    handler = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler)


class TestSamplingRate:

    def test_fs_fits_tx_unchanged(self):
        with warnings_logged() as logs:
            st = state.migrate(format2_state("pupildata.h5"), "PupilData")
        assert st["fs"] == 500.
        assert not [m for m in logs if "sampling rate" in m]

    def test_fs_corrected_from_tx(self):
        # as saved after downsample(fsd) with a non-integer fs/fsd
        st = format2_state("pupildata.h5")
        st["fs"] = 300.
        with warnings_logged() as logs:
            d = state.from_state(st)
        assert d.fs == pytest.approx(500.)
        assert any("sampling rate (300 Hz)" in m for m in logs)


EYEDATA_FILES = ["pupildata", "gazedata", "eyedata"]


class TestMigratedState:
    """Layout of the migrated (format-3) states."""

    @pytest.mark.parametrize("name", EYEDATA_FILES)
    def test_eyedata_layout(self, name):
        old = format2_state(f"{name}.h5")
        st = state.migrate(format2_state(f"{name}.h5"), old["class"])
        assert st["format_version"] == state.FORMAT_VERSION == 3
        assert st["class"] == old["class"]  # classes are merged in a later version
        for key in ("event_onsets", "event_labels", "event_meta", "blinks",
                    "interpolated_blinks", "excluded"):
            assert key not in st
        assert "mask" not in st["data"]
        assert set(st["data"]) == {"values", "layers", "layer_names", "rec_layers"}
        for key, arr in st["data"]["layers"].items():
            assert arr.dtype == np.uint16 and arr.shape == old["data"]["values"][key].shape
        np.testing.assert_array_equal(st["events"]["onsets"], old["event_onsets"])
        np.testing.assert_array_equal(st["events"]["labels"], old["event_labels"])
        assert st["events"]["units"] == "ms"
        assert set(st["events"]["meta"]) == set(old["event_meta"])
        for key in ("name", "fs", "params", "history", "info", "inplace"):
            assert st[key] == old[key], key
        np.testing.assert_array_equal(st["tx"], old["tx"])
        for key in old["data"]["values"]:
            np.testing.assert_array_equal(st["data"]["values"][key], old["data"]["values"][key])

    def test_input_not_changed(self):
        old = format2_state("eyedata.h5")
        state.migrate(old, "EyeData")
        assert "mask" in old["data"] and "blinks" in old and "event_onsets" in old
        assert old["format_version"] == 2

    @pytest.mark.parametrize("name", EYEDATA_FILES)
    def test_masks_by_origin(self, name):
        """The old mask is split into layers; masks of unknown origin
        (off-screen, divergence, manual) become the "user" layer."""
        old = format2_state(f"{name}.h5")
        d = state.from_state(format2_state(f"{name}.h5"))
        data = d.data
        for key, m in old["data"]["mask"].items():
            m = np.asarray(m, dtype=bool)
            # the old mask is the strict mask (interpolated samples masked)
            np.testing.assert_array_equal(d.mask(key, policy="strict"), m, err_msg=key)
            nan = np.isnan(old["data"]["values"][key])
            explained = nan | data.get_rec_layer("excluded")
            for layer in ("blink", "interpolated"):
                if layer in data.layer_names:
                    explained |= data.get_layer(key, layer)
            # the rest of the old mask is the user layer, nothing else
            user = data.get_layer(key, "user") if "user" in data.layer_names \
                else np.zeros_like(m)
            np.testing.assert_array_equal(user, m & ~explained, err_msg=key)
        assert "offscreen" not in data.layer_names  # origin lost
        assert "divergence" not in data.layer_names

    def test_which_origin_became_user(self):
        d = state.from_state(format2_state("eyedata.h5"))
        # mask_intervals on the left pupil, 2400-2440 ms
        user = d.data.get_layer("left_pupil", "user")
        np.testing.assert_array_equal(np.flatnonzero(user), np.arange(1200, 1220))
        # mask_offscreen_coords: left x at 1000-1029 (both left gaze variables)
        for key in ("left_x", "left_y"):
            np.testing.assert_array_equal(np.flatnonzero(d.data.get_layer(key, "user")),
                                          np.arange(1000, 1030))
        assert not d.data.get_layer("right_pupil", "user").any()

    def test_applied_blinks(self):
        """Blinks applied as mask are the blink layer only (not "user"), so
        that removing the blink layer unmasks them."""
        d = state.from_state(format2_state("pupildata.h5"))
        assert "_blink_applied" not in d.data.layer_names
        for key in ("left_pupil", "right_pupil"):
            blink = d.data.get_layer(key, "blink")
            assert blink.any()
            assert not (d.data.get_layer(key, "user") & blink).any()
            assert d.mask(key, policy="strict")[blink].all()  # some are interpolated
        d.clear_layer("blink")
        for key in ("left_pupil", "right_pupil"):
            np.testing.assert_array_equal(
                d.mask(key), np.isnan(d.data[key]) | d.data.get_rec_layer("excluded"))

    def test_blinks_not_applied_mask_the_data(self):
        """Blinks stored without applying them as mask mask the data after loading."""
        old = format2_state("pupildata.h5")
        old["data"]["mask"] = {k: np.zeros_like(m) for k, m in old["data"]["mask"].items()}
        d = state.from_state(old)
        for key in ("left_pupil", "right_pupil"):
            blink = d.data.get_layer(key, "blink")
            assert blink.any() and d.mask(key)[blink].all()
            assert not d.data.get_layer(key, "user").any()

    def test_development_layer_blink_applied(self):
        """Format-3 files of development versions with the layer `_blink_applied`."""
        d = pp.EyeData(left_pupil=np.arange(6.), sampling_rate=100)
        d.data.set_layer("left_pupil", "_blink_applied", [0, 1, 0, 0, 0, 0])
        d.data.set_layer("left_pupil", "blink", [0, 1, 1, 0, 0, 0])
        d.data.set_layer("left_pupil", "user", [0, 0, 0, 0, 1, 0])
        r = state.from_state(state.to_state(d))
        assert r.data.layer_names == ["blink", "user"]
        np.testing.assert_array_equal(r.mask("left", "pupil"), [0, 1, 1, 0, 1, 0])

    def test_excluded_is_recording_wide(self):
        old = format2_state("eyedata.h5")
        d = state.from_state(format2_state("eyedata.h5"))
        assert d.get_excluded().intervals == [tuple(r) for r in old["excluded"]["intervals"]]
        excl = d.data.get_rec_layer("excluded")
        np.testing.assert_array_equal(np.flatnonzero(excl),
                                      np.r_[0:100, 2590:2700])
        for key in d.data:
            assert not (d.data.get_layer(key, "user") & excl).any()

    def test_original_migrated(self):
        d = state.from_state(format2_state("eyedata.h5"))
        assert isinstance(d.original, pp.EyeData)
        st = state.to_state(d)
        assert st["original"]["format_version"] == 3
        assert "event_onsets" not in st["original"]

    def test_segments_epochs(self):
        old = format2_state("segmented.h5")
        st = state.migrate(format2_state("segmented.h5"), "SegmentedEyeData")
        ep = st["intervals"]
        assert ep["class"] == "Epochs"
        assert set(ep) == {"class", "onsets", "window", "labels", "meta", "label"}
        np.testing.assert_array_equal(ep["onsets"], old["intervals"]["event_onsets"])
        assert ep["window"] == tuple(old["intervals"]["interval_window"])
        s = state.from_state(st)
        assert isinstance(s.intervals, pp.Epochs)
        np.testing.assert_array_equal(np.asarray(s.intervals), old["intervals"]["intervals"])

    def test_segments_in_samples(self):
        st = format2_state("segmented.h5")
        iv = st["intervals"]
        fs = st["sampling_rate"]
        iv["units"] = "samples"
        iv["event_onsets"] = np.asarray(iv["event_onsets"]) * fs / 1000.
        iv["interval_window"] = tuple(np.asarray(iv["interval_window"]) * fs / 1000.)
        with warnings_logged() as logs:
            s = state.from_state(st)
        assert any("sample indices" in m for m in logs)
        assert s.intervals.window == (-200., 800.)
        np.testing.assert_allclose(s.intervals.onsets,
                                   format2_state("segmented.h5")["intervals"]["event_onsets"])

    def test_segments_not_event_locked(self):
        st = format2_state("segmented.h5")
        st["intervals"]["interval_window"] = None
        s = state.from_state(st)
        assert isinstance(s.intervals, pp.Intervals)
        assert state.to_state(s)["intervals"]["class"] == "Intervals"
        assert s.intervals.intervals == [tuple(r) for r in
                                         format2_state("segmented.h5")["intervals"]["intervals"]]


def assert_same_state(a, b):
    assert_state_equal(unittest.TestCase(), state.to_state(a), state.to_state(b))


def manifest_files():
    return sorted(load_manifest()["files"])


class TestFormat3RoundTrip:
    """Migrated objects saved as format 3 load unchanged (pickle and HDF5)."""

    @pytest.mark.parametrize("fname", manifest_files())
    @pytest.mark.parametrize("ext", ["pypil", "h5"])
    def test_fixture(self, fname, ext, tmp_path):
        with pp.nologging():
            obj = pp.read_pickle(os.path.join(FIXTURE_DIR, fname))
        if isinstance(obj, dict) and ext == "h5":
            pytest.skip("HDF5 stores eyedata and segmented objects only")
        out = str(tmp_path / f"out.{ext}")
        pp.write_pickle(obj, out)
        if ext == "h5":
            assert hdf5.read_state(out)["format_version"] == 3
        obj2 = pp.read_pickle(out)
        if isinstance(obj, dict):
            assert set(obj2) == set(obj)
            for k in obj:
                assert_same_state(obj[k], obj2[k])
        else:
            assert type(obj2) is type(obj)
            assert_same_state(obj, obj2)
        expected = load_manifest()["files"][fname]
        assert_matches(json.loads(json.dumps(facts(obj2))), expected, fname)

    @pytest.mark.parametrize("fname", ["pupildata.h5", "eyedata.h5", "segmented.h5"])
    def test_format3_file_not_migrated(self, fname, tmp_path):
        obj = pp.read_pickle(os.path.join(FIXTURE_DIR, fname))
        out = str(tmp_path / "out.h5")
        pp.write_pickle(obj, out)
        st = hdf5.read_state(out)
        assert state.migrate(st, st["class"]) is st
        # written again: the same state
        pp.write_pickle(pp.read_pickle(out), str(tmp_path / "again.h5"))
        assert_state_equal(unittest.TestCase(), st, hdf5.read_state(str(tmp_path / "again.h5")))

    def test_pickle_holds_format3_state(self):
        obj = pp.read_pickle(os.path.join(FIXTURE_DIR, "eyedata.pypil"))
        stored = []
        with patch.object(pp.EyeData, "__setstate__", lambda self, st: stored.append(st)):
            pickle.loads(pickle.dumps(obj))
        assert stored[0]["format_version"] == 3 and "events" in stored[0]

    @pytest.mark.parametrize("fname", ["eyedata_v0.pkl", "pupildata_v0.pkl", "gazedata_v0.pkl",
                                       "fakeeyedata_v0.pkl", "segmented_v0.pkl",
                                       "group_segmented_v0.pkl", "eyedata_v1.pypil",
                                       "eyedata_v1.h5", "segmented_v1.h5"])
    @pytest.mark.parametrize("ext", ["pypil", "h5"])
    def test_format0_and_1(self, fname, ext, tmp_path):
        """Older files go through the whole chain of migrations."""
        obj = pp.read_pickle(os.path.join(STATE_DIR, fname))
        out = str(tmp_path / f"out.{ext}")
        pp.write_pickle(obj, out)
        assert_same_state(obj, pp.read_pickle(out))


class TestRecordingVersions:

    def test_versions_of_loaded_objects(self):
        with state._recording_versions() as seen:
            pp.read_pickle(os.path.join(FIXTURE_DIR, "eyedata.pypil"))
        # the object and its original
        assert seen == [("EyeData", 2), ("EyeData", 2)]
        with state._recording_versions() as seen:
            pp.read_pickle(os.path.join(FIXTURE_DIR, "dict_of_eyedata.pypil"))
        assert {v for _, v in seen} == {2} and len(seen) == 3
        assert state._versions_seen is None
