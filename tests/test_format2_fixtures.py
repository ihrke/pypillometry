"""
Format-2 fixture files: files written by pypillometry before the 2.2 refactor.

``tests/data/format2/make_fixtures.py`` wrote the files and, for each of
them, the facts that a migration of the file must preserve
(``manifest.json``, computed with `facts` below). The tests load every file
with the current code and compare it with the manifest.

When the data model changes, `facts` is adapted to read the same facts from
the new objects (e.g., blinks from a mask layer); the manifest stays fixed.
"""
import json
import math
import os

import numpy as np
import pytest

import pypillometry as pp
from pypillometry.segmented import SegmentedEyeData

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "format2")


def jsonable(x):
    """Convert to plain JSON types (lists, floats, ints, str, None)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if x is None or isinstance(x, str):
        return x
    if isinstance(x, np.str_):
        return str(x)
    raise TypeError(f"cannot convert {type(x)}")


def intervals_mask(iv, n):
    """Boolean mask (n samples) of Intervals in samples."""
    m = np.zeros(n, dtype=bool)
    for on, off in iv.intervals:
        m[max(int(on), 0):min(int(off), n)] = True
    return m


def events_facts(onsets, labels, meta):
    return {"onsets": np.asarray(onsets, dtype=float), "labels": [str(lab) for lab in labels],
            "meta": {k: np.asarray(v) for k, v in (meta or {}).items()}}


def eyedata_facts(d):
    n = len(d.tx)
    out = {"class": type(d).__name__, "name": d.name, "n": n, "fs": d.fs,
           "tx_first": d.tx[0], "tx_last": d.tx[-1], "eyes": sorted(d.eyes),
           "variables": sorted(d.variables), "keys": sorted(d.data.keys())}
    blinks = d.get_blinks(units="samples")
    interp = d.get_interpolated_blinks(units="samples")
    excluded = d.get_excluded(units="samples")
    excl_mask = intervals_mask(excluded, n)
    out["blinks"] = {k: np.asarray(iv.intervals).reshape(-1, 2) for k, iv in blinks.items()}
    out["interpolated_blinks"] = {k: np.asarray(iv.intervals).reshape(-1, 2)
                                  for k, iv in interp.items()}
    out["excluded_ms"] = np.asarray(d.get_excluded().intervals).reshape(-1, 2)
    out["data_nansum"], out["masked"], out["masked_unexplained"] = {}, {}, {}
    for k in d.data.keys():
        v = np.asarray(d.data[k], dtype=float)
        # the mask of format 2 is the strict mask (interpolated samples masked)
        m = d.mask(k, policy="strict")
        explained = np.isnan(v) | excl_mask
        if k in blinks:
            explained |= intervals_mask(blinks[k], n)
        if k in interp:
            explained |= intervals_mask(interp[k], n)
        out["data_nansum"][k] = np.nansum(v)
        out["masked"][k] = int(m.sum())
        out["masked_unexplained"][k] = int((m & ~explained).sum())
    out["events"] = events_facts(d.events.onsets, d.events.labels, d.events.meta)
    setup = getattr(d, "experimental_setup", None)
    out["setup"] = None if setup is None else setup.to_dict()
    cal = getattr(d, "calibration", None)
    out["calibration"] = None if cal is None else {
        eye: {"eye": c.eye, "model": c.model, "points": c.points, "measured": c.measured,
              "errors": c.errors, "screen_resolution": c.screen_resolution}
        for eye, c in cal.items()}
    out["history_len"] = len(d.history)
    out["history"] = [h["funcname"] for h in d.history]
    out["info"] = d.info
    out["has_original"] = d.original is not None
    return out


def segmented_facts(s):
    ep = s.intervals  # Epochs (in ms)
    return {"class": type(s).__name__, "name": s.name, "variable": s.variable,
            "n_segments": s.n_segments, "n_timepoints": s.n_timepoints,
            "sampling_rate": s.sampling_rate, "tx_first": s.tx[0], "tx_last": s.tx[-1],
            "masked": int(np.ma.getmaskarray(s.data).sum()),
            "data_sum": float(np.ma.getdata(s.data).sum()),
            "intervals": {"units": "ms", "intervals": np.asarray(ep),
                          "interval_window": ep.window,
                          "events": events_facts(ep.onsets, ep.labels, ep.meta)}}


def facts(obj):
    """The facts about a loaded object that a migration must preserve."""
    if isinstance(obj, dict):
        return {"class": "dict", "entries": {k: facts(v) for k, v in obj.items()}}
    if isinstance(obj, SegmentedEyeData):
        return jsonable(segmented_facts(obj))
    return jsonable(eyedata_facts(obj))


def assert_matches(actual, expected, path="", rtol=1e-12):
    """Compare JSON-like structures (floats with tolerance, NaN == NaN)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for k in expected:
            assert_matches(actual[k], expected[k], f"{path}.{k}", rtol)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_matches(a, e, f"{path}[{i}]", rtol)
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)), path
        if math.isnan(expected):
            assert math.isnan(actual), path
        else:
            assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=1e-9), \
                f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


# one data class since 2.2: PupilData and GazeData objects load as EyeData
_NEW_CLASS = {"PupilData": "EyeData", "GazeData": "EyeData"}


def load_manifest():
    """The manifest, with the class names of the current version."""
    with open(os.path.join(FIXTURE_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    for entry in manifest["files"].values():
        for e in [entry, *entry.get("entries", {}).values()]:
            e["class"] = _NEW_CLASS.get(e["class"], e["class"])
    return manifest


def load_fixture(fname):
    """Load a fixture file (pickle or HDF5, any class)."""
    return pp.read_pickle(os.path.join(FIXTURE_DIR, fname))


def test_manifest_lists_all_files():
    files = {f for f in os.listdir(FIXTURE_DIR) if f.endswith((".pypil", ".h5"))}
    assert files == set(load_manifest()["files"])


def _manifest_files():
    # the manifest is missing only while make_fixtures.py runs (it imports this module)
    if not os.path.exists(os.path.join(FIXTURE_DIR, "manifest.json")):
        return []
    return sorted(load_manifest()["files"])


@pytest.mark.parametrize("fname", _manifest_files())
def test_fixture_matches_manifest(fname):
    expected = load_manifest()["files"][fname]
    with pp.nologging():
        obj = load_fixture(fname)
    assert_matches(json.loads(json.dumps(facts(obj))), expected, fname)
