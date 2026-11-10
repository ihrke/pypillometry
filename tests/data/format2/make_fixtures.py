"""
Script that wrote the format-version-2 fixtures in this folder.

Run it manually from the root of the repository::

    python tests/data/format2/make_fixtures.py

It builds small synthetic objects with the installed pypillometry, saves each
of them as pickle (``.pypil``, compressed) and HDF5 (``.h5``), and writes
``manifest.json`` with the facts that a migration of the files must preserve
(computed by `facts` in ``tests/test_format2_fixtures.py``).

The files are meant to stay fixed: they are what pypillometry wrote in state
format 2 (before the 2.2 refactor). Re-running the script with a later version
writes files in that version's format instead.
"""
import datetime
import glob
import json
import os
import subprocess
import sys
from importlib.metadata import version

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import pypillometry as pp  # noqa: E402
from pypillometry import u  # noqa: E402
from pypillometry.eyedata.spatial_calibration import SpatialCalibration  # noqa: E402
from pypillometry.state import FORMAT_VERSION  # noqa: E402
from tests.test_format2_fixtures import facts  # noqa: E402

N, FS = 3000, 500.
SETUP = dict(screen_resolution=(1280, 1024), physical_screen_size=("30 cm", "20 cm"),
             eye_to_screen_center="60 cm")


def signals(seed):
    """Pupil (with NaN blinks), gaze (with off-screen samples) and events.

    Values are rounded to 0.1 (as stored by eye trackers), which keeps the
    compressed files small.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N) * 1000. / FS
    out = {}
    for eye, blinks in (("left", [(500, 560), (1800, 1850), (2600, 2620)]),
                        ("right", [(510, 565), (1300, 1340), (2600, 2625)])):
        p = np.round(1000 + 30 * np.sin(2 * np.pi * t / 2500.) + rng.normal(0, 0.5, N), 1)
        for on, off in blinks:
            p[on:off] = np.nan
        out[f"{eye}_pupil"] = p
        out[f"{eye}_x"] = np.round(640 + rng.normal(0, 5, N), 1)
        out[f"{eye}_y"] = np.round(512 + rng.normal(0, 5, N), 1)
    out["left_x"][1000:1030] = 1350.   # off-screen
    out["right_y"][2200:2210] = -15.   # off-screen
    onsets = np.array([300., 1100., 1900., 2700., 3500., 4300., 5100.])
    labels = np.array(["stim_1_a", "resp_1", "stim_2_b", "resp_2", "stim_3_a",
                       "resp_3", "stim_4_b"])
    return t, out, onsets, labels


def with_meta(d):
    """Set event metadata: trial (from the labels) and condition."""
    ev = (d.get_events().parse_labels(r"_(?P<trial>\d+)")
          .parse_labels(r"stim_\d+_(?P<cond>[ab])")
          .set_meta("rt", [612.5, 580.0, np.nan], select="resp_"))
    return d.set_events(ev)


def make_pupildata():
    t, s, onsets, labels = signals(1)
    d = pp.PupilData(time=t, sampling_rate=FS, left_pupil=s["left_pupil"],
                     right_pupil=s["right_pupil"], event_onsets=onsets,
                     event_labels=labels, name="fixture_pupildata")
    d = d.pupil_blinks_detect(apply_mask=True).pupil_blinks_interpolate()
    return with_meta(d.exclude_intervals([(5800, 6000)]))


def make_gazedata():
    t, s, onsets, labels = signals(2)
    d = pp.GazeData(time=t, sampling_rate=FS, left_x=s["left_x"], left_y=s["left_y"],
                    right_x=s["right_x"], right_y=s["right_y"], event_onsets=onsets,
                    event_labels=labels, name="fixture_gazedata",
                    experimental_setup=pp.ExperimentalSetup(**SETUP))
    return d.mask_offscreen_coords()


def calibration():
    pts = np.array([[640., 512.], [640., 87.], [640., 936.], [115., 512.], [1165., 512.]])
    err = np.array([[0.3, -13.3, -1.6], [0.3, 13.2, -0.2], [0.42, 16.1, 9.2],
                    [0.25, 8.0, -3.0], [0.5, -20.0, 4.5]])
    return {eye: SpatialCalibration(eye, "HV5", pts, pts + err[:, 1:] * f, err * f,
                                    (1280, 1024))
            for eye, f in (("left", 1.0), ("right", 0.8))}


def make_eyedata():
    t, s, onsets, labels = signals(3)
    d = pp.EyeData(time=t, sampling_rate=FS, event_onsets=onsets, event_labels=labels,
                   experimental_setup=pp.ExperimentalSetup(**SETUP),
                   calibration=calibration(), name="fixture_eyedata", keep_orig=True,
                   info={"subject": "s01", "session": 2}, **s)
    d = d.pupil_blinks_detect(apply_mask=True).pupil_blinks_interpolate()
    # the excluded period touches the third blink (2600-2625), which is removed
    d = d.exclude_intervals([(0, 200), (5180, 5400)])
    d = with_meta(d.mask_offscreen_coords())
    # manual mask of an artifact in the left pupil
    d = d.mask_intervals(pp.Intervals([(2400, 2440)], units="ms"),
                         eyes=["left"], variables=["pupil"])
    return d


def make_segmented(d):
    iv = d.get_intervals("stim", window=(-200, 800) * u.ms)
    return d.get_segments(iv, "left_pupil")


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main():
    for pattern in ("*.pypil", "*.h5"):
        for f in glob.glob(os.path.join(HERE, pattern)):
            os.remove(f)
    with pp.nologging():
        pupil, gaze, eye = make_pupildata(), make_gazedata(), make_eyedata()
        objects = {"pupildata": pupil, "gazedata": gaze, "eyedata": eye,
                   "segmented": make_segmented(eye)}
    files = {}
    for name, obj in objects.items():
        for ext in ("pypil", "h5"):
            fname = f"{name}.{ext}"
            obj.write_file(os.path.join(HERE, fname))
            files[fname] = facts(obj)
    fname = "pupildata_uncompressed.pypil"
    pupil.write_file(os.path.join(HERE, fname), compress=False)
    files[fname] = facts(pupil)
    fname = "dict_of_eyedata.pypil"
    dct = {"eye": eye, "pupil": pupil}
    pp.write_pickle(dct, os.path.join(HERE, fname))
    files[fname] = facts(dct)
    manifest = {
        "description": "format-2 fixture files written by pypillometry before the "
                       "2.2 refactor (see tests/test_format2_fixtures.py)",
        "format_version": FORMAT_VERSION,
        "pypillometry_commit": git("rev-parse", "HEAD"),
        "package_modified": bool(git("status", "--porcelain", "--", "pypillometry")),
        "pypillometry_version": version("pypillometry"),
        "date": datetime.date.today().isoformat(),
        "files": files,
    }
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    for fname in sorted(files):
        print(f"{fname}: {os.path.getsize(os.path.join(HERE, fname))} bytes")


if __name__ == "__main__":
    main()
