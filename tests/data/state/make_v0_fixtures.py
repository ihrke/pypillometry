"""
Script that wrote the format-version-0 fixtures in this folder.

It was run with pypillometry at commit ff0d643 (before the state was
versioned), as ``python make_v0_fixtures.py tests/data/state``. With later
versions it writes files in the current format instead.
"""
import datetime, sys
import numpy as np
import pypillometry as pp
from pypillometry.eyedata.spatial_calibration import SpatialCalibration
from pypillometry.signal.fake import generate_foreshortening_data

out = sys.argv[1]
src = pp.get_example_data("rlmw_002_short").sub_slice(4.5, 8.5, units="sec")
setup = pp.ExperimentalSetup(screen_resolution=(1280, 1024),
                             physical_screen_size=("30 cm", "20 cm"),
                             eye_to_screen_center="60 cm",
                             camera_spherical=("20 deg", "-90 deg", "60 cm"))
pts = np.array([[640., 512.], [640., 87.], [640., 936.]])
err = np.array([[0.3, -13.3, -1.6], [0.3, 13.2, -0.2], [0.42, 16.1, 9.2]])
cal = {"left": SpatialCalibration("left", "HV3", pts, pts + err[:, 1:], err,
                                  (1280, 1024))}
info = {"eyelink_info": {"meas_date": datetime.datetime(2024, 5, 3, 10, 20, 30),
                         "screen_coords": np.array([1280, 1024]),
                         "sample_fields": ["xpos_left", "ps_left"],
                         "calibrations": [{"eye": "left", "model": "HV3",
                                           "onset": 1.5}]},
        "subject": "002"}
d = pp.EyeData(time=src.tx, left_x=src["left_x"].data, left_y=src["left_y"].data,
               left_pupil=src["left_pupil"].data, right_x=src["right_x"].data,
               right_y=src["right_y"].data, right_pupil=src["right_pupil"].data,
               event_onsets=src.event_onsets, event_labels=src.event_labels,
               sampling_rate=src.fs, experimental_setup=setup, calibration=cal,
               name="fixture_eyedata", keep_orig=True, info=info)
d = d.pupil_blinks_detect(apply_mask=True).pupil_blinks_interpolate()
d = d.scale().pupil_lowpass_filter(cutoff=5)
pp.write_pickle(d, f"{out}/eyedata_v0.pkl")

p = d.to_pupildata()
p.name = "fixture_pupildata"
pp.write_pickle(p, f"{out}/pupildata_v0.pkl")
g = d.to_gazedata()
g.name = "fixture_gazedata"
pp.write_pickle(g, f"{out}/gazedata_v0.pkl")

f = generate_foreshortening_data(duration=2, fs=100, seed=1)
f.name = "fixture_fake"
pp.write_pickle(f, f"{out}/fakeeyedata_v0.pkl")

iv = d.get_intervals("F_", interval=(-200, 800), units="ms")
seg = d.get_segments(iv, "left_pupil")
pp.write_pickle(seg, f"{out}/segmented_v0.pkl")
grp = pp.group_segments([seg, seg])
pp.write_pickle(grp, f"{out}/group_segmented_v0.pkl")
