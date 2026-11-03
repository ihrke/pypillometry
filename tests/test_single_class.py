"""One data class (EyeData) for every set of channels."""
import unittest

import numpy as np

import pypillometry as pp


def _gaze_only(n=100):
    rng = np.random.default_rng(1)
    return pp.EyeData(left_x=rng.normal(500, 10, n), left_y=rng.normal(400, 10, n),
                      sampling_rate=100)


def _pupil_only(n=100):
    return pp.EyeData(left_pupil=np.full(n, 5.0), sampling_rate=100)


class TestRequire(unittest.TestCase):
    def test_present(self):
        d = pp.get_rlmw_002_short()
        d._require("pupil")
        d._require("x", "y", "pupil")

    def test_missing_lists_channels(self):
        d = pp.get_rlmw_002_short().drop_data(variables=["x", "y"])
        with self.assertRaisesRegex(ValueError, "no x, y data; available channels: "
                                    "left_pupil, right_pupil"):
            d._require("pupil", "x", "y")


class TestConstructor(unittest.TestCase):
    def test_pupil_only(self):
        d = _pupil_only()
        self.assertIs(type(d), pp.EyeData)
        self.assertEqual(list(d.data.keys()), ["left_pupil"])

    def test_gaze_only(self):
        d = _gaze_only()
        self.assertIs(type(d), pp.EyeData)
        self.assertEqual(list(d.data.keys()), ["left_x", "left_y"])

    def test_mixed_eyes(self):
        d = pp.EyeData(left_pupil=np.ones(10), right_x=np.ones(10), right_y=np.ones(10),
                       sampling_rate=100)
        self.assertEqual(sorted(d.data.keys()), ["left_pupil", "right_x", "right_y"])

    def test_no_channel(self):
        with self.assertRaisesRegex(ValueError, "At least one channel"):
            pp.EyeData(sampling_rate=100)

    def test_x_without_y(self):
        with self.assertRaisesRegex(ValueError, "left_x and left_y"):
            pp.EyeData(left_x=np.ones(10), left_pupil=np.ones(10), sampling_rate=100)


class TestMissingChannels(unittest.TestCase):
    """Pupil and gaze methods say which data are missing."""

    def test_pupil_methods_on_gaze_only(self):
        d = _gaze_only()
        calls = [lambda: d.pupil_lowpass_filter(5), lambda: d.pupil_smooth_window(),
                 lambda: d.pupil_merge_eyes(), lambda: d.pupil_blinks_detect(),
                 lambda: d.pupil_blinks_interpolate(), lambda: d.pupil_estimate_baseline(),
                 lambda: d.pupil_estimate_response(),
                 lambda: d.plot.pupil_plot(), lambda: d.plot.pupil_plot_segments()]
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "no pupil data; available "
                                            "channels: left_x, left_y"):
                    call()

    def test_gaze_methods_on_pupil_only(self):
        d = _pupil_only()
        calls = [lambda: d.mask_eye_divergences(), lambda: d.mask_offscreen_coords(),
                 lambda: d.correct_pupil_foreshortening(),
                 lambda: d.fit_foreshortening("left"),
                 lambda: d.plot.plot_heatmap(), lambda: d.plot.plot_scanpath(),
                 lambda: d.plot.plot_pupil_foreshortening_error_surface()]
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "no x, y data; available "
                                            "channels: left_pupil"):
                    call()

    def test_generic_methods_work(self):
        for d in (_gaze_only(), _pupil_only()):
            d.downsample(50, inplace=False)
            d.crop(0, 500, inplace=False)
            d.quality_report()
            d.summary()


class TestSetupAndCalibration(unittest.TestCase):
    """Every eyedata object has an experimental setup and calibration (4.5)."""

    def setUp(self):
        self.d = pp.get_rlmw_002_short().crop(0, 2000, inplace=False)
        self.d.set_experimental_setup(screen_resolution=(1280, 1024),
                                      physical_screen_size=("30 cm", "20 cm"),
                                      eye_to_screen_center="60 cm")
        self.d.calibration = {"left": pp.SpatialCalibration(
            eye="left", model="HV9", points=np.zeros((2, 2)), measured=np.zeros((2, 2)),
            errors=np.zeros(2))}

    def test_pupil_only_from_constructor(self):
        d = _pupil_only()
        self.assertIsNone(d.experimental_setup)
        self.assertIsNone(d.calibration)
        d.set_experimental_setup(screen_resolution=(1280, 1024))
        self.assertEqual(d.experimental_setup.screen_resolution, (1280, 1024))

    def test_kept_without_gaze_on_save(self):
        import os
        import tempfile
        p = self.d.drop_data(variables=["x", "y"])
        for ext in (".pypil", ".h5"):
            with self.subTest(ext=ext), tempfile.TemporaryDirectory() as tmp:
                fname = os.path.join(tmp, "d" + ext)
                p.write_file(fname)
                q = pp.read_pickle(fname)
                self.assertEqual(sorted(q.data.keys()), ["left_pupil", "right_pupil"])
                self.assertEqual(q.experimental_setup.screen_resolution, (1280, 1024))
                self.assertEqual(list(q.calibration), ["left"])

    def test_state_without_setup(self):
        """States of development versions of format 3 without the two fields."""
        from pypillometry import state
        st = state.to_state(self.d)
        del st["experimental_setup"], st["calibration"]
        q = state.from_state(st)
        self.assertIsNone(q.experimental_setup)
        self.assertIsNone(q.calibration)


class TestDropData(unittest.TestCase):
    def test_keeps_class_and_attributes(self):
        from pypillometry.signal.fake import generate_foreshortening_data
        f = generate_foreshortening_data(duration=1, fs=100, seed=1)
        p = f.drop_data(variables=["x", "y"])
        self.assertIs(type(p), pp.FakeEyeData)
        self.assertEqual(p.variables, ["pupil"])
        self.assertIs(p.sim_fct, f.sim_fct)
        self.assertEqual(p.experimental_setup.to_dict(), f.experimental_setup.to_dict())
