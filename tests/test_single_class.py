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
