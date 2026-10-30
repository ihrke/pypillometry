import unittest
import numpy as np
import pypillometry as pp


class TestInplaceDefault(unittest.TestCase):
    """inplace resolution: method argument > object setting > package default"""

    def setUp(self):
        self.x = np.random.randn(100)

    def tearDown(self):
        pp.set_default_inplace(True)

    def make(self, **kwargs):
        return pp.PupilData(left_pupil=self.x, sampling_rate=100, **kwargs)

    def test_package_default_is_inplace(self):
        self.assertTrue(pp.get_default_inplace())
        d = self.make()
        self.assertIs(d.scale(), d)

    def test_package_default_copy(self):
        pp.set_default_inplace(False)
        d = self.make()
        r = d.scale()
        self.assertIsNot(r, d)
        np.testing.assert_array_equal(d["left_pupil"], self.x)

    def test_object_setting_overrides_package(self):
        d = self.make(inplace=False)
        self.assertIsNot(d.scale(), d)
        pp.set_default_inplace(False)
        d = self.make(inplace=True)
        self.assertIs(d.scale(), d)

    def test_method_argument_overrides_object(self):
        d = self.make(inplace=False)
        self.assertIs(d.scale(inplace=True), d)
        d = self.make(inplace=True)
        self.assertIsNot(d.scale(inplace=False), d)

    def test_setting_read_at_call_time(self):
        d = self.make()
        pp.set_default_inplace(False)
        self.assertIsNot(d.scale(), d)

    def test_copy_keeps_object_setting(self):
        d = self.make(inplace=False)
        self.assertFalse(d.copy().inplace)

    def test_to_pupildata_inherits_setting(self):
        d = pp.EyeData(left_x=self.x, left_y=self.x, left_pupil=self.x,
                       sampling_rate=100, inplace=True)
        self.assertTrue(d.to_pupildata("left").inplace)
        d = pp.EyeData(left_x=self.x, left_y=self.x, left_pupil=self.x,
                       sampling_rate=100)
        self.assertIsNone(d.to_pupildata("left").inplace)

    def test_internal_mask_calls_with_copy_default(self):
        # methods that mask internally must still mask the returned copy
        from pypillometry.eyedata.experimental_setup import ExperimentalSetup
        pp.set_default_inplace(False)
        x = np.array([100., 200, 300, 1500, 500])
        d = pp.GazeData(sampling_rate=10, left_x=x, left_y=x, right_x=x, right_y=x,
                        experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024)))
        r = d.mask_offscreen_coords(eyes="left")
        self.assertIsNot(r, d)
        self.assertEqual(r.data.mask["left_x"][3], 1)
        self.assertEqual(d.data.mask["left_x"][3], 0)
        d.data["right_x"] = x + np.array([0., 0, 0, 0, 900])
        r = d.mask_eye_divergences(threshold=.5)
        self.assertGreater(r.data.mask["left_x"].sum(), 0)
        self.assertEqual(d.data.mask["left_x"].sum(), 0)

    def test_setter_rejects_non_bool(self):
        with self.assertRaises(TypeError):
            pp.set_default_inplace(None)


if __name__ == '__main__':
    unittest.main()
