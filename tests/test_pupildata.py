import unittest
import sys
import numpy as np
sys.path.insert(0,"..")
import pypillometry as pp
import pytest

class TestPupilData(unittest.TestCase):
    def setUp(self):
        #self.d=pp.PupilData.from_file("data/test.pd")
        pass
    def test_from_file(self):
        d = pp.get_example_data("rlmw_002_short")
        print(d)
    def test_scale(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        d=d.scale("pupil")
        self.assertEqual(d.data["left","pupil"].mean(), 0)
        self.assertEqual(d.data["right","pupil"].mean(), 0)
        self.assertEqual(d.params["scale"]["mean"]["left"]["pupil"], 1.5)
        self.assertEqual(d.params["scale"]["mean"]["right"]["pupil"], 3.5)
    def test_scale_fixed(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        scalepars = {
            "left": {"pupil": 1},
            "right": {"pupil": 3}
        }
        sdpars = {
            "left": {"pupil": 0.5},
            "right": {"pupil": 0.5}
        }
        d=d.scale("pupil", mean=scalepars, sd=sdpars)
        self.assertEqual(d.data["left","pupil"].mean(), 1)
        self.assertEqual(d.data["right","pupil"].mean(), 1)
    def test_unscale(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        d=d.scale("pupil")
        d=d.unscale("pupil")
        self.assertEqual(d.data["left","pupil"].mean(), 1.5)
        self.assertEqual(d.data["right","pupil"].mean(), 3.5)

    def test_fill_time_discontinuities(self):
        # three segments sampled at 2 ms, separated by gaps of 500 ms
        segs = [np.arange(0, 1000, 2), np.arange(1500, 2500, 2), np.arange(3000, 4000, 2)]
        t = np.concatenate(segs).astype(float)
        x = np.concatenate([np.full(len(s), i+1.) for i, s in enumerate(segs)])
        d = pp.PupilData(left_pupil=x, time=t)
        self.assertEqual(len(d.tx), 2000)
        self.assertEqual(len(d.data["left_pupil"]), len(d.tx))
        np.testing.assert_allclose(np.diff(d.tx), 2.)
        # original samples are kept at their time points, gaps are filled
        # with NaN (masked)
        for i, s in enumerate(segs):
            ix = np.isin(d.tx, s)
            self.assertEqual(ix.sum(), len(s))
            np.testing.assert_array_equal(d.data["left_pupil"][ix], i+1.)
        self.assertEqual(np.isnan(d.data["left_pupil"]).sum(), 2000-len(t))
        self.assertEqual(d.data.mask["left_pupil"].sum(), 2000-len(t))
        

if __name__ == '__main__':
    unittest.main()