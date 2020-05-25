import unittest
import sys
import numpy as np
#sys.path.insert(0,"..")
#import pypillometry as pp
from .. import *

class TestPupilData(unittest.TestCase):
    def setUp(self):
        self.dfake=create_fake_pupildata(ntrials=100, fs=500)
        self.d=PupilData.from_file("data/test.pd")
    def test_from_file(self):
        d=PupilData.from_file("data/test.pd")
        self.assertEqual(d.__class__, PupilData)
    def test_create_fakedata(self):
        d=create_fake_pupildata(ntrials=100)
        self.assertEqual(d.__class__, FakePupilData)
    def test_history(self):
        d=create_fake_pupildata(ntrials=100)
        self.assertEqual(len(d.history), 0)
        d=d.drop_original()
        self.assertEqual(len(d.history), 1)
    def test_drop_original(self):
        d2=self.dfake.drop_original()
        self.assertIsNone(d2.original)
        self.assertLess(d2.size_bytes(), self.dfake.size_bytes())
    def test_reset_time(self):
        d=self.dfake.reset_time(t0=500)
        self.assertEqual(d.tx[0], 500)
        d=d.reset_time()
        self.assertEqual(d.tx[0], 0)
    def test_summary(self):
        self.dfake.summary()
        self.d.summary()
    def test_len(self):
        self.assertEqual(self.d.sy.size, len(self.d))
    def test_nevents(self):
        self.d.nevents()
    def test_nblinks(self):
        self.assertEqual(self.d.nblinks(), 0)
        d=self.d.blinks_detect()
        self.assertGreater(d.nblinks(), 0)
    def test_get_duration(self):
        d1=self.d.get_duration(units="ms")
        d2=self.d.get_duration(units="sec")
        d3=self.d.get_duration(units="min")
        d4=self.d.get_duration(units="h")
        self.assertLess(d2,d1)
        self.assertLess(d3,d2)
        self.assertLess(d4,d3)
        self.assertAlmostEqual(d2, d1/1000.)
        self.assertAlmostEqual(d3, d2/60.)
        self.assertAlmostEqual(d4, d3/60.)
    def test_sub_slice(self):
        d=self.dfake.sub_slice(1, 2, units="min")
        diff=d.tx[1]-d.tx[0]
        self.assertLess(abs(d.tx[0]-1*60*1000.), diff)
        self.assertLess(abs(d.get_duration(units="min")-1), diff)
    def test_scale(self):
        d=self.dfake.scale()
        self.assertAlmostEqual(np.mean(d.sy), 0)
        self.assertAlmostEqual(np.std(d.sy), 1)
        d2=d.unscale()
        self.assertAlmostEqual(np.mean(d2.sy), np.mean(self.dfake.sy))
        self.assertAlmostEqual(np.std(d2.sy), np.std(self.dfake.sy))
    def test_lowpass_filter(self):
        d=self.dfake.lowpass_filter(2)
    def test_smooth_window(self):
        d=self.dfake.smooth_window()
        d2=d.smooth_window(window="bartlett")
        d3=d.smooth_window(winsize=20)
    def test_downsample(self):
        d=self.dfake.downsample(100)
        self.assertEqual(d.fs, 100)
        with self.assertRaises(ZeroDivisionError):
            d.downsample(101)
        d2=self.dfake.downsample(5, dsfac=True)
        self.assertAlmostEqual(d.fs, d2.fs)
    def test_copy(self):
        d=self.dfake.copy()
        d.sy[0]=self.dfake.sy[0]+1.0
        self.assertNotEqual(d.sy[0], self.dfake.sy[0])
    def test_estimate_baseline(self):
        d=self.dfake.estimate_baseline()
        self.assertLess(np.sum(d.sy<d.baseline), 0.1*len(d))
    def test_stat_per_event(self):
        a1=self.dfake.stat_per_event([-100,0], return_missing=None)
        a2=self.dfake.stat_per_event([-100,0], return_missing="nmiss")
        a3=self.dfake.stat_per_event([-100,0], return_missing="prop")
        self.assertEqual(a1.__class__, np.ndarray)
        self.assertEqual(a2.__class__, tuple)
        self.assertEqual(a3.__class__, tuple)
    def test_estimate_response(self):
        d=self.dfake.estimate_response()
    def test_blinks_detect(self):
        d=self.d.blinks_detect()
    def test_blinks_merge(self):
        d=self.d.blinks_detect()
        d2=d.blinks_merge()
    def test_blinks_merge(self):
        d=self.d.blinks_detect().blinks_merge().blinks_interpolate()
    def test_blinks_merge(self):
        d=self.d.blinks_detect().blinks_merge().blinks_interp_mahot()
    def test_get_erpd(self):
        a=self.d.get_erpd("test", lambda x: True)
        self.assertEqual(a.__class__, ERPD)
    
    
    
        
        
    
        

if __name__ == '__main__':
    unittest.main()