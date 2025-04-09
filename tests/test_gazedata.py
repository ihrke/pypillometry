import unittest
import sys
import numpy as np
sys.path.insert(0,"..")
import pypillometry as pp
import pytest

class TestGazeData(unittest.TestCase):
    def setUp(self):
        #self.d=pp.GazeData.from_file("data/test.pd")
        pass
    def test_from_file(self):
        d = pp.get_example_data("rlmw_002_short")
        print(d)
    def test_scale(self):
        d = pp.GazeData(sampling_rate=10, left_x=[1,2,1,2], left_y=[3,4,3,4])
        d=d.scale(["x",'y'])
        self.assertEqual(d.data["left","x"].mean(), 0)
        self.assertEqual(d.data["left","y"].mean(), 0)
        self.assertEqual(d.params["scale"]["mean"]["left"]["x"], 1.5)
        self.assertEqual(d.params["scale"]["mean"]["left"]["y"], 3.5)
    def test_scale_fixed(self):
        d = pp.GazeData(sampling_rate=10, left_x=[1,2,1,2], left_y=[3,4,3,4])
        scalepars = {
            "left": {
                "x": 1,
                "y": 3
            }
        }
        sdpars = {
            "left": {
                "x": 0.5,
                "y": 0.5
            }
        }
        d=d.scale(["x",'y'], mean=scalepars, sd=sdpars)
        self.assertEqual(d.data["left","x"].mean(), 1)
        self.assertEqual(d.data["left","y"].mean(), 1)
    def test_unscale(self):
        d = pp.GazeData(sampling_rate=10, left_x=[1,2,1,2], left_y=[3,4,3,4])
        d=d.scale(["x",'y'])
        d=d.unscale(["x",'y'])
        self.assertEqual(d.data["left","x"].mean(), 1.5)
        self.assertEqual(d.data["left","y"].mean(), 3.5)

if __name__ == '__main__':
    unittest.main()