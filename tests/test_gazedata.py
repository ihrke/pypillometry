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
        pass
        #d=pp.GazeData.from_file("data/test.pd")
        #self.assertEqual(d.__class__, pp.GazeData)
    def test_scale(self):
        d = pp.GazeData(sampling_rate=10, left_x=[1,2,1,2], left_y=[3,4,3,4])
        d=d.scale(["x",'y'])
        self.assertEqual(d.data["left","x"].mean(), 0)
        self.assertEqual(d.data["left","y"].mean(), 0)
        self.assertEqual(d.scale_params["left","x","mean"], 1.5)
        self.assertEqual(d.scale_params["left","y","sd"], 0.5)
    def test_scale_fixed(self):
        d = pp.GazeData(sampling_rate=10, left_x=[1,2,1,2], left_y=[3,4,3,4])
        scalepars = pp.Parameters({("mean","left","x"):1, 
                                   ("mean","left","y"):3, 
                                   ("sd","left","x"):0.5, 
                                   ("sd","left","y"):0.5})
        d=d.scale("x", mean=scalepars["mean"], sd=scalepars["sd"])
        self.assertEqual(d.scale_params["left","x","mean"], 1)
        self.assertEqual(d.scale_params["left","x","sd"], 0.5)
    
        #pytest.set_trace()
    def test_unscale(self):
        d = pp.GazeData(sampling_rate=10, left_x=[1,2,1,2], left_y=[3,4,3,4])
        d2=d.scale(["x",'y'])
        d3=d2.unscale(["x",'y'])
        self.assertEqual(np.sum(d3.data["left","x"]-d.data["left","x"]), 0)
        self.assertEqual(np.sum(d3.data["left","y"]-d.data["left","y"]), 0)

if __name__ == '__main__':
    unittest.main()