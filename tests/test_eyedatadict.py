import unittest
import sys
import numpy as np
#sys.path.insert(0,"..")
import pypillometry as pp
from pypillometry import EyeDataDict

class TestEyeDataDict(unittest.TestCase):
    def setUp(self):
        pass
    def test_set(self):
        d=EyeDataDict()
        d["test"]=np.array([1,2,3])
        self.assertEqual(d["test"].__class__, np.ndarray)
        self.assertEqual(d["test"].dtype, np.float64)
        self.assertEqual(d["test"].shape, (3,))

    def test_set_wrong_len(self):  
        d=EyeDataDict()
        d["test"]=np.array([1,2,3])
        self.assertRaises(ValueError, d.__setitem__, "test2", np.array([1,2,3,4]))

    def test_set_wrong_type(self):
        d=EyeDataDict()
        self.assertRaises(ValueError, d.__setitem__, "test", np.array(["a","b","c"]))

    def test_init(self):
        d=EyeDataDict(a=np.array([1,2,3]), b=np.array([4,5,6]), c=np.array([7,8,9]))
        self.assertEqual(d["a"].__class__, np.ndarray)
        self.assertEqual(d["a"].dtype, np.float64)
        self.assertEqual(d["a"].shape, (3,))

    def test_init_wrong_dim(self):
        self.assertRaises(ValueError, EyeDataDict, a=np.array([[1,2,3], [4,5,6]]))
        
    def test_drop_emtpy(self):
        d=EyeDataDict(a=np.array([1,2,3]), b=np.array([4,5,6]), c=np.array([7,8,9]))
        d["d"]=np.array([])
        d["e"]=None
        self.assertRaises(KeyError, d.__getitem__, "d")
        self.assertRaises(KeyError, d.__getitem__, "e")

if __name__ == '__main__':
    unittest.main()