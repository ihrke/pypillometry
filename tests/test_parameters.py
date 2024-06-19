import unittest
import sys
sys.path.insert(0,"..")
from pypillometry import Parameters

class TestParameters(unittest.TestCase):
    def setUp(self):
        pass
    def test_init(self):
        p = Parameters({("mean","right"):10}, default_value=0)
        self.assertEqual(p["not"], 0)
        self.assertEqual(p["right","mean"], 10)
    def test_empty_init(self):
        p=Parameters()
        self.assertEqual(p["not"], None)
    def test_len(self):
        p = Parameters()        
        self.assertEqual(len(p),0)
        p2 = Parameters({("mean","right"):10, ("sd","left","baseline"):0.1})
        self.assertEqual(len(p2),2)
    def test_set(self):
        p = Parameters()
        p["mean","right"]=10
        self.assertEqual(p["right","mean"], p["mean","right"])
        self.assertEqual(p["right","mean"], 10)
    def test_wrongkey(self):
        p = Parameters()
        self.assertRaises(ValueError, p.__setitem__, 10, 10)

if __name__ == '__main__':
    unittest.main()