import unittest
import tempfile
import os
import sys
#sys.path.insert(0,"..")
#import pypillometry as pp
from .. import *

class TestIO(unittest.TestCase):
    def test_pd_read_pickle_file(self):
        d=pd_read_pickle("data/test.pd")
        self.assertEqual(d.fs, 500.0)
        self.assertEqual(len(d), 60001)
    def test_pd_read_pickle_http(self):
        d=pd_read_pickle("https://github.com/ihrke/pypillometry/blob/master/data/test.pd?raw=true")
        self.assertEqual(d.fs, 500.0)
        self.assertEqual(len(d), 60001)
    
    def test_pd_write_pickle(self):
        d=create_fake_pupildata(ntrials=10)
        fpath=tempfile.mkdtemp()
        fname=os.path.join(fpath, "test.pd")
        pd_write_pickle(d, fname)
        x=pd_read_pickle(fname)
        self.assertEqual(x.size_bytes(), d.size_bytes())
        self.assertEqual(x.name, d.name)
        

if __name__ == '__main__':
    unittest.main()