import unittest
import tempfile
import os, pickle, hashlib
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
        d=pd_read_pickle("data/test.pd")#create_fake_pupildata(ntrials=10)
        fpath=tempfile.mkdtemp()
        fname=os.path.join(fpath, "test2.pd")
        pd_write_pickle(d, fname)
        x=pd_read_pickle(fname)
        self.assertEqual(x.size_bytes(), d.size_bytes())
        self.assertEqual(x.name, d.name)
        dmd5=hashlib.md5(pickle.dumps(d,-1)).hexdigest()
        xmd5=hashlib.md5(pickle.dumps(x,-1)).hexdigest()
        self.assertEqual(dmd5,xmd5)

if __name__ == '__main__':
    unittest.main()