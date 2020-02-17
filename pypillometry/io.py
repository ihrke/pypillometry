"""
io.py
=====

Read/Write data from/to disk.
"""

try:
   import cPickle as pickle
except:
   import pickle
import requests
from .pupildata import *

def pd_write_pickle(pdobj, fname):
    """
    Store the :class:`.PupilData`-object `pdobj` in file using :mod:`pickle`.
    
    Parameters
    ----------
    
    pdobj: :class:`.PupilData`
        dataset to save
    fname: str
        filename to save to
    """
    with open(fname, "wb") as f:
        pickle.dump(pdobj,f)
    
def pd_read_pickle(fname):
    """
    Read the :class:`.PupilData`-object `pdobj` from file using :mod:`pickle`.
    
    Parameters
    ----------
    
    fname: str
        filename to save to
        
    Returns
    -------
    
    pdobj: :class:`.PupilData`
        loaded dataset 
    """
    with open(fname, 'rb') as f:
        pdobj=pickle.load(f)
    return pdobj
