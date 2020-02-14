"""
io.py
=====

Read/Write data from/to disk.
"""

import shelve
from .pupildata import *

def pd_write_shelve(pdobj, fname):
    """
    Store the :class:`.PupilData`-object `pdobj` in file using :mod:`shelve`.
    
    Parameters
    ----------
    
    pdobj: :class:`.PupilData`
        dataset to save
    fname: str
        filename to save to
    """
    with shelve.open('fname') as f:
        f['pdobj'] = pdobj
    
def pd_read_shelve(fname):
    """
    Read the :class:`.PupilData`-object `pdobj` from file using :mod:`shelve`.
    
    Parameters
    ----------
    
    fname: str
        filename to save to
        
    Returns
    -------
    
    pdobj: :class:`.PupilData`
        loaded dataset 
    """
    with shelve.open('fname') as f:
        pdobj=f['pdobj']
    return pdobj
