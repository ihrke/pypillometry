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

def eyedata_write_pickle(pdobj, fname):
    """
    Store the :class:`.GenericEyeData`-object `pdobj` in file using :mod:`pickle`.
    
    Parameters
    ----------
    
    pdobj: :class:`.GenericEyeData`
        dataset to save
    fname: str
        filename to save to
    """
    with open(fname, "wb") as f:
        pickle.dump(pdobj,f)
    
def eyedata_read_pickle(fname):
    """
    Read the :class:`.GenericEyeData`-object `pdobj` from file using :mod:`pickle`.
    
    Parameters
    ----------
    
    fname: str
        filename or URL to load data from
        
    Returns
    -------
    
    pdobj: :class:`.GenericEyeData`
        loaded dataset 
    """
    if fname.startswith("http"):
        # try loading from URL
        res=requests.get(fname)
        if res.status_code==200:
            pdobj=pickle.loads(res.content)
    else:
        with open(fname, 'rb') as f:
            pdobj=pickle.load(f)
    return pdobj
