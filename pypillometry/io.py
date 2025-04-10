"""
io.py
=====

Read/Write data from/to disk.
"""

try:
   import cPickle as pickle
except:
   import pickle
import os
import requests
from tqdm import tqdm


def read_study(osf_id: str, path: str, force_download: bool = False):
    """
    Read a study from OSF using the configuration file.
    
    Parameters
    ----------
    osf_id : str
        The OSF project ID
    path : str
        Local path where files should be downloaded/stored
    force_download : bool, optional
        If True, force re-download even if files exist locally. Default False.
        
    Returns
    -------
    dict
        Dictionary containing the loaded study data
    """
    # First download/read the config file
    config_url = f"https://osf.io/{osf_id}/files/pypillometry_conf.py" 
    config_path = os.path.join(path, "pypillometry_conf.py")
    
    if not os.path.exists(config_path) or force_download:
        response = requests.get(config_url)
        if response.status_code == 200:
            with open(config_path, 'wb') as f:
                f.write(response.content)
        else:
            raise ValueError(f"Could not download config file from {config_url}")
            
    # Load and parse config
    with open(config_path, 'r') as f:
        config = eval(f.read())
        
    # Download and read raw data files
    study_data = {}
    for data_file in config['raw_data']:
        file_url = f"https://osf.io/{osf_id}/files/{data_file}"
        file_path = os.path.join(path, data_file)
        
        if not os.path.exists(file_path) or force_download:
            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                total = int(response.headers.get('content-length', 0))
                with open(file_path, 'wb') as f, tqdm(
                    desc=f"Downloading {data_file}",
                    total=total,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)
            else:
                raise ValueError(f"Could not download {data_file}")
                
        # Use the specified read function to load the data
        read_func = eval(config['read_function'])
        study_data[data_file] = read_func(file_path)
        
    return study_data


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
