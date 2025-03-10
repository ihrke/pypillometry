"""Example datasets for pypillometry.

This module provides example datasets for pypillometry. The datasets are
used in the documentation and for testing purposes. Most of them are stored on
OSF: https://osf.io/p2u74/
"""

import numpy as np
import pandas as pd
from .eyedata import EyeData
from importlib.resources import files
from urllib.request import urlretrieve
import tempfile
import os.path
import requests
from tqdm import tqdm

example_datasets = {
    "rlmw_002": {
        "samples_asc":"https://osf.io/download/67cec1a4fb24855200fd6db7/",
        "events_asc":"https://osf.io/download/67cec1a183c337a5effd6fb6/",
        "description":"Binocular data from a 20-minute reinforcement learning task."
        "Data from a single participant, features strong horizontal eye-movements"
        "between the two stimuli presented on the left and right.",
    },
    "rlmw_002_short": {
        "sample_asc":files('pypillometry.data').joinpath('002_rlmw_samples_short.asc'),
        "events_asc":files('pypillometry.data').joinpath('002_rlmw_events_short.asc'),
        "description":"Short version (first 40 sec) of a 20-minute reinforcement learning task."
        "Data from a single participant, features strong horizontal eye-movements"
        "between the two stimuli presented on the left and right.",
    }

}


def download(url: str, fname: str, chunk_size=1024):
    """Download a file from a URL to a local file.

    See https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51.

    Parameters
    ----------
    url : str
        URL of the file to download.
    fname : str
        Local filename to save the file to.
    chunk_size : int, optional  
        Size of the chunks to download the file in. Default is 1024.
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def get_example_data(key):
    """Download example data from OSF.

    Parameters
    ----------
    key : str
        Key of the example dataset. Available keys are in the dictionary
        `example_datasets`.

    Returns
    -------
    EyeData, GazeData or PupilData
        The example dataset as an EyeData, GazeData or PupilData object.
    """
    if key not in example_datasets:
        raise ValueError(f"Key '{key}' not found in example datasets. "
                         f"Available keys are: {list(example_datasets.keys())}")

    # run the function with the same name as the key
    funcname = f"get_{key}"
    if funcname in globals():
        return globals()[funcname]()
    else:
        raise ValueError(f"Function '{funcname}' not found in example datasets.")


def get_rlmw_002():
    # download file from osf   
    with tempfile.TemporaryDirectory() as tmpdirname: 
        fname_samples = os.path.join(tmpdirname,"002_rlmw_samples.asc")
        fname_events = os.path.join(tmpdirname,"002_rlmw_events.asc")
        download(example_datasets["rlmw_002"]["samples_asc"], fname_samples)
        download(example_datasets["rlmw_002"]["events_asc"], fname_events)
        df=pd.read_table(fname_samples, index_col=False,
                        names=["time", "left_x", "left_y", "left_p",
                                "right_x", "right_y", "right_p"])

        ## Loading the events from the events file
        # read the whole file into variable `events` (list with one entry per line)
        with open(fname_events) as f:
            events=f.readlines()

    ## Eyelink tracker puts "   ." when no data is available for x/y coordinates
    left_x=df.left_x.values
    left_x[left_x=="   ."] = np.nan
    left_x = left_x.astype(float)

    left_y=df.left_y.values
    left_y[left_y=="   ."] = np.nan
    left_y = left_y.astype(float)

    right_x=df.right_x.values
    right_x[right_x=="   ."] = np.nan
    right_x = right_x.astype(float)

    right_y=df.right_y.values
    right_y[right_y=="   ."] = np.nan
    right_y = right_y.astype(float)


    # keep only lines starting with "MSG"
    events=[ev for ev in events if ev.startswith("MSG")]
    experiment_start_index=np.where(["experiment_start" in ev for ev in events])[0][0]
    events=events[experiment_start_index+1:]
    df_ev=pd.DataFrame([ev.split() for ev in events])
    df_ev=df_ev[[1,2]]
    df_ev.columns=["time", "event"]

    # Creating EyeData object that contains both X-Y coordinates
    # and pupil data
    d = EyeData(time=df.time, name="test short",
                screen_resolution=(1280,1024), physical_screen_size=(33.75,27),
                screen_eye_distance=60,
                left_x=left_x, left_y=left_y, left_pupil=df.left_p,
                right_x=right_x, right_y=right_y, right_pupil=df.right_p,
                event_onsets=df_ev.time, event_labels=df_ev.event,
                keep_orig=True)\
                .reset_time()
    d.set_experiment_info(screen_eye_distance=60, 
                        screen_resolution=(1280,1024), 
                        physical_screen_size=(30, 20))
    return d        




def get_rlmw_002_short():
    fname_samples = example_datasets["rlmw_002_short"]["sample_asc"]
    fname_events = example_datasets["rlmw_002_short"]["events_asc"]
    df=pd.read_table(fname_samples, index_col=False,
                    names=["time", "left_x", "left_y", "left_p",
                            "right_x", "right_y", "right_p"])

    ## Eyelink tracker puts "   ." when no data is available for x/y coordinates
    left_x=df.left_x.values
    left_x[left_x=="   ."] = np.nan
    left_x = left_x.astype(float)

    left_y=df.left_y.values
    left_y[left_y=="   ."] = np.nan
    left_y = left_y.astype(float)

    right_x=df.right_x.values
    right_x[right_x=="   ."] = np.nan
    right_x = right_x.astype(float)

    right_y=df.right_y.values
    right_y[right_y=="   ."] = np.nan
    right_y = right_y.astype(float)

    ## Loading the events from the events file
    # read the whole file into variable `events` (list with one entry per line)
    with open(fname_events) as f:
        events=f.readlines()

    # keep only lines starting with "MSG"
    events=[ev for ev in events if ev.startswith("MSG")]
    experiment_start_index=np.where(["experiment_start" in ev for ev in events])[0][0]
    events=events[experiment_start_index+1:]
    df_ev=pd.DataFrame([ev.split() for ev in events])
    df_ev=df_ev[[1,2]]
    df_ev.columns=["time", "event"]

    # Creating EyeData object that contains both X-Y coordinates
    # and pupil data
    d = EyeData(time=df.time, name="test short",
                screen_resolution=(1280,1024), physical_screen_size=(33.75,27),
                screen_eye_distance=60,
                left_x=left_x, left_y=left_y, left_pupil=df.left_p,
                right_x=right_x, right_y=right_y, right_pupil=df.right_p,
                event_onsets=df_ev.time, event_labels=df_ev.event,
                keep_orig=True)\
                .reset_time()
    d.set_experiment_info(screen_eye_distance=60, 
                        screen_resolution=(1280,1024), 
                        physical_screen_size=(30, 20))
    return d
        