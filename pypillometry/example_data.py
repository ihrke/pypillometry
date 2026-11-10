"""
example_data.py
===============  

Example datasets for pypillometry.

This module provides example datasets for pypillometry. The datasets are
used in the documentation and for testing purposes. Most of them are stored on
OSF (https://osf.io/p2u74/) or figshare
(https://figshare.com/articles/dataset/Example_pypillometry_dataset/28776617).
"""

import numpy as np
from .eyedata import EyeData
from .table import RecordDict
from importlib.resources import files
import inspect
import os
import os.path
from .io import fetch_url, fetch_figshare_file
from .convenience import _user_cache_dir

__all__ = [
    "get_available_datasets", "get_example_data", "get_rlmw_002",
    "get_rlmw_002_short", "get_rlmw_010_edf", "get_nback_001",
]

example_datasets = {
    "rlmw_002": {
        "source": "OSF",
        "filetype": "asc",
        "samples_asc":"https://osf.io/download/67cec1a4fb24855200fd6db7/",
        "events_asc":"https://osf.io/download/67cec1a183c337a5effd6fb6/",
        "description":"Binocular data from a 20-minute reinforcement learning task. "
        "Data from a single participant, features strong horizontal eye-movements "
        "between the two stimuli presented on the left and right.",
    },
    "rlmw_002_short": {
        "source": "local",
        "filetype": "asc",
        "sample_asc":files('pypillometry.data').joinpath('002_rlmw_samples_short.asc'),
        "events_asc":files('pypillometry.data').joinpath('002_rlmw_events_short.asc'),
        "description":"Short version (first 40 sec) of a 20-minute reinforcement learning task. "
        "Data from a single participant, features strong horizontal eye-movements "
        "between the two stimuli presented on the left and right.",
    },
    "rlmw_010_edf": {
        "source": "OSF",
        "filetype": "edf",
        "edf":"https://osf.io/trsuq/download",
        "description":"Binocular data from a 20-minute reinforcement learning task "
        "in EDF (Eyelink) format. "
        "Data from a single participant, features strong horizontal eye-movements "
        "between the two stimuli presented on the left and right.",
    },
    "nback_001": {
        "source": "figshare",
        "filetype": "pkl",
        "figshare": 28776617,
        "figshare_version": 1,
        "pypil": "sub-001_ses-1.pypil",
        "description": "Binocular data from a 56-minute N-back task "
        "(0-, 1- and 2-back blocks in three runs with grey, black and white "
        "screen background), saved as a pypillometry file. "
        "Data from a single participant, with events for fixation, stimulus "
        "and response in each trial.",
    },
}

class DatasetDict(RecordDict):
    """The example datasets: name -> properties (description, files).

    A dict that is displayed as a table with the name, source (OSF,
    figshare, local, ssh), file type (asc, edf, pkl, hdf5) and description
    of each dataset (see :class:`~pypillometry.RecordDict`).
    """

    def __init__(self, data=None):
        super().__init__(data, index_names=("dataset",))

    def _rows(self) -> list:
        rows = []
        for key, info in self.items():
            rows.append(((key,), {"source": info.get("source", "unknown"),
                                  "filetype": info.get("filetype", "unknown"),
                                  "description": info.get("description", "")}))
        return rows


def get_available_datasets():
    """Return available example datasets.
    
    Returns a dict (name -> properties) that is displayed as a table.
    
    Returns
    -------
    DatasetDict
        dataset name -> its properties (source, file type, description,
        files); displayed as a table with the name, source, file type and
        description of each dataset
    
    Examples
    --------
    >>> datasets = get_available_datasets()
    >>> datasets  # in Jupyter, this displays as a table  # doctest: +SKIP
    >>> "rlmw_002_short" in datasets  # but it is a normal dict
    True
    >>> datasets["rlmw_002"]["description"][:30]
    'Binocular data from a 20-minut'
    """
    return DatasetDict(example_datasets)

def get_example_data(key, cache_dir=None):
    """Load example data for a given example (see `example_datasets`).

    Parameters
    ----------
    key : str
        Key of the example dataset. Available keys are in the dictionary
        `example_datasets`.
    cache_dir : str, optional
        directory to store downloaded files in (reused in later calls);
        default ``$XDG_CACHE_HOME/pypillometry`` (``~/.cache/pypillometry``).
        Ignored for datasets included in the package.

    Returns
    -------
    EyeData
        The example dataset.
    """
    if key not in example_datasets:
        raise ValueError(f"Key '{key}' not found in example datasets. "
                         f"Available keys are: {list(example_datasets.keys())}. "
                         f"Call get_available_datasets() to see all available datasets.")

    # run the function with the same name as the key
    funcname = f"get_{key}"
    if funcname in globals():
        func = globals()[funcname]
        kwargs = {}
        if "cache_dir" in inspect.signature(func).parameters:
            kwargs["cache_dir"] = cache_dir
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore') 
            r = func(**kwargs)
        return r
    else:
        raise ValueError(f"Function '{funcname}' not found in example datasets.")


def get_rlmw_002(cache_dir=None):
    """Load the "rlmw_002" example dataset (downloaded from OSF).

    Binocular gaze and pupil data from a 20-minute reinforcement learning
    task (single participant). The ASC files are downloaded into
    `cache_dir` on the first call and reused later. Events are the "MSG" lines after
    "experiment_start"; missing pupil samples (coded as 0 by Eyelink) are
    set to NaN; time is reset to start at 0 and the original data
    are kept (`keep_orig=True`). Usually called via
    :func:`~pypillometry.get_example_data`.

    Parameters
    ----------
    cache_dir : str, optional
        directory for the downloaded files; default
        ``$XDG_CACHE_HOME/pypillometry`` (``~/.cache/pypillometry``)

    Returns
    -------
    EyeData
        the example dataset
    """
    import pandas as pd
    # download files from osf (or use the cached copies)
    cache_dir = cache_dir or _user_cache_dir()
    fname_samples = fetch_url(example_datasets["rlmw_002"]["samples_asc"],
                              cache_file=os.path.join(cache_dir, "002_rlmw_samples.asc"))
    fname_events = fetch_url(example_datasets["rlmw_002"]["events_asc"],
                             cache_file=os.path.join(cache_dir, "002_rlmw_events.asc"))

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

    ## Eyelink tracker codes missing pupil data (e.g., blinks) as 0
    left_p = np.where(df.left_p == 0, np.nan, df.left_p)
    right_p = np.where(df.right_p == 0, np.nan, df.right_p)


    # keep only lines starting with "MSG"
    events=[ev for ev in events if ev.startswith("MSG")]
    experiment_start_index=np.where(["experiment_start" in ev for ev in events])[0][0]
    events=events[experiment_start_index+1:]
    df_ev=pd.DataFrame([ev.split() for ev in events])
    df_ev=df_ev[[1,2]]
    df_ev.columns=["time", "event"]

    # Creating EyeData object that contains both X-Y coordinates
    # and pupil data
    from .eyedata.experimental_setup import ExperimentalSetup
    setup = ExperimentalSetup(
        screen_resolution=(1280, 1024),
        physical_screen_size=("30 cm", "20 cm"),
        eye_to_screen_center="60 cm",
    )
    d = EyeData(time=df.time, name="rlmw_002",
                experimental_setup=setup,
                left_x=left_x, left_y=left_y, left_pupil=left_p,
                right_x=right_x, right_y=right_y, right_pupil=right_p,
                event_onsets=df_ev.time, event_labels=df_ev.event,
                keep_orig=True)\
                .reset_time()
    return d        




def get_rlmw_002_short():
    """Load the "rlmw_002_short" example dataset (included in the package).

    Short version (first 40 sec) of the "rlmw_002" dataset: binocular gaze
    and pupil data from a reinforcement learning task. No download is
    needed. Events are the "MSG" lines after "experiment_start"; missing
    pupil samples (coded as 0 by Eyelink) are set to NaN; time is
    reset to start at 0 and the original data are kept (`keep_orig=True`).
    Usually called via :func:`~pypillometry.get_example_data`.

    Returns
    -------
    EyeData
        the example dataset
    """
    import pandas as pd
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

    ## Eyelink tracker codes missing pupil data (e.g., blinks) as 0
    left_p = np.where(df.left_p == 0, np.nan, df.left_p)
    right_p = np.where(df.right_p == 0, np.nan, df.right_p)

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
    from .eyedata.experimental_setup import ExperimentalSetup
    setup = ExperimentalSetup(
        screen_resolution=(1280, 1024),
        physical_screen_size=("30 cm", "20 cm"),
        eye_to_screen_center="60 cm",
    )
    d = EyeData(time=df.time, name="rlmw_002_short",
                experimental_setup=setup,
                left_x=left_x, left_y=left_y, left_pupil=left_p,
                right_x=right_x, right_y=right_y, right_pupil=right_p,
                event_onsets=df_ev.time, event_labels=df_ev.event,
                keep_orig=True)\
                .reset_time()
    return d
        

def get_rlmw_010_edf(cache_dir=None):
    """Load the "rlmw_010_edf" example dataset (downloaded from OSF).

    Binocular gaze and pupil data from a 20-minute reinforcement learning
    task (single participant) in EDF (Eyelink) format, read with
    :meth:`~pypillometry.EyeData.from_eyelink`. The EDF file is downloaded
    into `cache_dir` on the first call and reused later. Usually called via
    :func:`~pypillometry.get_example_data`.

    Parameters
    ----------
    cache_dir : str, optional
        directory for the downloaded files; default
        ``$XDG_CACHE_HOME/pypillometry`` (``~/.cache/pypillometry``)

    Returns
    -------
    EyeData
        the example dataset
    """
    cache_dir = cache_dir or _user_cache_dir()
    fname = fetch_url(example_datasets["rlmw_010_edf"]["edf"],
                      cache_file=os.path.join(cache_dir, "010_rlmw.edf"))
    d = EyeData.from_eyelink(fname)
    d.name = "rlmw_010"
    d.set_experimental_setup(
        physical_screen_size=("30 cm", "20 cm"),
        eye_to_screen_center="60 cm",
    )
    return d


def get_nback_001(cache_dir=None):
    """Load the "nback_001" example dataset (downloaded from figshare).

    Binocular gaze and pupil data from a 56-minute N-back task (single
    participant), saved as a pypillometry file. Events mark fixation,
    stimulus and response in each trial (e.g.,
    ``run-1_scr-grey_block-1_task-0b_trial-1_stim_target-1``). The file is
    downloaded into `cache_dir` on the first call and reused later. Usually
    called via :func:`~pypillometry.get_example_data`.

    Parameters
    ----------
    cache_dir : str, optional
        directory for the downloaded files; default
        ``$XDG_CACHE_HOME/pypillometry`` (``~/.cache/pypillometry``)

    Returns
    -------
    EyeData
        the example dataset
    """
    info = example_datasets["nback_001"]
    fname = fetch_figshare_file(info["figshare"], info["pypil"],
                                version=info["figshare_version"],
                                cache_dir=cache_dir or _user_cache_dir(),
                                cache_layout="mirror")
    d = EyeData.from_file(fname)
    d.name = "nback_001"
    return d
