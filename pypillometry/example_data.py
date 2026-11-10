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
        "files": {
            "samples": "https://osf.io/download/67cec1a4fb24855200fd6db7/",
            "events": "https://osf.io/download/67cec1a183c337a5effd6fb6/",
        },
        "description":"Binocular data from a 20-minute reinforcement learning task. "
        "Data from a single participant, features strong horizontal eye-movements "
        "between the two stimuli presented on the left and right.",
    },
    "rlmw_002_short": {
        "source": "local",
        "filetype": "asc",
        "files": {
            "samples": files('pypillometry.data').joinpath('002_rlmw_samples_short.asc'),
            "events": files('pypillometry.data').joinpath('002_rlmw_events_short.asc'),
        },
        "description":"Short version (first 40 sec) of a 20-minute reinforcement learning task. "
        "Data from a single participant, features strong horizontal eye-movements "
        "between the two stimuli presented on the left and right.",
    },
    "rlmw_010_edf": {
        "source": "OSF",
        "filetype": "edf",
        "files": {"data": "https://osf.io/trsuq/download"},
        "description":"Binocular data from a 20-minute reinforcement learning task "
        "in EDF (Eyelink) format. "
        "Data from a single participant, features strong horizontal eye-movements "
        "between the two stimuli presented on the left and right.",
    },
    "nback_001": {
        "source": "figshare",
        "filetype": "pkl",
        "figshare": {"id": 28776617, "version": 1},
        # names in the figshare item
        "files": {"data": "sub-001_ses-1.pypil", "behaviour": "sub-001_ses-1.csv"},
        "description": "Binocular data from a 56-minute N-back task "
        "(0-, 1- and 2-back blocks in three runs with grey, black and white "
        "screen background), saved as a pypillometry file. "
        "Data from a single participant, with events for fixation, stimulus "
        "and response in each trial; the behavioural data (stimulus, "
        "response, accuracy, reaction time) are in the events' metadata.",
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


def _read_rlmw_asc(fname_samples, fname_events, name):
    """EyeData from the samples and events ASC files of the "rlmw" task.

    Events are the "MSG" lines after "experiment_start"; missing values
    ("   ." for gaze, 0 for pupil) are set to NaN; time is reset to start
    at 0 and the original data are kept (`keep_orig=True`).
    """
    import pandas as pd
    from .eyedata.experimental_setup import ExperimentalSetup

    df = pd.read_table(fname_samples, index_col=False,
                       names=["time", "left_x", "left_y", "left_p",
                              "right_x", "right_y", "right_p"])

    # Eyelink puts "   ." when no gaze position is available
    def gaze(col):
        x = df[col].values
        x[x == "   ."] = np.nan
        return x.astype(float)

    # Eyelink codes missing pupil data (e.g., blinks) as 0
    def pupil(col):
        return np.where(df[col] == 0, np.nan, df[col])

    # events: the "MSG" lines after "experiment_start"
    with open(fname_events) as f:
        events = [ev for ev in f if ev.startswith("MSG")]
    start = np.where(["experiment_start" in ev for ev in events])[0][0]
    df_ev = pd.DataFrame([ev.split() for ev in events[start + 1:]])[[1, 2]]
    df_ev.columns = ["time", "event"]

    setup = ExperimentalSetup(
        screen_resolution=(1280, 1024),
        physical_screen_size=("30 cm", "20 cm"),
        eye_to_screen_center="60 cm",
    )
    return EyeData(time=df.time, name=name,
                   experimental_setup=setup,
                   left_x=gaze("left_x"), left_y=gaze("left_y"),
                   left_pupil=pupil("left_p"),
                   right_x=gaze("right_x"), right_y=gaze("right_y"),
                   right_pupil=pupil("right_p"),
                   event_onsets=df_ev.time, event_labels=df_ev.event,
                   keep_orig=True).reset_time()


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
    cache_dir = cache_dir or _user_cache_dir()
    urls = example_datasets["rlmw_002"]["files"]
    fname_samples = fetch_url(urls["samples"],
                              cache_file=os.path.join(cache_dir, "002_rlmw_samples.asc"))
    fname_events = fetch_url(urls["events"],
                             cache_file=os.path.join(cache_dir, "002_rlmw_events.asc"))
    return _read_rlmw_asc(fname_samples, fname_events, "rlmw_002")


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
    paths = example_datasets["rlmw_002_short"]["files"]
    return _read_rlmw_asc(paths["samples"], paths["events"], "rlmw_002_short")


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
    fname = fetch_url(example_datasets["rlmw_010_edf"]["files"]["data"],
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
    ``run-1_scr-grey_block-1_task-0b_trial-1_stim_target-1``).

    The behavioural log of the task is merged into the events' metadata
    (see :attr:`Events.meta <pypillometry.Events.meta>`): ``run``,
    ``screen``, ``block``, ``task`` (``0b``, ``1b``, ``2b`` or ``p``) and
    ``trial`` are parsed from the labels, and ``stimulus``, ``target``,
    ``response``, ``accuracy`` and ``rt`` (reaction time in s) are added to
    all events of each trial. The log's header (date, ambient light, ...)
    is in ``d.info["behaviour_log"]``.

    The files are downloaded into `cache_dir` on the first call and reused
    later. Usually called via :func:`~pypillometry.get_example_data`.

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
    info = example_datasets["nback_001"]
    figshare_id, version = info["figshare"]["id"], info["figshare"]["version"]
    cache_dir = cache_dir or _user_cache_dir()

    # download the files from figshare (or use the cached copies)
    fname_data = fetch_figshare_file(figshare_id, info["files"]["data"],
                                     version=version, cache_dir=cache_dir,
                                     cache_layout="mirror")
    fname_log = fetch_figshare_file(figshare_id, info["files"]["behaviour"],
                                    version=version, cache_dir=cache_dir,
                                    cache_layout="mirror")

    # eye-tracking data, saved with pypillometry
    d = EyeData.from_file(fname_data)
    d.name = "nback_001"

    # behavioural log: a header of "# key: value" lines, then one row per trial
    with open(fname_log) as f:
        comments = [line.lstrip("# ").strip() for line in f if line.startswith("#")]
    d.info["behaviour_log"] = dict(c.split(": ", 1) for c in comments if ": " in c)
    log = pd.read_csv(fname_log, comment="#")

    # run, screen, block, task and trial from the event labels, e.g.
    # "run-1_scr-grey_block-1_task-0b_trial-1_stim_target-1"
    events = (d.events.parse_labels(r"run-(?P<run>\d+)")
                      .parse_labels(r"scr-(?P<screen>[a-z]+)")
                      .parse_labels(r"block-(?P<block>\d+)")
                      .parse_labels(r"task-(?P<task>[^_]+)")
                      .parse_labels(r"trial-(?P<trial>\d+)"))

    # add each trial's row of the log to all its events (fixation, stimulus,
    # response), matched by run, block, task and trial
    events = events.merge_table(
        log, on=["run", "block", "task", "trial"],
        columns=["stimulus", "target", "response", "accuracy", "rt"])
    d.set_events(events)
    return d
