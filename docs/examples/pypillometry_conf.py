"""
Configuration file for the RLMW study.
This file contains information about the raw data files and how to read them.
"""
import pypillometry as pp
import pandas as pd
import os
import numpy as np


# Additional study metadata
study_info = {
    "name": "RLMW Study",
    "osf_id": "ca95r",
    "description": "Reinforcement learning study with mind wandering probes",
    "author": "Matthias Mittner",
    "doi": "",
    "date": "2024-04-10",
    "sampling_rate": 1000.0,  # Hz
    "time_unit": "ms",
    "screen_eye_distance": 60, # cm (distance between screen and eye)
    "screen_resolution": (1280,1024),  # pixels (width, height)
    "physical_screen_size": (30, 20) # cm (width, height)
}


# Dictionary of raw data files to be downloaded
# Keys are participant IDs, values are dictionaries containing paths to .asc files
raw_data = {
    "001": {
        "events": "data/eyedata/asc/001_rlmw_events.asc",
        "samples": "data/eyedata/asc/001_rlmw_samples.asc"
    },
    "002": {
        "events": "data/eyedata/asc/002_rlmw_events.asc",
        "samples": "data/eyedata/asc/002_rlmw_samples.asc"
    },
    "003": {
        "events": "data/eyedata/asc/003_rlmw_events.asc",
        "samples": "data/eyedata/asc/003_rlmw_samples.asc"
    }
}

# Data processing parameters
processing = {
    "blink_detection": {
        "threshold": 0.1,
        "min_duration": 50.0,  # ms
        "max_duration": 400.0  # ms
    },
    "interpolation": {
        "method": "linear",
        "limit": 100.0  # ms
    }
} 

# Function to use for reading the data files
# This should be a string that can be evaluated to get the actual function
def read_subject(info):
    """
    Read the data for a single subject. Input is each element of `raw_data`.
    """
    ## loading the raw samples from the asc file
    fname_samples=os.path.join(info["samples"])
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
    fname_events=os.path.join(info["events"])
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
    d = pp.EyeData(time=df.time, name=info["subject"],
                screen_resolution=study_info["screen_resolution"], 
                physical_screen_size=study_info["physical_screen_size"],
                screen_eye_distance=study_info["screen_eye_distance"],
                left_x=left_x, left_y=left_y, left_pupil=df.left_p,
                right_x=right_x, right_y=right_y, right_pupil=df.right_p,
                event_onsets=df_ev.time, event_labels=df_ev.event,
                keep_orig=True)\
                .reset_time()
    d.set_experiment_info(screen_eye_distance=study_info["screen_eye_distance"], 
                        screen_resolution=study_info["screen_resolution"], 
                        physical_screen_size=study_info["physical_screen_size"])
    return d    

# write down notes about each subject when going through the preprocs
notes={
    "001":"good",
    "002":"good but many blinks",
    "003":"ok, but some segments with many blinks (min 8, 12, ...)",
    "004":"ok, beginning is crap, many 'double blinks', a few 'dip blinks' which don't go all the way to zero but filter is ok",
    "005":"ok, some pretty long blinks and some dips but recovery mostly ok",
    "006":"ok, near ideal in the beginning, more blinks later",
    "007":"ok",
    "008":"ok but problems around 8.7-14 mins",
    "009":"ok, many multi-blinks but good recovery",
    "010":"ok, very slow opening of eye, difficult to correct - used rather large margin",
    "011":"very nice and regular blinking",
    "012":"ok, there are some weird 'spikes' upwards in the data, saccades? filter seems ok",
    "013":"ok",
    "014":"nice and regular",
    "015":"ok but not great, especially later parts",
    "016":"ok but not great",
    "017":"ok",
    "018":"consider exclusion, another one with slow opening of eye, used large margin, lots of missings",
    "019":"amazing, almost no blinks. Some spikes but filtered out ok",
    "020":"ok",
    "021":"ok, somewhat more messy during second half",
    "022":"ok",
    "023":"ok",
    "024":"ok but not great, some double-blinks",   
    "025":"ok",
    "026":"ok",
    "027":"ok but data quality in last part is getting worse",
    "028":"ok",
    "029":"consider exclusion, pretty bad",
    "030":"ok",
    "031":"ok but not great",
    "032":"consider exclusion, but not super bad",
    "033":"ok",
    "034":"ok",
    "035":"ok but not great",
    "036":"ok; very many but very short blinks... signal looks ok, though",
    "037":"ok",
    "038":"ok",
    "039":"exclude: ok quality but saccades are super prominent in this subject",
    "040":"ok",
    "041":"ok",
    "042":"ok", 
    "043":"ok, many blinks but ok recovery",
    "044":"exclude: not great qual and saccades are prominent",
    "045":"ok",
    "046":"ok; qual not great but preproc seems to deal with it",
    "047":"exclude: getting a lot worse at the end",
    "048":"ok, slow opening of eye, used large margin, but signal looks ok",
    "049":"exclude: huge saccades",
    "050":"ok"
}
exclude = ["018", "029", "032", "039", "044", "047", "049"]