"""Functions to handle intervals (blinks, erpds, etc.) in eyetracking data.
"""

import numpy as np

class IntervalStats(dict):
    """
    A dictionary with a specialized repr() function
    to display summary statistics of intervals.
    """
    def __repr__(self):
        n=self["n"] if "n" in self else 0
        mean=self["mean"] if "mean" in self else np.nan
        sd=self["sd"] if "sd" in self else np.nan
        minv=self["min"] if "min" in self else np.nan
        maxv=self["max"] if "max" in self else np.nan
        r = "%i intervals, %.2f +/- %.2f, [%.2f, %.2f]" % (n, mean, sd, minv, maxv)
        return r

def get_interval_stats(intervals):
    """
    Calculate summary statistics of intervals.
    
    Parameters
    ----------
    intervals: list of tuples
        list of intervals
    
    Returns
    -------
    IntervalStats
        dictionary with summary statistics
    """
    stats=IntervalStats()
    stats["n"]=len(intervals)

    durations = [i[1]-i[0] for i in intervals]
    stats["mean"]=np.mean(durations)
    stats["sd"]=np.std(durations)
    stats["min"]=np.min(durations)
    stats["max"]=np.max(durations)
    return stats