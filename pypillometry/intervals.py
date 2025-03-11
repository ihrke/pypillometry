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


def merge_intervals(intervals):
    """Merge overlapping intervals.

    Parameters
    ----------
    intervals : list of tuples
        List of intervals to merge.

    Returns
    -------
    list of tuples
        Merged intervals.
    """
    if not intervals:
        return []
    
    # Sort intervals based on the start point
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        
        # Check if intervals overlap
        if current[0] <= last_merged[1]:
            # Merge the intervals
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            # No overlap, add the current interval
            merged.append(current)
    
    return merged

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


def stat_event_interval(tx,sy,intervals,statfct=np.mean):
    """
    Return result of applying a statistical function to pupillometric data in a
    given interval relative to event-onsets. For example, extract mean 
    pupil-size in interval before trial onset.
        
    Parameters
    -----------
    
    tx : np.ndarray
        time-vector in milliseconds        
    sy : np.ndarray
        (baseline-corrected) pupil signal
    intervals : list of tuples (min,max)
        time-window in ms relative to event-onset (0 is event-onset)
    statfct : function
        function mapping np.array to a single number
    
    Returns
    --------
    
    result: np.array
        number of event-onsets long result array
    """
    intervals=np.array(intervals)
    starts=intervals[:,0]
    ends=intervals[:,1]

    res=np.zeros(len(starts))

    for i,interv in enumerate(zip(starts,ends)):
        start_ix=np.argmin(np.abs(interv[0]-tx))
        end_ix=np.argmin(np.abs(interv[1]-tx))
        if start_ix==end_ix:
            end_ix+=1
        res[i]=statfct(sy[start_ix:end_ix])
    return res
    
        