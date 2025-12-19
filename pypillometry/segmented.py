"""
segmented.py
============

Segmented eye data for event-related analysis.

"""
import numpy as np
import numpy.ma as ma
from typing import Optional, Tuple, Union
from collections.abc import Iterable
from loguru import logger
import scipy.stats

from .io import pd_write_pickle, pd_read_pickle


class SegmentedEyeData:
    """
    Class representing segmented eye data (e.g., event-related pupil dilation).
    
    Holds equal-length segments of a single eye data variable (e.g., "left_pupil", 
    "right_x") extracted around events. Tracks masked/missing data.

    Parameters
    ----------
    name : str
        Name of the dataset (e.g., "cue-locked", "conflict-trials")
    variable : str
        Variable name in "eye_variable" format (e.g., "left_pupil", "right_x")
    tx : np.ndarray
        Time axis relative to segment onset (in ms)
    data : np.ma.MaskedArray
        Masked array with shape (n_timepoints, n_segments)
    intervals : Intervals, optional
        Source Intervals object with event metadata
    sampling_rate : float, optional
        Original sampling rate in Hz
    """
    
    def __init__(self, 
                 name: str, 
                 variable: str,
                 tx: np.ndarray, 
                 data: ma.MaskedArray,
                 intervals=None,
                 sampling_rate: float = None):
        self.name = name
        self.variable = variable
        self.tx = np.asarray(tx)
        
        # Ensure data is a masked array
        if not isinstance(data, ma.MaskedArray):
            data = ma.masked_array(data, mask=np.isnan(data))
        self.data = data
        
        self.intervals = intervals
        self.sampling_rate = sampling_rate
        
        # Validate variable format
        if "_" not in variable:
            raise ValueError(f"Variable must be in 'eye_variable' format (e.g., 'left_pupil'), got '{variable}'")
    
    @property
    def eye(self) -> str:
        """Return the eye part of the variable (e.g., 'left' from 'left_pupil')."""
        return self.variable.split("_")[0]
    
    @property
    def var(self) -> str:
        """Return the variable part (e.g., 'pupil' from 'left_pupil')."""
        return "_".join(self.variable.split("_")[1:])
    
    @property
    def n_timepoints(self) -> int:
        """Number of time points per segment."""
        return self.data.shape[0]
    
    @property
    def n_segments(self) -> int:
        """Number of segments."""
        return self.data.shape[1]
    
    @property
    def window(self) -> Tuple[float, float]:
        """Time window in ms as (start, end)."""
        return (self.tx.min(), self.tx.max())
    
    @classmethod
    def from_masked_array(cls,
                          data: Union[np.ndarray, ma.MaskedArray],
                          tx: np.ndarray,
                          variable: str,
                          intervals=None,
                          name: str = "segments",
                          sampling_rate: float = None,
                          mask: np.ndarray = None) -> 'SegmentedEyeData':
        """
        Create SegmentedEyeData from a masked array.
        
        Parameters
        ----------
        data : np.ndarray or np.ma.MaskedArray
            Data array with shape (n_timepoints, n_segments)
        tx : np.ndarray
            Time axis relative to segment onset (in ms)
        variable : str
            Variable name in "eye_variable" format
        intervals : Intervals, optional
            Source Intervals object with event metadata
        name : str
            Name for the dataset
        sampling_rate : float, optional
            Original sampling rate in Hz
        mask : np.ndarray, optional
            Mask array (1=masked/missing, 0=valid). Only used if data is not already a MaskedArray.
            
        Returns
        -------
        SegmentedEyeData
            New instance with the provided data
        """
        if isinstance(data, ma.MaskedArray):
            masked_data = data
        else:
            if mask is None:
                mask = np.isnan(data)
            masked_data = ma.masked_array(data, mask=mask)
        
        return cls(
            name=name,
            variable=variable,
            tx=tx,
            data=masked_data,
            intervals=intervals,
            sampling_rate=sampling_rate
        )
    
    @classmethod
    def from_eyedata(cls, 
                     eyedata, 
                     intervals, 
                     variable: str,
                     name: str = None) -> 'SegmentedEyeData':
        """
        Extract segments from a GenericEyeData object using intervals.
        
        Parameters
        ----------
        eyedata : GenericEyeData
            Source eye data object (PupilData, GazeData, etc.)
        intervals : Intervals
            Intervals object defining the segments to extract
        variable : str
            Variable to extract in "eye_variable" format (e.g., "left_pupil")
        name : str, optional
            Name for the dataset. If None, uses intervals.label or "segments"
            
        Returns
        -------
        SegmentedEyeData
            New instance with extracted segments
        """
        from .intervals import Intervals
        
        if not isinstance(intervals, Intervals):
            raise TypeError("intervals must be an Intervals object. Use get_intervals() to create one.")
        
        # Validate variable format
        if "_" not in variable:
            raise ValueError(f"Variable must be in 'eye_variable' format (e.g., 'left_pupil'), got '{variable}'")
        
        # Check if variable exists in data
        if variable not in eyedata.data:
            available = list(eyedata.data.data.keys())
            raise KeyError(f"Variable '{variable}' not found in data. Available: {available}")
        
        # Convert Intervals to indices for data extraction
        if intervals.units is None:
            intervals_idx = intervals.intervals
            first_start, first_end = intervals_idx[0]
            duration_ix = first_end - first_start
            interval_ms = (first_start / eyedata.fs * 1000, first_end / eyedata.fs * 1000)
        else:
            fac_to_ms = 1.0 / eyedata._unit_fac(intervals.units)
            intervals_idx = []
            
            first_start, first_end = intervals.intervals[0]
            start_ms = first_start * fac_to_ms
            end_ms = first_end * fac_to_ms
            interval_ms = (start_ms, end_ms)
            
            for start, end in intervals.intervals:
                start_ms = start * fac_to_ms
                end_ms = end * fac_to_ms
                start_ix = np.argmin(np.abs(eyedata.tx - start_ms))
                end_ix = np.argmin(np.abs(eyedata.tx - end_ms))
                intervals_idx.append((start_ix, end_ix))
        
        # Calculate duration and time window
        duration_ix = intervals_idx[0][1] - intervals_idx[0][0]
        txw = np.linspace(interval_ms[0], interval_ms[1], num=duration_ix)
        
        n_segments = len(intervals_idx)
        
        # Extract segments
        data_arr = np.zeros((duration_ix, n_segments))
        mask_arr = np.ones((duration_ix, n_segments), dtype=bool)
        
        for i, (on, off) in enumerate(intervals_idx):
            onl, offl = 0, duration_ix  # "local" window indices
            
            if on < 0:  # pad with zeros in case timewindow starts before data
                onl = abs(on)
                on = 0
            if off >= eyedata.tx.size - 1:
                offl = (off - on)
                off = eyedata.tx.size
            
            data_arr[onl:offl, i] = eyedata.data[variable][on:off]
            mask_arr[onl:offl, i] = eyedata.data.mask[variable][on:off]
        
        masked_data = ma.masked_array(data_arr, mask=mask_arr)
        
        # Determine name
        if name is None:
            name = intervals.label if intervals.label is not None else "segments"
        
        return cls(
            name=name,
            variable=variable,
            tx=txw,
            data=masked_data,
            intervals=intervals,
            sampling_rate=eyedata.fs
        )
    
    def summary(self) -> dict:
        """
        Return a summary of the SegmentedEyeData object.
        
        Returns
        -------
        dict
            Dictionary with name, variable, n_segments, n_timepoints, window, 
            percent_masked, and sampling_rate
        """
        percent_masked = np.mean(self.data.mask) * 100 if self.data.mask is not np.bool_(False) else 0.0
        
        return dict(
            name=self.name,
            variable=self.variable,
            n_segments=self.n_segments,
            n_timepoints=self.n_timepoints,
            window=self.window,
            percent_masked=percent_masked,
            sampling_rate=self.sampling_rate
        )
    
    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        pars = self.summary()
        del pars["name"]
        s = f"SegmentedEyeData({self.name}):\n"
        flen = max(len(k) for k in pars.keys())
        for k, v in pars.items():
            if isinstance(v, float):
                s += f"  {k:<{flen}}: {v:.2f}\n"
            else:
                s += f"  {k:<{flen}}: {v}\n"
        return s
    
    def baseline_correct(self, 
                         baseline_win: Union[Tuple[float, float], float, None] = 0) -> 'SegmentedEyeData':
        """
        Apply baseline correction to the segments.
        
        Parameters
        ----------
        baseline_win : tuple (float, float), float, or None
            - If None, no baseline correction is applied
            - If tuple, the mean value in the window (ms) is subtracted from each segment
            - If float, the value at the time point closest to this value is used as baseline
            
        Returns
        -------
        SegmentedEyeData
            New instance with baseline-corrected data
        """
        if baseline_win is None:
            logger.warning("No baseline correction applied")
            return self
        
        # Create copy of data
        corrected_data = self.data.copy()
        
        if not isinstance(baseline_win, Iterable):
            # Single time point
            blwin_ix = np.argmin(np.abs(self.tx - baseline_win))
            for i in range(self.n_segments):
                baseline = corrected_data[blwin_ix, i]
                corrected_data[:, i] -= baseline
        else:
            # Time window
            blwin_ix = tuple(np.argmin(np.abs(bw - self.tx)) for bw in baseline_win)
            for i in range(self.n_segments):
                baseline = np.mean(corrected_data[blwin_ix[0]:blwin_ix[1], i])
                corrected_data[:, i] -= baseline
        
        logger.info(f"Baseline correction applied using window {baseline_win}")
        
        return SegmentedEyeData(
            name=self.name,
            variable=self.variable,
            tx=self.tx,
            data=corrected_data,
            intervals=self.intervals,
            sampling_rate=self.sampling_rate
        )
    
    def plot(self,
             meanfct=np.mean,
             varfct=scipy.stats.sem,
             plot_missing: bool = True,
             title: str = None,
             ax=None):
        """
        Plot mean and error ribbons.
        
        Parameters
        ----------
        meanfct : callable
            Function to compute mean across segments (default: np.mean)
        varfct : callable or None
            Function to compute error bands (e.g., np.std, scipy.stats.sem)
            If None, no error bands are plotted
        plot_missing : bool
            Plot percentage of missing data per time point
        title : str, optional
            Plot title. If None, uses the dataset name
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses current axes
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            ax1 = plt.gca()
        else:
            ax1 = ax
        
        if plot_missing:
            ax2 = ax1.twinx()
        
        # Compute mean and variance
        mean_data = meanfct(self.data, axis=1)
        sd_data = varfct(self.data, axis=1) if callable(varfct) else None
        
        # Compute percent missing
        if self.data.mask is not np.bool_(False):
            perc_missing = np.mean(self.data.mask, axis=1) * 100
        else:
            perc_missing = np.zeros(self.n_timepoints)
        
        # Plot
        if sd_data is not None:
            ax1.fill_between(self.tx, mean_data - sd_data, mean_data + sd_data, 
                           color="grey", alpha=0.3)
        ax1.plot(self.tx, mean_data, label=self.variable)
        ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        
        ax1.set_ylabel(f"mean {self.var}")
        ax1.set_xlabel("time (ms)")
        ax1.set_title(title if title is not None else self.name)
        ax1.legend()
        
        if plot_missing:
            ax2.plot(self.tx, perc_missing, alpha=0.3, color="orange")
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("% missing", color="orange")
    
    def write_file(self, fname: str):
        """
        Save to file using pickle.
        
        Parameters
        ----------
        fname : str
            Filename to save to
        """
        pd_write_pickle(self, fname)
    
    @classmethod
    def from_file(cls, fname: str) -> 'SegmentedEyeData':
        """
        Read a SegmentedEyeData object from a pickle file.
        
        Parameters
        ----------
        fname : str
            Filename to read from
            
        Returns
        -------
        SegmentedEyeData
            Loaded instance
        """
        return pd_read_pickle(fname)

