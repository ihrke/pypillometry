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

from .io import write_pickle, read_pickle
from .convenience import _warn_if_ignores_mask, _check_accepts_axis

__all__ = [
    "SegmentedEyeData", "GroupLevelSegmentedData", "group_segments",
]


class SegmentedEyeData:
    """
    Class representing segmented eye data (e.g., event-related pupil dilation).
    
    Holds equal-length segments of a single eye data variable (e.g., "left_pupil", 
    "right_x") extracted around events. Tracks masked/missing data.

    Usually created with :meth:`~pypillometry.GenericEyeData.get_segments`
    from the Intervals returned by
    :meth:`~pypillometry.GenericEyeData.get_intervals`, e.g.
    ``d.get_segments(d.get_intervals("F", interval=(-200, 800), units="ms"),
    "left_pupil")``. Segments of several subjects can be combined with
    :func:`~pypillometry.group_segments`.

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
        name : str, optional
            Name for the dataset (default "segments")
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
        
        # Calculate duration and time window; for event-locked intervals, use
        # the window, so that the length does not depend on the event timing
        if intervals.interval_window is not None:
            w0, w1 = intervals.interval_window
            if intervals.units is None:
                duration_ix = int(round(w1 - w0))
            else:
                fac_to_ms = 1.0 / eyedata._unit_fac(intervals.units)
                duration_ix = int(round((w1 - w0) * fac_to_ms * eyedata.fs / 1000.))
        else:
            duration_ix = intervals_idx[0][1] - intervals_idx[0][0]
        
        # Use interval_window for relative time axis if available (event-locked intervals)
        # Otherwise fall back to absolute interval times
        if intervals.interval_window is not None:
            # Convert interval_window to ms if needed
            if intervals.units is None:
                # interval_window is in indices, convert to ms
                ms_per_sample = 1000.0 / eyedata.fs
                txw = np.linspace(
                    intervals.interval_window[0] * ms_per_sample,
                    intervals.interval_window[1] * ms_per_sample,
                    num=duration_ix
                )
            else:
                # interval_window is in the same units as intervals, convert to ms
                fac_to_ms = 1.0 / eyedata._unit_fac(intervals.units)
                txw = np.linspace(
                    intervals.interval_window[0] * fac_to_ms,
                    intervals.interval_window[1] * fac_to_ms,
                    num=duration_ix
                )
        else:
            # No interval_window, use absolute times (not event-locked)
            txw = np.linspace(interval_ms[0], interval_ms[1], num=duration_ix)
        
        n_segments = len(intervals_idx)
        
        # Extract segments
        data_arr = np.zeros((duration_ix, n_segments))
        mask_arr = np.ones((duration_ix, n_segments), dtype=bool)
        
        for i, (on, _) in enumerate(intervals_idx):
            off = on + duration_ix
            onl, offl = 0, duration_ix  # "local" window indices
            
            if on < 0:  # pad with zeros in case timewindow starts before data
                onl = abs(on)
                on = 0
            if off > eyedata.tx.size:
                offl = duration_ix - (off - eyedata.tx.size)
                off = eyedata.tx.size
            
            # Ensure we extract exactly the right number of samples
            segment_len = min(offl - onl, off - on)
            data_arr[onl:onl+segment_len, i] = eyedata.data[variable][on:on+segment_len]
            mask_arr[onl:onl+segment_len, i] = eyedata.data.mask[variable][on:on+segment_len]
        
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
    
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        import base64
        from io import BytesIO
        
        summary = self.summary()
        
        # Build overview items
        window = summary['window']
        overview = [
            ("Variable", summary['variable']),
            ("Segments", f"{summary['n_segments']:,}"),
            ("Time points", f"{summary['n_timepoints']:,}"),
            ("Window", f"{window[0]:.0f} to {window[1]:.0f} ms"),
            ("Missing", f"{summary['percent_masked']:.1f}%"),
        ]
        if summary['sampling_rate']:
            overview.append(("Sampling rate", f"{summary['sampling_rate']:.0f} Hz"))
        
        # Add intervals info if available
        if self.intervals is not None and self.intervals.label:
            overview.insert(0, ("Intervals", self.intervals.label))
        
        # Generate sparkline of mean signal
        sparkline_html = ""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            mean_signal = np.mean(self.data, axis=1)
            
            fig = Figure(figsize=(3, 1), dpi=100)
            ax = fig.subplots()
            ax.plot(self.tx, mean_signal, color="#1f77b4", linewidth=1.5)
            ax.axvline(x=0, color="#d62728", linewidth=1, linestyle="--", alpha=0.7)
            ax.set_xlim(self.tx[0], self.tx[-1])
            ax.axis("off")
            fig.tight_layout(pad=0)
            
            canvas = FigureCanvasAgg(fig)
            buf = BytesIO()
            canvas.print_png(buf)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("ascii")
            
            sparkline_html = f'''
            <div style="margin-top:10px;">
              <div style="font-size:0.85em; color:#666; margin-bottom:4px;">Mean signal (red line = event onset)</div>
              <img src="data:image/png;base64,{img_b64}" alt="Mean signal" style="width:100%; max-width:300px; border-radius:4px;">
            </div>
            '''
        except Exception:
            pass  # Silently skip sparkline if matplotlib not available
        
        # Build HTML
        overview_html = "".join(
            f'''<div style="background:#f6f6f6; border-radius:4px; padding:6px 8px; border:1px solid #e0e0e0;">
                  <div style="font-size:0.85em; color:#666; text-transform:uppercase;">{label}</div>
                  <div style="font-weight:600;">{value}</div>
                </div>'''
            for label, value in overview
        )
        
        html = f'''
        <div style="font-family:system-ui, sans-serif; border:1px solid #ddd; border-radius:6px; padding:12px; max-width:400px;">
          <h3 style="margin-top:0;">SegmentedEyeData — <span style="color:#555;">{summary['name']}</span></h3>
          <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:8px;">
            {overview_html}
          </div>
          {sparkline_html}
        </div>
        '''
        return html
    
    def baseline_correct(self, 
                         window: Union[Tuple[float, float], float, None] = 0) -> 'SegmentedEyeData':
        """
        Apply baseline correction to the segments.
        
        The window is specified in milliseconds relative to the zero-point of the 
        intervals (typically the event onset). For example, `window=(-200, 0)` uses 
        the 200ms before the event as baseline.
        
        Parameters
        ----------
        window : tuple (float, float), float, or None
            Baseline window in ms relative to event onset (time 0):

            - If None, no baseline correction is applied
            - If tuple (start, end), the mean value in this time window is subtracted
            - If float, the value at the time point closest to this value is used

            Default is 0 (baseline at event onset).

        Returns
        -------
        SegmentedEyeData
            New instance with baseline-corrected data (the object itself if
            `window` is None)

        Examples
        --------
        >>> # Use pre-event period as baseline
        >>> segments_bc = segments.baseline_correct(window=(-200, 0))
        
        >>> # Use single time point as baseline
        >>> segments_bc = segments.baseline_correct(window=-100)
        """
        if window is None:
            logger.warning("No baseline correction applied")
            return self
        
        # Create copy of data
        corrected_data = self.data.copy()
        
        if not isinstance(window, Iterable):
            # Single time point
            blwin_ix = np.argmin(np.abs(self.tx - window))
            for i in range(self.n_segments):
                baseline = corrected_data[blwin_ix, i]
                corrected_data[:, i] -= baseline
        else:
            # Time window
            blwin_ix = tuple(np.argmin(np.abs(bw - self.tx)) for bw in window)
            for i in range(self.n_segments):
                baseline = np.mean(corrected_data[blwin_ix[0]:blwin_ix[1], i])
                corrected_data[:, i] -= baseline
        
        logger.info(f"Baseline correction applied using window {window}")
        
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
             show_missing: bool = None,
             show_zero_line: bool = True,
             show_legend: bool = None,
             title: str = None,
             label: str = None,
             ribbon_alpha: float = 0.3,
             ax=None,
             **kwargs):
        """
        Plot mean and error ribbons.
        
        Can be used standalone for a quick look, or with `ax` parameter to 
        overlay multiple segments on the same plot or across subplots.
        
        Parameters
        ----------
        meanfct : callable
            Function to compute mean across segments (default: np.mean)
        varfct : callable or None
            Function to compute error bands (e.g., np.std, scipy.stats.sem)
            If None, no error bands are plotted.
            Both functions are called as ``fct(data, axis=1)``, so they must
            accept an `axis` argument, and they must ignore masked values
            (e.g., np.ma.median, not np.median; a warning is logged otherwise).
        show_missing : bool or None
            Plot percentage of missing data per time point on secondary y-axis.
            If None, defaults to True in standalone mode, False in overlay mode.
        show_zero_line : bool
            Show vertical line at time 0 (default: True)
        show_legend : bool or None
            Show legend. If None, only shows legend in standalone mode (ax=None)
        title : str, optional
            Plot title. If None and ax is None, uses the dataset name.
            Ignored when ax is provided (for overlay use).
        label : str, optional
            Label for legend. If None, uses self.name
        ribbon_alpha : float
            Alpha (transparency) for error ribbon (default: 0.3)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses current axes (standalone mode)
        **kwargs : dict
            Additional arguments passed to ax.plot() (e.g., color, linestyle, linewidth)
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object for further customization
            
        Examples
        --------
        >>> # Quick standalone plot
        >>> segments.plot()
        
        >>> # Overlay multiple segments
        >>> fig, ax = plt.subplots()
        >>> seg1.plot(ax=ax, color='blue', label='Condition A')
        >>> seg2.plot(ax=ax, color='red', label='Condition B')
        >>> ax.legend()
        
        >>> # Customize appearance
        >>> segments.plot(varfct=np.std, color='green', linewidth=2, linestyle='--')
        """
        import matplotlib.pyplot as plt
        
        standalone = ax is None
        if standalone:
            ax1 = plt.gca()
        else:
            ax1 = ax
        
        # Determine label for legend
        plot_label = label if label is not None else self.name
        
        # Compute mean and variance
        _check_accepts_axis(meanfct, "meanfct")
        _warn_if_ignores_mask(meanfct, "meanfct", axis=1)
        mean_data = meanfct(self.data, axis=1)
        if callable(varfct):
            _check_accepts_axis(varfct, "varfct")
            _warn_if_ignores_mask(varfct, "varfct", axis=1)
        sd_data = varfct(self.data, axis=1) if callable(varfct) else None
        
        # Get color from kwargs for consistent ribbon color, or use default
        line_color = kwargs.get('color', kwargs.get('c', None))
        
        # Plot error ribbon
        if sd_data is not None:
            ribbon_color = line_color if line_color else "grey"
            ax1.fill_between(self.tx, mean_data - sd_data, mean_data + sd_data, 
                           color=ribbon_color, alpha=ribbon_alpha)
        
        # Plot mean line
        line, = ax1.plot(self.tx, mean_data, label=plot_label, **kwargs)
        
        # Show zero line
        if show_zero_line:
            ax1.axvline(x=0, color="grey", linestyle="--", alpha=0.5, zorder=0)
        
        # Labels (always set for clarity)
        ax1.set_xlabel("time (ms)")
        
        # Only set title and ylabel in standalone mode
        if standalone:
            ax1.set_ylabel(f"mean {self.var}")
            ax1.set_title(title if title is not None else self.name)
        
        # Show missing data on secondary axis (default: only in standalone mode)
        if show_missing is None:
            show_missing = standalone
        if show_missing:
            ax2 = ax1.twinx()
            if self.data.mask is not np.bool_(False):
                perc_missing = np.mean(self.data.mask, axis=1) * 100
            else:
                perc_missing = np.zeros(self.n_timepoints)
            ax2.plot(self.tx, perc_missing, alpha=0.3, color="orange")
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("% missing", color="orange")
            # Restore ax1 as current axes so plt.legend() works for user
            plt.sca(ax1)
        
        # Legend handling (after all plotting is done)
        if show_legend is None:
            show_legend = standalone
        if show_legend:
            ax1.legend()
        
        return ax1
    
    def write_file(self, fname: str, compress: bool = True):
        """
        Save to file using pickle (see :func:`~pypillometry.io.write_pickle`).
        
        Parameters
        ----------
        fname : str
            Filename to save to
        compress : bool, optional
            compress the file with zstd (default True)
        """
        write_pickle(self, fname, compress=compress)
    
    @classmethod
    def from_file(cls, fname: str, **kwargs) -> 'SegmentedEyeData':
        """
        Read a SegmentedEyeData object from a pickle file.
        
        Parameters
        ----------
        fname : str
            Filename (or HTTP(S)/SSH URL) to read from
        **kwargs
            passed to :func:`~pypillometry.io.read_pickle` (e.g.,
            `cache_file`, `cache_dir`, `cache_layout`, `autodelete`,
            `session`)
            
        Returns
        -------
        SegmentedEyeData
            Loaded instance
        """
        return read_pickle(fname, **kwargs)


class GroupLevelSegmentedData(SegmentedEyeData):
    """
    Group-level segmented eye data combining multiple subjects.
    
    This class inherits from SegmentedEyeData and adds group-level functionality.
    It combines multiple SegmentedEyeData objects (one per subject) into a single
    group-level object, either by stacking or aggregating.

    Usually created with :func:`~pypillometry.group_segments` (or
    :meth:`from_segments`) from a list or dict of
    :class:`~pypillometry.SegmentedEyeData` objects, one per subject.

    The data keep their mask: stacked segments keep their own mask, and in
    aggregate mode a subject's mean is masked where all of their segments are
    masked. In addition, `mask_percent` stores the average percentage of masked
    trials per subject at each timepoint.
    
    Parameters
    ----------
    name : str
        Name of the dataset
    variable : str
        Variable name in "eye_variable" format
    tx : np.ndarray
        Time axis relative to segment onset (in ms)
    data : np.ma.MaskedArray
        Masked array with shape (n_timepoints, n_subjects) or (n_timepoints, n_segments)
    mask_percent : np.ndarray
        Percentage of masked data per subject at each timepoint (shape: n_timepoints)
    n_subjects : int
        Number of subjects combined
    intervals : Intervals, optional
        Source Intervals object
    sampling_rate : float, optional
        Original sampling rate in Hz
    """
    
    def __init__(self,
                 name: str,
                 variable: str,
                 tx: np.ndarray,
                 data: ma.MaskedArray,
                 mask_percent: np.ndarray,
                 n_subjects: int,
                 intervals=None,
                 sampling_rate: float = None):
        # Call parent constructor
        super().__init__(
            name=name,
            variable=variable,
            tx=tx,
            data=data,
            intervals=intervals,
            sampling_rate=sampling_rate
        )
        self.mask_percent = mask_percent
        self.n_subjects = n_subjects
    
    @classmethod
    def from_segments(cls,
                      segments: list,
                      meanfct=np.mean,
                      name: str = None) -> 'GroupLevelSegmentedData':
        """
        Create GroupLevelSegmentedData by combining multiple SegmentedEyeData objects.
        
        Parameters
        ----------
        segments : list of SegmentedEyeData
            List of SegmentedEyeData objects (one per subject)
        meanfct : callable or None
            Function to aggregate within each subject, called as
            ``meanfct(data, axis=1)``; it must accept an `axis` argument and
            ignore masked values (e.g., np.mean or np.ma.median, but not
            np.median; a warning is logged otherwise).
            If None, all segments from all subjects are stacked (concatenated).
        name : str, optional
            Name for the group-level dataset. If None, uses ``"group_"`` + first
            segment's name
            
        Returns
        -------
        GroupLevelSegmentedData
            New instance with combined data
            
        Examples
        --------
        >>> # Aggregate within subjects, then combine
        >>> group = GroupLevelSegmentedData.from_segments([subj1, subj2, subj3])
        
        >>> # Stack all trials from all subjects
        >>> group = GroupLevelSegmentedData.from_segments([subj1, subj2, subj3], meanfct=None)
        """
        if not segments:
            raise ValueError("segments list cannot be empty")
        
        # Validate all segments have compatible properties
        first = segments[0]
        for i, seg in enumerate(segments[1:], start=2):
            if seg.variable != first.variable:
                raise ValueError(f"Variable mismatch: segment 1 has '{first.variable}', "
                               f"segment {i} has '{seg.variable}'")
            if seg.n_timepoints != first.n_timepoints:
                raise ValueError(f"Timepoint mismatch: segment 1 has {first.n_timepoints}, "
                               f"segment {i} has {seg.n_timepoints}")
            if not np.allclose(seg.tx, first.tx):
                raise ValueError(f"Time axis mismatch between segment 1 and segment {i}")
        
        n_subjects = len(segments)
        
        # Compute mask percentage for each subject at each timepoint
        # (average percentage of masked trials per subject)
        mask_percents = []
        for seg in segments:
            if seg.data.mask is not np.bool_(False):
                # Percentage of masked segments at each timepoint
                perc = np.mean(seg.data.mask, axis=1) * 100
            else:
                perc = np.zeros(seg.n_timepoints)
            mask_percents.append(perc)
        
        # Average mask percentage across subjects
        mask_percent = np.mean(mask_percents, axis=0)
        
        if meanfct is None:
            # Stack mode: concatenate all segments from all subjects (keeps masks)
            combined_data = ma.concatenate([seg.data for seg in segments], axis=1)
        else:
            # Aggregate mode: apply meanfct within each subject, then stack
            _check_accepts_axis(meanfct, "meanfct")
            _warn_if_ignores_mask(meanfct, "meanfct", axis=1)
            subject_means = [ma.asarray(meanfct(seg.data, axis=1)) for seg in segments]
            # Stack as columns (n_timepoints, n_subjects); a timepoint where all
            # of a subject's segments are masked stays masked
            combined_data = ma.column_stack(subject_means)
        
        # Determine name
        if name is None:
            name = f"group_{first.name}"
        
        return cls(
            name=name,
            variable=first.variable,
            tx=first.tx.copy(),
            data=combined_data,
            mask_percent=mask_percent,
            n_subjects=n_subjects,
            intervals=first.intervals,
            sampling_rate=first.sampling_rate
        )
    
    def summary(self) -> dict:
        """
        Return a summary of the GroupLevelSegmentedData object.
        
        Returns
        -------
        dict
            Dictionary with group-level statistics
        """
        base_summary = super().summary()
        base_summary['n_subjects'] = self.n_subjects
        base_summary['mean_mask_percent'] = np.mean(self.mask_percent)
        # Replace percent_masked with more meaningful group-level stat
        base_summary['percent_masked'] = base_summary['mean_mask_percent']
        return base_summary
    
    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        pars = self.summary()
        del pars["name"]
        s = f"GroupLevelSegmentedData({self.name}):\n"
        flen = max(len(k) for k in pars.keys())
        for k, v in pars.items():
            if isinstance(v, float):
                s += f"  {k:<{flen}}: {v:.2f}\n"
            else:
                s += f"  {k:<{flen}}: {v}\n"
        return s
    
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        import base64
        from io import BytesIO
        
        summary = self.summary()
        
        # Build overview items
        window = summary['window']
        overview = [
            ("Variable", summary['variable']),
            ("Subjects", f"{summary['n_subjects']}"),
            ("Columns", f"{summary['n_segments']:,}"),
            ("Time points", f"{summary['n_timepoints']:,}"),
            ("Window", f"{window[0]:.0f} to {window[1]:.0f} ms"),
            ("Avg missing", f"{summary['mean_mask_percent']:.1f}%"),
        ]
        if summary['sampling_rate']:
            overview.append(("Sampling rate", f"{summary['sampling_rate']:.0f} Hz"))
        
        # Generate sparkline of mean signal
        sparkline_html = ""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            mean_signal = np.mean(self.data, axis=1)
            
            fig = Figure(figsize=(3, 1), dpi=100)
            ax = fig.subplots()
            ax.plot(self.tx, mean_signal, color="#1f77b4", linewidth=1.5)
            ax.axvline(x=0, color="#d62728", linewidth=1, linestyle="--", alpha=0.7)
            ax.set_xlim(self.tx[0], self.tx[-1])
            ax.axis("off")
            fig.tight_layout(pad=0)
            
            canvas = FigureCanvasAgg(fig)
            buf = BytesIO()
            canvas.print_png(buf)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("ascii")
            
            sparkline_html = f'''
            <div style="margin-top:10px;">
              <div style="font-size:0.85em; color:#666; margin-bottom:4px;">Grand mean (red line = event onset)</div>
              <img src="data:image/png;base64,{img_b64}" alt="Grand mean" style="width:100%; max-width:300px; border-radius:4px;">
            </div>
            '''
        except Exception:
            pass
        
        # Build HTML
        overview_html = "".join(
            f'''<div style="background:#f6f6f6; border-radius:4px; padding:6px 8px; border:1px solid #e0e0e0;">
                  <div style="font-size:0.85em; color:#666; text-transform:uppercase;">{label}</div>
                  <div style="font-weight:600;">{value}</div>
                </div>'''
            for label, value in overview
        )
        
        html = f'''
        <div style="font-family:system-ui, sans-serif; border:1px solid #ddd; border-radius:6px; padding:12px; max-width:400px;">
          <h3 style="margin-top:0;">GroupLevelSegmentedData — <span style="color:#555;">{summary['name']}</span></h3>
          <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:8px;">
            {overview_html}
          </div>
          {sparkline_html}
        </div>
        '''
        return html
    
    def plot(self,
             meanfct=np.mean,
             varfct=scipy.stats.sem,
             show_missing: bool = None,
             show_zero_line: bool = True,
             show_legend: bool = None,
             title: str = None,
             label: str = None,
             ribbon_alpha: float = 0.3,
             ax=None,
             **kwargs):
        """
        Plot grand mean and error ribbons.

        Overrides :meth:`SegmentedEyeData.plot` to use `mask_percent` for the
        missing-data display. Mean and error are computed across the columns
        of `data` (subjects when aggregated, all segments when stacked).

        Parameters
        ----------
        meanfct : callable
            Function to compute mean across columns (default: np.mean)
        varfct : callable or None
            Function to compute error bands (e.g., np.std, scipy.stats.sem;
            default: scipy.stats.sem). If None, no error bands are plotted.
            Both functions are called as ``fct(data, axis=1)``, so they must
            accept an `axis` argument, and they must ignore masked values
            (e.g., np.ma.median, not np.median; a warning is logged otherwise).
        show_missing : bool or None
            Plot `mask_percent` (average percentage of missing data per
            subject) on secondary y-axis. If None, defaults to True in
            standalone mode, False in overlay mode.
        show_zero_line : bool
            Show vertical line at time 0 (default: True)
        show_legend : bool or None
            Show legend. If None, only shows legend in standalone mode (ax=None)
        title : str, optional
            Plot title. If None and ax is None, uses the dataset name.
            Ignored when ax is provided (for overlay use).
        label : str, optional
            Label for legend. If None, uses self.name
        ribbon_alpha : float
            Alpha (transparency) for error ribbon (default: 0.3)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses current axes (standalone mode)
        **kwargs : dict
            Additional arguments passed to ax.plot() (e.g., color, linestyle, linewidth)

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object for further customization
        """
        import matplotlib.pyplot as plt
        
        standalone = ax is None
        if standalone:
            ax1 = plt.gca()
        else:
            ax1 = ax
        
        # Determine label for legend
        plot_label = label if label is not None else self.name
        
        # Compute mean and variance
        _check_accepts_axis(meanfct, "meanfct")
        _warn_if_ignores_mask(meanfct, "meanfct", axis=1)
        mean_data = meanfct(self.data, axis=1)
        if callable(varfct):
            _check_accepts_axis(varfct, "varfct")
            _warn_if_ignores_mask(varfct, "varfct", axis=1)
        sd_data = varfct(self.data, axis=1) if callable(varfct) else None
        
        # Get color from kwargs
        line_color = kwargs.get('color', kwargs.get('c', None))
        
        # Plot error ribbon
        if sd_data is not None:
            ribbon_color = line_color if line_color else "grey"
            ax1.fill_between(self.tx, mean_data - sd_data, mean_data + sd_data, 
                           color=ribbon_color, alpha=ribbon_alpha)
        
        # Plot mean line
        ax1.plot(self.tx, mean_data, label=plot_label, **kwargs)
        
        # Show zero line
        if show_zero_line:
            ax1.axvline(x=0, color="grey", linestyle="--", alpha=0.5, zorder=0)
        
        ax1.set_xlabel("time (ms)")
        
        if standalone:
            ax1.set_ylabel(f"mean {self.var}")
            ax1.set_title(title if title is not None else self.name)
        
        # Show missing data using mask_percent
        if show_missing is None:
            show_missing = standalone
        if show_missing:
            ax2 = ax1.twinx()
            ax2.plot(self.tx, self.mask_percent, alpha=0.3, color="orange")
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("avg % missing", color="orange")
            plt.sca(ax1)
        
        # Legend handling
        if show_legend is None:
            show_legend = standalone
        if show_legend:
            ax1.legend()
        
        return ax1


def group_segments(segments, meanfct=np.mean, name: str = None) -> GroupLevelSegmentedData:
    """
    Combine multiple SegmentedEyeData objects into a GroupLevelSegmentedData.
    
    Convenience function that wraps GroupLevelSegmentedData.from_segments().
    
    Parameters
    ----------
    segments : list or dict of SegmentedEyeData
        SegmentedEyeData objects to combine. If dict, values are used.
    meanfct : callable or None
        Function to aggregate within each subject, called as
        ``meanfct(data, axis=1)``; it must accept an `axis` argument and
        ignore masked values (e.g., np.mean or np.ma.median, but not
        np.median; a warning is logged otherwise).
        If None, all segments from all subjects are stacked (concatenated).
    name : str, optional
        Name for the group-level dataset. If None, uses ``"group_"`` + first
        segment's name

    Returns
    -------
    GroupLevelSegmentedData
        Combined group-level data
        
    Examples
    --------
    >>> import pypillometry as pp
    >>> # From a list
    >>> group = pp.group_segments([subj1_seg, subj2_seg, subj3_seg])
    
    >>> # From a dict
    >>> segments = {"S01": subj1_seg, "S02": subj2_seg}
    >>> group = pp.group_segments(segments)
    
    >>> # Stack all trials instead of averaging
    >>> group = pp.group_segments(segments, meanfct=None)
    """
    # Handle dict input
    if isinstance(segments, dict):
        segments = list(segments.values())
    
    return GroupLevelSegmentedData.from_segments(segments, meanfct=meanfct, name=name)

