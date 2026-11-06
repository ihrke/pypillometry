"""
segmented.py
============

Segmented eye data for event-related analysis.

"""
import numpy as np
import numpy.ma as ma
from typing import Optional, Tuple, Union
from loguru import logger
import scipy.stats

from .io import write_pickle, read_pickle
from . import config, state
from .units import parse_time, _ms_to_samples
from .convenience import _warn_if_ignores_mask, _check_accepts_axis, _plain_values

__all__ = [
    "SegmentedEyeData", "GroupLevelSegmentedData", "group_segments",
]


def _extract_epochs(eyedata, epochs, variable, policy="default"):
    """
    Time axis and segments (masked array, time points x epochs) of `variable`,
    masked where the samples are unusable under the mask `policy`.

    The length and the time axis come from the window: ``w0 + arange(n) *
    1000/fs`` ms, with ``n`` the window length in samples (rounded). Each
    segment starts at the first sample at or after ``onset + w0``. Parts
    outside the data are masked (their values are NaN).
    """
    w0, w1 = epochs.window
    n = eyedata._samples(w1 - w0, rounding="nearest")
    tx = w0 + np.arange(n) * 1000. / eyedata.fs
    starts = np.atleast_1d(_ms_to_samples(eyedata, epochs.onsets + w0))
    ix = starts[None, :] + np.arange(n)[:, None]
    inside = (ix >= 0) & (ix < len(eyedata.tx))
    values = np.full(ix.shape, np.nan)
    mask = np.ones(ix.shape, dtype=bool)
    values[inside] = eyedata.data[variable][ix[inside]]
    mask[inside] = eyedata.data.get_policy_mask(variable, policy)[ix[inside]]
    return tx, ma.masked_array(values, mask=mask)


def _epochs_from_spans(eyedata, intervals):
    """
    Epochs for Intervals of the same length: onsets at the starts, window
    (0, length) in ms; their event labels and metadata are kept.
    """
    from .epochs import Epochs
    if len(intervals) == 0:
        raise ValueError("no intervals to extract segments from")
    lengths = np.diff(np.asarray(intervals, dtype=float), axis=1)[:, 0]
    if not np.allclose(lengths, lengths[0], rtol=1e-9, atol=0):
        raise ValueError(
            f"get_segments() needs intervals of the same length, got lengths from "
            f"{lengths.min():g} to {lengths.max():g} {intervals.units}; for "
            f"event-locked segments, use a window around single events, e.g., "
            f"get_epochs(event, window=(-200, 800) * u.ms)")
    if lengths[0] <= 0:
        raise ValueError("get_segments() needs intervals that are longer than 0")
    spans = np.asarray(intervals.to_units("ms", eyedata), dtype=float)
    labels = None if intervals.event_labels is None else \
        [str(lab) for lab in intervals.event_labels]
    return Epochs(spans[:, 0], (0., float(np.mean(spans[:, 1] - spans[:, 0]))),
                  labels=labels, meta=intervals.event_meta, label=intervals.label)


class SegmentedEyeData(state.VersionedState):
    """
    Class representing segmented eye data (e.g., event-related pupil dilation).
    
    Holds equal-length segments of a single eye data variable (e.g., "left_pupil", 
    "right_x") extracted around events. Tracks masked/missing data.

    Usually created with :meth:`~pypillometry.EyeData.get_segments`
    from the :class:`~pypillometry.Epochs` returned by
    :meth:`~pypillometry.EyeData.get_epochs`, e.g.
    ``d.get_segments(d.get_epochs("F", window=(-200, 800), units="ms"),
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
    epochs : Epochs, optional
        the epochs the segments were extracted from (event onsets, window,
        labels and metadata); segments from old files made from intervals
        that were not event-locked keep these Intervals
    sampling_rate : float, optional
        Original sampling rate in Hz
    """
    
    # state (see pypillometry.state); `epochs` is Epochs or Intervals (stored
    # as "intervals", the attribute's name up to 2.2)
    _STATE_FIELDS = ("name", "variable", "tx", ("data", "masked"),
                     ("intervals", "tagged", "epochs"), "sampling_rate")

    def __init__(self, 
                 name: str, 
                 variable: str,
                 tx: np.ndarray, 
                 data: ma.MaskedArray,
                 epochs=None,
                 sampling_rate: float = None):
        self.name = name
        self.variable = variable
        self.tx = np.asarray(tx)
        
        # Ensure data is a masked array
        if not isinstance(data, ma.MaskedArray):
            data = ma.masked_array(data, mask=np.isnan(data))
        self.data = data
        
        self.epochs = epochs
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
                          epochs=None,
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
        epochs : Epochs, optional
            the epochs the segments were extracted from
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
            epochs=epochs,
            sampling_rate=sampling_rate
        )
    
    @classmethod
    def from_eyedata(cls, 
                     eyedata, 
                     intervals, 
                     variable: str,
                     name: str = None,
                     policy="default") -> 'SegmentedEyeData':
        """
        Extract segments from a EyeData object using intervals.
        
        Parameters
        ----------
        eyedata : EyeData
            Source eye data object
        intervals : Epochs or Intervals
            the segments to extract; for Epochs (e.g., from
            :meth:`~pypillometry.EyeData.get_epochs`),
            the length and the time axis (``window[0] + arange(n) * 1000/fs``
            ms) come from the window, and the parts outside the data are
            masked. Intervals must all have the same length; they are used
            as Epochs with the onsets at their starts and the window
            ``(0, length)`` (kept as `epochs` of the result)
        variable : str
            Variable to extract in "eye_variable" format (e.g., "left_pupil")
        name : str, optional
            Name for the dataset. If None, uses intervals.label or "segments"
        policy : str, MaskPolicy or list of str
            mask policy of the segments: the samples that are unusable under
            it are masked (see :meth:`~pypillometry.EyeData.mask`);
            default ``"default"`` (interpolated samples are used)
            
        Returns
        -------
        SegmentedEyeData
            New instance with extracted segments
        """
        from .intervals import Intervals
        from .epochs import Epochs

        if not isinstance(intervals, (Intervals, Epochs)):
            raise TypeError("intervals must be Epochs or Intervals, e.g., from get_epochs()")
        
        # Validate variable format
        if "_" not in variable:
            raise ValueError(f"Variable must be in 'eye_variable' format (e.g., 'left_pupil'), got '{variable}'")
        
        # Check if variable exists in data
        if variable not in eyedata.data:
            available = list(eyedata.data.data.keys())
            raise KeyError(f"Variable '{variable}' not found in data. Available: {available}")

        if name is None:
            name = intervals.label if intervals.label is not None else "segments"
        if isinstance(intervals, Intervals):
            intervals = _epochs_from_spans(eyedata, intervals)
        txw, masked_data = _extract_epochs(eyedata, intervals, variable, policy)
        return cls(name=name, variable=variable, tx=txw, data=masked_data,
                   epochs=intervals, sampling_rate=eyedata.fs)
    
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
        
        return _plain_values(dict(
            name=self.name,
            variable=self.variable,
            n_segments=self.n_segments,
            n_timepoints=self.n_timepoints,
            window=self.window,
            percent_masked=percent_masked,
            sampling_rate=self.sampling_rate
        ))
    
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
        if self.epochs is not None and self.epochs.label:
            overview.insert(0, ("Epochs", self.epochs.label))
        
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
                         window: Union[Tuple[float, float], float, None] = 0,
                         inplace: Optional[bool] = None) -> 'SegmentedEyeData':
        """
        Apply baseline correction to the segments.
        
        The window is specified in milliseconds relative to the zero-point of the 
        intervals (typically the event onset). For example, `window=(-200, 0)` uses 
        the 200ms before the event as baseline.
        
        Parameters
        ----------
        window : tuple (float, float), float, str, Quantity or None
            Baseline window relative to event onset (time 0), in ms or with
            units (e.g., ``(-200, 0) * u.ms``):

            - If None, no baseline correction is applied
            - If tuple (start, end), the mean value in this time window is subtracted
            - If float, the value at the time point closest to this value is used

            Default is 0 (baseline at event onset).

            The window must be within the time axis of the segments.
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, return a new object with the corrected data;
            if `None`, use the package-level setting (see
            :func:`~pypillometry.set_default_inplace`)

        Returns
        -------
        SegmentedEyeData
            the baseline-corrected segments (`self` or a new object,
            depending on `inplace`; `self` if `window` is None)

        Raises
        ------
        ValueError
            if the window (or time point) is not within the segments, or its
            start is not before its end

        Examples
        --------
        >>> import pypillometry as pp
        >>> from pypillometry import u
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
        >>> segments = d.get_segments(epochs, "left_pupil")
        >>> # Use pre-event period as baseline
        >>> segments_bc = segments.baseline_correct(window=(-200, 0), inplace=False)
        >>> segments = segments.baseline_correct(window=(-200, 0) * u.ms)

        >>> # Use single time point as baseline
        >>> segments = segments.baseline_correct(window=-100)
        """
        if window is None:
            logger.warning("No baseline correction applied")
            return self
        
        # Create copy of data
        corrected_data = self.data.copy()
        window = parse_time(window)
        self._check_baseline_window(window)

        if np.ndim(window) == 0:
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

        if inplace is None:
            inplace = config.get_default_inplace()
        if inplace:
            self.data = corrected_data
            return self
        return SegmentedEyeData(
            name=self.name,
            variable=self.variable,
            tx=self.tx,
            data=corrected_data,
            epochs=self.epochs,
            sampling_rate=self.sampling_rate
        )
    
    def _check_baseline_window(self, window):
        """Raise a ValueError if the baseline `window` (ms) is not within the segments."""
        step = 1000. / self.sampling_rate if self.sampling_rate else \
            float(np.median(np.diff(self.tx))) if len(self.tx) > 1 else 0.
        # a time point stands for the samples within half a step around it;
        # the end of a window is exclusive
        lo, hi = self.tx[0] - step / 2, self.tx[-1] + step / 2
        seg = f"the segments cover {self.tx[0]:g} to {self.tx[-1]:g} ms"
        if np.ndim(window) == 0:
            if not lo <= window <= hi:
                raise ValueError(f"baseline time {window:g} ms is outside the segments; {seg}")
            return
        if len(window) != 2 or window[0] >= window[1]:
            raise ValueError(f"baseline window must be (start, end) with start < end, "
                             f"got {tuple(window)}")
        if window[0] < lo or window[1] > hi + step / 2:
            raise ValueError(f"baseline window ({window[0]:g}, {window[1]:g}) ms is not "
                             f"within the segments; {seg}")

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
        Event-related pupil dilation after each stimulus. Pass ``ax=`` to overlay
        several segments (e.g., two conditions or eyes) in one plot:

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> d = d | pp.pupil.blinks_detect() | pp.pupil.blinks_interpolate()
           >>> epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
           >>> erpd = d.get_segments(epochs, "left_pupil").baseline_correct((-500, 0))
           >>> ax = erpd.plot()
           >>> # overlay two eyes
           >>> right = d.get_segments(epochs, "right_pupil").baseline_correct((-500, 0))
           >>> fig, ax = plt.subplots()
           >>> ax = erpd.plot(ax=ax, label="left eye")
           >>> ax = right.plot(ax=ax, label="right eye")
           >>> legend = ax.legend()
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
        Save to file (see :func:`~pypillometry.io.write_pickle`).

        The format follows from the extension: HDF5 for ``.h5`` and
        ``.hdf5``, pickle otherwise.

        Parameters
        ----------
        fname : str
            Filename to save to
        compress : bool, optional
            compress the file (default True; pickle: zstd, HDF5: gzip)
        """
        write_pickle(self, fname, compress=compress)
    
    @classmethod
    def from_file(cls, fname: str, **kwargs) -> 'SegmentedEyeData':
        """
        Read a SegmentedEyeData object from a pickle or HDF5 file.
        
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
    epochs : Epochs, optional
        the epochs of the first segments
    sampling_rate : float, optional
        Original sampling rate in Hz
    """
    
    # state (see pypillometry.state): adds to that of SegmentedEyeData
    _STATE_FIELDS = ("mask_percent", "n_subjects")

    def __init__(self,
                 name: str,
                 variable: str,
                 tx: np.ndarray,
                 data: ma.MaskedArray,
                 mask_percent: np.ndarray,
                 n_subjects: int,
                 epochs=None,
                 sampling_rate: float = None):
        # Call parent constructor
        super().__init__(
            name=name,
            variable=variable,
            tx=tx,
            data=data,
            epochs=epochs,
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
        One SegmentedEyeData per subject (here: the example recording twice):

        >>> import pypillometry as pp
        >>> from pypillometry import u
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
        >>> subj1 = d.get_segments(epochs, "left_pupil", name="S01")
        >>> subj2 = d.get_segments(epochs, "left_pupil", name="S02")

        >>> # Aggregate within subjects, then combine
        >>> group = pp.GroupLevelSegmentedData.from_segments([subj1, subj2])
        >>> group.data.shape  # (time points, subjects)
        (1500, 2)

        >>> # Stack all trials from all subjects
        >>> group = pp.GroupLevelSegmentedData.from_segments([subj1, subj2], meanfct=None)
        >>> group.data.shape  # (time points, trials)
        (1500, 20)
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
            epochs=first.epochs,
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
        base_summary['mean_mask_percent'] = float(np.mean(self.mask_percent))
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
    One SegmentedEyeData per subject (here: the example recording twice):

    >>> import pypillometry as pp
    >>> from pypillometry import u
    >>> d = pp.get_example_data("rlmw_002_short")
    >>> epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
    >>> subj1_seg = d.get_segments(epochs, "left_pupil", name="S01")
    >>> subj2_seg = d.get_segments(epochs, "left_pupil", name="S02")

    >>> # From a list
    >>> group = pp.group_segments([subj1_seg, subj2_seg])

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

