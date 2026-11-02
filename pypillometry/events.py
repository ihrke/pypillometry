"""Functions to handle events in eyetracking data.
"""

import numpy as np
from typing import Optional, List, Tuple

from loguru import logger
from pypillometry.convenience import normalize_unit
from pypillometry.units import (UNSET, Quantity, _has_units, _time_values_and_units,
                                parse_time)

__all__ = [
    "Events", "ClockSync", "sync_clocks",
]


class Events:
    """
    Container for events with metadata.
    
    This class wraps arrays of event onsets and labels and provides additional
    functionality such as unit conversion, filtering, and display capabilities.

    Onsets are always times (not sample indices), so they stay valid when
    the data are sliced or resampled.

    The events of a dataset are obtained with
    :meth:`~pypillometry.GenericEyeData.get_events`. An (e.g., filtered)
    Events object can be passed as `event_select` to
    :meth:`~pypillometry.GenericEyeData.get_intervals` or stored in a dataset
    with :meth:`~pypillometry.GenericEyeData.set_events` or added with
    :meth:`~pypillometry.GenericEyeData.add_events`. Events from a
    behaviour log can be created with :meth:`from_table` and mapped to the
    eye-tracker clock with :func:`~pypillometry.sync_clocks`.

    Parameters
    ----------
    onsets : np.ndarray, list or Quantity
        Event onset times (stored as float array); can carry units, e.g.
        ``[100, 500] * u.ms``
    labels : np.ndarray or list
        Event labels (stored as array of strings), same length as `onsets`
    units : str, Unit or None, optional
        Time units of the onsets ("ms", "sec", "min" or "h").
        Default is "ms", or the units of `onsets` if they carry units
        (otherwise, onsets with units are converted to `units`).
    data_time_range : tuple or None, optional
        Time range (min, max) of the original dataset (in same units as onsets)

    Raises
    ------
    ValueError
        If `onsets` and `labels` differ in length, or if the units are not
        time units (e.g., None or samples).

    Examples
    --------
    >>> events = Events(onsets=[100, 500, 1000], labels=["A", "B", "C"], units="ms")
    >>> len(events)
    3
    >>> for onset, label in events:
    ...     print(f"{label} at {onset}")
    """
    
    def __init__(self, 
                 onsets: np.ndarray | List,
                 labels: np.ndarray | List,
                 units: Optional[str] = UNSET,
                 data_time_range: Optional[Tuple[float, float]] = None):
        """
        Initialize an Events object (see class docstring for parameters).
        """
        if units is UNSET and not _has_units(onsets):
            units = "ms"
        onsets, self.units = _time_values_and_units(onsets, units)
        if self.units == "samples":
            raise ValueError("Events need time units (e.g., units='ms'), "
                             "not sample indices")
        self.onsets = np.array(onsets, dtype=float)
        self.labels = np.array(labels, dtype=str)
        
        self.data_time_range = data_time_range
        
        # Validate that onsets and labels have same length
        if len(self.onsets) != len(self.labels):
            raise ValueError(f"onsets and labels must have same length, got {len(self.onsets)} and {len(self.labels)}")
    
    def __len__(self) -> int:
        """Return number of events."""
        return len(self.onsets)
    
    def __iter__(self):
        """Iterate over (onset, label) pairs."""
        return zip(self.onsets, self.labels)
    
    def __getitem__(self, idx):
        """
        Get event(s) by index.
        
        Parameters
        ----------
        idx : int, slice, or array-like
            Index, slice, or boolean/integer array to retrieve
            
        Returns
        -------
        Events
            New Events object with the selected event(s)
        """
        if isinstance(idx, (int, np.integer)):
            # Wrap single index in a list to get 1-element arrays
            return Events(
                onsets=self.onsets[[idx]],
                labels=self.labels[[idx]],
                units=self.units,
                data_time_range=self.data_time_range
            )
        elif isinstance(idx, (slice, np.ndarray, list)):
            return Events(
                onsets=self.onsets[idx],
                labels=self.labels[idx],
                units=self.units,
                data_time_range=self.data_time_range
            )
        else:
            raise TypeError(f"indices must be integers, slices, or array-like, not {type(idx)}")
    
    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert events to numpy arrays.
        
        Returns
        -------
        tuple of np.ndarray
            Tuple of (onsets, labels) as numpy arrays
            
        Examples
        --------
        >>> events = Events([100, 200], ["A", "B"], units="ms")
        >>> onsets, labels = events.to_array()
        """
        return self.onsets.copy(), self.labels.copy()

    @classmethod
    def from_table(cls, table, columns, units: Optional[str] = UNSET) -> 'Events':
        """
        Create Events from a table with one row per trial.

        Behaviour logs usually have one row per trial and one column per
        event time (e.g., stimulus onset, response). Each column becomes
        one kind of event; the column (or dict key) is the label. Missing
        times (NaN, e.g., no response) give no event. The events are
        sorted by onset.

        Parameters
        ----------
        table : pandas.DataFrame or dict
            table with one column of times per kind of event
        columns : list or dict
            list of column names (labels are the column names), or dict
            mapping labels to column names or to arrays of times (e.g.,
            ``{"response": df.stim_on + df.rt}``)
        units : str, Unit or None, optional
            time units of the values ("ms", "sec", "min" or "h");
            default "ms"

        Returns
        -------
        Events
            all events of all columns, sorted by onset

        Raises
        ------
        ValueError
            if the columns have different lengths

        Examples
        --------
        >>> table = {"stim": [1.0, 3.0], "rt": [0.5, np.nan]}
        >>> ev = Events.from_table(table, {"stim": "stim",
        ...                                "response": np.add(table["stim"], table["rt"])},
        ...                        units="sec")
        >>> ev.labels
        array(['stim', 'response', 'stim'], dtype='<U8')
        """
        if not isinstance(columns, dict):
            columns = {name: name for name in columns}
        onsets, labels = [], []
        nrows = None
        for label, col in columns.items():
            times = table[col] if isinstance(col, str) else col
            times = np.asarray(times, dtype=float)
            if nrows is not None and len(times) != nrows:
                raise ValueError(f"column for '{label}' has {len(times)} "
                                 f"values, expected {nrows}")
            nrows = len(times)
            times = times[~np.isnan(times)]
            onsets.append(times)
            labels.append(np.full(len(times), label, dtype=object))
        onsets = np.concatenate(onsets) if onsets else np.array([])
        labels = np.concatenate(labels).astype(str) if labels else np.array([], dtype=str)
        # stable sort, so events at the same time keep the column order
        order = np.argsort(onsets, kind="stable")
        return cls(onsets[order], labels[order], units=units)

    def to_units(self, target_units: Optional[str]) -> 'Events':
        """
        Convert event onsets to different time units.
        
        Parameters
        ----------
        target_units : str
            Target units: "ms", "sec", "min" or "h"
            
        Returns
        -------
        Events
            New Events object with converted units
            
        Raises
        ------
        ValueError
            If `target_units` is not a time unit
        
        Examples
        --------
        >>> events = Events([1000, 2000], ["A", "B"], units="ms")
        >>> events_sec = events.to_units("sec")
        >>> events_sec.onsets
        array([1., 2.])
        """
        # Normalize target units (source units already normalized in __init__)
        target_units = normalize_unit(target_units)
        
        if target_units == "samples":
            raise ValueError("Events need time units (e.g., 'ms'), not sample "
                             "indices; use the data object to find samples")
        
        if self.units == target_units:
            # No conversion needed, return self
            return self
        
        # Define conversion factors (everything relative to ms)
        units_to_ms = {
            "ms": 1.0,
            "sec": 1000.0,
            "min": 60000.0,
            "h": 3600000.0
        }
        
        if self.units not in units_to_ms:
            raise ValueError(f"Unknown source units: {self.units}")
        if target_units not in units_to_ms:
            raise ValueError(f"Unknown target units: {target_units}")
        
        # Convert: first to ms, then to target units
        fac = units_to_ms[self.units] / units_to_ms[target_units]
        converted_onsets = self.onsets * fac
        
        # Convert data_time_range if present
        if self.data_time_range is not None:
            converted_range = (self.data_time_range[0] * fac, self.data_time_range[1] * fac)
        else:
            converted_range = None
        
        return Events(
            onsets=converted_onsets,
            labels=self.labels.copy(),
            units=target_units,
            data_time_range=converted_range
        )
    
    def __repr__(self) -> str:
        """
        String representation of Events.
        
        Returns
        -------
        str
            Formatted string with number of events and units
        """
        parts = []
        
        # Add basic info
        if len(self) == 0:
            parts.append("Events: 0 events")
        elif len(self) == 1:
            parts.append("Events: 1 event")
        else:
            parts.append(f"Events: {len(self)} events")
        
        parts.append(f"units={self.units}")
        
        # Add time range if available
        if self.data_time_range is not None:
            range_str = f"[{self.data_time_range[0]:.1f}, {self.data_time_range[1]:.1f}]"
            parts.append(f"range={range_str}")
        
        return " | ".join(parts)
    
    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebooks.
        
        Displays events as a nicely formatted HTML table with columns for
        index, onset time, and event label.
        
        Returns
        -------
        str
            HTML string for displaying in Jupyter notebook
        """
        try:
            import pandas as pd
        except ImportError:
            # Fallback to text representation if pandas not available
            return f"<pre>{repr(self)}</pre>"
        
        if len(self) == 0:
            return f"<p><i>Events: 0 events (units={self.units})</i></p>"
        
        # Determine unit display
        unit_str = self.units
        
        # Create DataFrame
        df = pd.DataFrame({
            f'Onset ({unit_str})': self.onsets,
            'Label': self.labels
        })

        # Convert to HTML respecting current pandas display options
        import inspect

        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")

        to_html_kwargs = {
            "index": True,
            "index_names": ['#'],
        }

        to_html_signature = inspect.signature(pd.DataFrame.to_html)
        if "max_rows" in to_html_signature.parameters and max_rows is not None:
            to_html_kwargs["max_rows"] = max_rows
        if "min_rows" in to_html_signature.parameters and min_rows is not None:
            to_html_kwargs["min_rows"] = min_rows

        html = df.to_html(**to_html_kwargs)
        
        # Add summary info below table
        summary = f'<p style="font-size: 0.9em; color: #666; margin-top: 8px;">'
        summary += f'Total: {len(self)} events'
        if self.data_time_range is not None:
            summary += f' | Data range: [{self.data_time_range[0]:.1f}, {self.data_time_range[1]:.1f}] {unit_str}'
        summary += '</p>'
        
        return html + summary

    def plot(
        self,
        show_labels: str = "auto",
        marker: str = ".",
        units: Optional[str] = "min",
        **kwargs,
    ):
        """
        Plot events as points on a time axis.

        Parameters
        ----------
        show_labels : {"auto", "all", "none"}, optional
            Labeling strategy. "auto" (default) shows a subset spaced apart,
            "all" annotates every event, "none" disables labels.
        marker : str, optional
            Matplotlib marker style for the events (default '.').
        units : str or None, optional
            Units to display in the plot. Defaults to "min". Use None to keep existing units.
        **kwargs :
            Additional keyword arguments passed to :func:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        None
            Plot is drawn on the current matplotlib axes.

        Examples
        --------
        The stimulus events of the example dataset:

        .. plot::

           >>> import pypillometry as pp
           >>> from pypillometry import u
           >>> import matplotlib.pyplot as plt
           >>> d = pp.get_example_data("rlmw_002_short")
           >>> plt.figure(figsize=(10, 3))
           >>> d.get_events().filter("S_").plot(units="sec")
        """
        import matplotlib.pyplot as plt

        if show_labels not in {"auto", "all", "none"}:
            raise ValueError("show_labels must be one of {'auto', 'all', 'none'}")

        ax = plt.gca()

        target_units = units if units is not None else self.units
        if target_units != self.units:
            converted = self.to_units(target_units)
        else:
            converted = self

        # Always label axes even if no events so the plot remains informative.
        x_label = f"Time ({converted.units})"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Event #")

        if len(self) == 0:
            ax.set_ylim(0, 1)
            return

        y_positions = np.arange(1, len(self) + 1)

        plot_kwargs = {"linestyle": "None", "marker": marker, "markersize": 3}
        plot_kwargs.update(kwargs)

        ax.plot(converted.onsets, y_positions, **plot_kwargs)

        if converted.data_time_range is not None:
            x_min, x_max = converted.data_time_range
        else:
            x_min = float(np.min(converted.onsets))
            x_max = float(np.max(converted.onsets))

        span = x_max - x_min
        if span == 0:
            padding = max(1.0, abs(x_max) * 0.05)
        else:
            padding = span * 0.05

        x_limits = (x_min - padding, x_max + padding)
        ax.set_xlim(*x_limits)
        ax.set_ylim(0, len(self) + 1)
        ax.set_yticks([1, len(self)] if len(self) > 1 else [1])

        if show_labels != "none":
            effective_spacing = 0.0
            if show_labels == "auto":
                logger.debug("Plotting only a subset of labels, use show_labels='all' to show all labels")
                fig = ax.figure
                try:
                    fig.canvas.draw()
                except Exception:
                    pass
                bbox = ax.get_window_extent()
                axis_width_pixels = bbox.width if bbox.width > 0 else 1.0
                desired_pixels = 10.0
                span = x_limits[1] - x_limits[0]
                effective_spacing = span * desired_pixels / axis_width_pixels if axis_width_pixels else 0.0

            label_offset = padding * 0.05 if padding > 0 else 0.0
            last_label_x = -np.inf
            for onset, y_pos, label in zip(converted.onsets, y_positions, self.labels):
                spacing_requirement = (
                    show_labels == "all"
                    or effective_spacing <= 0
                    or abs(onset - last_label_x) >= effective_spacing
                )
                if spacing_requirement:
                    ax.text(
                        onset + label_offset,
                        y_pos,
                        str(label),
                        va="center",
                        ha="left",
                        fontsize=8,
                        alpha=0.8,
                        rotation=90,
                        rotation_mode="anchor",
                    )
                    last_label_x = onset
        return
    
    def filter(self, selector, exact: bool = False) -> 'Events':
        """
        Filter events and return a new Events object.
        
        This method supports multiple types of selectors for flexible filtering:
        - String: matches labels containing the substring (or equal to
          it, if `exact` is True)
        - Callable: applies function to each label, keeps events where it returns True
        - Tuple: filters by time range (min_time, max_time)
        
        Parameters
        ----------
        selector : str, callable, or tuple
            Filtering criterion:

            - str: substring to search for in labels
            - callable: function that takes a label and returns bool
            - tuple: (min_time, max_time) to filter by onset time range
              (inclusive, in the units of the Events object unless it
              carries units, e.g. ``(10, 30) * u.sec``)
        exact : bool, optional
            if True, a string selector matches only labels that are equal
            to it; if False (default), labels that contain it

        Returns
        -------
        Events
            New Events object containing only the filtered events
            
        Examples
        --------
        >>> events = Events([100, 500, 1000], ["stim", "resp", "stim"], units="ms")
        >>> stim_events = events.filter("stim")
        >>> len(stim_events)
        2
        
        >>> events.filter("stim", exact=True).onsets  # label equal to "stim"
        array([ 100., 1000.])

        >>> # Filter by custom function
        >>> events = Events([100, 200], ["A1", "B2"], units="ms")
        >>> numeric = events.filter(lambda label: label[-1].isdigit())
        
        >>> # Filter by time range
        >>> events = Events([100, 500, 1000], ["A", "B", "C"], units="ms")
        >>> middle = events.filter((200, 800))
        >>> len(middle)
        1
        """
        if isinstance(selector, str):
            # String selector: substring matching
            mask = np.array([label == selector if exact else selector in label
                             for label in self.labels], dtype=bool)
        
        elif callable(selector):
            # Function selector: apply to each label
            mask = np.array([bool(selector(label)) for label in self.labels], dtype=bool)
        
        elif isinstance(selector, (tuple, Quantity)):
            # Time range selector
            if len(selector) != 2:
                raise ValueError("Time range selector must be a tuple of (min_time, max_time)")
            if _has_units(selector):
                selector = parse_time(selector, to=self.units, default=None)

            min_time, max_time = selector
            if min_time >= max_time:
                raise ValueError(f"min_time must be < max_time, got {min_time} >= {max_time}")
            
            mask = np.logical_and(self.onsets >= min_time, self.onsets <= max_time)
        
        else:
            raise TypeError(
                f"selector must be str, callable, or tuple, got {type(selector)}"
            )
        
        # Create new Events with filtered data
        return Events(
            onsets=self.onsets[mask],
            labels=self.labels[mask],
            units=self.units,
            data_time_range=self.data_time_range
        )



class ClockSync:
    """
    Linear map between two clocks, fitted by :func:`sync_clocks`.

    Maps times `t` on the source clock (e.g., a behaviour log) to the target
    clock (e.g., the eye tracker): ``offset + slope * t``. The slope accounts
    for the unit conversion and for the drift between the clocks.

    Calling the object maps times: plain numbers are taken to be in
    `src_units` and are returned as an array in `dst_units`; numbers with
    units and :class:`Events` are converted first and returned in
    `dst_units` (as :class:`Events` for :class:`Events`).

    Attributes
    ----------
    offset : float
        target time (in `dst_units`) of source time 0
    slope : float
        target units per source unit
    drift : float
        how much faster the target clock runs (e.g., 1e-5 is 36 ms per
        hour); 0 if only the offset was fitted
    residuals : np.ndarray
        target times minus mapped source times (in `dst_units`); they are
        small (around the timing precision) if the pairs are correct
    src, dst : np.ndarray
        the paired times (in `src_units` and `dst_units`)
    src_units, dst_units : str
        units of the source and target times

    Examples
    --------
    >>> sync = sync_clocks([10, 70, 130], [5000, 65000.6, 125001.2],
    ...                    src_units="sec", dst_units="ms")
    >>> sync(100.0)
    95000.9
    """

    def __init__(self, offset, slope, src, dst, src_units, dst_units):
        self.offset = float(offset)
        self.slope = float(slope)
        self.src = np.asarray(src, dtype=float)
        self.dst = np.asarray(dst, dtype=float)
        self.src_units = src_units
        self.dst_units = dst_units
        self.residuals = self.dst - (self.offset + self.slope * self.src)

    @property
    def drift(self) -> float:
        """Relative drift of the target clock against the source clock."""
        return self.slope / parse_time(1, to=self.dst_units,
                                       default=self.src_units) - 1

    def __call__(self, times):
        """Map times from the source clock to the target clock."""
        if isinstance(times, Events):
            ev = times.to_units(self.src_units)
            return Events(self(ev.onsets), ev.labels.copy(),
                          units=self.dst_units)
        if _has_units(times):
            times = parse_time(times, to=self.src_units, default=None)
        mapped = self.offset + self.slope * np.asarray(times, dtype=float)
        return float(mapped) if mapped.ndim == 0 else mapped

    def summary(self) -> str:
        """One-line summary of the fit."""
        drift = self.drift * parse_time(1, to="ms", default="h")
        return (f"offset={self.offset:.6g} {self.dst_units}, "
                f"drift={drift:+.3g} ms/h, "
                f"max |residual|={np.max(np.abs(self.residuals)):.3g} "
                f"{self.dst_units} (n={len(self.src)})")

    def __repr__(self) -> str:
        lines = [f"ClockSync({self.src_units} -> {self.dst_units}): {self.summary()}",
                 f"  {'src':>12} {'dst':>14} {'residual':>10}"]
        rows = [f"  {s:12.6g} {d:14.8g} {r:10.3g}"
                for s, d, r in zip(self.src, self.dst, self.residuals)]
        if len(rows) > 10:  # first and last pairs only
            rows = rows[:5] + [f"  ... ({len(rows) - 10} more)"] + rows[-5:]
        return "\n".join(lines + rows)


def sync_clocks(src, dst, src_units=UNSET, dst_units=UNSET,
                slope: bool = True, tol=5) -> ClockSync:
    """
    Fit a linear map from one clock to another from paired times.

    Behaviour logs (e.g., from Psychtoolbox, PsychoPy or Presentation)
    and eye trackers use different clocks. They differ in their zero point
    and drift apart slowly, so an offset alone can be off by several ms
    within a session. Given the times of the same moments on both clocks
    (e.g., the start of each block), this fits
    ``dst = offset + slope * src`` by least squares.

    The pairs are matched by order. A summary of the fit is logged; the
    residuals show whether the pairing is right (large residuals mean that
    a pair is shifted or missing).

    Parameters
    ----------
    src : array-like, Quantity or Events
        times on the source clock (e.g., from the behaviour log)
    dst : array-like, Quantity or Events
        times of the same moments on the target clock, in the same order
        (e.g., ``d.get_events().filter("Start Block")``)
    src_units, dst_units : str or Unit, optional
        units of plain numbers in `src` and `dst` (e.g., "sec", "ms");
        needed unless the times carry units or are :class:`Events`
    slope : bool, optional
        if True (default), fit offset and slope (drift); if False, only
        the offset. With a single pair, only the offset is fitted.
    tol : float, str or Quantity, optional
        a warning is given if a residual is larger than this (plain
        numbers in ms); default 5 ms

    Returns
    -------
    ClockSync
        the fitted map; call it on times or :class:`Events` to map them
        to the target clock

    Raises
    ------
    ValueError
        if `src` and `dst` differ in length, are empty or contain NaN,
        or if units are missing

    Examples
    --------
    >>> starts = behav.groupby("block").trialstart.min()     # in sec
    >>> sync = pp.sync_clocks(starts, d.get_events().filter("Start Block"),
    ...                       src_units="sec")
    >>> d.add_events(sync(pp.Events.from_table(behav, ["stim_on"], units="sec")))
    """
    def values(times, units, name):
        if isinstance(times, Events):
            return times.onsets, times.units
        if not _has_units(times) and units is UNSET:
            raise ValueError(f"{name} needs units, e.g. {name}_units='sec'")
        times, units = _time_values_and_units(times, units)
        if units == "samples":
            raise ValueError(f"{name} must be times, not sample indices")
        return np.asarray(times, dtype=float), units

    src, src_units = values(src, src_units, "src")
    dst, dst_units = values(dst, dst_units, "dst")
    if len(src) != len(dst):
        raise ValueError(f"src has {len(src)} times and dst has {len(dst)}; "
                         "they must be paired one-to-one")
    if len(src) == 0:
        raise ValueError("need at least one pair of times")
    if np.isnan(src).any() or np.isnan(dst).any():
        raise ValueError("src and dst must not contain NaN")

    # slope without drift (unit conversion)
    unit_slope = parse_time(1, to=dst_units, default=src_units)
    if slope and len(src) == 1:
        logger.warning("sync_clocks(): one pair of times only gives the "
                       "offset; drift is not corrected")
        slope = False
    if slope:
        b, a = np.polyfit(src, dst, 1)
    else:
        b = unit_slope
        a = np.mean(dst - b * src)
    sync = ClockSync(a, b, src, dst, src_units, dst_units)

    logger.info(f"sync_clocks(): {sync.summary()}")
    if slope and len(src) == 2:
        logger.warning("sync_clocks(): two pairs fit exactly; the residuals "
                       "cannot check the pairing")
    tol = parse_time(tol, to=dst_units, default="ms")
    big = np.abs(sync.residuals) > tol
    if big.any():
        pairs = np.flatnonzero(big).tolist()
        pairs = str(pairs) if len(pairs) <= 5 else f"{str(pairs[:5])[:-1]}, ...]"
        logger.warning(f"sync_clocks(): {big.sum()} of {len(big)} residuals larger "
                       f"than {tol:.3g} {dst_units} (pairs {pairs}); "
                       "are the times paired correctly?")
    return sync
