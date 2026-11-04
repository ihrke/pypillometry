"""
pupil.py
========

Methods of :class:`~pypillometry.EyeData` for pupil data.
"""

import itertools
from typing import Optional, Tuple, Union
from .generic import keephistory
from .namespace import Namespace
from .pipes import step
#from .. import convenience
from ..signal import baseline
from ..signal import preproc
from ..signal import pupil
from ..signal.pupil import _bridge_gaps
#from .. import io
from ..intervals import Intervals
from ..units import parse_distance, parse_angle
from .foreshortening_calibration import (
    ForeshorteningCalibration,
    _determine_knots,
    _create_bspline_basis,
    _objective_function,
    _objective_gradient,
    _split_params
)
from ..signal.baseline import butter_lowpass_filter, downsample

import json
from loguru import logger
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate
import scipy
from scipy import optimize


def _merge_overlapping_blink_windows(blinks, n):
    """Merge blinks whose interpolation windows overlap.

    Each blink `(t2, t3)` is interpolated from the anchor points
    ``t1 = t2-(t3-t2)`` and ``t4 = t3+(t3-t2)``. Neighbouring blinks are
    merged if the window ``[t1, t4]`` of one reaches into the other.

    Windows are computed from the original blinks, not from the merged
    ones (a merged blink is longer, so its window would be wider and
    would reach further blinks, merging the whole recording).

    Returns
    -------
    list of tuple
        ``(t1, t2, t3, t4)`` for each merged blink: `t2`/`t3` are the
        on-/offset of the merged blink, `t1`/`t4` the anchor points (from
        the windows of the first/last original blink in the group)
    """
    def window(on, off):
        return max(0, 2*on-off), min(2*off-on, n-1)

    out = []
    for on, off in sorted((int(b[0]), int(b[1])) for b in blinks):
        t1, t4 = window(on, off)
        if out:
            pt1, pon, poff, pt4 = out[-1]
            if pt4 >= on or t1 <= poff:
                out[-1] = (pt1, pon, max(poff, off), max(pt4, t4))
                continue
        out.append((t1, on, off, t4))
    return out

class PupilNamespace(Namespace):
    """
    Methods for pupil data, accessed as ``d.pupil`` (e.g.,
    ``d.pupil.blinks_detect()``).

    The methods need pupil data and raise a ValueError if there is none.
    The transforming methods return the data object (see
    :class:`~pypillometry.eyedata.namespace.Namespace`).
    """

    _name = "pupil"

    @step
    @keephistory
    def lowpass_filter(self,  cutoff: float, order: int=2, eyes=[], policy="default",
                             inplace=None):
        """
        Lowpass-filter pupil signal using a Butterworth-filter.
        
        Only the usable samples under the mask `policy` are filtered
        (by default, interpolated samples are used; see
        :meth:`~pypillometry.GenericEyeData.mask`). The unusable samples
        (masked or NaN) are bridged linearly for the filter
        (see :func:`~pypillometry.signal.pupil.lowpass_filter_iterative`),
        so their stored values (e.g., zeros in blinks) do not enter the
        filter; they keep their values. The filtered samples replace the
        pupil data; the mask layers are kept.
    
        Parameters
        ----------
        cutoff : float
            lowpass-filter cutoff frequency in Hz
        order : int
            filter order
        eyes : list
            list of eyes to filter; if empty, all available eyes are filtered
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are usable (see
            :meth:`~pypillometry.GenericEyeData.mask`); default ``"default"``
        inplace : bool or None
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes           
            if `None`, use the object-level setting         

        Returns
        -------
        EyeData
            the object with the filtered pupil data
        """
        from ..signal.pupil import lowpass_filter_iterative
        
        self._d._require("pupil")
        obj = self._d._get_inplace(inplace)
        eyes=self._d._eyes_with(eyes, "pupil")

        for eye in eyes:
            x = obj.values(eye, "pupil", policy)  # NaN: unusable
            usable = ~np.isnan(x)
            out = obj.data[eye, "pupil"].copy()
            out[usable] = lowpass_filter_iterative(x, cutoff, obj.fs, order)[usable]
            obj.data.set_with_mask(f"{eye}_pupil", out, preserve_mask=True)
        return obj

    @step
    @keephistory
    def smooth_window(self, eyes=[], window: str="hanning", winsize: float=11,
                            policy="default", inplace=None):
        """
        Apply smoothing of the signal using a moving window.
        See :func:`~pypillometry.signal.preproc.smooth_window`.

        Only the usable samples under the mask `policy` are smoothed; the
        unusable samples (masked or NaN) are bridged linearly for the
        smoothing and keep their stored values (as in
        :meth:`lowpass_filter`). The mask layers are kept.
        
        Parameters
        ----------
        eyes : list
            list of eyes to smooth; if empty, all available eyes are smoothed
        window : str
            (the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'); 
            flat window will produce a moving average smoothing.
        winsize : float, str or Quantity
            the length of the window in ms (or with units, e.g. ``"20 ms"``)
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are usable (see
            :meth:`~pypillometry.GenericEyeData.mask`); default ``"default"``
        inplace : bool or None
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                            
            if `None`, use the object-level setting

        Returns
        -------
        EyeData
            the object with the smoothed pupil data
        """
        self._d._require("pupil")
        obj = self._d._get_inplace(inplace)
        eyes=self._d._eyes_with(eyes, "pupil")

        # convert winsize to index based on sampling rate
        winsize_ix=self._d._samples(winsize, rounding="up")

        # process requested eyes
        for eye in eyes:
            x = obj.values(eye, "pupil", policy)  # NaN: unusable
            usable = ~np.isnan(x)
            if not usable.any():
                continue
            out = obj.data[eye, "pupil"].copy()
            smoothed = preproc.smooth_window(_bridge_gaps(x), winsize_ix, window)
            out[usable] = smoothed[usable]
            obj.data.set_with_mask(f"{eye}_pupil", out, preserve_mask=True)

        return obj

    @step
    @keephistory
    def merge_eyes(self, method: str = "local_offset", 
                         lowpass_cutoff: float = 0.2,
                         max_iter: int = 10,
                         keep_eyes: bool = True, 
                         policy="default",
                         name: str = "mean",
                         inplace: bool | None = None):
        """
        Merge pupil data from both eyes into a new eye (default ``"mean"``).
        
        This method combines left and right pupil signals, handling missing data
        (masked values) intelligently. Where both eyes are valid, the signals are
        averaged to reduce noise. Where only one eye is valid, that eye's data is
        used (with offset correction if applicable). Valid samples are the
        usable samples under the mask `policy` (by default, interpolated
        samples are used, see :meth:`~pypillometry.GenericEyeData.mask`).
        
        Parameters
        ----------
        method : str, optional
            Method for merging the two eyes. Options:
            
            - "local_offset" (default): Compute a slowly-varying offset between eyes
              using lowpass filtering. This handles drift over the session. The right
              eye is shifted to match the left eye's level before merging.
            - "offset": Use a single global offset (mean difference) computed over
              the entire signal where both eyes are valid.
            - "both_available": Only use data where both eyes are valid (take mean).
              Timepoints where only one eye is valid remain masked.
            
            # TODO: Implement "signal_quality" method that weights eyes by their
            # local signal quality (SNR), giving more weight to the cleaner eye.
            
        lowpass_cutoff : float, optional
            Cutoff frequency (Hz) for the lowpass filter used in "local_offset" method.
            Lower values produce a more stable offset that only tracks very slow drift.
            Default is 0.2 Hz (~5 second time constant).
        max_iter : int, optional
            Maximum iterations for iterative lowpass filtering in "local_offset" method.
            Default is 10.
        keep_eyes : bool, optional
            If True (default), keep the original left/right pupil data.
            If False, remove the original left/right pupil data after merging
            (gaze data are kept).
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are valid (see
            :meth:`~pypillometry.GenericEyeData.mask`); default ``"default"``
        name : str, optional
            name of the new eye (default ``"mean"``, as in
            :meth:`~pypillometry.GenericEyeData.merge_eyes`); the merged
            pupil is stored as ``"<name>_pupil"``
        inplace : bool or None, optional
            If True, modify in-place. If False, return a copy.
            If None, use the object's default setting.
            
        Returns
        -------
        EyeData
            The object with merged pupil data stored as "mean_pupil".
            
        Raises
        ------
        ValueError
            If only one eye is available (nothing to merge).
            If an unknown method is specified.
            
        Notes
        -----
        The merged signal is stored as "mean_pupil" (with the default `name`)
        and can be accessed via `data["mean", "pupil"]` or `data["mean_pupil"]`.
        
        The mask for the merged signal reflects which timepoints have valid data:
        - For "local_offset" and "offset": masked only where BOTH eyes are masked
        - For "both_available": masked where EITHER eye is masked
        Masked samples still hold the merged stored values (e.g., interpolated
        blinks), computed as for valid samples (NaN only if both eyes are NaN).
        
        The "local_offset" method is recommended for most use cases as it:
        1. Uses all available data (doesn't discard single-eye timepoints)
        2. Handles session-long drift in the inter-ocular difference
        3. Produces smooth transitions at boundaries between single/both-eye regions
        
        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short")
        >>> # Default: local offset correction
        >>> d1 = d.pupil.merge_eyes(inplace=False)
        >>> merged = d1["mean", "pupil"]

        >>> # Use global offset (simpler, assumes no drift)
        >>> d2 = d.pupil.merge_eyes(method="offset", inplace=False)

        >>> # Only use timepoints where both eyes are valid
        >>> d3 = d.pupil.merge_eyes(method="both_available", inplace=False)

        >>> # Remove the original pupil data after merging (gaze is kept)
        >>> d4 = d.pupil.merge_eyes(keep_eyes=False, inplace=False)
        >>> list(d4.data.keys())
        ['left_x', 'left_y', 'right_x', 'right_y', 'mean_pupil']
        
        See Also
        --------
        merge_eyes : Generic merge method in GenericEyeData (masks if all eyes invalid)
        pypillometry.signal.pupil.pupil_eye_offset : Compute local offset between eyes
        """
        self._d._require("pupil")
        obj = self._d._get_inplace(inplace)
        
        # Check that we have both eyes
        available_eyes = obj.data.get_available_eyes(variable="pupil")
        if len(available_eyes) < 2:
            raise ValueError(f"Need both eyes to merge, but only have: {available_eyes}")
        
        if "left" not in available_eyes or "right" not in available_eyes:
            raise ValueError(f"Need 'left' and 'right' eyes, but have: {available_eyes}")
        
        valid_methods = ["local_offset", "offset", "both_available"]
        if method not in valid_methods:
            raise ValueError(f"Unknown method '{method}'. Must be one of: {valid_methods}")

        # get data and mask from object
        left_data = obj.data["left_pupil"]
        right_data = obj.data["right_pupil"]
        left_mask = obj.mask("left", "pupil", policy)
        right_mask = obj.mask("right", "pupil", policy)
        
        # Determine valid regions
        both_valid = ~left_mask & ~right_mask
        only_left = ~left_mask & right_mask
        only_right = left_mask & ~right_mask
        neither_valid = left_mask & right_mask
        
        n_both = np.sum(both_valid)
        n_left_only = np.sum(only_left)
        n_right_only = np.sum(only_right)
        n_neither = np.sum(neither_valid)
        
        logger.debug(f"Merge regions: both={n_both}, left_only={n_left_only}, "
                    f"right_only={n_right_only}, neither={n_neither}")
        
        # Initialize merged array
        merged = np.full(len(left_data), np.nan)
        merged_mask = np.ones(len(left_data), dtype=int)
        
        if method == "both_available":
            # Only use timepoints where both eyes are valid
            merged[both_valid] = (left_data[both_valid] + right_data[both_valid]) / 2
            merged_mask[both_valid] = 0
            
        elif method == "offset":
            # Compute global offset where both valid
            if n_both == 0:
                logger.warning("No timepoints with both eyes valid. Using zero offset.")
                offset = 0.0
            else:
                offset = np.mean(left_data[both_valid] - right_data[both_valid])
            
            logger.debug(f"Global offset (left - right): {offset:.2f}")
            
            # Merge: use true midpoint where both valid
            # Where only one eye: shift it to midpoint level (half the offset)
            merged[both_valid] = (left_data[both_valid] + right_data[both_valid]) / 2
            merged[only_left] = left_data[only_left] - offset / 2  # shift down to midpoint
            merged[only_right] = right_data[only_right] + offset / 2  # shift up to midpoint
            merged_mask[~neither_valid] = 0
            
        elif method == "local_offset":
            # Compute local (time-varying) offset
            offset = pupil.pupil_eye_offset(
                left_data, right_data, obj.fs,
                left_mask=left_mask, right_mask=right_mask,
                lowpass_cutoff=lowpass_cutoff, max_iter=max_iter
            )
            
            # Merge: use true midpoint where both valid
            # Where only one eye: shift it to midpoint level (half the offset)
            merged[both_valid] = (left_data[both_valid] + right_data[both_valid]) / 2
            merged[only_left] = left_data[only_left] - offset[only_left] / 2  # shift down to midpoint
            merged[only_right] = right_data[only_right] + offset[only_right] / 2  # shift up to midpoint
            merged_mask[~neither_valid] = 0
        
        # Where the result is masked, keep the merged stored values (e.g.,
        # interpolated blinks) instead of NaN
        offset = 0.0 if method == "both_available" else offset
        left_shifted = left_data - np.broadcast_to(offset, left_data.shape) / 2
        right_shifted = right_data + np.broadcast_to(offset, right_data.shape) / 2
        lfin, rfin = np.isfinite(left_data), np.isfinite(right_data)
        fill = np.where(lfin & rfin, (left_data + right_data) / 2,
                        np.where(lfin, left_shifted, right_shifted))
        masked = merged_mask.astype(bool)
        merged[masked] = fill[masked]
        
        # Store merged data
        obj.data.set_with_mask((name, "pupil"), merged, mask=merged_mask)
        
        # Optionally remove original eye data
        if not keep_eyes:
            del obj.data["left_pupil"]
            del obj.data["right_pupil"]
            logger.debug("Removed original left/right pupil data")
        
        return obj

    def _convert_velocity_thresholds(self, pupil_data, winsize_ix, vel_onset, vel_offset):
        """
        Convert velocity thresholds from per-ms or percentile to per-sample units.
        
        Parameters
        ----------
        pupil_data : np.array
            pupil data array (used for percentile calculation)
        winsize_ix : int
            window size in samples for velocity calculation
        vel_onset : float or str
            onset threshold (per-ms float or percentile string like "5%")
        vel_offset : float or str
            offset threshold (per-ms float or percentile string like "95%")
            
        Returns
        -------
        tuple (float, float)
            (vel_onset_samples, vel_offset_samples) thresholds in per-sample units
        """
        # Helper to parse percentile string
        def parse_percentile(val):
            if isinstance(val, str) and val.endswith('%'):
                return float(val[:-1])
            return None
        
        onset_pct = parse_percentile(vel_onset)
        offset_pct = parse_percentile(vel_offset)
        
        # If either threshold is a percentile, we need to compute velocity profile
        if onset_pct is not None or offset_pct is not None:
            # Fill NaN/zero values with interpolation for velocity calculation
            invalid_mask = np.isnan(pupil_data) | (pupil_data == 0)
            pupil_filled = pupil_data.copy()
            valid_indices = np.where(~invalid_mask)[0]
            invalid_indices = np.where(invalid_mask)[0]
            if len(invalid_indices) > 0 and len(valid_indices) > 0:
                pupil_filled[invalid_indices] = np.interp(
                    invalid_indices, valid_indices, pupil_data[valid_indices]
                )
            
            # Compute velocity profile
            vel = preproc.velocity_savgol(pupil_filled, winsize_ix, polyorder=2, direction="center")
            # Mask out invalid regions for percentile calculation
            vel_valid = vel[~invalid_mask]
        
        # Convert onset threshold
        if onset_pct is not None:
            vel_onset_samples = np.percentile(vel_valid, onset_pct)
            logger.debug(f"vel_onset {vel_onset} -> {vel_onset_samples:.4f} (percentile)")
        else:
            # Convert from per-ms to per-sample
            vel_onset_samples = float(vel_onset) * (1000 / self._d.fs)
            logger.debug(f"vel_onset {vel_onset}/ms -> {vel_onset_samples:.4f}/sample")
        
        # Convert offset threshold
        if offset_pct is not None:
            vel_offset_samples = np.percentile(vel_valid, offset_pct)
            logger.debug(f"vel_offset {vel_offset} -> {vel_offset_samples:.4f} (percentile)")
        else:
            # Convert from per-ms to per-sample
            vel_offset_samples = float(vel_offset) * (1000 / self._d.fs)
            logger.debug(f"vel_offset {vel_offset}/ms -> {vel_offset_samples:.4f}/sample")
        
        return vel_onset_samples, vel_offset_samples

    @step
    @keephistory
    def blinks_detect(self, eyes=[], min_duration:float=20, blink_val:float=0,
                      winsize: float=11, vel_onset=-5.0, vel_offset=5.0, 
                      vel_onset_min_duration: float=5.0, vel_offset_min_duration: float=5.0,
                      max_duration: float=1000,
                      strategies: list=["zero","velocity"],
                      ignore_existing_mask=False, inplace=None):
        """
        Detect blinks in the pupillary signal using several strategies.
        First, blinks are detected as consecutive sequence of `blink_val` 
        (f.eks., 0 or NaN). Second, blinks are defined as everything between
        two crossings of the velocity profile (from negative to positive).
        
        Detected blinks are stored in the object, as the ``"blink"``
        layer of the pupil (retrieve them with
        :meth:`~pypillometry.GenericEyeData.get_blinks`), and are masked
        under the mask policies (see
        :meth:`~pypillometry.GenericEyeData.mask`). They replace the
        blinks detected before (and their mask), and the blinks detected
        before are not treated as masked during detection, so re-running the
        detection with other parameters gives the same result as running it
        once (``clear_layer("blink")`` removes them).

        Finally, detected blinks have to be at least `min_duration` ms.
        
        Parameters
        ----------
        eyes : list
            list of eyes to process; if empty, all available eyes are processed
        min_duration : float, str or Quantity
            minimum duration (in ms, or with units) for a sequence of missing numbers to be treated as blink
        blink_val : float
            "missing value" code for the "zero" strategy. Missing samples
            (NaN) are always detected; samples equal to `blink_val` (default
            0) are detected in addition. Use `np.nan` to detect only NaN
            (e.g., for baseline-corrected data where 0 is a valid value).
            The "zero" strategy always uses the raw data, so regions masked
            for other reasons are not detected as blinks.
        winsize : float, str or Quantity
            window-size for Savitzky-Golay velocity estimation (in ms, or with units)
        vel_onset : float or str
            velocity threshold to detect blink onset (negative value).

            - If float: threshold in pupil-units per millisecond (e.g., -5.0)
            - If str: percentile of velocity distribution (e.g., "5%" for bottom 5%)
        vel_offset : float or str
            velocity threshold to detect blink offset (positive value).

            - If float: threshold in pupil-units per millisecond (e.g., 5.0)
            - If str: percentile of velocity distribution (e.g., "95%" for top 5%)
        vel_onset_min_duration : float, str or Quantity
            minimum duration (in ms, or with units) that velocity must exceed threshold to detect as onset
            (to avoid noise-induced false detections)
        vel_offset_min_duration : float, str or Quantity
            minimum duration (in ms, or with units) that velocity must exceed threshold to detect as offset
            (to avoid noise-induced false detections)
        max_duration : float, str, Quantity or None
            maximum duration (in ms, or with units) of a blink detected by the
            "velocity" strategy; an onset without an offset within this time
            is dropped. `None` means no limit. Does not apply to the "zero"
            strategy.
        strategies : list
            list of strategies to use;
            so far, use a list containing any combination of "zero" and "velocity"
            (other names raise a ValueError)
        ignore_existing_mask : bool
            if `True`, use raw data ignoring existing masks
            if `False` (default), only detect blinks in the usable data under
            the ``"strict"`` mask policy (masked regions are skipped, see
            :meth:`~pypillometry.GenericEyeData.mask`); interpolated samples
            count as missing, so that detection after
            :meth:`blinks_interpolate` finds the interpolated blinks again
            (instead of reading the reconstructed values)
        inplace : bool or None
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                                    
            if `None`, use the object-level setting
            
        Returns
        -------
        :class:`~pypillometry.EyeData`
            the object with the blinks (`self` or a copy, depending on
            `inplace`)

        Examples
        --------
        >>> d = d.pupil.blinks_detect()  # doctest: +SKIP
        >>> d.get_blinks(variables="pupil", units="ms")  # doctest: +SKIP
        """
        self._d._require("pupil")
        obj = self._d._get_inplace(inplace)
        eyes=self._d._eyes_with(eyes, "pupil")

        # Convert ms to samples
        winsize_ix=self._d._samples(winsize)
        if winsize_ix % 2==0:
            winsize_ix += 1
        min_duration_ix=self._d._samples(min_duration)
        vel_onset_min_samples = max(1, self._d._samples(vel_onset_min_duration))
        vel_offset_min_samples = max(1, self._d._samples(vel_offset_min_duration))
        max_duration_ix = (None if max_duration is None
                           else self._d._samples(max_duration))

        
        # check for unknown strategies
        if isinstance(strategies, str):
            strategies = [strategies]
        if not strategies or any(strat not in ("zero", "velocity") for strat in strategies):
            raise ValueError("strategies must be a non-empty list of 'zero' and/or "
                             f"'velocity', got {strategies!r}")
        
        for eye in eyes:
            logger.debug(f"Detecting blinks for eye {eye}")
            
            # the new blinks replace the old ones, which must not mask the data
            obj.set_blinks(None, eyes=[eye], variables=["pupil"])

            # pupil data: unusable samples are NaN, unless the existing mask
            # is ignored; interpolated samples count as missing (they are
            # reconstructed blinks), so re-detection after interpolation
            # finds the blinks again
            pupil_data = obj.values(eye, "pupil", "raw" if ignore_existing_mask else "strict")
            interpolated = (np.zeros(len(pupil_data), dtype=bool) if ignore_existing_mask
                            else obj.data.get_layer(f"{eye}_pupil", "interpolated"))
            
            ## detect blinks with the different strategies
            if "velocity" in strategies:
                # Convert velocity thresholds to per-sample units
                vel_onset_samples, vel_offset_samples = self._convert_velocity_thresholds(
                    pupil_data, winsize_ix, vel_onset, vel_offset
                )
                blinks_vel=preproc.detect_blinks_velocity(pupil_data, winsize_ix, vel_onset_samples, vel_offset_samples, vel_onset_min_samples, vel_offset_min_samples, min_duration_ix,
                    max_duration_ix)
                logger.debug(f"Detected {len(blinks_vel)} blinks with velocity strategy")
            else: 
                blinks_vel=np.array([])
                
            if "zero" in strategies:
                # Look for runs of NaN or `blink_val` in the raw data. Use the
                # raw data even when `ignore_existing_mask=False`: other masks
                # (e.g., offscreen gaze) must not be turned into blinks.
                raw = obj.data[eye, "pupil"]
                if not np.isnan(blink_val):
                    raw = np.where(raw == blink_val, np.nan, raw)
                raw = np.where(interpolated, np.nan, raw)
                blinks_zero=preproc.detect_blinks_zero(raw, 1, np.nan)
                logger.debug(f"Detected {len(blinks_zero)} blinks with zero strategy")
            else:
                blinks_zero=np.array([])

            ## merge the two blinks
            logger.debug(f"Merging {len(blinks_vel)} blinks with {len(blinks_zero)} blinks")
            blinks=preproc.helper_merge_blinks(blinks_vel, blinks_zero)
            logger.debug(f"Merged {len(blinks)} blinks")
            
            # Detected offsets are inclusive, Intervals are end-exclusive.
            # Filter by minimum duration and create Intervals object
            filtered_blinks = [(int(on), int(off)+1) for (on,off) in blinks
                               if off+1-on>=min_duration_ix]
            # blinks in or next to excluded periods are not blinks
            excl = obj._excluded_mask()
            if excl.any():
                filtered_blinks = [(on, off) for on, off in filtered_blinks
                                   if not excl[max(on-1, 0):off+1].any()]
            logger.debug(f"Filtered {len(filtered_blinks)} blinks by minimum duration")
            intervals_obj = Intervals(
                intervals=filtered_blinks,
                units="samples",
                label=f"{eye}_pupil_blinks")
            
            obj.set_blinks(intervals_obj, eyes=[eye], variables=["pupil"])
        return obj
    
    @step
    @keephistory
    def blinks_merge_close(
        self,
        eyes: list = [],
        distance: float = 100,
        units: str = "ms",
        inplace: bool | None = None,
    ):
        """
        Merge blinks that are close together.

        The merged blinks replace the ``"blink"`` layer of the pupil
        (see :meth:`~pypillometry.GenericEyeData.set_blinks`); read them with
        :meth:`~pypillometry.GenericEyeData.get_blinks`.
        
        Parameters
        ----------
        eyes : str or list of str, optional
            eye(s) to process; if empty, all eyes with pupil data
        distance : float, str or Quantity, optional
            Merge blinks closer than this distance, e.g. ``100 * u.ms`` or
            ``"0.1 sec"``; plain numbers are in units given by ``units``.
            Default 100.
        units : str, optional
            Units for ``distance`` if it is a plain number. Can be "ms",
            "sec", "min", or "h". Default "ms".
        inplace : bool or None, optional
            if `True`, make change in-place and return the object;
            if `False`, make and return copy before making changes;
            if `None`, use the object-level setting (specified in the constructor).
            
        Returns
        -------
        :class:`~pypillometry.GenericEyeData`
            the object with the merged blinks (`self` or a copy, depending
            on `inplace`)
        """
        self._d._require("pupil")
        obj = self._d._get_inplace(inplace)
        pairs = obj.data.select(eyes, "pupil")

        distance_ix = obj._samples(distance, units)

        logger.debug(f"Distance '{distance}' in index units: {distance_ix}")

        for eye, var in pairs:
            # all runs, also those next to excluded periods (set_blinks replaces the layer)
            blinks = obj._layer_runs(f"{eye}_{var}", "blink")
            if len(blinks) == 0:
                continue
            logger.debug(f"Processing blinks for eye {eye} and variable {var}: {len(blinks)} blinks")
            blinks_array = np.array(blinks).astype(int)
            
            newblinks = []
            i = 1
            cblink = blinks_array[0, :].copy()
            while i < blinks_array.shape[0]:
                if (blinks_array[i, 0] - cblink[1]) <= distance_ix:
                    cblink[1] = blinks_array[i, 1]
                else:
                    newblinks.append(cblink)
                    cblink = blinks_array[i, :].copy()
                i += 1
            newblinks.append(cblink)
            
            # Store merged blinks as Intervals
            merged_intervals = Intervals(
                intervals=[(int(s), int(e)) for s, e in newblinks],
                units="samples",
                label=f"{eye}_{var}_blinks")
            obj.set_blinks(merged_intervals, eyes=[eye], variables=[var])
        return obj

    @step
    @keephistory
    def blinks_interpolate(self, eyes: str|list=[],
                                 store_as: str="pupil", 
                                 method="mahot", winsize: float=25, 
                                 vel_onset=-5.0, vel_offset=5.0, 
                                 margin: Tuple[float,float]=(10,30), 
                                 blinkwindow: float=500,
                                 interp_type: str="cubic",
                                 inplace=None):
        """Interpolating blinks in the pupillary signal.

        Implements the blink-interpolation method by Mahot (2013).
        
        Mahot, 2013:
        https://figshare.com/articles/A_simple_way_to_reconstruct_pupil_size_during_eye_blinks/688001.

        The blinks to interpolate are taken from the blinks stored in the object,
        so blinks have to be detected first (e.g., using :meth:`blinks_detect`).
        The on- and offsets of each blink are refined using the velocity profile
        before interpolating. Blinks that are so close that the anchor points
        of one fall into the other are merged and interpolated together.
        The interpolated samples get the ``"interpolated"`` layer (on the
        variable they are stored in, see `store_as`): they are used under the
        default mask policy and masked under the ``"strict"`` one (see
        :meth:`~pypillometry.GenericEyeData.mask`); the interpolated blinks
        can be retrieved with
        :meth:`~pypillometry.GenericEyeData.get_interpolated_blinks`.

        This procedure relies heavily on eye-balling (reconstructing visually convincing signal),
        so it can help to inspect the result visually (e.g., using
        :meth:`d.plot.plot_blinks() <pypillometry.plot.PlotNamespace.plot_blinks>`) to set
        good parameter values for `winsize`, `vel_onset`, `vel_offset` and `margin`
        (see paper linked above).

        Parameters
        ----------
        eyes : str or list
            str or list of eyes to process; if empty, all available eyes are processed
        store_as : str
            how to store the interpolated data; either "pupil" (default) which replaces
            the original pupil data or a string that will be used as the new variable
            name in the data (e.g., "pupilinterp")
        method : str
            method to use; so far, only "mahot" is implemented
        winsize : float, str or Quantity
            size of the Savitzky-Golay window (in ms, or with units)
        vel_onset : float or str
            velocity threshold to detect blink onset (negative value).

            - If float: threshold in pupil-units per millisecond (e.g., -5.0)
            - If str: percentile of velocity distribution (e.g., "5%" for bottom 5%)
        vel_offset : float or str
            velocity threshold to detect blink offset (positive value).

            - If float: threshold in pupil-units per millisecond (e.g., 5.0)
            - If str: percentile of velocity distribution (e.g., "95%" for top 5%)
        margin : Tuple[float,float] or Quantity
            margin that is subtracted/added to onset and offset (in ms, or
            with units, e.g. ``(10, 30) * u.ms``)
        blinkwindow : float, str or Quantity
            how much time before and after each blink to include (in ms, or
            with units)
        interp_type : str
            type of interpolation accepted by :func:`scipy.interpolate.interp1d()`
        inplace : bool or None
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes              
            if `None`, use the object-level setting

        Returns
        -------
        EyeData
            the object with the interpolated pupil data
        """
        self._d._require("pupil")
        obj = self._d._get_inplace(inplace)
        eyes=self._d._eyes_with(eyes, "pupil")

        # parameters in sampling units (from ms)
        winsize_ix=self._d._samples(winsize, rounding="up")
        margin_ix=self._d._samples(margin, rounding="up")
        blinkwindow_ix=self._d._samples(blinkwindow)
        if winsize_ix % 2==0: ## ensure smoothing window is odd
            winsize_ix+=1 

        for eye in eyes:
            logger.debug(f"Interpolating blinks for eye {eye}")
            syr=obj.data[eye,"pupil"].copy() ## interpolated signal
            interpolated = np.zeros(len(syr), dtype=bool)  # interpolated samples
            bls = self._d.get_blinks(eye, "pupil", units="samples")[f"{eye}_pupil"]
            
            # Use raw data for velocity computation - blink_onsets_mahot handles NaN/zeros internally
            # Don't use mask-applied data here because the mask includes blink regions,
            # which would smooth out the velocity transitions we need to detect boundaries
            pupil_data = obj.data[eye,"pupil"]
            
            # Convert velocity thresholds to per-sample units
            vel_onset_samples, vel_offset_samples = self._convert_velocity_thresholds(
                pupil_data, winsize_ix, vel_onset, vel_offset
            )
            blink_onsets=preproc.blink_onsets_mahot(pupil_data, bls, 
                                                    winsize_ix, 
                                                    vel_onset_samples, vel_offset_samples,
                                                    margin_ix, blinkwindow_ix)
            logger.debug(f"Refined {len(blink_onsets)} blink boundaries")
            # the anchor points of a blink must not fall into a neighbouring
            # blink (they would be NaN), so merge blinks with overlapping windows
            blink_onsets = _merge_overlapping_blink_windows(blink_onsets, len(self._d))

            # Collect successfully interpolated blinks
            interpolated_intervals = []
            
            # loop through blinks
            for ix,(t1,t2,t3,t4) in enumerate(blink_onsets):
                # Ensure all 4 time points are distinct and in order
                if t1==t2:
                    t2+=1
                if t3==t4:
                    t3-=1
                
                # Check if blink is too short to interpolate (need at least 4 distinct points)
                if not (t1 < t2 < t3 < t4):
                    logger.warning(f"Skipping blink {ix} for {eye}: blink too short to interpolate (t1={t1}, t2={t2}, t3={t3}, t4={t4})")
                    continue
                
                txpts=[obj.tx[pt] for pt in [t1,t2,t3,t4]]
                sypts=[obj.data[eye,"pupil"][pt] for pt in [t1,t2,t3,t4]]
                intfct=interp1d(txpts,sypts, kind=interp_type)
                islic=slice(t2, t3)
                syr[islic]=intfct(obj.tx[islic])
                # mark the interpolated datapoints
                interpolated[islic] = True
                
                # Record this successfully interpolated blink
                interpolated_intervals.append((t2, t3))

            # store interpolated data (with the masks and layers of the
            # pupil) and set the interpolated layer
            key, target = f"{eye}_pupil", f"{eye}_{store_as}"
            if target == key:
                obj.data.set_with_mask(key, syr, preserve_mask=True)
            else:
                obj.data._set_channel(target, syr, obj.data.layers[key])
            obj.data.set_layer(target, "interpolated", interpolated)
            logger.debug(f"Interpolated {len(interpolated_intervals)} blinks for {eye}")

        return obj
            


    @step
    @keephistory
    def estimate_baseline(self, eyes=[], variable="pupil",
                                method: str="envelope_iter_bspline_2", policy="default",
                                inplace=None, **kwargs):
        """
        Apply one of the baseline-estimation methods.

        The baseline is estimated from the usable samples under the mask
        `policy` (by default, interpolated samples are used, see
        :meth:`~pypillometry.GenericEyeData.mask`). The unusable samples
        (masked or NaN, e.g., blinks that were not interpolated or the
        excluded periods) are bridged linearly between their neighbours.
        Interpolate blinks first (see :meth:`blinks_interpolate`) to
        use a better reconstruction than a straight line.
        
        Parameters
        ----------
        eyes : list or str
            str or list of eyes to process; if empty, all available eyes are processed
        variable : str
            default is to use the "pupil" but it could be used to process, e.g., 
            interpolated pupil data stored in a different variables, e.g., "pupilinterp"
        method : str
            - "envelope_iter_bspline_1":
              :func:`~pypillometry.signal.baseline.baseline_envelope_iter_bspline`
              with one iteration
            - "envelope_iter_bspline_2" (default):
              :func:`~pypillometry.signal.baseline.baseline_envelope_iter_bspline`
              with two iterations
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are usable (see
            :meth:`~pypillometry.GenericEyeData.mask`); default ``"default"``
        inplace : bool or None
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                        
            if `None`, use the object-level setting
        **kwargs : dict
            named arguments passed to the low-level function in
            :mod:`pypillometry.signal.baseline`.
            
        Returns
        -------
        EyeData
            object with baseline estimated (stored in ``data[eye, "baseline"]``)

        Raises
        ------
        ValueError
            if `method` is unknown or no sample is usable

        References
        ----------
        Mittner, M., & Groot, J. M. (2025). A novel method for modeling tonic and
        phasic pupil dynamics in humans. *Behavior Research Methods*, 57(9), 233.
        https://doi.org/10.3758/s13428-025-02755-7
        """
        self._d._require(variable)
        obj = self._d._get_inplace(inplace)
        if not isinstance(variable, str):
            logger.warning("variable must be a string; using default 'pupil'")
            variable="pupil"
        eyes=self._d._eyes_with(eyes, variable)

        for eye in eyes:
            logger.debug("Estimating baseline for eye %s"%eye)
            sy = self._d.values(eye, variable, policy)
            unusable = np.isnan(sy)
            if unusable.all():
                raise ValueError(f"'{eye}_{variable}' has no usable samples; the "
                                 "baseline cannot be estimated")
            if unusable.any():
                logger.info(f"{eye}_{variable}: {unusable.sum()} unusable samples "
                            "(masked or NaN) bridged linearly for the baseline")
                sy = _bridge_gaps(sy)

            if method=="envelope_iter_bspline_2":
                txd,syd,base2,base1=baseline.baseline_envelope_iter_bspline(self._d.tx, sy,
                                                                            self._d.events.onsets,self._d.fs,**kwargs)
                f=interpolate.interp1d(txd, base2, kind="cubic", bounds_error=False, fill_value="extrapolate")
                obj.data[eye,"baseline"]=f(self._d.tx)
            elif method=="envelope_iter_bspline_1": 
                txd,syd,base2,base1=baseline.baseline_envelope_iter_bspline(self._d.tx, sy,
                                                                            self._d.events.onsets,self._d.fs,**kwargs)
                f=interpolate.interp1d(txd, base1, kind="cubic", bounds_error=False, fill_value="extrapolate")
                obj.data[eye,"baseline"]=f(self._d.tx)
            else:
                raise ValueError("Undefined method for baseline estimation: %s"%method)         
            
        return obj
        
    @step
    @keephistory
    def estimate_response(self, 
                          eyes=[], 
                          npar: str|float="free", tmax: str|float="free", 
                          verbose: int=50,
                          bounds: dict={"npar":(1,20), "tmax":(100,2000)},
                          min_valid: float=0.5,
                          policy="default",
                          inplace=None):
        """
        Estimate pupil-response based on event-onsets, see
        :func:`~pypillometry.signal.pupil.pupil_response`.

        The model has one response per event in ``d.events``: *all* events
        are used, whatever their label. To use only some of them, set the
        events first, e.g. ``d.set_events(d.events.filter("stim"))`` on a
        copy of the data.

        If a baseline has been estimated (see :meth:`estimate_baseline`),
        it is subtracted from the pupil signal before estimating the response;
        otherwise, a baseline of zero is used.

        Only the usable samples under the mask `policy` are used for the
        fit: by default, NaN, masked samples (e.g., blinks applied as mask)
        and the excluded periods (see
        :meth:`~pypillometry.GenericEyeData.exclude_intervals`, e.g., breaks)
        are not used, interpolated samples (e.g., blinks) are. For each
        event, the share of its response that falls on usable samples
        (`valid`) and the share that falls on real data (`observed`:
        usable under the ``"strict"`` policy, i.e., not interpolated either)
        are stored with the parameters (weighted by the response, see
        :func:`~pypillometry.signal.pupil.pupil_response_coverage`). Events
        with a `valid` share below `min_valid` are left out of the fit;
        their coefficient is NaN.

        Parameters
        ----------
        eyes : list or str
            str or list of eyes to process; if empty, all available eyes are processed
        npar : float or str
            npar-parameter for the canonical response-function or "free";
            in case of "free", the function optimizes for this parameter
        tmax : float or str
            tmax-parameter for the canonical response-function or "free";
            in case of "free", the function optimizes for this parameter
        verbose : int
            verbosity level, passed on to
            :func:`~pypillometry.signal.pupil.pupil_response`
        bounds : dict
            in case that one or both parameters are estimated, give the lower
            and upper bounds for the parameters        
        min_valid : float
            events whose share `valid` is below this are left out of the
            fit (coefficient NaN), as their amplitude cannot be estimated
            reliably; default 0.5; 0 uses all events
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are usable (see
            :meth:`~pypillometry.GenericEyeData.mask`); default ``"default"``
        inplace : bool or None
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                        
            if `None`, use the object-level setting

        Returns
        -------
        EyeData
            the object with the estimated response

        Raises
        ------
        ValueError
            if the object has no events

        Notes
        -----
        The predicted response is stored in ``data[eye, "response"]``; the
        estimated parameters (`npar`, `tmax`, the coefficients `coef` and the
        `bounds`) are stored in ``params["response"]["<eye>_pupil"]`` (e.g.,
        ``params["response"]["left_pupil"]``), together with
        the shares `valid` and `observed` (0-1, one per event; NaN for
        events without response in the data, e.g., at the very end).

        References
        ----------
        Mittner, M., & Groot, J. M. (2025). A novel method for modeling tonic and
        phasic pupil dynamics in humans. *Behavior Research Methods*, 57(9), 233.
        https://doi.org/10.3758/s13428-025-02755-7
        """
        self._d._require("pupil")
        obj = self._d._get_inplace(inplace)
        eyes=self._d._eyes_with(eyes, "pupil")

        if len(obj.events) == 0:
            raise ValueError("estimate_response needs events (one response per event); "
                             "the object has none (see set_events)")
        # replace the parameters of the processed eyes only
        obj.params["response"] = dict(obj.params.get("response", {}))
        for eye in eyes:
            logger.debug("Estimating response for eye %s"%eye)
            if not eye+"_baseline" in obj.data.keys():
                logger.warning("Eye %s: no baseline estimated yet, using zero as baseline"%eye)
                base=np.zeros(len(obj.tx))
            else:
                base=obj.data[eye,"baseline"]
        
            syd = obj.data[eye,"pupil"]-base
            valid = ~obj.mask(eye, "pupil", policy) & np.isfinite(syd)
            pred, coef, npar_est, tmax_est, x1=pupil.pupil_response(obj.tx, syd, 
                                                            obj.events.onsets, obj.fs, 
                                                            npar=npar, tmax=tmax, verbose=verbose,
                                                            bounds=bounds, valid=valid,
                                                            min_valid=min_valid)
            share_valid = pupil.pupil_response_coverage(x1, valid)
            nout = int(np.sum(np.isnan(coef)))
            if nout > 0:
                logger.info(f"Eye {eye}: {nout} of {len(coef)} events left out of the "
                            f"response fit (less than {min_valid:g} of their response "
                            "on usable samples); their coefficient is NaN")
            share_observed = pupil.pupil_response_coverage(
                x1, ~obj.mask(eye, "pupil", "strict"))
            
            obj.data[eye+"_response"]=pred

            obj.params["response"][f"{eye}_pupil"]={"npar":npar_est,
                                "npar_free":True if npar=="free" else False,
                                "tmax":tmax_est,
                                "tmax_free":True if tmax=="free" else False,
                                "coef":coef,
                                "bounds":bounds,
                                "valid":share_valid,
                                "observed":share_observed,
                                "min_valid":min_valid
                            }
        
        return obj
    

    @step
    @keephistory
    def correct_foreshortening(self, eyes=[], midpoint=None, store_as="pupil", inplace=None):
        r"""Correct pupil data using a simple algorithm to correct for foreshortening effects.

        Correct the pupil data for foreshortening effects caused
        by saccades/eye movements. This method is based on a simple algorithm
        that assumes the pupil appears perfectly circular when fixating `midpoint`:
        for each sample, the distance :math:`\Delta` (in mm) of the gaze position
        from `midpoint` is computed and the pupil signal is multiplied by the
        correction factor :math:`\sqrt{1+\Delta^2/d^2}`, where :math:`d` is the
        eye-to-screen distance from the experimental setup.
        For a more complete model that also estimates the camera position,
        see :meth:`fit_foreshortening`.

        Relevant publication (not the description of the algorithm used here):    
        https://link.springer.com/article/10.3758/s13428-015-0588-x

        - [ ] TODO: when using interpolated pupil data, x/y data may still be missing. Make sure that the 
          pupil data is not touched where x/y is missing

        Parameters 
        ----------
        eyes : list
            Which eyes to correct. If empty, correct all available eyes
            that have pupil and gaze (x, y) data.
        midpoint : tuple or None
            The center of the screen (x,y) in pixels where it is assumed that the pupil is completely circular.
            If None, the midpoint is taken to be the center of the screen as registered
            in the experimental_setup. 
        store_as : str
            name of the variable in which to store the corrected pupil data
            (as ``data[eye, store_as]``); the default "pupil" replaces the
            original pupil data
        inplace : bool or None
            Whether to modify the object in place or return a new object.
            `true`: modify in place
            `false`: return a new object
            `None`: use the object's default setting (initialized in __init__)

        Returns
        -------
        EyeData
            An EyeData object with the corrected pupil

        Raises
        ------
        ValueError
            if no experimental setup is set (see :meth:`~pypillometry.GenericEyeData.set_experimental_setup`)
        """
        self._d._require("pupil", "x", "y")
        obj=self._d._get_inplace(inplace)
        eyes=self._d._eyes_with(eyes, ["pupil", "x", "y"])

        if not isinstance(store_as, str):
            logger.warning("store_as must be a string; using 'pupil' instead")
            store_as="pupil"

        # Require experimental_setup for this correction
        if self._d.experimental_setup is None:
            raise ValueError(
                "Cannot correct pupil foreshortening: experimental_setup not set. "
                "Use set_experimental_setup() to configure it."
            )
        setup = self._d.experimental_setup

        if midpoint is None:
            midpoint = (setup.screen_width / 2, setup.screen_height / 2)
        
        scaling_factor_x = setup.mm_per_pixel_x
        scaling_factor_y = setup.mm_per_pixel_y

        # calculate distance of x,y from midpoint
        for eye in eyes:
            vx="_".join([eye, "x"])
            vy="_".join([eye, "y"])
            xdist = np.abs(self._d.data[vx] - midpoint[0]) * scaling_factor_x
            ydist = np.abs(self._d.data[vy] - midpoint[1]) * scaling_factor_y
            dist = np.sqrt(xdist**2 + ydist**2)
            corr = np.sqrt((dist**2) / (setup.d**2) + 1)  # correction factor
            # correct the masked samples too and keep the layers of the
            # pupil (NaN gaze gives NaN)
            corrected = self._d.data[eye, "pupil"] * corr
            key, target = f"{eye}_pupil", f"{eye}_{store_as}"
            if target == key:
                obj.data.set_with_mask(key, corrected, preserve_mask=True)
            else:
                obj.data._set_channel(target, corrected, obj.data.layers[key])

        return obj

    def fit_foreshortening(
        self,
        eye: str,
        r: Optional[Union[float, str]] = None,
        d: Optional[Union[float, str]] = None,
        intervals: Optional[Intervals] = None,
        target_fs: float = 50.0,
        lowpass_freq: float = 4.0,
        lambda_smooth: float = 1.0,
        knots_per_second: float = 1.0,
        initial_theta: Optional[Union[float, str]] = None,
        initial_phi: Optional[Union[float, str]] = None,
        verbose: bool = True
    ) -> ForeshorteningCalibration:
        """
        Fit foreshortening correction model to estimate camera geometry and true pupil size.
        
        This method implements Stage 1 of the foreshortening correction algorithm,
        which jointly estimates the camera position (theta, phi) and the temporal
        dynamics of true pupil size A0(t) using B-splines. The data is automatically
        preprocessed (filtered and downsampled) before fitting.
        
        Parameters
        ----------
        eye : str
            Which eye to fit ('left' or 'right')
        r : float, str, or Quantity, optional
            Eye-to-camera distance. If not provided, uses the 
            `r` from experimental_setup.

            - Plain number: assumed to be mm (with warning)
            - String: e.g., "600 mm", "60 cm"
            - Quantity: e.g., 600 * u.mm
        d : float, str, or Quantity, optional
            Eye-to-screen distance. If not provided, uses the 
            `d` from experimental_setup.

            - Plain number: assumed to be mm (with warning)
            - String: e.g., "700 mm", "70 cm"
            - Quantity: e.g., 700 * u.mm
        intervals : Intervals, optional
            Time intervals to use for fitting. If None, uses all available data.
            Useful for fitting only on calibration periods. Intervals in any
            time units or in sample indices (``units="samples"``) are accepted; a
            plain list of `(start, end)` tuples is taken as ms. Intervals are
            end-exclusive.
        target_fs : float, default 50.0
            Target sampling rate in Hz after downsampling for efficiency.
            Pupil dynamics are typically well-preserved at 50 Hz.
        lowpass_freq : float, default 4.0
            Lowpass filter cutoff frequency in Hz. Removes high-frequency noise
            before downsampling. Should be below target_fs/2 (Nyquist).
        lambda_smooth : float, default 1.0
            Smoothness regularization weight. Higher values produce smoother
            A0(t) estimates. Tune via cross-validation if needed.
        knots_per_second : float, default 1.0
            Number of B-spline knots per second. Controls temporal resolution
            of A0(t) model. Typical values: 0.5-2.0.
        initial_theta : float, str, or Quantity, optional
            Initial guess for camera polar angle. If None, uses
            pi/2 (horizontal) as starting point.

            - Plain number: assumed to be radians (with warning)
            - String: e.g., "20 degrees", "0.349 radians"
            - Quantity: e.g., 20 * u.deg
        initial_phi : float, str, or Quantity, optional
            Initial guess for camera azimuthal angle. If None,
            uses 0 (camera on x-axis) as starting point.

            - Plain number: assumed to be radians (with warning)
            - String: e.g., "-90 degrees", "-1.57 radians"
            - Quantity: e.g., -90 * u.deg
        verbose : bool, default True
            Print optimization progress and fit statistics.
        
        Returns
        -------
        calibration : ForeshorteningCalibration
            Fitted calibration object containing camera geometry (theta, phi),
            spline coefficients for A0(t), and fit quality metrics.
        
        Raises
        ------
        ValueError
            If specified eye doesn't exist or lacks gaze/pupil data, if the
            experimental setup is not set or lacks required information, if
            `lowpass_freq` is not below the Nyquist frequency or if fewer
            than 100 valid samples are available.
        
        Notes
        -----
        The model is:
            A_measured(x,y,t) = A0(t) * cos(alpha(x,y; theta, phi))
        
        where:

        - A0(t) = sum_k a_k * B_k(t) is modeled with B-splines
        - cos(alpha) is the foreshortening factor depending on viewing angle
        - theta, phi parameterize camera position in spherical coordinates
        
        The optimization minimizes:
            sum_i (A_i - A_pred_i)^2 + lambda_smooth * sum_k (a_{k+1} - a_k)^2
        
        Examples
        --------
        >>> import pypillometry as pp
        >>> data = pp.EyeData.from_eyelink('recording.edf')  # doctest: +SKIP
        >>> 
        >>> # Set experimental setup
        >>> data.set_experimental_setup(  # doctest: +SKIP
        ...     screen_resolution=(1920, 1080),
        ...     physical_screen_size=("52 cm", "29 cm"),
        ...     eye_to_screen_center="65 cm",
        ...     camera_offset=("0 cm", "-30 cm", "0 cm"),
        ... )
        >>> 
        >>> # Fit on calibration period (first 2 minutes)
        >>> cal_intervals = pp.Intervals([(0, 120000)], units="ms")  # doctest: +SKIP
        >>> calib = data.pupil.fit_foreshortening(  # doctest: +SKIP
        ...     eye='left',
        ...     intervals=cal_intervals
        ... )
        >>> 
        >>> # Or provide r and d explicitly
        >>> calib = data.pupil.fit_foreshortening(  # doctest: +SKIP
        ...     eye='left',
        ...     r="60 cm",
        ...     d="70 cm",
        ...     intervals=cal_intervals
        ... )
        >>> 
        >>> # Inspect fit quality
        >>> print(calib)  # doctest: +SKIP
        >>> print(f"R² = {calib.fit_metrics['r2']:.3f}")  # doctest: +SKIP
        >>> 
        >>> # Compute correction for all data
        >>> correction_factor = calib.get_correction_factor(  # doctest: +SKIP
        ...     data['left', 'x'], 
        ...     data['left', 'y']
        ... )
        >>> corrected_pupil = data['left', 'pupil'] * correction_factor  # doctest: +SKIP
        
        See Also
        --------
        ForeshorteningCalibration : Container for fitted parameters
        correct_foreshortening : Simple foreshortening correction method
        """
        # Validate eye
        self._d._require("pupil", "x", "y")
        if not isinstance(eye, str):
            raise TypeError(f"eye must be the name of one eye (e.g., 'left'), got {eye!r}")
        if eye not in self._d.eyes:
            raise ValueError(f"Eye '{eye}' not found in dataset. Available: {self._d.eyes}")
        
        # Check for required data
        x_var = f"{eye}_x"
        y_var = f"{eye}_y"
        pupil_var = f"{eye}_pupil"
        
        if x_var not in self._d.data or y_var not in self._d.data:
            raise ValueError(f"Gaze data (x, y) not available for {eye} eye")
        if pupil_var not in self._d.data:
            raise ValueError(f"Pupil data not available for {eye} eye")
        
        # Require experimental_setup for screen dimensions
        if self._d.experimental_setup is None:
            raise ValueError(
                "Cannot fit foreshortening: experimental_setup not set. "
                "Use set_experimental_setup() to configure it."
            )
        setup = self._d.experimental_setup
        
        # Get geometric parameters from experimental_setup or explicit args
        if r is None:
            if not setup.has_camera_position():
                raise ValueError(
                    "Camera position not set in experimental_setup. "
                    "Either provide r explicitly or set camera_offset in experimental_setup."
                )
            r = setup.r
            if verbose:
                logger.info(f"Using camera distance from experimental_setup: {r:.1f} mm")
        else:
            r = parse_distance(r)
        
        if d is None:
            if not setup.has_eye_distance():
                raise ValueError(
                    "Eye-to-screen distance not set in experimental_setup. "
                    "Either provide d explicitly or set eye_to_screen_center in experimental_setup."
                )
            d = setup.d
            if verbose:
                logger.info(f"Using screen distance from experimental_setup: {d:.1f} mm")
        else:
            d = parse_distance(d)
        
        # Parse angle parameters if provided
        if initial_theta is not None:
            initial_theta = parse_angle(initial_theta)
        
        if initial_phi is not None:
            initial_phi = parse_angle(initial_phi)
        
        # Validate geometric parameters
        if r <= 0:
            raise ValueError(f"Eye-to-camera distance r must be positive, got {r}")
        if d <= 0:
            raise ValueError(f"Eye-to-screen distance d must be positive, got {d}")
        
        if verbose:
            logger.info(f"Eye-to-camera distance: {r} mm")
            logger.info(f"Eye-to-screen distance: {d} mm")
        
        # Extract data (using masked array access to exclude blinks/artifacts)
        time = self._d["time_ms"].copy()
        x_gaze = self._d[eye, 'x'].copy()
        y_gaze = self._d[eye, 'y'].copy()
        pupil = self._d[eye, 'pupil'].copy()
        
        # Apply interval mask if provided
        if intervals is not None:
            if not isinstance(intervals, Intervals):
                intervals = Intervals(intervals, units="ms")  # plain list in ms
            mask = intervals.to_mask(self._d).astype(bool)
            
            time = time[mask]
            x_gaze = x_gaze[mask]
            y_gaze = y_gaze[mask]
            pupil = pupil[mask]
            
            if verbose:
                logger.info(f"Using {np.sum(mask)} samples from specified intervals")
        
        # mm relative to screen center, positive y = up
        x_gaze_mm, y_gaze_mm = setup.pixels_to_mm(x_gaze, y_gaze, centered=True)
        
        # Remove invalid samples (masked, NaN, pupil <= 0); the masks must be
        # checked explicitly, comparisons on masked arrays keep masked samples
        masked = (np.ma.getmaskarray(x_gaze_mm) | np.ma.getmaskarray(y_gaze_mm)
                  | np.ma.getmaskarray(pupil))
        x_gaze_mm = np.ma.getdata(x_gaze_mm)
        y_gaze_mm = np.ma.getdata(y_gaze_mm)
        pupil = np.ma.getdata(pupil)
        valid = (~masked & np.isfinite(x_gaze_mm) & np.isfinite(y_gaze_mm)
                 & np.isfinite(pupil) & (pupil > 0))

        time = time[valid]
        x_gaze_mm = x_gaze_mm[valid]
        y_gaze_mm = y_gaze_mm[valid]
        pupil = pupil[valid]
        
        n_valid = len(time)
        if n_valid < 100:
            raise ValueError(f"Insufficient valid samples for fitting: {n_valid} < 100")
        
        if verbose:
            logger.info(f"Valid samples after preprocessing: {n_valid}")
        
        # Preprocessing: lowpass filter
        if verbose:
            logger.info(f"Applying {lowpass_freq} Hz lowpass filter")
        
        nyquist = self._d.fs / 2
        if lowpass_freq >= nyquist:
            raise ValueError(f"Lowpass frequency ({lowpass_freq} Hz) must be < Nyquist ({nyquist} Hz)")
        
        # Apply Butterworth lowpass filter (order 3)
        pupil_filt = butter_lowpass_filter(pupil, cutoff=lowpass_freq, fs=self._d.fs, order=3)
        
        # Downsample
        if self._d.fs > target_fs:
            downsample_factor = int(np.round(self._d.fs / target_fs))
            
            if verbose:
                logger.info(f"Downsampling by factor {downsample_factor}: {self._d.fs} Hz → {self._d.fs/downsample_factor:.1f} Hz")
            
            time = downsample(time, downsample_factor)
            x_gaze_mm = downsample(x_gaze_mm, downsample_factor)
            y_gaze_mm = downsample(y_gaze_mm, downsample_factor)
            pupil_filt = downsample(pupil_filt, downsample_factor)
        
        n_samples = len(time)
        
        # Setup B-spline basis
        t_min, t_max = time[0], time[-1]
        duration_sec = (t_max - t_min) / 1000.0
        
        knots = _determine_knots(t_min, t_max, knots_per_second)
        basis_matrix = _create_bspline_basis(time, knots, degree=3)
        n_basis = basis_matrix.shape[1]
        
        if verbose:
            logger.info(f"B-spline basis: {n_basis} functions over {duration_sec:.1f} seconds")
        
        # Initialize parameters
        # Spline coefficients: initialize to mean pupil size
        a_init = np.full(n_basis, np.mean(pupil_filt))
        
        # Camera angles: use provided or default
        theta_init = initial_theta if initial_theta is not None else np.pi / 2  # Horizontal
        phi_init = initial_phi if initial_phi is not None else 0.0  # x-axis
        
        params_init = np.concatenate([a_init, [theta_init, phi_init]])
        
        # Setup bounds for L-BFGS-B
        bounds = [(0.1 * np.mean(pupil_filt), 10 * np.mean(pupil_filt))] * n_basis  # Spline coeffs > 0
        bounds += [(0, np.pi/2)]  # theta: polar angle from z-axis, camera must be forward of eye (0=on-axis, π/2=perpendicular)
        bounds += [(-np.pi, np.pi)]  # phi: azimuthal in x-y plane (-π/2=below, 0=right, π/2=above, ±π=left)
        
        # check that the initial values are within the bounds
        for i, (lb, ub) in enumerate(bounds):
            if params_init[i] < lb or params_init[i] > ub:
                logger.warning(f"Initial parameter {i} is out of bounds: {params_init[i]} not in [{lb}, {ub}]")
                params_init[i] = (lb + ub) / 2

        # Run optimization
        if verbose:
            logger.info("Starting optimization...")
        
        result = optimize.minimize(
            fun=_objective_function,
            x0=params_init,
            args=(x_gaze_mm, y_gaze_mm, time, pupil_filt, basis_matrix, r, d, lambda_smooth, 0.0),
            method='L-BFGS-B',
            bounds=bounds,
            jac=_objective_gradient,
            options={'maxiter': 500}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        elif verbose:
            logger.info(f"Optimization converged in {result.nit} iterations")
        
        # Extract fitted parameters
        params_opt = result.x
        spline_coeffs_opt, theta_opt, phi_opt = _split_params(params_opt, n_basis)
        
        # Compute fit metrics
        from .foreshortening_calibration import _compute_cos_alpha_vectorized
        
        A0_fit = basis_matrix @ spline_coeffs_opt
        cos_alpha_fit = _compute_cos_alpha_vectorized(x_gaze_mm, y_gaze_mm, theta_opt, phi_opt, r, d, 0.0)
        pupil_pred = A0_fit * cos_alpha_fit
        
        residuals = pupil_filt - pupil_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((pupil_filt - np.mean(pupil_filt)) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        fit_metrics = {
            'r2': r2,
            'rmse': rmse,
            'n_samples': n_samples,
            'loss': result.fun,
            'n_iterations': result.nit,
            'converged': result.success
        }
        
        if verbose:
            logger.info(f"Fit quality: R² = {r2:.4f}, RMSE = {rmse:.4f}")
            logger.info(f"Camera position: theta = {np.degrees(theta_opt):.1f}°, phi = {np.degrees(phi_opt):.1f}°")
        
        # Create ForeshorteningCalibration object
        calibration = ForeshorteningCalibration(
            eye=eye,
            theta=theta_opt,
            phi=phi_opt,
            experimental_setup=setup,
            spline_coeffs=spline_coeffs_opt,
            spline_knots=knots,
            spline_degree=3,
            fit_intervals=intervals,
            fit_metrics=fit_metrics
        )
        
        return calibration
