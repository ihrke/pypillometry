"""
pupildata.py
============

Methods of :class:`~pypillometry.EyeData` for pupil data.
"""

import itertools
from typing import Optional, Tuple
from .generic import keephistory
#from .. import convenience
from ..signal import baseline
from ..signal import preproc
from ..signal import pupil
from ..signal.pupil import _bridge_gaps
#from .. import io
from ..intervals import Intervals

import json
from loguru import logger
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate
import scipy


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

class _PupilMethods:
    """
    Methods of :class:`~pypillometry.EyeData` for pupil data.

    Internal: not a data class (no state and no constructor). The methods
    need pupil data and raise a ValueError if there is none.
    """

    @keephistory
    def pupil_lowpass_filter(self,  cutoff: float, order: int=2, eyes=[], policy="default",
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
        PupilData
            the object with the filtered pupil data
        """
        from ..signal.pupil import lowpass_filter_iterative
        
        self._require("pupil")
        obj = self._get_inplace(inplace)
        eyes=self._eyes_with(eyes, "pupil")

        for eye in eyes:
            x = obj.values(eye, "pupil", policy)  # NaN: unusable
            usable = ~np.isnan(x)
            out = obj.data[eye, "pupil"].copy()
            out[usable] = lowpass_filter_iterative(x, cutoff, obj.fs, order)[usable]
            obj.data.set_with_mask(f"{eye}_pupil", out, preserve_mask=True)
        return obj

    @keephistory
    def pupil_smooth_window(self, eyes=[], window: str="hanning", winsize: float=11,
                            policy="default", inplace=None):
        """
        Apply smoothing of the signal using a moving window.
        See :func:`~pypillometry.signal.preproc.smooth_window`.

        Only the usable samples under the mask `policy` are smoothed; the
        unusable samples (masked or NaN) are bridged linearly for the
        smoothing and keep their stored values (as in
        :meth:`pupil_lowpass_filter`). The mask layers are kept.
        
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
        PupilData
            the object with the smoothed pupil data
        """
        self._require("pupil")
        obj = self._get_inplace(inplace)
        eyes=self._eyes_with(eyes, "pupil")

        # convert winsize to index based on sampling rate
        winsize_ix=self._samples(winsize, rounding="up")

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

    @keephistory
    def pupil_merge_eyes(self, method: str = "local_offset", 
                         lowpass_cutoff: float = 0.2,
                         max_iter: int = 10,
                         keep_eyes: bool = True, 
                         policy="default",
                         inplace: bool | None = None):
        """
        Merge pupil data from both eyes into a single "merged" variable.
        
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
            If False, remove the original eye data after merging.
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are valid (see
            :meth:`~pypillometry.GenericEyeData.mask`); default ``"default"``
        inplace : bool or None, optional
            If True, modify in-place. If False, return a copy.
            If None, use the object's default setting.
            
        Returns
        -------
        PupilData
            The object with merged pupil data stored as "merged_pupil".
            
        Raises
        ------
        ValueError
            If only one eye is available (nothing to merge).
            If an unknown method is specified.
            
        Notes
        -----
        The merged signal is stored as "merged_pupil" and can be accessed via
        `data["merged", "pupil"]` or `data["merged_pupil"]`.
        
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
        >>> # Default: local offset correction
        >>> data.pupil_merge_eyes()
        >>> merged = data["merged", "pupil"]
        
        >>> # Use global offset (simpler, assumes no drift)
        >>> data.pupil_merge_eyes(method="offset")
        
        >>> # Only use timepoints where both eyes are valid
        >>> data.pupil_merge_eyes(method="both_available")
        
        >>> # Remove original eye data after merging
        >>> data.pupil_merge_eyes(keep_eyes=False)
        
        See Also
        --------
        merge_eyes : Generic merge method in GenericEyeData (masks if all eyes invalid)
        pypillometry.signal.pupil.pupil_eye_offset : Compute local offset between eyes
        """
        self._require("pupil")
        obj = self._get_inplace(inplace)
        
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
        obj.data.set_with_mask("merged_pupil", merged, mask=merged_mask)
        
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
            vel_onset_samples = float(vel_onset) * (1000 / self.fs)
            logger.debug(f"vel_onset {vel_onset}/ms -> {vel_onset_samples:.4f}/sample")
        
        # Convert offset threshold
        if offset_pct is not None:
            vel_offset_samples = np.percentile(vel_valid, offset_pct)
            logger.debug(f"vel_offset {vel_offset} -> {vel_offset_samples:.4f} (percentile)")
        else:
            # Convert from per-ms to per-sample
            vel_offset_samples = float(vel_offset) * (1000 / self.fs)
            logger.debug(f"vel_offset {vel_offset}/ms -> {vel_offset_samples:.4f}/sample")
        
        return vel_onset_samples, vel_offset_samples

    @keephistory
    def pupil_blinks_detect(self, eyes=[], min_duration:float=20, blink_val:float=0,
                      winsize: float=11, vel_onset=-5.0, vel_offset=5.0, 
                      vel_onset_min_duration: float=5.0, vel_offset_min_duration: float=5.0,
                      max_duration: float=1000,
                      strategies: list=["zero","velocity"],
                      apply_mask=False, ignore_existing_mask=False, inplace=None):
        """
        Detect blinks in the pupillary signal using several strategies.
        First, blinks are detected as consecutive sequence of `blink_val` 
        (f.eks., 0 or NaN). Second, blinks are defined as everything between
        two crossings of the velocity profile (from negative to positive).
        
        Detected blinks are always stored in the object, as the ``"blink"``
        layer of the pupil (retrieve them with
        :meth:`~pypillometry.GenericEyeData.get_blinks`). They replace the
        blinks detected before (and their mask), and the blinks detected
        before are not treated as masked during detection, so re-running the
        detection with other parameters gives the same result as running it
        once. When `apply_mask=True`,
        the blinks are also applied as masks to the data. When `apply_mask=False`
        (default), the detected blinks are returned as Intervals objects and
        stored in the object itself (also with `inplace=False`; the data and
        the mask are not changed).

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
        apply_mask : bool
            if `True`, apply detected blinks as masks to the data and return the object
            if `False` (default), return detected blinks as Intervals (or dict of Intervals)
        ignore_existing_mask : bool
            if `True`, use raw data ignoring existing masks
            if `False` (default), only detect blinks in the usable data under
            the default mask policy (masked regions are skipped; interpolated
            samples are used, see :meth:`~pypillometry.GenericEyeData.mask`)
        inplace : bool or None
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes                                                    
            if `None`, use the object-level setting;
            only used with `apply_mask=True`
            
        Returns
        -------
        PupilData or Intervals or dict
            If `apply_mask=True`: returns the object (for chaining)
            If `apply_mask=False` and single eye: returns Intervals object
            If `apply_mask=False` and multiple eyes: returns dict of Intervals objects
            (keys like ``"left_pupil"``)
        """
        # without apply_mask, the blinks are returned and stored in self
        self._require("pupil")
        obj = self._get_inplace(inplace) if apply_mask else self
        eyes=self._eyes_with(eyes, "pupil")

        # Convert ms to samples
        winsize_ix=self._samples(winsize)
        if winsize_ix % 2==0:
            winsize_ix += 1
        min_duration_ix=self._samples(min_duration)
        vel_onset_min_samples = max(1, self._samples(vel_onset_min_duration))
        vel_offset_min_samples = max(1, self._samples(vel_offset_min_duration))
        max_duration_ix = (None if max_duration is None
                           else self._samples(max_duration))

        
        # check for unknown strategies
        for strat in strategies:
            if strat not in ["zero", "velocity"]:
                logger.warning("Strategy '%s' unknown"%strat)
        
        detected_intervals = {}
        
        for eye in eyes:
            logger.debug(f"Detecting blinks for eye {eye}")
            
            # the new blinks replace the old ones, which must not mask the data
            obj.set_blinks(None, eyes=[eye], variables=["pupil"])

            # pupil data: unusable samples (default mask policy) are NaN,
            # unless the existing mask is ignored
            pupil_data = obj.values(eye, "pupil", "raw" if ignore_existing_mask else "default")
            
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
            
            # Store detected blinks internally (always); return them as stored
            # (blinks that touch are one blink)
            obj.set_blinks(intervals_obj, eyes=[eye], variables=["pupil"], apply_mask=apply_mask)
            detected_intervals[f"{eye}_pupil"] = obj.get_blinks(eye, "pupil")
        
        # Return based on apply_mask
        if apply_mask:
            return obj
        else:
            # Return single Intervals or dict depending on number of eyes
            if len(eyes) == 1:
                return detected_intervals[f"{eyes[0]}_pupil"]
            else:
                return detected_intervals    
    
    @keephistory
    def pupil_blinks_interpolate(self, eyes: str|list=[],
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
        so blinks have to be detected first (e.g., using :meth:`pupil_blinks_detect`).
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
        :meth:`~pypillometry.PupilPlotter.plot_blinks`) to set
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
        PupilData
            the object with the interpolated pupil data
        """
        self._require("pupil")
        obj = self._get_inplace(inplace)
        eyes=self._eyes_with(eyes, "pupil")

        # parameters in sampling units (from ms)
        winsize_ix=self._samples(winsize, rounding="up")
        margin_ix=self._samples(margin, rounding="up")
        blinkwindow_ix=self._samples(blinkwindow)
        if winsize_ix % 2==0: ## ensure smoothing window is odd
            winsize_ix+=1 

        for eye in eyes:
            logger.debug(f"Interpolating blinks for eye {eye}")
            syr=obj.data[eye,"pupil"].copy() ## interpolated signal
            interpolated = np.zeros(len(syr), dtype=bool)  # interpolated samples
            bls = self.get_blinks(eye, "pupil")
            
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
            blink_onsets = _merge_overlapping_blink_windows(blink_onsets, len(self))

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
            


    @keephistory
    def pupil_estimate_baseline(self, eyes=[], variable="pupil",
                                method: str="envelope_iter_bspline_2", policy="default",
                                inplace=None, **kwargs):
        """
        Apply one of the baseline-estimation methods.

        The baseline is estimated from the usable samples under the mask
        `policy` (by default, interpolated samples are used, see
        :meth:`~pypillometry.GenericEyeData.mask`). The unusable samples
        (masked or NaN, e.g., blinks that were not interpolated or the
        excluded periods) are bridged linearly between their neighbours.
        Interpolate blinks first (see :meth:`pupil_blinks_interpolate`) to
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
        PupilData
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
        self._require(variable)
        obj = self._get_inplace(inplace)
        if not isinstance(variable, str):
            logger.warning("variable must be a string; using default 'pupil'")
            variable="pupil"
        eyes=self._eyes_with(eyes, variable)

        for eye in eyes:
            logger.debug("Estimating baseline for eye %s"%eye)
            sy = self.values(eye, variable, policy)
            unusable = np.isnan(sy)
            if unusable.all():
                raise ValueError(f"'{eye}_{variable}' has no usable samples; the "
                                 "baseline cannot be estimated")
            if unusable.any():
                logger.info(f"{eye}_{variable}: {unusable.sum()} unusable samples "
                            "(masked or NaN) bridged linearly for the baseline")
                sy = _bridge_gaps(sy)

            if method=="envelope_iter_bspline_2":
                txd,syd,base2,base1=baseline.baseline_envelope_iter_bspline(self.tx, sy,
                                                                            self.events.onsets,self.fs,**kwargs)
                f=interpolate.interp1d(txd, base2, kind="cubic", bounds_error=False, fill_value="extrapolate")
                obj.data[eye,"baseline"]=f(self.tx)
            elif method=="envelope_iter_bspline_1": 
                txd,syd,base2,base1=baseline.baseline_envelope_iter_bspline(self.tx, sy,
                                                                            self.events.onsets,self.fs,**kwargs)
                f=interpolate.interp1d(txd, base1, kind="cubic", bounds_error=False, fill_value="extrapolate")
                obj.data[eye,"baseline"]=f(self.tx)
            else:
                raise ValueError("Undefined method for baseline estimation: %s"%method)         
            
        return obj
        
    @keephistory
    def pupil_estimate_response(self, 
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

        If a baseline has been estimated (see :meth:`pupil_estimate_baseline`),
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
        PupilData
            the object with the estimated response

        Notes
        -----
        The predicted response is stored in ``data[eye, "response"]``; the
        estimated parameters (`npar`, `tmax`, the coefficients `coef` and the
        `bounds`) are stored in ``params["response"][eye]``, together with
        the shares `valid` and `observed` (0-1, one per event; NaN for
        events without response in the data, e.g., at the very end).

        References
        ----------
        Mittner, M., & Groot, J. M. (2025). A novel method for modeling tonic and
        phasic pupil dynamics in humans. *Behavior Research Methods*, 57(9), 233.
        https://doi.org/10.3758/s13428-025-02755-7
        """
        self._require("pupil")
        obj = self._get_inplace(inplace)
        eyes=self._eyes_with(eyes, "pupil")

        obj.params["response"]=dict()
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

            obj.params["response"][eye]={"npar":npar_est,
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
    
