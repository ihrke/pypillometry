"""
pupil.py
========

Methods of :class:`~pypillometry.EyeData` for pupil data.
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union
from dataclasses import replace
from .history import keephistory
from .namespace import Namespace
from .eyedatadict import _without_layer
from .pipes import step
#from .. import convenience
from ..signal import baseline
from ..signal import preproc
from ..signal import pupil
from ..signal import transplant
from ..signal import autotune
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

from loguru import logger
import numpy as np

if TYPE_CHECKING:
    from .eyedata import EyeData


#: parameters of the methods of ``blinks_interpolate()`` and their defaults
_INTERP_DEFAULTS = {
    "mahot": dict(winsize=25, vel_onset=-5.0, vel_offset=5.0, margin=(10, 30),
                  interp_type="cubic", max_duration=2000),
    "kalman": dict(optimize=False),
}


def _interp_params(method, kwargs):
    """Parameters of `method` (defaults updated by `kwargs`) for
    ``blinks_interpolate()``. Parameters of other methods are ignored with a
    warning; unknown parameters raise a TypeError."""
    if method not in _INTERP_DEFAULTS:
        raise ValueError(f"unknown method {method!r} for blinks_interpolate(); "
                         f"use one of {', '.join(map(repr, _INTERP_DEFAULTS))}")
    pars = dict(_INTERP_DEFAULTS[method])
    for name, value in kwargs.items():
        if name in pars:
            pars[name] = value
            continue
        users = [m for m, d in _INTERP_DEFAULTS.items() if name in d]
        if not users:
            raise TypeError(f"blinks_interpolate() got an unexpected keyword argument {name!r}")
        logger.warning(f"{name!r} is only used by method={users[0]!r} of "
                       f"blinks_interpolate(), not by {method!r}; ignored")
    return pars

class PupilNamespace(Namespace["EyeData"]):
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
        :meth:`~pypillometry.EyeData.mask`). The unusable samples
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
            :meth:`~pypillometry.EyeData.mask`); default ``"default"``
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
            x = obj.get_values(eye, "pupil", policy)  # NaN: unusable
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
            :meth:`~pypillometry.EyeData.mask`); default ``"default"``
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
            x = obj.get_values(eye, "pupil", policy)  # NaN: unusable
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
        usable samples under the mask `policy` (see
        :meth:`~pypillometry.EyeData.mask`), where recorded samples
        come first: an interpolated sample of one eye is used only if the
        other eye has no recorded sample there. The merged pupil gets the
        layers that are set in all eyes that went into a sample (e.g.,
        ``"interpolated"`` where both were interpolated, ``"blink"`` where
        both eyes are masked by a blink); masked samples without such a
        common layer get the ``"user"`` layer.
        
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
            :meth:`~pypillometry.EyeData.mask`); default ``"default"``
        name : str, optional
            name of the new eye (default ``"mean"``, as in
            :meth:`~pypillometry.EyeData.merge_eyes`); the merged
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
        merge_eyes : Generic merge method in EyeData (masks if all eyes invalid)
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
        keys = ["left_pupil", "right_pupil"]
        if method == "both_available":
            # both eyes must be usable (recorded or not)
            used = np.stack([~obj.data.get_policy_mask(k, policy) for k in keys])
        else:
            used = obj.data._merge_usage(keys, policy)
        left_mask, right_mask = ~used[0], ~used[1]
        
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
        
        # Store merged data with the layers the used eyes have in common
        obj.data._store_merged(f"{name}_pupil", merged, keys, used, masked, policy)
        
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
                      vel_onset_min_duration: float=10.0, vel_offset_min_duration: float=10.0,
                      max_duration: float=2000, margin=0,
                      strategies: list=["zero","velocity"],
                      ignore_existing_mask=False, inplace=None):
        """
        Detect blinks in the pupillary signal using several strategies.
        First, blinks are detected as consecutive sequence of `blink_val` 
        (f.eks., 0 or NaN). Second, blinks are defined as everything between
        two crossings of the velocity profile (from negative to positive).
        
        Detected blinks are stored in the object, as the ``"blink"``
        layer of the pupil (retrieve them with
        :meth:`~pypillometry.EyeData.get_blinks`), and are masked
        under the mask policies (see
        :meth:`~pypillometry.EyeData.mask`). They replace the
        blinks detected before (and their mask), and the blinks detected
        before are not treated as masked during detection, so re-running the
        detection with other parameters gives the same result as running it
        once (``clear_layer("blink")`` removes them). Eyes whose pupil has
        interpolated samples (after :meth:`blinks_interpolate`) are skipped
        with a warning and keep their blinks: the values there are
        reconstructed, so detect blinks before interpolating them.

        Finally, detected blinks have to be at least `min_duration` and at
        most `max_duration` long; longer runs of missing samples (e.g., loss
        of the pupil or breaks) stay missing, but are not blinks.
        They can be extended by a `margin` on each side, e.g. to include
        the slow parts of the transient around the blink.
        
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
            maximum duration (in ms, or with units) of a blink (default 2 s,
            generous for blinks that are merged later); for the "velocity"
            strategy, an onset without an offset within this time is
            dropped. `None` means no limit.
        margin : float, str, Quantity or tuple
            time (in ms, or with units) by which each detected blink is
            extended to the left and right, after the `min_duration` filter;
            a tuple `(before, after)` gives different margins on each side.
            Blinks that overlap after extending are merged. Default 0 (no
            extension). This is independent of the `margin` of
            :meth:`blinks_interpolate`, which is applied on top of it.
        strategies : list
            list of strategies to use;
            so far, use a list containing any combination of "zero" and "velocity"
            (other names raise a ValueError)
        ignore_existing_mask : bool
            if `True`, use raw data ignoring existing masks
            if `False` (default), only detect blinks in the usable data under
            the ``"strict"`` mask policy (masked regions are skipped, see
            :meth:`~pypillometry.EyeData.mask`)
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
        margin_ix = self._d._samples(margin, rounding="up")
        if np.ndim(margin_ix) == 0:
            margin_ix = (margin_ix, margin_ix)
        if len(margin_ix) != 2:
            raise ValueError(f"margin must be a number or a (before, after) tuple, got {margin!r}")
        if min(margin_ix) < 0:
            raise ValueError(f"margin must not be negative, got {margin!r}")

        
        # check for unknown strategies
        if isinstance(strategies, str):
            strategies = [strategies]
        if not strategies or any(strat not in ("zero", "velocity") for strat in strategies):
            raise ValueError("strategies must be a non-empty list of 'zero' and/or "
                             f"'velocity', got {strategies!r}")
        
        for eye in eyes:
            if obj.data.get_layer(f"{eye}_pupil", "interpolated").any():
                logger.warning(f"{eye}_pupil has interpolated samples; blink detection "
                               "skipped (detect blinks before interpolating them)")
                continue
            logger.debug(f"Detecting blinks for eye {eye}")
            
            # the new blinks replace the old ones, which must not mask the data
            obj.set_blinks(None, eyes=[eye], variables=["pupil"])

            # pupil data: unusable samples are NaN, unless the existing mask
            # is ignored
            pupil_data = obj.get_values(eye, "pupil", "raw" if ignore_existing_mask else "strict")
            
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
            filtered_blinks = [(int(on)-margin_ix[0], int(off)+1+margin_ix[1])
                               for (on,off) in blinks
                               if off+1-on>=min_duration_ix
                               and (max_duration_ix is None or off+1-on<=max_duration_ix)]
            # blinks are cut at excluded periods (e.g., breaks without data
            # are not blinks); blinks next to them are kept. Going through
            # the mask also clips extended blinks to the data and merges
            # the ones that overlap.
            excl = obj._excluded_mask()
            if excl.any() or any(margin_ix):
                inblink = Intervals(filtered_blinks, "samples").to_mask(obj).astype(bool)
                filtered_blinks = Intervals.from_mask(inblink & ~excl).intervals
            logger.debug(f"Filtered {len(filtered_blinks)} blinks by duration")
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
        (see :meth:`~pypillometry.EyeData.set_blinks`); read them with
        :meth:`~pypillometry.EyeData.get_blinks`.
        
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
        :class:`~pypillometry.EyeData`
            the object with the merged blinks (`self` or a copy, depending
            on `inplace`)
        """
        self._d._require("pupil")
        obj = self._d._get_inplace(inplace)
        pairs = obj.data.select(eyes, "pupil")

        distance_ix = obj._samples(distance, units)

        logger.debug(f"Distance '{distance}' in index units: {distance_ix}")

        for eye, var in pairs:
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

    def _transplant(self, obj, eye, window, merge_distance, min_duration, max_duration,
                    pad, cores=None, **kwargs):
        """:func:`~pypillometry.signal.transplant.transplant_blinks` for the
        pupil of `eye` in `obj` (parameters in ms)"""
        if obj.data.get_layer(f"{eye}_pupil", "interpolated").any():
            raise ValueError(f"{eye}_pupil has interpolated samples; transplant blinks "
                             "before interpolating them")
        window_ix = obj._samples(window, rounding="up")
        if np.ndim(window_ix) == 0:
            window_ix = (window_ix, window_ix)
        # no blinks from or to samples that are unusable for other reasons
        # (other masks, excluded periods), and none to the detected blinks
        # (they are larger than their cores or have none)
        key = f"{eye}_pupil"
        other = obj.data.get_policy_mask(key, _without_layer("strict", "blink"))
        return transplant.transplant_blinks(
            obj.data[eye, "pupil"], window_ix,
            merge_distance=obj._samples(merge_distance),
            min_len=max(1, obj._samples(min_duration)),
            max_len=None if max_duration is None else obj._samples(max_duration),
            pad=obj._samples(pad, rounding="up"),
            missing=other & ~np.isnan(obj.data[key]), avoid=obj.data.get_layer(key, "blink"),
            cores=cores, **kwargs)

    @step
    @keephistory
    def blinks_transplant(self, eyes=[], window=(200, 500), n=None,
                          merge_distance: float=100, min_duration: float=20,
                          max_duration: float=1000, pad: float=100,
                          mode: str="additive", shrink: bool=True, blink_val: float=0,
                          original_as: str="pupilorig", seed=None, inplace=None):
        """
        Transplant the blinks of the pupil signal into its clean stretches.

        This gives data with realistic blinks at places where the signal
        without the blink is known, e.g. to evaluate the interpolation of
        blinks (see :meth:`blinks_transplant_error`) or to tune its
        parameters (see :meth:`blinks_autotune`).

        The blinks are the runs of missing samples (NaN or `blink_val`) of
        at least `min_duration`, the cores of the blinks; the detected
        blinks are not used. Runs that are closer than `merge_distance` are
        one blink. Each blink that is surrounded by usable signal is cut
        out with the `window` around its core, and a slow baseline (the
        line through both ends of the window) is removed. What remains,
        the artifact of the blink, is added at random places where the
        signal is clean: no missing or masked samples (under the
        ``"strict"`` mask policy) in the `window` around them. See
        :func:`~pypillometry.signal.transplant.transplant_blinks`.

        In the returned object, the pupil has the transplanted blinks in
        addition to its own. Their cores are added to the blinks (the
        ``"blink"`` layer), so :meth:`blinks_interpolate` treats them like
        the others. The windows of the transplanted blinks, with `pad`
        around them, get the layer ``"transplant"`` (see
        :meth:`~pypillometry.EyeData.get_layer_intervals`), which does not
        mask them. The signal without the transplanted blinks is kept as
        another variable (`original_as`).

        Parameters
        ----------
        eyes : str or list
            eyes to process; if empty, all eyes with pupil data
        window : float, str, Quantity or tuple
            time `(before, after)` the core of a blink (in ms, or with
            units) that belongs to the blink and is transplanted with it;
            should be generous, so that it includes the whole artifact
        n : int, optional
            number of blinks to transplant (per eye); default (and at
            most): as many as fit
        merge_distance : float, str or Quantity
            missing samples that are closer than this (in ms, or with
            units) are one blink
        min_duration, max_duration : float, str or Quantity
            minimum and maximum duration (in ms, or with units) of the
            cores of the blinks that are transplanted; `max_duration=None`:
            no limit
        pad : float, str or Quantity
            clean signal (in ms, or with units) that is kept free on each
            side of the window of a transplanted blink
        mode : str
            ``"additive"`` (default): the difference to the baseline is
            transplanted; ``"multiplicative"``: the ratio to the baseline
            (e.g., an eyelid that covers a part of the pupil), which needs
            a positive signal
        shrink : bool
            remove the fluctuations of the signal around the original
            blinks from their artifacts (default; see
            :func:`~pypillometry.signal.transplant.shrink_blink_artifacts`)
        blink_val : float
            "missing value" code in addition to NaN (default 0)
        original_as : str or None
            name of the variable that keeps the pupil without the
            transplanted blinks (default ``"pupilorig"``, i.e.
            ``"left_pupilorig"`` etc.); `None`: do not keep it
        seed : int, optional
            seed of the random numbers (which blink goes where)
        inplace : bool or None
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes
            if `None`, use the object-level setting

        Returns
        -------
        EyeData
            the object with the transplanted blinks

        Raises
        ------
        ValueError
            if the pupil has interpolated samples (transplant the blinks
            before interpolating them)

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short").pupil.blinks_detect()
        >>> dt = d.pupil.blinks_transplant(n=10, seed=1, inplace=False)
        >>> len(dt.get_layer_intervals("transplant", "left", "pupil")["left_pupil"])
        10
        >>> dt = dt.pupil.blinks_interpolate()
        >>> err = dt.pupil.blinks_transplant_error()
        >>> list(err)
        ['left_pupil', 'right_pupil']
        """
        self._d._require("pupil")
        obj = self._d._get_inplace(inplace)
        rng = np.random.default_rng(seed)
        for eye in self._d._eyes_with(eyes, "pupil"):
            key = f"{eye}_pupil"
            tr = self._transplant(obj, eye, window, merge_distance, min_duration,
                                  max_duration, pad, n=n, mode=mode, shrink=shrink,
                                  blink_val=blink_val, seed=rng)
            logger.info(f"{key}: transplanted {len(tr)} blinks ({len(tr.donors)} different ones)")
            if original_as is not None:
                obj.data._set_channel(f"{eye}_{original_as}", obj.data[key].copy(),
                                      obj.data.layers[key].copy())
            obj.data.set_with_mask(key, tr.signal, preserve_mask=True)
            n_samples = len(tr.signal)
            obj.data.set_layer(key, "blink", transplant._mask(tr.blinks, n_samples))
            obj.data.set_layer(key, "transplant", transplant._mask(tr.sites, n_samples),
                               replace=True)
        return obj

    def blinks_transplant_error(self, eyes=[], original: str="pupilorig",
                                per_blink: bool=False, policy="default"):
        """
        Error of the reconstruction of transplanted blinks.

        Root mean squared difference between the pupil and the pupil
        without the transplanted blinks (see :meth:`blinks_transplant`), over
        all samples around the transplanted blinks (the ``"transplant"``
        layer): the missing samples, the artifacts around them and the
        clean samples next to them. Samples that are unusable under the
        mask `policy` (e.g., blinks that were not interpolated) are
        interpolated linearly first.

        Parameters
        ----------
        eyes : str or list
            eyes to process; if empty, all eyes with pupil data
        original : str
            name of the variable with the pupil without the transplanted
            blinks (`original_as` of :meth:`blinks_transplant`)
        per_blink : bool
            return the error of each transplanted blink instead of the
            error over all of them
        policy : str, MaskPolicy or list of str
            mask policy that decides which samples are usable (see
            :meth:`~pypillometry.EyeData.mask`); default ``"default"``

        Returns
        -------
        dict
            the error for each eye (keys like ``"left_pupil"``): a number,
            or with `per_blink` an array (in the order of
            ``d.get_layer_intervals("transplant")``)

        Raises
        ------
        ValueError
            if there are no transplanted blinks
        """
        self._d._require("pupil")
        result = {}
        for eye in self._d._eyes_with(eyes, "pupil"):
            key = f"{eye}_pupil"
            if f"{eye}_{original}" not in self._d.data:
                raise ValueError(f"no variable {eye}_{original}; transplant blinks with "
                                 "blinks_transplant() first")
            sites = self._d._layer_runs(key, "transplant").intervals
            if len(sites) == 0:
                raise ValueError(f"{key} has no transplanted blinks")
            result[key] = transplant.transplant_error(
                self._d.get_values(eye, "pupil", policy), self._d.data[eye, original],
                sites, per_site=per_blink)
        return result

    def blinks_autotune(self, eyes=[], method: str="mahot", window=(200, 500), n=None,
                        test_fraction: float=1/3, start=None, bounds=None, fixed=None,
                        merge_distance: float=100, min_duration: float=20,
                        max_duration: float=1000, pad: float=100,
                        mode: str="additive", shrink: bool=True, blink_val: float=0,
                        seed=None, **kwargs):
        """
        Tune the parameters of :meth:`blinks_interpolate` for this recording.

        The parameters of the ``"mahot"`` method (`winsize`, `vel_onset`,
        `vel_offset`, `margin` and `blinkwindow`) decide how much of the
        signal around the missing samples of a blink is replaced. Instead
        of setting them by eye, they are tuned on transplanted blinks (see
        :meth:`blinks_transplant`): the blinks of the recording are added
        to its clean stretches, where the signal without the blink is
        known, and the parameters that reconstruct it best are searched
        (see :func:`~pypillometry.signal.autotune.tune_blinks_mahot`).
        The error is the difference between the reconstructed and the
        clean signal in the `window` around the transplanted blinks (with
        `pad`): the root mean squared error of each blink, averaged over
        the blinks.

        A part of the blinks (`test_fraction`) is not used for tuning:
        they are transplanted separately, and the tuned parameters are
        only returned if they are better for them than the starting
        parameters (otherwise those are returned, with a warning). The
        blinks of all `eyes` are used together; call the method for one
        eye at a time to get parameters for each.

        The object is not changed and does not need detected blinks: the
        blinks that are transplanted are the runs of missing samples. The
        result depends on the random places of the transplanted blinks
        (see `seed`).

        Parameters
        ----------
        eyes : str or list
            eyes to use; if empty, all eyes with pupil data
        method : str
            method of :meth:`blinks_interpolate` to tune; only ``"mahot"``
        window : float, str, Quantity or tuple
            time `(before, after)` the missing samples of a blink (in ms,
            or with units) that is transplanted with it; should be
            generous, so that it includes the whole artifact. It also
            limits the search ranges (see `bounds`).
        n : int, optional
            number of blinks to transplant per eye, for tuning and for
            testing each; default (and at most): as many as fit
        test_fraction : float
            fraction of the blinks that is only used for testing the
            tuned parameters (0: none)
        start : dict, optional
            starting parameters, as the arguments of
            :meth:`blinks_interpolate` (e.g., ``dict(vel_onset=-10)``);
            default: its defaults. Velocity thresholds in percent are
            not supported.
        bounds : dict, optional
            search range `(lower, upper)` for ``"winsize"``,
            ``"vel_onset"``, ``"vel_offset"``, ``"margin_before"``,
            ``"margin_after"`` or ``"blinkwindow"`` (units as in
            :meth:`blinks_interpolate`); default: from the data (see
            :func:`~pypillometry.signal.autotune.tune_blinks_mahot`)
        fixed : dict, optional
            parameters that are not tuned, with their values, as the
            arguments of :meth:`blinks_interpolate` (e.g.,
            ``dict(blinkwindow=500)``)
        merge_distance, min_duration, max_duration, pad, mode, shrink, blink_val
            how the blinks are transplanted, see :meth:`blinks_transplant`
        seed : int, optional
            seed of the random numbers (places of the transplanted
            blinks, blinks used for testing, search)
        **kwargs
            further arguments of
            :func:`~pypillometry.signal.autotune.tune_blinks_mahot`
            (`n_init`, `n_grid`, `sweeps`)

        Returns
        -------
        BlinkTuneResult
            ``result.params`` are the parameters to use, as arguments of
            :meth:`blinks_interpolate`; the result also has the errors of
            the tuned and the starting parameters and all evaluated
            parameters (see
            :class:`~pypillometry.signal.autotune.BlinkTuneResult`)

        Examples
        --------
        >>> import pypillometry as pp
        >>> d = pp.get_example_data("rlmw_002_short").pupil.blinks_detect()
        >>> res = d.pupil.blinks_autotune(seed=1)  # doctest: +SKIP
        >>> d = d.pupil.blinks_interpolate(**res.params)  # doctest: +SKIP
        """
        if method != "mahot":
            raise ValueError(f"cannot tune method {method!r}; only 'mahot' has "
                             "parameters to tune")
        d = self._d
        d._require("pupil")
        eyes = d._eyes_with(eyes, "pupil")
        if not 0 <= test_fraction < 1:
            raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
        rng = np.random.default_rng(seed)
        per_ms = 1000 / d.fs    # ms per sample

        def in_samples(pars, what):
            """parameters of blinks_interpolate() (ms) in samples"""
            pars = dict(pars)
            for name in ("vel_onset", "vel_offset"):
                if isinstance(pars.get(name, 0), str):
                    raise ValueError(f"{what}: {name} in percent is not supported")
            names = {"winsize", "vel_onset", "vel_offset", "margin", "blinkwindow"}
            if what == "bounds":
                names = (names - {"margin"}) | {"margin_before", "margin_after"}
            if set(pars) - names:
                raise ValueError(f"{what}: unknown parameter(s) {', '.join(set(pars) - names)}; "
                                 f"use {', '.join(sorted(names))}")
            out = {}
            for name, value in pars.items():
                if name.startswith("vel_"):
                    out[name] = tuple(np.asarray(value, dtype=float) * per_ms) \
                        if np.ndim(value) else float(value) * per_ms
                elif name == "winsize" and what != "bounds":
                    w = d._samples(value, rounding="up")
                    out["smooth_winsize"] = w + (w % 2 == 0)
                else:
                    rounding = "down" if name == "blinkwindow" and what != "bounds" else "up"
                    out["smooth_winsize" if name == "winsize" else name] = \
                        d._samples(value, rounding=rounding)
            return out

        def in_ms(pars):
            """parameters in samples as arguments of blinks_interpolate()"""
            if pars is None:
                return None
            out = {}
            for name, value in pars.items():
                factor = 1 / per_ms if name.startswith("vel_") else per_ms
                value = tuple(float(v) for v in np.asarray(value) * factor) \
                    if np.ndim(value) else float(value * factor)
                out["winsize" if name == "smooth_winsize" else name] = value
            return out

        start_pars = dict(_INTERP_DEFAULTS["mahot"], blinkwindow=500)
        del start_pars["interp_type"]
        start_pars.update(start or {})

        # transplant the blinks of each eye: some for tuning, the others for testing
        train, test = [], []
        targs = dict(window=window, merge_distance=merge_distance, min_duration=min_duration,
                     max_duration=max_duration, pad=pad, mode=mode, shrink=shrink,
                     blink_val=blink_val)
        for eye in eyes:
            donors = self._transplant(d, eye, n=0, seed=rng, **targs).donors
            perm = rng.permutation(len(donors))
            n_test = int(round(test_fraction * len(donors)))
            if n_test > 0:
                test.append(self._transplant(d, eye, n=n, seed=rng,
                                             cores=donors[np.sort(perm[:n_test])], **targs))
            train.append(self._transplant(d, eye, n=n, seed=rng,
                                          cores=donors[np.sort(perm[n_test:])], **targs))
            logger.debug(f"{eye}_pupil: {len(donors)} blinks to transplant, "
                         f"{n_test} of them for testing")
        if sum(len(tr) for tr in train) == 0:
            raise ValueError("no blinks could be transplanted: the recording needs blinks "
                             "(missing samples) that are surrounded by signal, and clean "
                             "stretches (longer than the window) to put them")

        res = autotune.tune_blinks_mahot(
            train, test, start=in_samples(start_pars, "start"),
            bounds=None if bounds is None else in_samples(bounds, "bounds"),
            fixed=None if fixed is None else in_samples(fixed, "fixed"), seed=rng, **kwargs)

        # in the units of blinks_interpolate()
        hist = {("winsize" if k == "smooth_winsize" else k):
                (v if k == "loss" else v / per_ms if k.startswith("vel_") else v * per_ms)
                for k, v in res.history.items()}
        res = replace(res, params=in_ms(res.params), tuned=in_ms(res.tuned),
                      start=in_ms(res.start), bounds=in_ms(res.bounds), history=hist,
                      units="ms")
        logger.info(f"Tuned the blink interpolation on {res.n_blinks} transplanted blinks: "
                    f"error {res.loss_start:.4g} -> {res.loss:.4g}"
                    + (f" ({res.test_loss_start:.4g} -> {res.test_loss:.4g} on "
                       f"{res.n_test} test blinks)" if res.n_test else ""))
        return res

    def _interpolate_kalman(self, obj, eye, blinkwindow_ix, optimize):
        """blinks of `eye` in `obj` interpolated with
        :func:`~pypillometry.signal.preproc.interpolate_blinks_kalman`;
        returns the signal and the interpolated samples"""
        bls = obj.get_blinks(eye, "pupil", units="samples")[f"{eye}_pupil"]
        excl = obj._excluded_mask()
        blinks = Intervals.from_mask(bls.to_mask(obj).astype(bool) & ~excl).intervals
        # unusable samples (other masks, excluded periods) are no observations
        missing = excl | np.isnan(obj.get_values(eye, "pupil", "strict"))
        syr, interpolated, model = preproc.interpolate_blinks_kalman(
            obj.data[eye, "pupil"], blinks, obj.fs, blinkwindow_ix, missing=missing,
            optimize=optimize)
        if model is not None:
            pars = ", ".join(f"{k}={v:.4g}" for k, v in model.params.items())
            (logger.info if optimize else logger.debug)(
                f"{eye}_pupil: Kalman interpolation model {pars}")
        return syr, interpolated

    def _interpolate_mahot(self, obj, eye, blinkwindow_ix, winsize, vel_onset, vel_offset,
                           margin, interp_type, max_duration):
        """blinks of `eye` in `obj` interpolated with
        :func:`~pypillometry.signal.preproc.interpolate_blinks_mahot`;
        returns the signal and the interpolated samples"""
        # parameters in sampling units (from ms)
        winsize_ix=self._d._samples(winsize, rounding="up")
        margin_ix=self._d._samples(margin, rounding="up")
        max_ix = None if max_duration is None else self._d._samples(max_duration)
        if winsize_ix % 2==0: ## ensure smoothing window is odd
            winsize_ix+=1 
        bls = self._d.get_blinks(eye, "pupil", units="samples")[f"{eye}_pupil"]
        
        # Use raw data for velocity computation - blink_onsets_mahot handles NaN/zeros internally
        # Don't use mask-applied data here because the mask includes blink regions,
        # which would smooth out the velocity transitions we need to detect boundaries
        pupil_data = obj.data[eye,"pupil"]
        
        # Convert velocity thresholds to per-sample units
        vel_onset_samples, vel_offset_samples = self._convert_velocity_thresholds(
            pupil_data, winsize_ix, vel_onset, vel_offset
        )
        syr, interpolated, skipped = preproc.interpolate_blinks_mahot(
            pupil_data, bls, winsize_ix, vel_onset_samples, vel_offset_samples,
            margin_ix, blinkwindow_ix, interp_type=interp_type, tx=obj.tx,
            missing=obj._excluded_mask(), max_duration=max_ix)
        for ix, (t1, t2, t3, t4), reason in skipped:
            logger.warning(f"Skipping blink {ix} for {eye}: {reason} "
                           f"(t1={t1}, t2={t2}, t3={t3}, t4={t4})")
        return syr, interpolated

    @step
    @keephistory
    def blinks_interpolate(self, eyes: str|list=[],
                                 store_as: str="pupil",
                                 method: str="mahot",
                                 blinkwindow: float=500,
                                 inplace=None, **kwargs):
        """Interpolating blinks in the pupillary signal.

        The blinks to interpolate are taken from the blinks stored in the object,
        so blinks have to be detected first (e.g., using :meth:`blinks_detect`).
        The interpolated samples get the ``"interpolated"`` layer (on the
        variable they are stored in, see `store_as`): they are used under the
        default mask policy and masked under the ``"strict"`` one (see
        :meth:`~pypillometry.EyeData.mask`); the interpolated blinks
        can be retrieved with
        :meth:`~pypillometry.EyeData.get_interpolated`.

        Methods:

        - ``"mahot"`` (default): the method by Mahot (2013). The on- and
          offsets of each blink are refined using the velocity profile, and
          the blink is replaced by a cubic curve through four anchor points
          around it. Blinks that are so close that the anchor points of one
          fall into the other are merged and interpolated together.
          This procedure relies heavily on eye-balling (reconstructing
          visually convincing signal), so it can help to inspect the result
          visually (e.g., using
          :meth:`d.plot.plot_blinks() <pypillometry.plot.PlotNamespace.plot_blinks>`)
          to set good parameter values for `winsize`, `vel_onset`,
          `vel_offset` and `margin` (see paper linked below).
        - ``"kalman"``: the blink samples are replaced by their expected
          value given the signal around them (Kalman smoother). Around each
          blink, `blinkwindow` before and after it is used, after removing a
          linear trend; the signal is modelled as a smooth Gauss-Markov
          process (a fast and a slow Matérn process, smoothness 3/2) plus
          measurement noise. The time scales of the components are fixed
          and the noise level is estimated from the data (with
          ``optimize=True``, all parameters are fitted, which is much
          slower). Exactly the
          stored blinks are replaced, so they should include the transients
          around the blinks (see the `margin` of :meth:`blinks_detect`).

        Each method has its own parameters (see below), passed as keyword
        arguments. Parameters of another method are ignored with a warning,
        so that the method can be switched quickly.

        Parameters
        ----------
        eyes : str or list
            str or list of eyes to process; if empty, all available eyes are processed
        store_as : str
            how to store the interpolated data; either "pupil" (default) which replaces
            the original pupil data or a string that will be used as the new variable
            name in the data (e.g., "pupilinterp")
        method : str
            method to use: ``"mahot"`` (default) or ``"kalman"``
        blinkwindow : float, str or Quantity
            how much time before and after each blink to use (in ms, or with
            units); for ``"mahot"``, the window in which the on- and offsets
            are searched; for ``"kalman"``, the signal the blink is
            predicted from
        inplace : bool or None
            if `True`, make change in-place and return the object
            if `False`, make and return copy before making changes
            if `None`, use the object-level setting
        **kwargs
            parameters of the method (see below)

        Other Parameters
        ----------------
        winsize : float, str or Quantity
            ``"mahot"``: size of the Savitzky-Golay window (in ms, or with
            units); default 25
        vel_onset : float or str
            ``"mahot"``: velocity threshold to detect blink onset (negative
            value); default -5.0

            - If float: threshold in pupil-units per millisecond (e.g., -5.0)
            - If str: percentile of velocity distribution (e.g., "5%" for bottom 5%)
        vel_offset : float or str
            ``"mahot"``: velocity threshold to detect blink offset (positive
            value); default 5.0

            - If float: threshold in pupil-units per millisecond (e.g., 5.0)
            - If str: percentile of velocity distribution (e.g., "95%" for top 5%)
        margin : Tuple[float,float] or Quantity
            ``"mahot"``: margin that is subtracted/added to onset and offset
            (in ms, or with units, e.g. ``(10, 30) * u.ms``); default
            ``(10, 30)``
        interp_type : str
            ``"mahot"``: type of interpolation accepted by
            :func:`scipy.interpolate.interp1d()`; default ``"cubic"``. A
            blink whose anchor points are close on one side (because of a
            neighbouring blink) gets a monotone curve
            (:class:`scipy.interpolate.PchipInterpolator`) instead of the
            cubic, which would overshoot.
        max_duration : float, str, Quantity or None
            ``"mahot"``: maximum duration (in ms, or with units) of a stretch
            that is interpolated; blinks that are longer (with their margins,
            or after merging with neighbouring blinks whose anchor points
            overlap) are skipped with a warning; default 2000, `None`: no
            limit
        optimize : bool
            ``"kalman"``: fit all parameters of the model to the data
            instead of using the defaults; much slower, and usually changes
            the result little; default False

        Returns
        -------
        EyeData
            the object with the interpolated pupil data

        Raises
        ------
        ValueError
            if `method` is unknown
        TypeError
            if a keyword argument is not a parameter of any method

        References
        ----------
        Mahot, 2013:
        https://figshare.com/articles/A_simple_way_to_reconstruct_pupil_size_during_eye_blinks/688001.
        """
        pars = _interp_params(method, kwargs)
        self._d._require("pupil")
        obj = self._d._get_inplace(inplace)
        eyes=self._d._eyes_with(eyes, "pupil")
        blinkwindow_ix=self._d._samples(blinkwindow)

        for eye in eyes:
            logger.debug(f"Interpolating blinks for eye {eye} ({method})")
            if method == "kalman":
                syr, interpolated = self._interpolate_kalman(obj, eye, blinkwindow_ix, **pars)
            else:
                syr, interpolated = self._interpolate_mahot(obj, eye, blinkwindow_ix, **pars)

            # store interpolated data (with the masks and layers of the
            # pupil) and set the interpolated layer
            key, target = f"{eye}_pupil", f"{eye}_{store_as}"
            if target == key:
                obj.data.set_with_mask(key, syr, preserve_mask=True)
            else:
                obj.data._set_channel(target, syr, obj.data.layers[key])
            obj.data.set_layer(target, "interpolated", interpolated)
            logger.debug(f"Interpolated {interpolated.sum()} samples for {eye}")

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
        :meth:`~pypillometry.EyeData.mask`). The unusable samples
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
            :meth:`~pypillometry.EyeData.mask`); default ``"default"``
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
        from scipy import interpolate
        self._d._require(variable)
        obj = self._d._get_inplace(inplace)
        if not isinstance(variable, str):
            logger.warning("variable must be a string; using default 'pupil'")
            variable="pupil"
        eyes=self._d._eyes_with(eyes, variable)

        for eye in eyes:
            logger.debug("Estimating baseline for eye %s"%eye)
            sy = self._d.get_values(eye, variable, policy)
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
        :meth:`~pypillometry.EyeData.exclude_intervals`, e.g., breaks)
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
            :meth:`~pypillometry.EyeData.mask`); default ``"default"``
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
            if no experimental setup is set (see :meth:`~pypillometry.EyeData.set_experimental_setup`)
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
        from scipy import optimize
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
