"""
preproc.py
==========

preprocessing functions (blinks, smoothing, missing data...)
"""
import numpy as np
from loguru import logger

__all__ = [
    "smooth_window", "velocity_savgol", "detect_blinks_velocity",
    "detect_blinks_zero", "blink_onsets_mahot", "interpolate_blinks_mahot",
    "interpolate_blinks_kalman", "zscale",
]


def smooth_window(x, window_len=11, window='hanning', direction='center'):
    """
    Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The direction parameter controls whether the smoothing uses past, future, 
    or both samples relative to each point. 

    Note: If using backward or forward smoothing, the smoothed signal will be shifted 
    by half the window size. This is intentional for onset and offset detection, 
    where the smoothed signal is used to detect the onset and offset of the blink.
    
    Adapted from SciPy Cookbook: `<https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html>`_.
    
    Parameters
    ----------
    x : np.ndarray
        The input signal
    window_len : int
        The dimension of the smoothing window; should be an odd integer
    window : str
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
        'flat' window will produce a moving average smoothing.
    direction : str
        - 'center': symmetric smoothing using both past and future samples (default)
        - 'backward': only uses current and past samples (useful for onset detection
          before missing data, where future NaN shouldn't affect the estimate)
        - 'forward': only uses current and future samples (useful for offset detection
          after missing data, where past NaN shouldn't affect the estimate)

    Returns
    -------
    np.ndarray
        The smoothed signal (same length as input)
        
    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> signal = np.sin(np.linspace(0, 10, 200)) + rng.normal(0, 0.1, 200)
    >>> # Standard centered smoothing
    >>> smoothed = smooth_window(signal, window_len=11, window='hanning')
    >>> smoothed.shape
    (200,)
    >>> # Backward-looking for onset detection
    >>> smoothed_back = smooth_window(signal, window_len=11, direction='backward')
    >>> # Forward-looking for offset detection
    >>> smoothed_fwd = smooth_window(signal, window_len=11, direction='forward')
    """
    x = np.asarray(x, dtype=float)
    
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    
    if np.any(np.isnan(x)):
        raise ValueError("Input signal contains NaN values. Handle missing data before smoothing "
                        "(e.g., interpolate or use a masked approach).")

    if window_len < 3:
        return x.copy()

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window should be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    
    if direction not in ['center', 'backward', 'forward']:
        raise ValueError("direction should be one of 'center', 'backward', 'forward'")

    # Create window weights
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    w = w / w.sum()

    if direction == 'center':
        # Symmetric padding with reflected signal
        s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
        y = np.convolve(w, s, mode='same')
        return y[(window_len-1):(-window_len+1)]
    
    elif direction == 'backward':
        # Backward-looking: y[i] depends on x[i-window_len+1:i+1] (current and past)
        # This causes a phase lag, which is intentional for onset detection
        padded = np.r_[np.full(window_len-1, x[0]), x]
        y = np.convolve(w, padded, mode='valid')
        return y
    
    elif direction == 'forward':
        # Forward-looking: y[i] depends on x[i:i+window_len] (current and future)
        # This causes a phase lead, which is intentional for offset detection
        padded = np.r_[x, np.full(window_len-1, x[-1])]
        y = np.convolve(w, padded, mode='valid')
        return y


def zscale(y):
    """
    Z-transform an array, ignoring NaNs.

    Parameters
    ----------
    y : np.ndarray
        input array

    Returns
    -------
    np.ndarray
        `y` with mean subtracted and divided by its standard deviation
        (both computed with :func:`numpy.nanmean`/:func:`numpy.nanstd`);
        NaNs are kept
    """
    return (y-np.nanmean(y))/np.nanstd(y)


def velocity_savgol(x, window_len=11, polyorder=2, direction='center'):
    """
    Compute velocity (first derivative) using Savitzky-Golay filter.
    
    This computes velocity by fitting a polynomial to a window of samples
    and evaluating its derivative, which is more robust than simple differencing.
    The direction parameter controls whether the velocity estimate uses past, 
    future, or both samples relative to each point.
    
    Parameters
    ----------
    x : np.ndarray
        The input signal
    window_len : int
        The dimension of the smoothing window; should be an odd integer.
        Must be greater than polyorder.
    polyorder : int
        Order of the polynomial to fit. Default is 2 (quadratic).
    direction : str
        'center' - symmetric, uses both past and future samples (default)
        'backward' - only uses current and past samples (useful for onset detection)
        'forward' - only uses current and future samples (useful for offset detection)
        
    Returns
    -------
    np.ndarray
        The velocity (first derivative) in units per sample (same length as input)
        
    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> signal = np.sin(np.linspace(0, 10, 200)) + rng.normal(0, 0.1, 200)
    >>> # Standard centered velocity
    >>> vel = velocity_savgol(signal, window_len=11)
    >>> vel.shape
    (200,)
    >>> # Backward-looking for onset detection
    >>> vel_back = velocity_savgol(signal, window_len=11, direction='backward')
    >>> # Forward-looking for offset detection
    >>> vel_fwd = velocity_savgol(signal, window_len=11, direction='forward')
    """
    from scipy.signal import savgol_coeffs
    from scipy.ndimage import convolve1d
    x = np.asarray(x, dtype=float)
    
    if x.ndim != 1:
        raise ValueError("velocity_savgol only accepts 1 dimension arrays.")
    
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    
    # Ensure minimum window size for Savitzky-Golay (must be > polyorder and >= 3)
    min_window = max(3, polyorder + 1)
    if window_len < min_window:
        logger.warning(f"window_len={window_len} too small for polyorder={polyorder}, increasing to {min_window}")
        window_len = min_window
        
    # Ensure window_len is odd
    if window_len % 2 == 0:
        window_len += 1
        
    if direction not in ['center', 'backward', 'forward']:
        raise ValueError("direction should be one of 'center', 'backward', 'forward'")
    
    # Set position parameter based on direction
    # pos is the index within the window where the output value is evaluated
    if direction == 'center':
        pos = window_len // 2  # center of window (default)
    elif direction == 'backward':
        pos = window_len - 1   # rightmost position (current sample at end of window)
    else:  # forward
        pos = 0                # leftmost position (current sample at start of window)
    
    # Get Savitzky-Golay coefficients for first derivative
    coeffs = savgol_coeffs(window_len, polyorder, deriv=1, pos=pos)
    
    # Apply filter using convolution with appropriate boundary handling
    # Use 'nearest' mode to extend boundary values
    vel = convolve1d(x, coeffs, mode='nearest')
    
    return vel


def helper_merge_blinks(b1,b2):
    """
    Merge two sets of blink intervals into one set of non-overlapping intervals.

    Intervals from `b1` and `b2` that overlap or touch (onset of one
    `<=` offset of the previous one) are combined into a single interval.

    Parameters
    ----------
    b1 : np.ndarray
        blink onset/offset indices (nblinks x 2); may be empty
    b2 : np.ndarray
        blink onset/offset indices (nblinks x 2); may be empty

    Returns
    -------
    np.ndarray
        merged blink onset/offset indices (nblinks x 2), sorted by onset;
        if one of the inputs is empty, the other one is returned unchanged
    """
    if b1.size==0:
        return b2
    elif b2.size==0:
        return b1
    on=np.sort(np.concatenate( (b1[:,0], b2[:,0]) ))
    off=np.sort(np.concatenate( (b1[:,1], b2[:,1]) ))
    b=np.vstack((on,off)).T

    newb=[]
    on,off=b[0,:]
    for i in range(1,b.shape[0]):
        if b[i,0]<=off:
            # absorb onset from next 
            off=max(off,b[i,1])
        else:
            newb.append([on,off])
            on,off=b[i,:]
    off=b[-1,1]
    newb.append([on,off])
    return np.array(newb)

def detect_blinks_velocity(sy, smooth_winsize, vel_onset, vel_offset, min_onset_len=5, min_offset_len=5, min_blink_samples=1,
                           max_blink_samples=None):
    """
    Detect blinks as everything between a fast downward and a fast upward-trending PD-changes.

    Each offset is paired with the last onset before it: an onset that is
    followed by another onset before the next offset has no offset of its own
    and is dropped.

    Uses asymmetric Savitzky-Golay velocity estimation to handle NaN values properly:
    - Backward-looking velocity for onset detection (so future NaN doesn't affect the estimate)
    - Forward-looking velocity for offset detection (so past NaN doesn't affect the estimate)
    
    This works similarly to :py:func:`blink_onsets_mahot()`.
    
    Parameters
    ----------
    sy : np.ndarray
        pupil data (can contain NaN or zeros for missing values)
    smooth_winsize : int
        size of the Savitzky-Golay window in samples (odd)
    vel_onset : float
        velocity-threshold to detect the onset of the blink; in units per sample (negative value)
    vel_offset : float
        velocity-threshold to detect the offset of the blink; in units per sample (positive value)
    min_onset_len : int
        number of consecutive samples that must cross the threshold to detect an onset
        (runs must have at least `min_onset_len` samples)
    min_offset_len : int
        number of consecutive samples that must cross the threshold to detect an offset
        (runs must have at least `min_offset_len` samples)
    min_blink_samples : int
        minimum duration of a blink in samples; offsets that would create shorter intervals
        are skipped during matching (to avoid spurious short offsets stealing real onsets)
    max_blink_samples : int or None
        maximum duration of a blink in samples; an onset without an offset
        within this many samples is not a blink and is dropped (otherwise it
        would be paired with an arbitrarily distant offset); `None` means no
        limit

    Returns
    -------
    np.ndarray
        array (nblinks x 2) of blink onset/offset sample indices (both
        inclusive); the offset is the last sample of the matching offset run,
        or the last sample of `sy` if no offset was found after the last
        onset (and `max_blink_samples` allows it); empty array if no blinks
        were found
    """
    n = len(sy)
    
    # Track NaN/zero locations (invalid data)
    invalid_mask = np.isnan(sy) | (sy == 0)
    
    # Fill invalid values with linear interpolation for velocity estimation
    # (the asymmetric Savitzky-Golay will prevent NaN from affecting edges)
    sy_filled = sy.copy()
    valid_indices = np.where(~invalid_mask)[0]
    if len(valid_indices) == 0:
        logger.debug("No valid samples - returning empty array")
        return np.array([])
        
    # Simple linear interpolation for NaN regions
    invalid_indices = np.where(invalid_mask)[0]
    if len(invalid_indices) > 0:
        sy_filled[invalid_indices] = np.interp(
            invalid_indices, valid_indices, sy[valid_indices]
        )
    
    # Generate velocity profiles using asymmetric Savitzky-Golay
    # Backward-looking velocity for onset detection (only uses past samples)
    vel_backward = velocity_savgol(sy_filled, smooth_winsize, polyorder=2, direction="backward")
    
    # Forward-looking velocity for offset detection (only uses future samples)  
    vel_forward = velocity_savgol(sy_filled, smooth_winsize, polyorder=2, direction="forward")
    
    # Don't detect inside invalid regions
    vel_backward[invalid_mask] = 0
    vel_forward[invalid_mask] = 0
    
    logger.debug(f"Generated asymmetric velocity profiles, length {n}")

    # Find onset candidates using backward velocity (fast drop before blink)
    onsets = np.where(vel_backward <= vel_onset)[0]
    logger.debug(f"Found {len(onsets)} potential onset points")
    
    if len(onsets) == 0:
        logger.debug("No onsets found - returning empty array")
        return np.array([])
        
    onsets_ixx = np.r_[10, np.diff(onsets)] > 1
    onsets_len = np.diff(np.r_[np.where(onsets_ixx)[0], onsets.size])
    onsets = onsets[onsets_ixx]
    onsets = onsets[onsets_len >= min_onset_len]
    logger.debug(f"After filtering, {len(onsets)} onsets remain")

    if len(onsets) == 0:
        logger.debug("No valid onsets - returning empty array")
        return np.array([])

    # Find offset candidates using forward velocity (fast rise after blink)
    offsets = np.where(vel_forward >= vel_offset)[0]
    logger.debug(f"Found {len(offsets)} potential offset points")

    # Without offsets, all blinks end at the last sample (see below)
    offset_starts = offset_ends = np.array([], dtype=int)
    if offsets.size > 0:
        # Mark starts and ends of runs
        offsets_start_ixx = np.r_[10, np.diff(offsets)] > 1
        offsets_end_ixx = np.r_[np.diff(offsets), 10] > 1
        offsets_len = np.diff(np.r_[np.where(offsets_start_ixx)[0], offsets.size])

        # Filter by minimum length (applied to both starts and ends of runs)
        valid_len_mask = offsets_len >= min_offset_len
        offset_starts = offsets[offsets_start_ixx][valid_len_mask]
        offset_ends = offsets[offsets_end_ixx][valid_len_mask]
    logger.debug(f"After filtering, {len(offset_starts)} offsets remain")

    # Find corresponding on- and off-sets
    # Use offset STARTS for comparison, but offset ENDS as actual values
    # Skip offsets that would create intervals shorter than min_blink_samples
    blinks = []
    on = onsets[0]
    while on is not None:
        # Find offset runs that START after this onset AND create long enough intervals
        valid_offset_mask = (offset_starts > on) & (offset_ends - on >= min_blink_samples)
        if valid_offset_mask.any():
            # Get the first valid offset run and use its END
            first_valid_idx = np.where(valid_offset_mask)[0][0]
            off = offset_ends[first_valid_idx]
            # a later onset before this offset: this onset has no offset of
            # its own (e.g., a dip from an eye movement) and must not take
            # the offset of the next blink
            later = onsets[(onsets > on) & (onsets < offset_starts[first_valid_idx])
                           & (off - onsets >= min_blink_samples)]
            if later.size > 0:
                on = later[0]
                continue
        else:
            off = n - 1
        if max_blink_samples is not None and off - on + 1 > max_blink_samples:
            # no offset close enough: drop this onset, try the next one
            ons = onsets[onsets > on]
            on = ons[0] if ons.size > 0 else None
            continue
        blinks.append([on, off])
        ons = onsets[onsets > off]
        on = ons[0] if ons.size > 0 else None
    logger.debug(f"Found {len(blinks)} blink pairs")
        
    # If onsets fall in an invalid region, grow backward until first valid sample
    # (Don't grow offsets forward - offset_ends already mark the end of recovery)
    blinks2 = []
    for (on, off) in blinks:
        while on > 0 and invalid_mask[on]:
            on -= 1
        blinks2.append([on, off])
    
    result = np.array(blinks2)
    logger.debug(f"Returning {len(result)} blinks after invalid-region adjustment")
    return result

def detect_blinks_zero(sy, min_duration, blink_val=0):
    """
    Detect blinks as consecutive sequence of `blink_val` (e.g., 0 or NaN) of at least
    `min_duration` successive values in the signal `sy`.
    Detected blinks are put a matrix `blinks` (nblinks x 2) where start and end
    are stored as indexes.

    `blink_val=np.nan` matches NaN values; any other value is compared with
    `sy == blink_val`.

    Parameters
    ----------
    sy : np.ndarray
        signal
    min_duration : int
        minimum number of samples for a sequence of missing numbers to be
        treated as blink
    blink_val : float
        "missing value" code, e.g. 0 or `np.nan` (default 0)

    Returns
    -------
    np.ndarray
        array (nblinks x 2) containing the indices of the start/end of the
        blinks (both inclusive); empty array if no blinks were found
    """
    missing = np.isnan(sy) if np.isnan(blink_val) else (sy==blink_val)
    # pad with False so that runs at the start/end also have a start and end
    x=np.diff(np.r_[0, missing.astype(int), 0])
    starts=np.where(x==1)[0]
    ends=np.where(x==-1)[0]-1
    
    logger.debug(f"Found {len(starts)} potential blinks")
    
    blinks=[ [start,end] for start,end in zip(starts,ends) if end-start+1>=min_duration]
    logger.debug(f"Found {len(blinks)} blinks after duration filtering")
    
    return np.array(blinks)
    
def _blink_core(sy, start, end):
    """
    Core of the blink stored as `start`-`end` (end-exclusive): the stretch
    from the first to the last missing (NaN or 0) sample in it, or the
    smallest pupil if it has no missing samples. Returns `(first, last)`
    (both inclusive). It does not depend on how far the stored blink was
    extended (e.g., by the `margin` of the detection).
    """
    seg = sy[start:end]
    missing = np.where(np.isnan(seg) | (seg == 0))[0]
    if missing.size > 0:
        return start + missing[0], start + missing[-1]
    if seg.size == 0:
        return start, max(start, end - 1)
    m = start + int(np.argmin(seg))
    return m, m


def _velocity_runs(cands, max_gap, min_len):
    """
    Runs of the (sorted) sample indices `cands`, as an array (nruns x 2) of
    first/last index (both inclusive). Runs separated by at most `max_gap`
    samples are joined (e.g., a drop that briefly falls below the
    threshold); runs shorter than `min_len` samples are dropped (after
    joining).
    """
    if cands.size == 0:
        return np.zeros((0, 2), dtype=int)
    brk = np.flatnonzero(np.diff(cands) > max_gap + 1)
    first = cands[np.concatenate(([0], brk + 1))]
    last = cands[np.concatenate((brk, [cands.size - 1]))]
    keep = last - first + 1 >= min_len
    return np.stack((first[keep], last[keep]), axis=1)


def blink_onsets_mahot(sy, blinks, smooth_winsize, vel_onset, vel_offset, margin, blinkwindow,
                       min_onset_len=1, min_offset_len=1, max_gap=None):
    """
    Method for finding the on- and offset for each blink (including the transient).
    See https://figshare.com/articles/A_simple_way_to_reconstruct_pupil_size_during_eye_blinks/688001.

    The search starts from the core of each blink: the stretch from its first
    to its last missing (NaN or 0) sample, or its smallest pupil if it has no
    missing samples. The onset is the start of the closest closing transient
    (velocity below `vel_onset`) that starts before the core, the offset the
    end of the closest opening transient (velocity above `vel_offset`) that
    ends after it. The search (and the margins) stop at the neighbouring
    blinks, so that a transient of a neighbour is not taken for one of the
    blink. The result therefore does not depend on how far the
    blinks were extended when they were detected, and for a blink that
    contains several transients (e.g., merged blinks), the outermost ones
    are used.
    
    Uses asymmetric Savitzky-Golay velocity estimation:
    - Backward-looking velocity for onset detection (so future zeros/NaN don't affect the estimate)
    - Forward-looking velocity for offset detection (so past zeros/NaN don't affect the estimate)
    
    Parameters
    ----------
    sy : np.ndarray
        pupil data (can contain NaN or zeros for missing values)
    blinks : np.ndarray or Intervals
        blink start/end sample indices (nblinks x 2, end-exclusive), e.g. an
        :class:`~pypillometry.Intervals` object in index units; anything
        that iterates over `(start, end)` pairs works
    smooth_winsize : int
        size of the Savitzky-Golay window in samples (odd)
    vel_onset : float
        velocity-threshold to detect the onset of the blink; in units per sample (negative value)
    vel_offset : float
        velocity-threshold to detect the offset of the blink; in units per sample (positive value)
    margin : tuple of int
        `(before, after)` margin that is subtracted/added to onset and offset (in samples)
    blinkwindow : int
        how far before and after the core of each blink to search for the
        on-/offset (in samples)
    min_onset_len : int
        minimum number of consecutive samples in an onset run to be considered valid
    min_offset_len : int
        minimum number of consecutive samples in an offset run to be considered valid
    max_gap : int or None
        runs of threshold crossings separated by at most this many samples
        are joined into one transient; `None` (default) uses `smooth_winsize`

    Returns
    -------
    np.ndarray
        array (n x 2) of refined onset/offset sample indices, one row per
        input blink; if no transient is found before (after) the core of a
        blink, the start (end) of the core is used
    """
    # Generate asymmetric velocity profiles
    # Backward-looking for onset detection (doesn't peek into blink)
    vel_backward = velocity_savgol(sy, smooth_winsize, polyorder=2, direction="backward")
    # Forward-looking for offset detection (doesn't peek into blink)
    vel_forward = velocity_savgol(sy, smooth_winsize, polyorder=2, direction="forward")
    if max_gap is None:
        max_gap = smooth_winsize
    n = len(sy)

    pairs = [(int(start), int(end)) for start, end in blinks]
    # the neighbours of each blink: the end of the previous one and the
    # start of the next one (in the order of their starts)
    order = sorted(range(len(pairs)), key=lambda i: pairs[i])
    prev_end, next_start, last = {}, {}, 0
    for k, i in enumerate(order):
        prev_end[i] = last
        last = max(last, pairs[i][1])
        next_start[i] = pairs[order[k + 1]][0] if k + 1 < len(order) else n
    newblinks = []
    for ix, (start, end) in enumerate(pairs):
        cstart, cend = _blink_core(sy, start, end)
        lo, hi = min(prev_end[ix], cstart), max(next_start[ix], cend + 1)
        winstart, winend = max(lo, cstart - blinkwindow), min(cend + blinkwindow + 1, hi, n)

        # onset: closest closing transient that starts before the core
        runs = _velocity_runs(np.where(vel_backward[winstart:winend] <= vel_onset)[0] + winstart,
                              max_gap, min_onset_len)
        runs = runs[runs[:, 0] < cstart]
        if runs.shape[0] > 0:
            onset = runs[-1, 0]
        else:
            # e.g. the signal does not drop before the blink: keep the
            # core instead of dropping the blink
            logger.debug(f"Blink {ix} (samples {start}-{end}): no onset crossing, using blink core")
            onset = cstart
        onset = max(0, lo, onset - margin[0])

        # offset: closest opening transient that ends after the core
        runs = _velocity_runs(np.where(vel_forward[winstart:winend] >= vel_offset)[0] + winstart,
                              max_gap, min_offset_len)
        runs = runs[runs[:, 1] > cend]
        if runs.shape[0] > 0:
            offset = runs[0, 1]
        else:
            logger.debug(f"Blink {ix} (samples {start}-{end}): no offset crossing, using blink core")
            offset = cend
        offset = min(n - 1, hi - 1, offset + margin[1])

        newblinks.append([onset, offset])

    return np.array(newblinks)


def _merge_overlapping_blink_windows(blinks, n):
    """Anchor points of the blinks, merging only blinks that overlap.

    Each blink `(t2, t3)` is interpolated from the anchor points
    ``t1 = t2-(t3-t2)`` and ``t4 = t3+(t3-t2)``. Blinks whose stretches
    `(t2, t3)` overlap or touch are merged (the anchor points are from the
    first and the last of them). If the anchor points of a blink would
    reach into a neighbouring blink, they are moved closer instead: at most
    to the on- or offset of the neighbour (which is not replaced by its
    curve); with anchor points that close, the blink is interpolated with
    a monotone curve instead of the cubic (see `_anchors_close`). Merging these blinks instead would replace the signal between
    them, and the anchor points of the merged blink would reach further
    blinks (chains of several seconds in real data).

    Returns
    -------
    list of tuple
        ``(t1, t2, t3, t4)`` for each (merged) blink
    """
    groups = []     # [t1, t2, t3, t4] before moving the anchor points
    for on, off in sorted((int(b[0]), int(b[1])) for b in blinks):
        t1, t4 = 2 * on - off, 2 * off - on
        if groups and on <= groups[-1][2]:
            groups[-1][2] = max(groups[-1][2], off)
            groups[-1][3] = max(groups[-1][3], t4)
            continue
        groups.append([t1, on, off, t4])
    out = []
    for i, (t1, t2, t3, t4) in enumerate(groups):
        lo = groups[i - 1][2] if i > 0 else 0
        hi = groups[i + 1][1] if i + 1 < len(groups) else n - 1
        out.append((max(t1, lo), t2, t3, min(t4, hi)))
    return out

def _anchors_close(t1, t2, t3, t4):
    """Whether an anchor point of the blink `(t2, t3)` is less than half of
    its length away from it (moved because of a neighbouring blink or the
    end of the data, see `_merge_overlapping_blink_windows`)."""
    return min(t2 - t1, t4 - t3) < (t3 - t2) / 2


def _lagrange(xp, yp, x):
    """Polynomial through the points `(xp, yp)`, evaluated at `x`."""
    out = np.zeros(len(x))
    for i in range(len(xp)):
        w = np.full(len(x), float(yp[i]))
        for j in range(len(xp)):
            if j != i:
                w *= (x - xp[j]) / (xp[i] - xp[j])
        out += w
    return out


def interpolate_blinks_mahot(sy, blinks, smooth_winsize, vel_onset, vel_offset, margin,
                             blinkwindow, interp_type="cubic", tx=None, missing=None,
                             max_duration=None):
    """
    Interpolate blinks with the method by Mahot (2013).

    The on- and offset of each blink (including the transients) are found
    in the velocity profile (see :func:`blink_onsets_mahot`), and the samples
    between them are replaced by a curve through four anchor points: the
    onset `t2`, the offset `t3`, and the points one blink duration before
    and after them (``t1 = t2-(t3-t2)``, ``t4 = t3+(t3-t2)``). Blinks that
    are so close that the anchor points of one fall into the other are
    merged and interpolated together. Blinks that are too short or have an
    unusable anchor point are skipped.

    Parameters
    ----------
    sy : np.ndarray
        pupil data (can contain NaN or zeros for missing values)
    blinks : np.ndarray or Intervals
        blink start/end sample indices (nblinks x 2, end-exclusive); only
        the core of each blink is used (see :func:`blink_onsets_mahot`)
    smooth_winsize, vel_onset, vel_offset, margin, blinkwindow
        see :func:`blink_onsets_mahot` (samples, and units per sample)
    interp_type : str
        type of interpolation accepted by :func:`scipy.interpolate.interp1d`
    tx : np.ndarray, optional
        time of each sample; default: the sample index
    missing : np.ndarray of bool, optional
        samples that cannot be used as anchor points (e.g., excluded
        periods); NaN samples never can
    max_duration : int, optional
        maximum number of samples that are replaced for one blink (with its
        margins, or after merging it with neighbouring blinks); longer ones
        are skipped (e.g., chains of merged blinks); `None`: no limit

    Returns
    -------
    filled : np.ndarray
        copy of `sy` with the blinks replaced
    interpolated : np.ndarray of bool
        the replaced samples
    skipped : list
        the blinks that were not interpolated, as tuples
        ``(index, (t1, t2, t3, t4), reason)``

    References
    ----------
    Mahot, 2013:
    https://figshare.com/articles/A_simple_way_to_reconstruct_pupil_size_during_eye_blinks/688001.
    """
    from scipy.interpolate import interp1d
    sy = np.asarray(sy, dtype=float)
    n = len(sy)
    tx = np.arange(n, dtype=float) if tx is None else np.asarray(tx)
    filled, interpolated, skipped = sy.copy(), np.zeros(n, dtype=bool), []
    if len(blinks) == 0:
        return filled, interpolated, skipped
    unusable = np.isnan(sy)
    if missing is not None:
        unusable |= np.asarray(missing, dtype=bool)

    onsets = blink_onsets_mahot(sy, blinks, smooth_winsize, vel_onset, vel_offset,
                                margin, blinkwindow)
    logger.debug(f"Refined {len(onsets)} blink boundaries")
    # the anchor points of a blink must not fall into a neighbouring
    # blink (they would be NaN), so merge blinks with overlapping windows
    for ix, (t1, t2, t3, t4) in enumerate(_merge_overlapping_blink_windows(onsets, n)):
        # Ensure all 4 time points are distinct and in order
        if t1 == t2:
            t2 += 1
        if t3 == t4:
            t3 -= 1
        pts = [t1, t2, t3, t4]
        if not (t1 < t2 < t3 < t4):
            skipped.append((ix, tuple(pts), "blink too short to interpolate"))
            continue
        # e.g., several blinks merged into one long stretch: a curve through
        # four points cannot replace it
        if max_duration is not None and t3 - t2 > max_duration:
            skipped.append((ix, tuple(pts), f"longer than max_duration ({t3 - t2} samples)"))
            continue
        # e.g., a blink that ends in an excluded period (a break)
        if unusable[pts].any():
            skipped.append((ix, tuple(pts), "no usable data at its anchor points"))
            continue
        if interp_type == "cubic" and _anchors_close(t1, t2, t3, t4):
            # a cubic through anchor points that are close on one side and
            # far on the other overshoots; a monotone curve cannot
            from scipy.interpolate import PchipInterpolator
            filled[t2:t3] = PchipInterpolator(tx[pts], sy[pts])(tx[t2:t3])
        elif interp_type == "cubic":
            # the cubic polynomial through the four points (what interp1d
            # gives for them, but much faster)
            filled[t2:t3] = _lagrange(tx[pts], sy[pts], tx[t2:t3])
        else:
            filled[t2:t3] = interp1d(tx[pts], sy[pts], kind=interp_type)(tx[t2:t3])
        interpolated[t2:t3] = True
    return filled, interpolated, skipped


def _blink_groups(blinks, window, n):
    """
    Local windows for Kalman interpolation: blinks (`(start, end)`,
    end-exclusive) whose `window`-neighbourhoods overlap share one window.
    Returns a list of `(wstart, wend, [blinks])`.
    """
    groups = []
    for start, end in sorted((int(a), int(b)) for a, b in blinks):
        ws, we = max(0, start - window), min(n, end + window)
        if groups and ws < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], we)
            groups[-1][2].append((start, end))
        else:
            groups.append([ws, we, [(start, end)]])
    return [tuple(g) for g in groups]


def _local_trend(y, valid, gap_start, gap_end):
    """
    Linear trend fitted to the valid samples of `y`; a constant if all valid
    samples are on one side of the gap `gap_start`-`gap_end` (no
    extrapolation of a slope into the gap).
    """
    t = np.arange(len(y))
    tv = t[valid]
    deg = 1 if (tv < gap_start).any() and (tv >= gap_end).any() else 0
    coef = np.polyfit(tv, y[valid], deg)
    return np.polyval(coef, t)


def _noise_sd(residuals):
    """
    SD of the white measurement noise from the second differences of the
    (detrended) windows: for white noise, the variance of the second
    difference is 6 times the noise variance; the signal hardly contributes
    at the sampling rates of eye trackers. Robust (MAD) estimate.
    """
    d2 = np.concatenate([r[2:] - 2 * r[1:-1] + r[:-2] for r in residuals])
    d2 = d2[~np.isnan(d2)]
    if d2.size < 2:
        return np.nan
    return 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)


def interpolate_blinks_kalman(sy, blinks, fs, window, missing=None, optimize=False,
                              max_fit_windows=100):
    """
    Interpolate blinks with a Gauss-Markov model (Kalman smoother).

    Around each blink, a local window of `window` samples on each side is
    taken (blinks with overlapping windows share one window) and a linear
    trend fitted to its valid samples is removed. The residuals are modelled
    as the sum of a fast and a slow Matérn process (smoothness 3/2) plus
    white measurement noise. The blink samples are replaced by the smoothed
    mean (Kalman filter + RTS smoother) plus the trend.

    By default, the time scales of the components are fixed (5 and 300 ms),
    their standard deviations are 0.1 and 1 times the SD of the residuals,
    and the SD of the noise is estimated from the second differences of the
    residuals. The smoothed mean depends only on the ratios of these SDs, so
    the defaults do not depend on the units of the pupil data. With
    `optimize=True`, all parameters are fitted instead (maximum likelihood
    over the windows, exact for the missing samples), which is much slower
    and changes the result little.

    Parameters
    ----------
    sy : np.ndarray
        pupil data (NaN or 0 for missing values)
    blinks : iterable of (start, end)
        blinks in samples (end-exclusive); these samples are replaced
    fs : float
        sampling rate in Hz
    window : int
        number of samples before and after each blink to use
    missing : np.ndarray of bool, optional
        further samples that are not used as observations (e.g., excluded
        periods or other blinks); the blinks, NaN and 0 are always missing
    optimize : bool
        fit all parameters of the model instead of using the defaults
    max_fit_windows : int
        with `optimize=True`, at most this many (evenly spaced) windows are
        used to fit the parameters

    Returns
    -------
    filled : np.ndarray
        copy of `sy` with the blinks replaced
    interpolated : np.ndarray of bool
        the replaced samples
    model : GaussMarkovModel or None
        the model used (None if there was nothing to interpolate)
    """
    from scipy.optimize import minimize
    from .gaussmarkov import GaussMarkovModel, Matern, kalman_filter, kalman_loglik, rts_smoother

    sy = np.asarray(sy, dtype=float)
    n = len(sy)
    filled, interpolated = sy.copy(), np.zeros(n, dtype=bool)
    inblink = np.zeros(n, dtype=bool)
    for start, end in blinks:
        inblink[max(0, int(start)):min(n, int(end))] = True
    obs_missing = inblink | np.isnan(sy) | (sy == 0)
    if missing is not None:
        obs_missing |= np.asarray(missing, dtype=bool)

    # detrended windows
    wins = []
    for ws, we, bls in _blink_groups(blinks, window, n):
        valid = ~obs_missing[ws:we]
        if valid.sum() < 2:
            logger.warning(f"Skipping blinks in samples {ws}-{we}: no usable data around them")
            continue
        y = sy[ws:we]
        trend = _local_trend(np.where(valid, y, 0.0), valid, bls[0][0] - ws, bls[-1][1] - ws)
        wins.append((ws, we, bls, np.where(valid, y - trend, np.nan), trend))
    if not wins:
        return filled, interpolated, None

    # default model: fixed time scales, SDs relative to the residuals, noise
    # estimated. A fast and a slow component: with only a slow one, the
    # fast fluctuations are not accounted for and the fill is worse; with
    # only a fast one, the fill reverts to the trend within a few tens of ms.
    residuals = [w[3] for w in wins]
    sd = max(np.nanstd(np.concatenate(residuals)), 1e-12)
    noise = _noise_sd(residuals)
    noise = min(max(noise if np.isfinite(noise) else 0.01 * sd, 1e-4 * sd), sd)
    model = GaussMarkovModel([Matern(sigma=0.1 * sd, ell=5.0, nu=1.5),
                              Matern(sigma=sd, ell=300.0, nu=1.5)], noise=noise)

    if optimize:
        # all parameters (log scale, bounded) on up to max_fit_windows windows
        fitwins = [residuals[i] for i in
                   np.unique(np.linspace(0, len(wins) - 1, min(len(wins), max_fit_windows)).astype(int))]
        bounds = [(np.log(1e-3 * sd), np.log(100 * sd)),         # fast: sigma
                  (np.log(1000.0 / fs), np.log(200.0)),          # fast: ell (ms)
                  (np.log(1e-3 * sd), np.log(100 * sd)),         # slow: sigma
                  (np.log(20.0), np.log(10000.0)),               # slow: ell (ms)
                  (np.log(1e-4 * sd), np.log(10 * sd))]          # noise

        def negll(theta):
            try:
                ssm = model.from_theta(theta).to_lgssm(fs)
                ll = sum(kalman_loglik(ssm, r) for r in fitwins)
            except np.linalg.LinAlgError:
                return np.inf
            return -ll if np.isfinite(ll) else np.inf

        theta0 = np.clip(model.theta, [b[0] for b in bounds], [b[1] for b in bounds])
        opt = minimize(negll, theta0, method="L-BFGS-B", bounds=bounds)
        if np.isfinite(opt.fun) and (opt.success or opt.fun < negll(theta0)):
            model = model.from_theta(opt.x)
        else:
            logger.warning(f"Fitting the Kalman interpolation model failed ({opt.message}); "
                           "using the default parameters")
    logger.debug(f"Kalman interpolation model: {model.params}")

    # smooth each window and fill the blinks
    ssm = model.to_lgssm(fs)
    for ws, we, bls, r, trend in wins:
        sres = rts_smoother(ssm, kalman_filter(ssm, r))
        est = sres.signal + trend
        for start, end in bls:
            a, b = max(0, start), min(n, end)
            filled[a:b] = est[a - ws:b - ws]
            interpolated[a:b] = True
    return filled, interpolated, model
