"""
transplant.py
=============

Transplantation of blinks: real blinks (the missing samples and the
artifacts around them) are cut out of a signal and added to clean stretches
of a signal, where the true signal is known. This gives data with realistic
blinks and a ground truth, e.g. to evaluate or tune the interpolation of
blinks.

All functions work on plain numpy arrays, in samples. Intervals are arrays
(n x 2) of start/end sample indices (end-exclusive).
"""
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

__all__ = [
    "blink_cores", "blink_donors", "extract_blink_artifacts", "blink_artifact_profile",
    "shrink_blink_artifacts", "clean_stretches",
    "place_blink_artifacts", "insert_blink_artifacts", "transplant_blinks",
    "transplant_error", "BlinkTransplant",
]


#: number of blink-free windows used to estimate the fluctuations of the
#: signal (see `blink_artifact_profile`), and the minimum number needed
_N_NULL, _MIN_NULL = 500, 10


def _runs(mask):
    """Runs of True in `mask` as an int array (n x 2) of start/end (end-exclusive)."""
    x = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return np.c_[np.where(x == 1)[0], np.where(x == -1)[0]]


def _mask(runs, n):
    """Boolean array of length `n` that is True in the `runs` (n x 2, end-exclusive)."""
    mask = np.zeros(n, dtype=bool)
    for a, b in runs:
        mask[a:b] = True
    return mask


def _missing(sy, blink_val=0):
    """Missing samples of `sy`: NaN, or equal to `blink_val` (unless it is NaN)."""
    sy = np.asarray(sy, dtype=float)
    miss = np.isnan(sy)
    if blink_val is not None and not np.isnan(blink_val):
        miss |= sy == blink_val
    return miss


def _pair(value, name):
    """`(before, after)` in samples from a number or a pair."""
    pair = (value, value) if np.ndim(value) == 0 else tuple(value)
    if len(pair) != 2 or min(pair) < 0:
        raise ValueError(f"{name} must be a non-negative number or a (before, after) pair, "
                         f"got {value!r}")
    return int(pair[0]), int(pair[1])


def _decreasing(y):
    """Closest (least squares) non-increasing sequence to `y` (isotonic
    regression, pool-adjacent-violators)."""
    vals, sizes = [], []
    for v in np.asarray(y, dtype=float):
        vals.append(v)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            n = sizes[-2] + sizes[-1]
            vals[-2:] = [(vals[-2] * sizes[-2] + vals[-1] * sizes[-1]) / n]
            sizes[-2:] = [n]
    return np.repeat(vals, sizes)


def blink_cores(sy, merge_distance=0, blink_val=0):
    """
    Cores of the blinks: the runs of missing samples.

    Runs that are separated by at most `merge_distance` valid samples are one
    core (e.g., several quick blinks, or a blink with a few samples of
    signal in it), so that a core is surrounded by signal on both sides.

    Parameters
    ----------
    sy : np.ndarray
        pupil data
    merge_distance : int
        runs separated by at most this many samples are merged
    blink_val : float
        "missing value" code in addition to NaN (default 0); `np.nan` or
        `None` if only NaN is missing

    Returns
    -------
    np.ndarray
        array (n x 2) of start/end sample indices (end-exclusive) of the
        cores, from the first to the last missing sample of each
    """
    runs = _runs(_missing(sy, blink_val))
    if len(runs) < 2:
        return runs
    new = np.r_[True, runs[1:, 0] - runs[:-1, 1] > merge_distance]
    return np.c_[runs[new, 0], runs[np.r_[new[1:], True], 1]]


def blink_donors(cores, missing, window, min_len=1, max_len=None):
    """
    Select the blinks that can be transplanted.

    A blink can be transplanted if the window around its core is inside the
    data and has no missing samples other than the core itself (e.g., no
    other blink), so that the artifact around the blink can be separated
    from the signal.

    Parameters
    ----------
    cores : np.ndarray
        cores of the blinks (n x 2, end-exclusive), see :func:`blink_cores`
    missing : np.ndarray of bool
        samples without usable signal (the cores and, e.g., samples that
        are masked for other reasons)
    window : int or tuple of int
        number of samples `(before, after)` the core that belong to the
        blink
    min_len, max_len : int
        only cores with at least `min_len` and at most `max_len` samples
        are used (e.g., no single missing samples, no long periods without
        data); `max_len=None`: no limit

    Returns
    -------
    np.ndarray of bool
        True for the cores that can be transplanted
    """
    cores = np.asarray(cores, dtype=int).reshape(-1, 2)
    missing = np.asarray(missing, dtype=bool)
    pre, post = _pair(window, "window")
    # number of missing samples before each sample
    nmiss = np.r_[0, np.cumsum(missing)]
    ok = np.zeros(len(cores), dtype=bool)
    for i, (a, b) in enumerate(cores):
        if b - a < min_len or (max_len is not None and b - a > max_len):
            continue
        if a - pre < 0 or b + post > len(missing):
            continue
        ok[i] = (nmiss[a] - nmiss[a - pre] == 0) and (nmiss[b + post] - nmiss[b] == 0)
    return ok


def extract_blink_artifacts(sy, cores, window, edge=None, mode="additive", blink_val=0):
    """
    Cut the artifacts of blinks out of a signal.

    For each blink, the window around its core is cut out and a slow
    baseline is removed: the straight line through the means of the first
    and the last `edge` samples of the window. What remains is the
    artifact of the blink (plus the fluctuations of the signal and the
    noise in the window). It fades out to zero over the `edge` samples
    at both ends (cosine taper), so it can be added to another signal
    without steps (see :func:`insert_blink_artifacts`).

    Parameters
    ----------
    sy : np.ndarray
        pupil data
    cores : np.ndarray
        cores of the blinks to cut out (n x 2, end-exclusive); their
        windows must be inside the data and should have no other missing
        samples (see :func:`blink_donors`)
    window : int or tuple of int
        number of samples `(before, after)` the core to cut out; should
        be generous, so that the whole artifact is inside the window
    edge : int, optional
        number of samples at each end of the window for the baseline and
        the taper; default: a quarter of the smaller side of the window
    mode : str
        ``"additive"`` (default): the artifact is the difference to the
        baseline; ``"multiplicative"``: the artifact is the ratio to the
        baseline (e.g., an eyelid that covers a part of the pupil), which
        needs a positive signal
    blink_val : float
        "missing value" code in addition to NaN (default 0)

    Returns
    -------
    list of np.ndarray
        one artifact per blink, with `before + len(core) + after` samples;
        NaN at the missing samples. Additive artifacts are 0 where there
        is no artifact, multiplicative ones 1.
    """
    if mode not in ("additive", "multiplicative"):
        raise ValueError(f"mode must be 'additive' or 'multiplicative', got {mode!r}")
    sy = np.asarray(sy, dtype=float)
    pre, post = _pair(window, "window")
    if edge is None:
        edge = max(1, min(pre, post) // 4)
    if not 1 <= edge <= min(pre, post):
        raise ValueError(f"edge must be between 1 and the window size, got {edge}")
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(edge) + 0.5) / edge)
    artifacts = []
    for a, b in np.asarray(cores, dtype=int).reshape(-1, 2):
        if a - pre < 0 or b + post > len(sy):
            raise ValueError(f"the window of the blink at samples {a}-{b} is not inside the data")
        y = sy[a - pre:b + post].copy()
        y[_missing(y, blink_val)] = np.nan
        n = len(y)
        left, right = np.nanmean(y[:edge]), np.nanmean(y[-edge:])
        if np.isnan(left) or np.isnan(right):
            raise ValueError(f"no signal at the edges of the window of the blink at samples {a}-{b}")
        # line through the centres of the edges
        baseline = left + (right - left) * (np.arange(n) - (edge - 1) / 2) / (n - edge)
        taper = np.ones(n)
        taper[:edge], taper[-edge:] = ramp, ramp[::-1]
        if mode == "additive":
            artifacts.append((y - baseline) * taper)
        else:
            if baseline.min() <= 0:
                raise ValueError("mode='multiplicative' needs a positive signal")
            artifacts.append(1 + (y / baseline - 1) * taper)
    return artifacts


def blink_artifact_profile(artifacts, null_artifacts, window):
    """
    The typical artifact of the blinks and how reliable the single ones are.

    A single artifact (see :func:`extract_blink_artifacts`) also contains
    the fluctuations of the signal around that blink, which cannot be told
    apart from its artifact. Over all blinks, they can: the fluctuations
    average out in the median artifact, and their size is known from
    "artifacts" that were cut out where there is no blink
    (`null_artifacts`). Where the artifacts differ more from each other
    than these do, the blinks differ. The weight of the single artifacts
    is the share of their variance (across blinks) that is not explained
    by the fluctuations: ``1 - s**2 / v**2`` for the robust standard
    deviations `v` of the artifacts and `s` of the null artifacts at each
    sample (at least 0), made non-increasing with the distance to the core.

    Parameters
    ----------
    artifacts : list of np.ndarray
        artifacts of the blinks
    null_artifacts : list of np.ndarray
        artifacts that were cut out the same way from clean signal
    window : int or tuple of int
        number of samples `(before, after)` the core in the artifacts

    Returns
    -------
    dict
        ``"median"``: the median artifact, ``"weight"``: the weight of the
        single artifacts (between 0 and 1); each a tuple `(before, after)`
        of arrays for the samples before the core (the last one is next
        to the core) and after it (the first one is next to it)
    """
    pre, post = _pair(window, "window")
    if len(artifacts) == 0 or len(null_artifacts) == 0:
        raise ValueError("no artifacts")
    median, weight = [], []
    for before in (True, False):
        # the samples of one side, ordered by their distance to the core
        # (the core has a different size in each artifact)
        def side(arts):
            return np.array([a[:pre][::-1] if before else a[len(a) - post:] for a in arts])
        arts, null = side(artifacts), side(null_artifacts)
        m = np.median(arts, axis=0)
        v = 1.4826 * np.median(np.abs(arts - m), axis=0)
        s = 1.4826 * np.median(np.abs(null - np.median(null, axis=0)), axis=0)
        w = _decreasing(np.clip(1 - s ** 2 / np.maximum(v ** 2, 1e-300), 0, 1))
        median.append(m[::-1] if before else m)
        weight.append(w[::-1] if before else w)
    return {"median": tuple(median), "weight": tuple(weight)}


def shrink_blink_artifacts(artifacts, profile):
    """
    Remove the fluctuations of the signal from the artifacts of blinks.

    Each artifact is replaced by the median artifact plus its difference
    to the median artifact times the weight (see
    :func:`blink_artifact_profile`): close to the core, where the blinks
    differ, the artifacts are kept; further away, where they only differ
    by the fluctuations of the signal around them, they become the median
    artifact.

    Parameters
    ----------
    artifacts : list of np.ndarray
        artifacts of the blinks (see :func:`extract_blink_artifacts`)
    profile : dict
        median artifact and weights (see :func:`blink_artifact_profile`)

    Returns
    -------
    list of np.ndarray
        the artifacts without the fluctuations
    """
    (m0, m1), (w0, w1) = profile["median"], profile["weight"]
    out = []
    for a in artifacts:
        a = a.copy()
        a[:len(m0)] = m0 + (a[:len(m0)] - m0) * w0
        if len(m1):
            a[-len(m1):] = m1 + (a[-len(m1):] - m1) * w1
        out.append(a)
    return out


def clean_stretches(missing, window=0, min_len=1):
    """
    Stretches of clean signal: no missing samples and far from any.

    Parameters
    ----------
    missing : np.ndarray of bool
        samples without usable signal (e.g., blinks)
    window : int or tuple of int
        number of samples `(before, after)` each missing sample that are
        not clean (e.g., the artifacts around the blinks)
    min_len : int
        minimum number of samples of a stretch

    Returns
    -------
    np.ndarray
        array (n x 2) of start/end sample indices (end-exclusive)
    """
    missing = np.asarray(missing, dtype=bool)
    pre, post = _pair(window, "window")
    dirty = missing.copy()
    for a, b in _runs(missing):
        dirty[max(0, a - pre):b + post] = True
    runs = _runs(~dirty)
    return runs[runs[:, 1] - runs[:, 0] >= min_len]


def place_blink_artifacts(stretches, lengths, n=None, pad=0, seed=None):
    """
    Choose random, non-overlapping places for artifacts in clean stretches.

    The stretches are filled with randomly chosen artifacts at random
    distances; each artifact is used about equally often.

    Parameters
    ----------
    stretches : np.ndarray
        clean stretches (n x 2, end-exclusive), see :func:`clean_stretches`
    lengths : sequence of int
        number of samples of each artifact
    n : int, optional
        number of places; default (and at most): as many as fit
    pad : int
        number of clean samples to keep free on each side of each artifact
        (twice as many between two artifacts)
    seed : int or np.random.Generator, optional
        seed of the random numbers

    Returns
    -------
    np.ndarray
        int array (n x 2), sorted by position: the first sample of each
        place and the index of the artifact that goes there
    """
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    places = []
    if len(lengths) == 0:
        return np.zeros((0, 2), dtype=int)
    order, k = rng.permutation(len(lengths)), 0
    for a, b in np.asarray(stretches, dtype=int).reshape(-1, 2):
        cursor = a
        while True:
            art = order[k % len(order)]
            need = lengths[art] + 2 * pad
            if cursor + need > b:
                break
            # random distance, at most the size of the place itself
            cursor += int(rng.integers(0, min(b - cursor - need, need) + 1))
            places.append((cursor + pad, art))
            cursor += need
            k += 1
            if k % len(order) == 0:
                order = rng.permutation(len(lengths))
    places = np.array(places, dtype=int).reshape(-1, 2)
    if n is not None and n < len(places):
        places = places[np.sort(rng.choice(len(places), n, replace=False))]
    return places


def insert_blink_artifacts(sy, artifacts, places, mode="additive", missing_value=np.nan):
    """
    Put artifacts of blinks into a signal.

    Parameters
    ----------
    sy : np.ndarray
        the (clean) signal
    artifacts : list of np.ndarray
        artifacts, see :func:`extract_blink_artifacts`
    places : np.ndarray
        int array (n x 2): the first sample and the index of the artifact
        for each place, see :func:`place_blink_artifacts`
    mode : str
        ``"additive"`` (default) or ``"multiplicative"``: how the
        artifacts were extracted
    missing_value : float
        value of the missing samples of the artifacts (default NaN)

    Returns
    -------
    np.ndarray
        copy of `sy` with the artifacts
    """
    if mode not in ("additive", "multiplicative"):
        raise ValueError(f"mode must be 'additive' or 'multiplicative', got {mode!r}")
    out = np.array(sy, dtype=float)
    for start, art in np.asarray(places, dtype=int).reshape(-1, 2):
        a = artifacts[art]
        if start < 0 or start + len(a) > len(out):
            raise ValueError(f"the artifact at sample {start} is not inside the data")
        seg = out[start:start + len(a)]
        seg[:] = np.where(np.isnan(a), missing_value, seg + a if mode == "additive" else seg * a)
    return out


@dataclass
class BlinkTransplant:
    """
    A signal with transplanted blinks (see :func:`transplant_blinks`).

    Attributes
    ----------
    signal : np.ndarray
        the signal with the transplanted blinks
    clean : np.ndarray
        the signal without them (the ground truth)
    blinks : np.ndarray
        the cores of the transplanted blinks (n x 2, end-exclusive): from
        the first to the last missing sample
    windows : np.ndarray
        the windows that hold the transplanted artifacts (n x 2)
    sites : np.ndarray
        the windows with the clean samples reserved around them (n x 2);
        the errors of a reconstruction are computed in the sites (see
        :func:`transplant_error`)
    donor : np.ndarray
        index (into `donors`) of the blink that was transplanted to each site
    donors : np.ndarray
        the cores of the blinks that were cut out (n x 2)
    artifacts : list of np.ndarray
        their artifacts (see :func:`extract_blink_artifacts`)
    window : tuple of int
        number of samples `(before, after)` the cores in the windows
    mode : str
        ``"additive"`` or ``"multiplicative"``
    profile : dict or None
        the median artifact and the weights of the single artifacts (see
        :func:`blink_artifact_profile`) that were used to remove the
        fluctuations of the signal from the artifacts
    """
    signal: np.ndarray
    clean: np.ndarray
    blinks: np.ndarray
    windows: np.ndarray
    sites: np.ndarray
    donor: np.ndarray
    donors: np.ndarray
    artifacts: list = field(repr=False)
    window: tuple = (0, 0)
    mode: str = "additive"
    profile: dict = field(default=None, repr=False)

    def __len__(self):
        return len(self.blinks)

    def __repr__(self):
        return (f"BlinkTransplant({len(self.blinks)} transplanted blinks from "
                f"{len(self.donors)} donors, {len(self.signal)} samples, mode={self.mode!r})")

    def error(self, filled, per_site=False):
        """RMSE of the reconstruction `filled` (see :func:`transplant_error`)."""
        return transplant_error(filled, self.clean, self.sites, per_site=per_site)


def transplant_blinks(sy, window, edge=None, merge_distance=0, min_len=1, max_len=None,
                      n=None, pad=0, mode="additive", shrink=True, missing=None,
                      avoid=None, blink_val=0, source=None, cores=None, seed=None):
    """
    Transplant the blinks of a signal into its clean stretches.

    The blinks are found as runs of missing samples (their cores, see
    :func:`blink_cores`). The blinks that are surrounded by signal are cut
    out with a window around the core (:func:`blink_donors`,
    :func:`extract_blink_artifacts`) and put at random places where the
    signal is clean (:func:`clean_stretches`,
    :func:`place_blink_artifacts`, :func:`insert_blink_artifacts`). There,
    the signal without the blink is known, so a reconstruction of the
    transplanted blinks can be compared to it (see :func:`transplant_error`).

    The window around a blink does not only contain its artifact but also
    the fluctuations of the signal there, which would be transplanted with
    it. With `shrink`, they are removed where the blinks do not differ
    from each other (see :func:`shrink_blink_artifacts`), so the window can
    be generous. Close to the cores, where the blinks differ, the
    fluctuations and the noise are still transplanted with the artifacts.

    Parameters
    ----------
    sy : np.ndarray
        pupil data (the blinks are missing samples: NaN or `blink_val`)
    window : int or tuple of int
        number of samples `(before, after)` the core of a blink that are
        transplanted with it; should be generous, so that it includes the
        whole artifact
    edge : int, optional
        number of samples at each end of the window for the baseline and
        the taper (see :func:`extract_blink_artifacts`)
    merge_distance : int
        missing samples separated by at most this many samples are one
        blink (see :func:`blink_cores`)
    min_len, max_len : int
        minimum and maximum number of samples of the cores that are
        transplanted (see :func:`blink_donors`)
    n : int, optional
        number of blinks to transplant; default (and at most): as many as fit
    pad : int
        number of clean samples to keep free on each side of each window
    mode : str
        ``"additive"`` (default) or ``"multiplicative"`` (see
        :func:`extract_blink_artifacts`)
    shrink : bool
        remove the fluctuations of the signal from the artifacts where the
        blinks do not differ (default); needs clean signal in `source` to
        estimate the fluctuations, otherwise the artifacts are used as
        they are
    missing : np.ndarray of bool, optional
        further samples without usable signal (e.g., masked for other
        reasons); the blinks next to them are not transplanted and nothing
        is transplanted next to them
    avoid : np.ndarray of bool, optional
        samples of `sy` where no blinks are transplanted, e.g. other
        artifacts (in addition to the `window` around all missing samples)
    blink_val : float
        "missing value" code in addition to NaN (default 0). The
        transplanted cores get this value if the signal uses it (and has
        no NaN), otherwise NaN.
    source : np.ndarray, optional
        signal to take the blinks from; default: `sy` itself. E.g., for
        transplanting the blinks of a recording into a simulated signal.
    cores : np.ndarray, optional
        cores of the blinks to transplant (n x 2, end-exclusive), e.g. a
        subset of the blinks; default: all blinks (see :func:`blink_cores`).
        Those that are not surrounded by signal are left out.
    seed : int or np.random.Generator, optional
        seed of the random numbers

    Returns
    -------
    BlinkTransplant
        the signal with the transplanted blinks, the clean signal and
        where the blinks are

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> y = 1000 + np.cumsum(rng.normal(0, 1, 5000))
    >>> y[980:1000] -= np.linspace(0, 500, 20)   # a blink at samples 1000-1050
    >>> y[1050:1090] -= np.linspace(500, 0, 40)
    >>> y[1000:1050] = np.nan
    >>> tr = transplant_blinks(y, window=(100, 150), n=5, seed=1)
    >>> len(tr.blinks)
    5
    >>> filled = np.where(np.isnan(tr.signal), 1000, tr.signal)   # some reconstruction
    >>> rmse = tr.error(filled)
    """
    sy = np.asarray(sy, dtype=float)
    src = sy if source is None else np.asarray(source, dtype=float)
    pre, post = _pair(window, "window")

    # donors
    src_missing = _missing(src, blink_val)
    if cores is None:
        cores = blink_cores(src, merge_distance, blink_val)
    cores = np.asarray(cores, dtype=int).reshape(-1, 2)
    unusable = src_missing.copy()
    if missing is not None and source is None:
        unusable |= np.asarray(missing, dtype=bool)
    donors = cores[blink_donors(cores, unusable, (pre, post), min_len, max_len)]
    artifacts = extract_blink_artifacts(src, donors, (pre, post), edge, mode, blink_val)
    logger.debug(f"{len(donors)} of {len(cores)} blinks can be transplanted")
    rng = np.random.default_rng(seed)
    profile = None
    if shrink and len(donors) > 0:
        # "artifacts" without a blink, with a core of typical size
        size = pre + int(np.median(donors[:, 1] - donors[:, 0])) + post
        null = place_blink_artifacts(clean_stretches(unusable, (pre, post)), [size],
                                     _N_NULL, seed=rng)[:, 0]
        if len(null) >= _MIN_NULL:
            null = extract_blink_artifacts(src, np.c_[null + pre, null + size - post],
                                           (pre, post), edge, mode, blink_val)
            profile = blink_artifact_profile(artifacts, null, (pre, post))
            artifacts = shrink_blink_artifacts(artifacts, profile)
        else:
            logger.warning("Not enough clean signal to estimate its fluctuations: "
                           "they are transplanted with the blinks")

    # places in the clean signal
    sy_missing = _missing(sy, blink_val)
    dirty = sy_missing.copy()
    if missing is not None:
        dirty |= np.asarray(missing, dtype=bool)
    stretches = clean_stretches(dirty, (pre, post))
    if avoid is not None:
        stretches = _runs(_mask(stretches, len(sy)) & ~np.asarray(avoid, dtype=bool))
    places = place_blink_artifacts(stretches, [len(a) for a in artifacts], n, pad, rng)

    uses_val = (blink_val is not None and not np.isnan(blink_val)
                and not np.isnan(src).any() and src_missing.any())
    signal = insert_blink_artifacts(sy, artifacts, places, mode,
                                    blink_val if uses_val else np.nan)
    start, donor = places[:, 0], places[:, 1]
    length = np.array([len(artifacts[i]) for i in donor], dtype=int)
    windows = np.c_[start, start + length]
    return BlinkTransplant(signal=signal, clean=sy.copy(),
                           blinks=np.c_[start + pre, start + length - post],
                           windows=windows, sites=windows + [-pad, pad],
                           donor=donor, donors=donors, artifacts=artifacts,
                           window=(pre, post), mode=mode,
                           profile=profile)


def transplant_error(filled, clean, sites, per_site=False):
    """
    Error of a reconstruction of transplanted blinks.

    Root mean squared difference between the reconstructed and the clean
    signal over all samples of the sites, i.e., the samples of the cores,
    of the artifacts around them and of the clean signal next to them (a
    reconstruction can replace more samples than necessary). Samples that
    are still missing in `filled` (NaN) are interpolated linearly first,
    so that reconstructions that leave blinks out can be compared.

    Parameters
    ----------
    filled : np.ndarray
        the signal with the reconstructed blinks
    clean : np.ndarray
        the signal without the transplanted blinks
    sites : np.ndarray
        the sites of the transplanted blinks (n x 2, end-exclusive), see
        :class:`BlinkTransplant`
    per_site : bool
        return the error of each site instead of the error over all sites

    Returns
    -------
    float or np.ndarray
        the error (NaN without sites), or one error per site
    """
    filled = np.asarray(filled, dtype=float)
    clean = np.asarray(clean, dtype=float)
    sites = np.asarray(sites, dtype=int).reshape(-1, 2)
    if len(sites) == 0:
        return np.zeros(0) if per_site else np.nan
    nan = np.isnan(filled)
    if nan.any():
        valid = np.where(~nan)[0]
        if valid.size == 0:
            raise ValueError("the reconstructed signal has no valid samples")
        filled = filled.copy()
        filled[nan] = np.interp(np.where(nan)[0], valid, filled[valid])
    sse = np.array([np.nansum((filled[a:b] - clean[a:b]) ** 2) for a, b in sites])
    nsamp = sites[:, 1] - sites[:, 0]
    if per_site:
        return np.sqrt(sse / nsamp)
    return float(np.sqrt(sse.sum() / nsamp.sum()))
