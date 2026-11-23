"""
autotune.py
===========

Tuning of the parameters of the interpolation of blinks on transplanted
blinks (see :mod:`~pypillometry.signal.transplant`): blinks that were put
into clean signal, where the signal without the blink is known.

All functions work on plain numpy arrays, in samples.
"""
from dataclasses import dataclass, field, replace

import numpy as np
from loguru import logger

from .preproc import interpolate_blinks_mahot
from .transplant import BlinkTransplant, blink_cores

__all__ = ["transplant_error_mahot", "tune_blinks_mahot", "BlinkTuneResult"]

# the tuned values: the parameters of interpolate_blinks_mahot(), with the
# absolute value of vel_onset and both margins
_NAMES = ("smooth_winsize", "vel_onset", "vel_offset", "margin_before", "margin_after",
          "blinkwindow")
# maximum number of changes of a parameter per round of the search
_MAX_MOVES = 20
# searched on a logarithmic scale
_LOG = np.array([True, True, True, False, False, True])


def _as_list(transplants):
    return [transplants] if isinstance(transplants, BlinkTransplant) else list(transplants)


def _to_vector(params):
    """Values (see `_NAMES`) from parameters of interpolate_blinks_mahot()."""
    return np.array([params["smooth_winsize"], abs(params["vel_onset"]), abs(params["vel_offset"]),
                     params["margin"][0], params["margin"][1], params["blinkwindow"]], dtype=float)


def _to_params(x):
    """Valid parameters of interpolate_blinks_mahot() from values (see `_NAMES`)."""
    winsize = max(3, int(round(x[0])))
    return dict(smooth_winsize=winsize + (winsize % 2 == 0),
                vel_onset=-float(f"{x[1]:.4g}"), vel_offset=float(f"{x[2]:.4g}"),
                margin=(max(0, int(round(x[3]))), max(0, int(round(x[4])))),
                blinkwindow=max(1, int(round(x[5]))))


def transplant_error_mahot(transplants, smooth_winsize, vel_onset, vel_offset, margin,
                           blinkwindow, interp_type="cubic", per_blink=False):
    """
    Error of the Mahot interpolation of transplanted blinks.

    The transplanted blinks are interpolated with
    :func:`~pypillometry.signal.preproc.interpolate_blinks_mahot` and
    compared to the signal without them (see
    :func:`~pypillometry.signal.transplant.transplant_error`). The error is
    the mean over the blinks of their root mean squared errors, which
    depends less on a few blinks with a large error than the root mean
    squared error over all of them.

    As in the interpolation of a recording, the other missing samples of
    the signal (e.g., its own blinks) are interpolated together with the
    transplanted blinks, so that a blink whose anchor points fall into
    them is merged with them instead of being skipped. Only the error at
    the transplanted blinks counts.

    Parameters
    ----------
    transplants : BlinkTransplant or list of BlinkTransplant
        signal(s) with transplanted blinks (e.g., one per eye), see
        :func:`~pypillometry.signal.transplant.transplant_blinks`
    smooth_winsize, vel_onset, vel_offset, margin, blinkwindow, interp_type
        parameters of the interpolation (in samples, and units per sample)
    per_blink : bool
        return the errors of all blinks instead of their mean

    Returns
    -------
    float or np.ndarray
        the mean error per blink, or the error of each blink
    """
    errors = []
    for tr in _as_list(transplants):
        filled, _, _ = interpolate_blinks_mahot(tr.signal, _all_blinks(tr), smooth_winsize,
                                                vel_onset, vel_offset, margin, blinkwindow,
                                                interp_type)
        errors.append(tr.error(filled, per_site=True))
    errors = np.concatenate(errors) if errors else np.zeros(0)
    if per_blink:
        return errors
    return float(errors.mean()) if errors.size else np.nan


def _all_blinks(tr):
    """The transplanted blinks of `tr` and the other runs of missing
    samples of its signal (n x 2, sorted)."""
    if len(tr) == 0:
        return tr.blinks
    other = blink_cores(tr.signal)
    if len(other):
        # runs that overlap a transplanted blink are part of it
        k = np.searchsorted(tr.blinks[:, 0], other[:, 1], side="left")
        hit = (k > 0) & (tr.blinks[np.maximum(k - 1, 0), 1] > other[:, 0])
        other = other[~hit]
    blinks = np.vstack([tr.blinks, other])
    return blinks[np.argsort(blinks[:, 0], kind="stable")]


def _context(transplants, hi):
    """Number of samples around the sites of transplanted blinks that the
    Mahot interpolation can use with parameters up to `hi` (see `_NAMES`):
    the search for the on-/offsets (`blinkwindow` and the margins), the
    anchor points (one interpolated stretch further out) and the smoothing
    window of the velocity."""
    core = max((int(np.max(tr.blinks[:, 1] - tr.blinks[:, 0])) for tr in transplants
                if len(tr)), default=0)
    winsize, _, _, before, after, blinkwindow = (int(np.ceil(v)) for v in hi)
    stretch = core + 2 * blinkwindow + before + after
    return blinkwindow + max(before, after) + stretch + winsize + 10   # + rounding


def _crop(tr, context):
    """The transplant `tr` with only the samples within `context` of its
    sites. The interpolation of the transplanted blinks and its error do not
    change if `context` covers all samples it uses (see `_context`), and
    are much faster for a long signal with few transplanted blinks."""
    n = len(tr.signal)
    if len(tr) == 0:
        return tr
    keep = []
    for a, b in zip(np.maximum(tr.sites[:, 0] - context, 0),
                    np.minimum(tr.sites[:, 1] + context, n)):
        if keep and a <= keep[-1][1]:
            keep[-1][1] = max(keep[-1][1], b)
        else:
            keep.append([a, b])
    if sum(b - a for a, b in keep) >= n:
        return tr
    # new index of the first sample of each kept stretch
    starts = np.array([a for a, b in keep])
    shift = starts - np.r_[0, np.cumsum([b - a for a, b in keep])[:-1]]

    def move(iv):
        k = np.searchsorted(starts, iv[:, 0], side="right") - 1
        return iv - shift[k][:, None]
    take = np.concatenate([np.arange(a, b) for a, b in keep])
    return replace(tr, signal=tr.signal[take], clean=tr.clean[take], blinks=move(tr.blinks),
                   windows=move(tr.windows), sites=move(tr.sites))


def _default_bounds(transplants):
    """Search ranges (lower, upper) of the values (see `_NAMES`) from the
    windows of the transplanted blinks and the velocities in them."""
    pre = max(tr.window[0] for tr in transplants)
    post = max(tr.window[1] for tr in transplants)
    noise, fastest = [], []
    for tr in transplants:
        for (a, b), (c0, c1) in zip(tr.windows, tr.blinks):
            noise.append(np.diff(tr.clean[a:b]))
            # the changes in the artifacts (not into or out of the core)
            fastest.append(np.nanmax(np.abs(np.r_[np.diff(tr.signal[a:c0]),
                                                  np.diff(tr.signal[c1:b]), 0])))
    noise = np.concatenate(noise) if noise else np.zeros(1)
    noise = 1.4826 * np.nanmedian(np.abs(noise - np.nanmedian(noise)))
    fastest = np.percentile(fastest, 99) if fastest else 1.0
    if not noise > 0:    # e.g., a signal without noise
        noise = fastest / 1000 if fastest > 0 else 1e-6
    vel = (0.1 * noise, max(fastest, noise))
    winmax = max(5, pre // 2)
    return {"smooth_winsize": (3, winmax + (winmax % 2 == 0)),
            "vel_onset": vel, "vel_offset": vel,
            "margin_before": (0, max(1, pre)), "margin_after": (0, max(1, post)),
            "blinkwindow": (max(1, pre // 20), pre + post)}


@dataclass
class BlinkTuneResult:
    """
    Result of tuning the parameters of the interpolation of blinks.

    Attributes
    ----------
    params : dict
        the parameters to use: the tuned ones (`tuned`) if they are better
        than the starting ones on the blinks that were not used for tuning
        (or if there are none), otherwise the starting ones (`start`)
    tuned : dict
        the tuned parameters
    start : dict or None
        the parameters the search started with
    loss, loss_start : float
        the error (mean error per blink, see
        :func:`transplant_error_mahot`) with the tuned and the starting
        parameters, on the blinks that were used for tuning (for the
        search and for the refinement)
    test_loss, test_loss_start : float
        the same on the blinks that were not used for tuning (NaN without)
    success : bool
        whether the tuned parameters are used (see `params`)
    bounds : dict
        the search range `(lower, upper)` of each value
    history : dict
        all evaluated parameters: one array per value (``"smooth_winsize"``,
        ``"vel_onset"``, ``"vel_offset"``, ``"margin_before"``,
        ``"margin_after"``, ``"blinkwindow"``), their ``"loss"`` and the
        ``"stage"`` (1: search on a set of `train`, 2: choice and
        refinement; the losses are on different blinks) and the
        ``"search"`` (index of the set of `train`; -1 in stage 2)
    n_blinks, n_refine, n_test : int
        number of transplanted blinks used for the searches (all sets), for
        the refinement (in addition) and for testing
    n_searches : int
        number of searches (sets of `train`)
    units : str
        units of the parameters, the bounds and the history:
        ``"samples"``, or ``"ms"`` (as the arguments of
        :meth:`~pypillometry.eyedata.pupil.PupilNamespace.blinks_interpolate`)
    """
    params: dict
    tuned: dict
    start: dict
    loss: float
    loss_start: float
    test_loss: float
    test_loss_start: float
    success: bool
    bounds: dict = field(repr=False)
    history: dict = field(repr=False)
    n_blinks: int = 0
    n_refine: int = 0
    n_test: int = 0
    n_searches: int = 1
    units: str = "samples"

    def __repr__(self):
        def fmt(p):
            if p is None:
                return "-"
            return ", ".join(f"{k}=" + ("(%.4g, %.4g)" % tuple(v) if np.ndim(v) else f"{v:.4g}")
                             for k, v in p.items())
        search = (f"{self.n_blinks} for {self.n_searches} searches" if self.n_searches > 1
                  else f"{self.n_blinks} for the search")
        lines = [f"BlinkTuneResult ({self.units}; transplanted blinks: {search}, "
                 f"{self.n_refine} more for refining, {self.n_test} for testing)",
                 f"  tuned : {fmt(self.tuned)}",
                 f"          error {self.loss:.4g}, on test blinks {self.test_loss:.4g}",
                 f"  start : {fmt(self.start)}",
                 f"          error {self.loss_start:.4g}, on test blinks {self.test_loss_start:.4g}",
                 f"  params: the {'tuned' if self.success else 'starting'} parameters"]
        return "\n".join(lines)




def _coordinate_search(loss, u, best, free, width, n_grid, rounds, tol):
    """From `u` (with error `best`), make the best change of one parameter
    at a time on a grid of `n_grid` values within `width` around it, until
    no change reduces the error by more than `tol`; `rounds` times, halving
    the width each time. Returns the parameters, their error and the width
    after the last round."""
    for r in range(rounds):
        for _ in range(_MAX_MOVES):
            move, move_loss = None, best * (1 - tol)
            for j in free:
                grid = np.unique(np.clip(u[j] + np.linspace(-width, width, n_grid), 0, 1))
                cands = np.repeat(u[None], len(grid), axis=0)
                cands[:, j] = grid
                losses = np.array([loss(c) for c in cands])
                if losses.min() < move_loss:
                    move, move_loss = cands[np.argmin(losses)], losses.min()
            if move is None:
                break
            u, best = move, move_loss
        logger.debug(f"Tuning: round {r + 1}: {best:.4g}")
        width /= 2
    return u, best, width


def tune_blinks_mahot(train, test=None, refine=None, start=None, bounds=None, fixed=None,
                      n_init=None, n_grid=9, sweeps=3, tol=0.01, seed=None):
    """
    Tune the parameters of the Mahot interpolation on transplanted blinks.

    Searches for the parameters of
    :func:`~pypillometry.signal.preproc.interpolate_blinks_mahot` with
    the smallest error on transplanted blinks (see
    :func:`transplant_error_mahot`). The error is not a smooth function of
    the parameters, so they are searched without derivatives: from the
    starting parameters, each parameter is tried at `n_grid` values around
    its current value, and the one change that reduces the error most is
    made. This is repeated until no change reduces the error by more than
    `tol`, in `sweeps` rounds with finer and finer grids (the first one
    covers the whole search range). Different parameters can give nearly
    the same error (e.g., a lower threshold or a larger margin); with
    `tol`, the search stays close to the starting parameters then.
    Without starting parameters (or with `n_init`), the search starts
    from the best of `n_init` parameter sets that cover the search range
    evenly (a Sobol sequence).

    The search needs many evaluations, so it can be run on a few blinks
    (`train`, e.g. 10 different blinks, each transplanted a few times).
    With few blinks, the result depends on which blinks they are, so the
    search can be run several times on different blinks (`train` as a list
    of such sets). The results (and the starting parameters) are compared
    on further transplanted blinks (`refine`, or else on all sets of
    `train`), and the best one is refined on its own blinks and `refine`
    with one more round on the finest grid.

    With many parameters and few blinks, the tuned parameters can fit the
    blinks they were tuned on better than other blinks. If transplanted
    blinks that were not used for tuning are given (`test`), the tuned
    parameters are only used if they are better on those than the starting
    parameters.

    Parameters
    ----------
    train : BlinkTransplant, list of BlinkTransplant, or list of lists
        signal(s) with transplanted blinks to search the parameters on (see
        :func:`~pypillometry.signal.transplant.transplant_blinks`); a list
        of lists (or of lists and BlinkTransplant) is several sets, with one
        search on each
    test : BlinkTransplant or list of BlinkTransplant, optional
        signal(s) with other transplanted blinks, to check the tuned
        parameters (e.g., using other blinks of the same recording, see
        `cores` of :func:`~pypillometry.signal.transplant.transplant_blinks`)
    refine : BlinkTransplant or list of BlinkTransplant, optional
        signal(s) with further transplanted blinks, to choose the best of
        the searches and refine it on (together with its `train` set)
    start : dict, optional
        starting parameters: `smooth_winsize`, `vel_onset`, `vel_offset`,
        `margin` and `blinkwindow` (in samples, and units per sample)
    bounds : dict, optional
        search range `(lower, upper)` for ``"smooth_winsize"``,
        ``"vel_onset"``, ``"vel_offset"`` (absolute values),
        ``"margin_before"``, ``"margin_after"`` or ``"blinkwindow"``.
        Default: from the data. The windows of the transplanted blinks
        `(before, after)` limit the margins (up to the window), the
        smoothing window (3 samples to half of `before`) and `blinkwindow`
        (up to the whole window); the velocity thresholds are between a
        tenth of the noise of the clean signal (the robust standard
        deviation of its sample-to-sample changes) and the fastest changes
        in the artifacts.
    fixed : dict, optional
        parameters that are not tuned, with their values (e.g.,
        ``{"blinkwindow": 250}``); `margin` as `(before, after)`
    n_init : int, optional
        number of parameter sets (rounded up to a power of 2) to choose
        the start of the search from, in addition to `start`; default: 0
        with starting parameters, otherwise 64
    n_grid : int
        number of values per parameter and round
    sweeps : int
        number of rounds of the search
    tol : float
        minimum relative reduction of the error for changing a parameter
    seed : int or np.random.Generator, optional
        seed of the random numbers (only used with `n_init`)

    Returns
    -------
    BlinkTuneResult
        the tuned parameters (``result.params`` can be passed to
        :func:`~pypillometry.signal.preproc.interpolate_blinks_mahot`),
        their errors and all evaluated parameters

    Examples
    --------
    >>> from pypillometry.signal import transplant_blinks, interpolate_blinks_mahot
    >>> rng = np.random.default_rng(1)
    >>> y = 1000 + np.cumsum(rng.normal(0, 1, 20000))
    >>> for s in range(1000, 20000, 2000):       # blinks
    ...     y[s-20:s] -= np.linspace(0, 500, 20)
    ...     y[s+50:s+90] -= np.linspace(500, 0, 40)
    ...     y[s:s+50] = np.nan
    >>> tr = transplant_blinks(y, window=(100, 150), pad=50, seed=1)
    >>> res = tune_blinks_mahot(tr, n_init=16, sweeps=1, seed=1)
    >>> filled, interpolated, _ = interpolate_blinks_mahot(tr.signal, tr.blinks, **res.params)
    """
    from scipy.stats import qmc
    train = _as_list(train)
    if any(isinstance(t, (list, tuple)) for t in train):
        sets = [_as_list(t) for t in train]
    else:
        sets = [train]
    train = [tr for trs in sets for tr in trs]
    test = [] if test is None else _as_list(test)
    refine = [] if refine is None else _as_list(refine)
    n_blinks, n_test = sum(len(tr) for tr in train), sum(len(tr) for tr in test)
    n_refine = sum(len(tr) for tr in refine)
    if n_blinks == 0:
        raise ValueError("no transplanted blinks to tune the parameters on")
    rng = np.random.default_rng(seed)

    # search ranges; fixed parameters have a range without size
    ranges = _default_bounds(train)
    for name, (lo, hi) in (bounds or {}).items():
        if name not in ranges:
            raise ValueError(f"unknown parameter {name!r} in bounds; use {', '.join(_NAMES)}")
        lo, hi = sorted((abs(lo), abs(hi)))
        if hi <= 0 and _LOG[_NAMES.index(name)]:
            raise ValueError(f"the bounds of {name} must not be 0")
        ranges[name] = (lo, hi)
    fixed = dict(fixed or {})
    if "margin" in fixed:
        fixed["margin_before"], fixed["margin_after"] = fixed.pop("margin")
    for name, value in fixed.items():
        if name not in ranges:
            raise ValueError(f"unknown parameter {name!r} in fixed; use "
                             "smooth_winsize, vel_onset, vel_offset, margin, blinkwindow")
        ranges[name] = (abs(value), abs(value))
    lo = np.array([ranges[name][0] for name in _NAMES], dtype=float)
    hi = np.array([ranges[name][1] for name in _NAMES], dtype=float)
    free = np.where(hi > lo)[0]
    # the lower bound of a logarithmic scale must be positive
    lo_log = np.where(_LOG, np.maximum(lo, 1e-12 * np.maximum(hi, 1e-300)), lo)

    hi_log = np.maximum(hi, lo_log)

    # only the signal around the transplanted blinks is needed
    context = _context(train + refine + test, hi)
    sets, test, refine = ([[_crop(tr, context) for tr in trs] for trs in sets],
                          [_crop(tr, context) for tr in test],
                          [_crop(tr, context) for tr in refine])

    def to_x(u):
        """values from the unit cube"""
        with np.errstate(invalid="ignore", divide="ignore"):
            log = np.exp(np.log(lo_log) + u * (np.log(hi_log) - np.log(lo_log)))
        return np.where(_LOG, log, lo + u * (hi - lo))

    def to_u(x):
        x = np.clip(x, lo_log, hi_log)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(_LOG, (np.log(x) - np.log(lo_log)) / (np.log(hi_log) - np.log(lo_log)),
                         (x - lo) / (hi - lo))
        return np.where(hi > lo, u, 0.0)

    history = []

    def make_loss(transplants, stage, search=-1):
        """error on `transplants` of the parameters at a point of the unit cube"""
        cache = {}

        def loss(u):
            params = _to_params(to_x(u))
            key = (params["smooth_winsize"], params["vel_onset"], params["vel_offset"],
                   params["margin"], params["blinkwindow"])
            if key not in cache:
                cache[key] = transplant_error_mahot(transplants, **params)
                history.append((_to_vector(params), cache[key], stage, search))
            return cache[key]
        return loss

    # stage 1: search on each set of `train`, from the starting parameters
    # or the best of an even cover of the search range
    if n_init is None:
        n_init = 64 if start is None else 0
    if start is None and n_init < 1:
        raise ValueError("the search needs starting parameters or n_init > 0")
    points = np.zeros((0, len(_NAMES)))
    if n_init > 0:
        points = qmc.Sobol(len(_NAMES), seed=rng).random_base2(int(np.ceil(np.log2(n_init))))
    if start is not None:
        start = _to_params(_to_vector(start))
        start_u = to_u(_to_vector(start))
        start_u[hi <= lo] = 0.0
        points = np.vstack([start_u, points])
    points[:, hi <= lo] = 0.0
    found = []
    for i, trs in enumerate(sets):
        loss = make_loss(trs, 1, i)
        losses = np.array([loss(u) for u in points])
        u, best = points[np.argmin(losses)].copy(), losses.min()
        u, best, width = _coordinate_search(loss, u, best, free, 0.5, n_grid, sweeps, tol)
        found.append(u)
        logger.debug(f"Tuning: search {i + 1} of {len(sets)}: {best:.4g}")

    # stage 2: the best of the searches (or the starting parameters, as the
    # searches can fit their few blinks too closely) on further blinks,
    # refined on them and its own blinks with one more round on the finest grid
    k, best_u, tuning = 0, found[0], sets[0] + refine
    if n_refine or len(sets) > 1:
        choose = make_loss(refine if n_refine else train, 2)
        cands = found + ([start_u] if start is not None else [])
        losses = [choose(u) for u in cands]
        k = int(np.argmin(losses))
        best_u = cands[k]
        if k == len(found):
            logger.debug("Tuning: refining from the starting parameters")
            k = 0
        tuning = (sets[k] if n_refine else []) + (refine if n_refine else train)
        logger.debug(f"Tuning: refining on {sum(len(tr) for tr in tuning)} blinks")
        loss = make_loss(tuning, 2)
        best_u, best, _ = _coordinate_search(loss, best_u, loss(best_u), free, 2 * width,
                                             n_grid, 1, tol)

    tuned = _to_params(to_x(best_u))
    loss_start = transplant_error_mahot(tuning, **start) if start is not None else np.nan
    test_loss = transplant_error_mahot(test, **tuned) if n_test else np.nan
    test_loss_start = (transplant_error_mahot(test, **start)
                       if n_test and start is not None else np.nan)
    # use the tuned parameters unless they are worse on the test blinks
    success = not (test_loss >= test_loss_start)
    if not success:
        logger.warning("The tuned parameters are not better than the starting parameters on "
                       f"the blinks that were not used for tuning (error {test_loss:.4g} vs. "
                       f"{test_loss_start:.4g}): keeping the starting parameters")
    hist = {name: np.array([h[0][i] for h in history]) for i, name in enumerate(_NAMES)}
    hist["vel_onset"] = -hist["vel_onset"]
    hist["loss"] = np.array([h[1] for h in history])
    hist["stage"] = np.array([h[2] for h in history], dtype=int)
    hist["search"] = np.array([h[3] for h in history], dtype=int)
    out_bounds = {name: (float(lo[i]), float(hi[i])) for i, name in enumerate(_NAMES)}
    out_bounds["vel_onset"] = (-out_bounds["vel_onset"][1], -out_bounds["vel_onset"][0])
    return BlinkTuneResult(params=dict(tuned if success else start), tuned=tuned, start=start,
                           loss=float(best), loss_start=float(loss_start),
                           test_loss=float(test_loss), test_loss_start=float(test_loss_start),
                           success=bool(success), bounds=out_bounds, history=hist,
                           n_blinks=n_blinks, n_refine=n_refine, n_test=n_test,
                           n_searches=len(sets))
