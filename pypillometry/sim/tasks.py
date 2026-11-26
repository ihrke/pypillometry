"""
Ready-made tasks for simulations (see :class:`~pypillometry.sim.Task`).

Each function returns a :class:`~pypillometry.sim.Task` with typical
parameters, which can be changed with its arguments (or extended by
chaining).

- :func:`pvt`: psychomotor vigilance task
- :func:`sart`: sustained attention to response task
- :func:`ftrsgt`: finger-tapping random-sequence generation task with
  thought probes
- :func:`nback`: n-back task
"""
from .task import Task

__all__ = ["pvt", "sart", "ftrsgt", "nback"]


def pvt(duration="10 min", isi="uniform(2 s, 10 s)", rt="lognormal(300 ms, 80 ms)",
        p_respond=0.99, timeout="30 s"):
    """
    Psychomotor vigilance task: a visual target at random intervals, which
    stays until the response.

    Parameters
    ----------
    duration : float, str or distribution
        duration of the task (default 10 min)
    isi : float, str or distribution
        time from the response (or the end of the target) to the next target
    rt : float, str or distribution
        reaction time
    p_respond : float
        probability of a response
    timeout : float, str or distribution
        duration of a target without response

    Returns
    -------
    Task
        events ``target`` (visual, until the response) and ``press``
    """
    return (Task("PVT")
            .block(duration=duration, isi=isi)
            .stimulus("target", "visual", duration="until response", timeout=timeout)
            .response("press", after="target", rt=rt, p=p_respond))


def sart(n=225, soa="1150 ms", p_nogo=1 / 9, rt="lognormal(350 ms, 70 ms)",
         p_go=0.97, p_commission=0.3):
    """
    Sustained attention to response task: digits followed by a mask; press
    for every digit except 3 (no-go).

    Parameters
    ----------
    n : int
        number of trials (default 225)
    soa : float, str or distribution
        time between the digits (default 1150 ms: 250 ms digit, 900 ms
        mask)
    p_nogo : float
        probability of a no-go trial (digit 3)
    rt : float, str or distribution
        reaction time
    p_go : float
        probability of a press in go trials (1 - omission errors)
    p_commission : float
        probability of a press in no-go trials (commission errors)

    Returns
    -------
    Task
        events ``digit`` (with the metadata ``digit``), ``mask`` and
        ``press``; conditions ``go`` and ``nogo``
    """
    return (Task("SART")
            .block(n=n, soa=soa, conditions={"go": 1 - p_nogo, "nogo": p_nogo})
            .stimulus("digit", "visual", duration="250 ms",
                      meta={"digit": {"go": [1, 2, 4, 5, 6, 7, 8, 9], "nogo": [3]}})
            .stimulus("mask", "visual", at="250 ms", duration="900 ms")
            .response("press", after="digit", rt=rt, p={"go": p_go, "nogo": p_commission}))


def ftrsgt(duration="20 min", soa="750 ms", asynchrony="normal(-40 ms, 40 ms)",
           probe_interval="uniform(40 s, 80 s)", rating_rt="lognormal(3 s, 1.5 s)",
           ratings=(1, 2, 3, 4)):
    """
    Finger-tapping random-sequence generation task: tap with the left or
    right finger in synchrony with a beep, in a random sequence, with
    thought probes at random intervals.

    Parameters
    ----------
    duration : float, str or distribution
        the task (tapping and probes) is repeated until this duration
        (default 20 min)
    soa : float, str or distribution
        time between the beeps (default 750 ms)
    asynchrony : float, str or distribution
        time of the tap relative to the beep (negative: before it)
    probe_interval : float, str or distribution
        time of tapping between two thought probes
    rating_rt : float, str or distribution
        time from the probe to the rating
    ratings : sequence
        possible ratings (drawn with equal probabilities)

    Returns
    -------
    Task
        events ``beep`` (auditory), ``tap`` (with the metadata ``side``,
        ``"left"`` or ``"right"``), ``probe`` (visual, until the rating)
        and ``rating`` (with the metadata ``rating``)
    """
    return (Task("FT-RSGT")
            .block(duration=probe_interval, soa=soa)
            .stimulus("beep", "auditory", duration="75 ms")
            .response("tap", after="beep", rt=asynchrony, meta={"side": ["left", "right"]})
            .block(n=1)
            .stimulus("probe", "visual", duration="until response")
            .response("rating", after="probe", rt=rating_rt, meta={"rating": list(ratings)})
            .rest("1 s")
            .repeat(duration=duration))


def nback(level=2, n=120, soa="2500 ms", p_target=1 / 3,
          letters="BCDFGHJKLMNPQRSTVWXZ", rt="lognormal(650 ms, 150 ms)", p_respond=0.97,
          p_correct=(0.9, 0.95)):
    """
    n-back task: a letter every `soa`; respond to every letter with
    "match" (same letter as `level` letters before) or "nomatch".

    Parameters
    ----------
    level : int
        n of the n-back (default 2)
    n : int
        number of trials
    soa : float, str or distribution
        time between the letters (shown for 500 ms)
    p_target : float
        probability of a target (match) trial
    letters : str or sequence
        the letters
    rt : float, str or distribution
        reaction time
    p_respond : float
        probability of a response
    p_correct : tuple of float
        probability of the correct key in target and non-target trials

    Returns
    -------
    Task
        events ``letter`` (with the metadata ``letter``) and ``press``
        (with the metadata ``key``, ``"match"`` or ``"nomatch"``);
        conditions ``target`` and ``nontarget``
    """
    hit, reject = p_correct
    return (Task(f"{level}-back")
            .block(n=n, soa=soa, conditions={"target": p_target, "nontarget": 1 - p_target})
            .stimulus("letter", "visual", duration="500 ms",
                      meta={"letter": {"nback": level, "values": list(letters)}})
            .response("press", after="letter", rt=rt, p=p_respond,
                      meta={"key": {"target": {"match": hit, "nomatch": 1 - hit},
                                    "nontarget": {"nomatch": reject, "match": 1 - reject}}}))
