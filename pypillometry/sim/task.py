"""
:class:`Task`: the design of an experiment, built by chaining blocks and events.
"""
import copy
from dataclasses import dataclass, replace

import numpy as np

from .dist import parse_dist
from .steps import SimStep

__all__ = ["Task"]

#: columns of the metadata of the simulated events (besides the user's)
_COLUMNS = ("kind", "modality", "trial", "block", "condition", "duration", "luminance", "rt")
_MODALITY_COLORS = {"visual": "#4c72b0", "auditory": "#dd8452", "motor": "#55a868"}
UNTIL_RESPONSE = "until response"


def _color(modality):
    return _MODALITY_COLORS.get(modality, "#8172b3")


def _p(p, condition):
    """probability of an event in a condition (dict: 0 for conditions not given)"""
    if isinstance(p, dict):
        return float(p.get(condition, 0.0))
    return float(p)


def _fmt_p(p):
    if isinstance(p, dict):
        return " / ".join(f"{c} {v:g}" for c, v in p.items())
    return f"{p:g}"


def _fmt_time(spec):
    return str(parse_dist(spec))


@dataclass(repr=False)
class Task(SimStep):
    """
    Design of an experiment: blocks of trials with stimuli and responses.

    A task is built by chaining: :meth:`block` starts a block of trials,
    :meth:`stimulus` and :meth:`response` add events to each trial of the
    current block, :meth:`rest` adds a pause and :meth:`repeat` repeats the
    blocks. Each method returns a new task. Times are fixed (``"250 ms"``,
    plain numbers are ms) or distributions (``"uniform(2 s, 10 s)"``,
    ``"lognormal(350 ms, 70 ms)"``, see
    :func:`~pypillometry.sim.parse_dist`).

    In a simulation (:meth:`Simulation.task
    <pypillometry.sim.Simulation.task>`), the task makes the events of the
    data, with the metadata ``kind`` (``"stimulus"`` or ``"response"``),
    ``modality``, ``trial`` (numbered over the task), ``block``,
    ``condition``, ``duration`` (of stimuli, ms), ``luminance``, ``rt``
    (of responses: the time from the event they follow, ms) and the values
    given in `meta` of the events. Ready-made tasks are in
    :mod:`pypillometry.sim.tasks`.

    Parameters
    ----------
    name : str
        name of the task

    Examples
    --------
    >>> import pypillometry as pp
    >>> sart = (pp.sim.Task("SART")
    ...         .block(n=225, soa="1150 ms", conditions={"go": 8/9, "nogo": 1/9})
    ...         .stimulus("digit", "visual", duration="250 ms")
    ...         .stimulus("mask", "visual", at="250 ms", duration="900 ms")
    ...         .response("press", after="digit", rt="lognormal(350 ms, 70 ms)",
    ...                   p={"go": 0.97, "nogo": 0.3}))
    >>> import numpy as np
    >>> ev, end = sart.generate(np.random.default_rng(1))
    >>> sorted(str(x) for x in set(ev["label"]))
    ['digit', 'mask', 'press']
    """
    name: str = "task"
    blocks: tuple = ()
    repeat_n: int = None
    repeat_duration: str = None
    start: str = 0
    stage = "events"

    # -- building ---------------------------------------------------------------
    def _with(self, **changes):
        return replace(copy.deepcopy(self), **changes)

    def _current(self):
        if not self.blocks or "rest" in self.blocks[-1]:
            raise ValueError("start a block with .block() first")
        return self.blocks[-1]

    def block(self, n=None, duration=None, soa=None, isi=None, conditions=None):
        """
        Start a block of trials.

        Parameters
        ----------
        n : int, optional
            number of trials
        duration : float, str or distribution, optional
            duration of the block (trials start until it is over); `n`
            and/or `duration` must be given
        soa : float, str or distribution, optional
            time from the start of one trial to the start of the next
        isi : float, str or distribution, optional
            time from the end of one trial (its last event, or the end of
            the last stimulus) to the start of the next; default 0 if
            neither `soa` nor `isi` is given
        conditions : dict or list, optional
            conditions of the trials with their probabilities (e.g.
            ``{"go": 8/9, "nogo": 1/9}``; a list for equal probabilities)

        Returns
        -------
        Task
        """
        if n is None and duration is None:
            raise ValueError("a block needs n (trials) and/or a duration")
        if soa is not None and isi is not None:
            raise ValueError("give soa or isi, not both")
        for spec in (duration, soa, isi):
            if spec is not None:
                parse_dist(spec)
        if isinstance(conditions, (list, tuple)):
            conditions = {str(c): 1.0 / len(conditions) for c in conditions}
        if conditions is not None:
            conditions = {str(c): float(p) for c, p in conditions.items()}
            if any(p < 0 for p in conditions.values()) or sum(conditions.values()) <= 0:
                raise ValueError("the probabilities of the conditions must be positive")
        block = {"n": None if n is None else int(n), "duration": duration, "soa": soa,
                 "isi": isi, "conditions": conditions, "events": []}
        return self._with(blocks=self.blocks + (block,))

    def rest(self, duration):
        """
        Add a pause without events.

        Parameters
        ----------
        duration : float, str or distribution
            duration of the pause

        Returns
        -------
        Task
        """
        parse_dist(duration)
        return self._with(blocks=self.blocks + ({"rest": duration},))

    def _add_event(self, event):
        out = self._with()
        block = out._current()
        labels = [e["label"] for e in block["events"]]
        if event["label"] in labels:
            raise ValueError(f"the block has an event {event['label']!r} already")
        if event["after"] is not None and event["after"] not in labels:
            raise ValueError(f"after={event['after']!r}: no such event before it in the "
                             f"block (events: {labels})")
        conditions = block["conditions"] or {}
        if isinstance(event["p"], dict):
            unknown = set(event["p"]) - set(conditions)
            if unknown:
                raise ValueError(f"p: unknown conditions {sorted(unknown)}")
            event["p"] = {str(c): float(v) for c, v in event["p"].items()}
        else:
            event["p"] = float(event["p"])
        for name in event["meta"] or {}:
            if name in _COLUMNS or name in ("onset", "label"):
                raise ValueError(f"meta: {name!r} is a column of the events already")
        block["events"].append(event)
        return out

    def stimulus(self, label, modality="visual", at=0, after=None, duration=None,
                 timeout="30 s", p=1.0, luminance=None, meta=None):
        """
        Add a stimulus to each trial of the current block.

        Parameters
        ----------
        label : str
            label of the event
        modality : str
            ``"visual"`` (default), ``"auditory"`` or another modality
        at : float, str or distribution
            onset relative to the start of the trial, or to the event
            `after` (default 0)
        after : str, optional
            label of an earlier event of the trial that `at` refers to
        duration : float, str or distribution, optional
            duration of the stimulus; ``"until response"``: until the first
            response that follows it (or `timeout`)
        timeout : float, str or distribution
            duration of a stimulus ``"until response"`` without a response
            (default 30 s)
        p : float or dict
            probability of the stimulus (per condition with a dict; 0 for
            conditions that are not given)
        luminance : float, optional
            change of the luminance at the onset (for visual stimuli; used
            by future pupil responses)
        meta : dict, optional
            further metadata, name -> values: a value, a list (drawn with
            equal probabilities), a dict value -> probability, a dict
            condition -> one of these, or ``{"nback": k, "values": [...]}``
            for the stimuli of an n-back task (in condition ``"target"``
            the value of k trials before, otherwise another value)

        Returns
        -------
        Task
        """
        if duration is not None and duration != UNTIL_RESPONSE:
            parse_dist(duration)
        parse_dist(at)
        return self._add_event({"label": str(label), "kind": "stimulus", "modality": modality,
                                "at": at, "after": after, "duration": duration,
                                "timeout": timeout, "p": p, "luminance": luminance,
                                "meta": _meta_spec(meta)})

    def response(self, label, after=None, rt=0, p=1.0, modality="motor", meta=None):
        """
        Add a response to each trial of the current block.

        Parameters
        ----------
        label : str
            label of the event
        after : str, optional
            label of the earlier event of the trial that it responds to; if
            None, `rt` is relative to the start of the trial
        rt : float, str or distribution
            time from the event `after` (the reaction time; may be negative,
            e.g. for taps that anticipate a tone)
        p : float or dict
            probability of the response (per condition with a dict; 0 for
            conditions that are not given)
        modality : str
            default ``"motor"``
        meta : dict, optional
            further metadata (see :meth:`stimulus`)

        Returns
        -------
        Task
        """
        parse_dist(rt)
        return self._add_event({"label": str(label), "kind": "response", "modality": modality,
                                "at": rt, "after": after, "duration": None, "timeout": None,
                                "p": p, "luminance": None, "meta": _meta_spec(meta)})

    def repeat(self, n=None, duration=None):
        """
        Repeat the blocks of the task.

        Parameters
        ----------
        n : int, optional
            number of times the blocks are run
        duration : float, str or distribution, optional
            the blocks are run again until this duration is reached

        Returns
        -------
        Task
        """
        if n is None and duration is None:
            raise ValueError("give n and/or duration")
        if duration is not None:
            parse_dist(duration)
        return self._with(repeat_n=None if n is None else int(n), repeat_duration=duration)

    # -- events -----------------------------------------------------------------
    def apply(self, run, rng):
        run.events = self.generate(rng)[0]

    def generate(self, rng):
        """
        Simulate the events of the task.

        Parameters
        ----------
        rng : np.random.Generator

        Returns
        -------
        events : dict
            column -> array: ``onset`` (ms), ``label`` and the metadata
            (see :class:`Task`), sorted by onset
        end : float
            end of the task (ms)
        """
        if not any("events" in b for b in self.blocks):
            raise ValueError(f"the task {self.name!r} has no blocks with events")
        rows, trial, run = [], 0, 0
        t = float(parse_dist(self.start).sample(rng))
        until = None if self.repeat_duration is None else \
            float(parse_dist(self.repeat_duration).sample(rng))
        while True:
            start = t
            for b, block in enumerate(self.blocks):
                if "rest" in block:
                    t += max(float(parse_dist(block["rest"]).sample(rng)), 0.0)
                else:
                    t, trial = _run_block(block, b + 1, t, trial, rng, rows)
            run += 1
            if self.repeat_n is None and until is None:
                break
            if self.repeat_n is not None and run >= self.repeat_n:
                break
            if until is not None and (t >= until or t <= start):
                break
        return _table(rows), t

    # -- display ----------------------------------------------------------------
    def __repr__(self):
        lines = [f"Task({self.name!r}): {len(self.blocks)} block(s)"
                 + (f", starts at {_fmt_time(self.start)}" if self.start != 0 else "")
                 + (f", repeated {self.repeat_n} times" if self.repeat_n else "")
                 + (f", repeated for {_fmt_time(self.repeat_duration)}"
                    if self.repeat_duration is not None else "")]
        for i, block in enumerate(self.blocks, 1):
            lines.append(f"  {i}. {_block_text(block)}")
            for e in block.get("events", []):
                lines.append(f"     - {_event_text(e)}")
        return "\n".join(lines)

    def plot(self, axes=None, figsize=None):
        """
        Schematic of the trials of each block of the task.

        One panel per block with events: one lane per event, stimuli as
        bars (coloured by modality), the onsets that vary as a band from
        the 10% to the 90% quantile around the mean, the spacing of the
        trials and the probabilities of the events.

        Parameters
        ----------
        axes : sequence of Axes, optional
            axes to draw on (one per block with events); default: a new
            figure
        figsize : tuple, optional
            size of the new figure

        Returns
        -------
        matplotlib Figure
        """
        from ..plot._axes import get_axes
        blocks = [(i, b) for i, b in enumerate(self.blocks, 1) if "events" in b]
        if figsize is None:
            figsize = (9, sum(0.9 + 0.45 * max(len(b["events"]), 1) for _, b in blocks))
        own = axes is None
        fig, axes = get_axes(axes, len(blocks), what="block", figsize=figsize)
        for ax, (i, block) in zip(axes, blocks):
            _plot_block(ax, block, f"{self.name}: block {i} – {_block_text(block)}")
        if own:
            fig.tight_layout()
        return fig

    def _repr_html_(self):
        parts = [f'<div style="font-family:system-ui, sans-serif; border:2px solid #f0a04b; '
                 f'background:#fff7ec; border-radius:6px; padding:12px; max-width:min(640px, 60vw);">'
                 f'<h3 style="margin-top:0;">Task — <span style="color:#555;">{_esc(self.name)}'
                 f'</span></h3>']
        if self.repeat_n or self.repeat_duration is not None:
            rep = (f"{self.repeat_n} times" if self.repeat_n else
                   f"for {_fmt_time(self.repeat_duration)}")
            parts.append(f'<div style="margin-bottom:6px;">blocks repeated {rep}</div>')
        for i, block in enumerate(self.blocks, 1):
            parts.append(f'<div style="font-weight:600; margin-top:6px;">{i}. '
                         f'{_esc(_block_text(block))}</div>')
            if block.get("events"):
                parts.append(_block_svg(block))
        legend = " ".join(f'<span style="color:{c};">■</span> {m}'
                          for m, c in _MODALITY_COLORS.items())
        parts.append(f'<div style="font-size:0.8em; color:#666; margin-top:4px;">{legend}; '
                     f'bands: 10–90% of the onsets</div></div>')
        return "".join(parts)


# -- helpers: generation --------------------------------------------------------
def _meta_spec(meta):
    """metadata specification with string keys for weighted values"""
    if meta is None:
        return None
    out = {}
    for name, spec in meta.items():
        out[str(name)] = _plain_spec(spec)
    return out


def _plain_spec(spec):
    if isinstance(spec, dict) and "nback" not in spec:
        return {str(k): (_plain_spec(v) if isinstance(v, (dict, list, tuple)) else v)
                for k, v in spec.items()}
    if isinstance(spec, tuple):
        return list(spec)
    if isinstance(spec, dict):
        return {"nback": int(spec["nback"]), "values": list(spec["values"])}
    return spec


def _choose(spec, rng):
    """one value from a value, list or dict value -> probability"""
    if isinstance(spec, (list, tuple)):
        return spec[rng.integers(len(spec))]
    if isinstance(spec, dict):
        values = list(spec)
        p = np.array([float(spec[v]) for v in values])
        return values[rng.choice(len(values), p=p / p.sum())]
    return spec


def _meta_value(name, spec, condition, conditions, rng, history):
    if isinstance(spec, dict) and "nback" in spec:
        k, values = spec["nback"], list(spec["values"])
        hist = history.setdefault(name, [])
        if condition == "target" and len(hist) >= k:
            v = hist[-k]
        else:
            others = [x for x in values if len(hist) < k or x != hist[-k]]
            v = others[rng.integers(len(others))]
        hist.append(v)
        return v
    if isinstance(spec, dict) and conditions and set(spec) <= set(conditions):
        if condition not in spec:
            return None
        spec = spec[condition]
    return _choose(spec, rng)


def _run_block(block, b, t, trial, rng, rows):
    conditions = block["conditions"]
    names = list(conditions) if conditions else [""]
    probs = np.array([conditions[c] for c in names]) if conditions else np.ones(1)
    probs = probs / probs.sum()
    nback = {name: spec["nback"] for e in block["events"] for name, spec in (e["meta"] or {}).items()
             if isinstance(spec, dict) and "nback" in spec}
    end = None if block["duration"] is None else \
        t + float(parse_dist(block["duration"]).sample(rng))
    history = {}
    onset, k = t, 0
    while (block["n"] is None or k < block["n"]) and (end is None or onset < end):
        condition = names[rng.choice(len(names), p=probs)]
        if condition == "target" and nback and k < max(nback.values()):
            # no target before there are enough stimuli
            others = [c for c in names if c != "target"]
            if others:
                condition = others[rng.integers(len(others))]
        occurred, trial_rows = {}, []
        for e in block["events"]:
            if rng.random() >= _p(e["p"], condition):
                continue
            ref = onset if e["after"] is None else occurred.get(e["after"])
            if ref is None:
                continue
            at = float(parse_dist(e["at"]).sample(rng))
            row = {"onset": ref + at, "label": e["label"], "kind": e["kind"],
                   "modality": e["modality"], "trial": trial, "block": b,
                   "condition": condition, "duration": np.nan,
                   "luminance": np.nan if e["luminance"] is None else float(e["luminance"]),
                   "rt": at if (e["kind"] == "response" and e["after"] is not None) else np.nan}
            if e["kind"] == "stimulus" and e["duration"] is not None \
                    and e["duration"] != UNTIL_RESPONSE:
                row["duration"] = max(float(parse_dist(e["duration"]).sample(rng)), 0.0)
            for name, spec in (e["meta"] or {}).items():
                row[name] = _meta_value(name, spec, condition, conditions, rng, history)
            occurred[e["label"]] = row["onset"]
            trial_rows.append((e, row))
        for e, row in trial_rows:
            if e["kind"] == "stimulus" and e["duration"] == UNTIL_RESPONSE:
                resp = [r["onset"] for f, r in trial_rows
                        if f["kind"] == "response" and f["after"] == e["label"]]
                row["duration"] = (resp[0] - row["onset"]) if resp else \
                    float(parse_dist(e["timeout"]).sample(rng))
        ends = [r["onset"] + (0 if np.isnan(r["duration"]) else r["duration"])
                for _, r in trial_rows]
        trial_end = max(ends + [onset])
        rows.extend(r for _, r in trial_rows)
        if block["soa"] is not None:
            onset = onset + max(float(parse_dist(block["soa"]).sample(rng)), 0.0)
        else:
            gap = 0.0 if block["isi"] is None else float(parse_dist(block["isi"]).sample(rng))
            onset = trial_end + max(gap, 0.0)
        trial += 1
        k += 1
        if block["soa"] is None and block["isi"] is None and onset <= trial_end and not trial_rows:
            break       # nothing happens and no time passes
    return onset, trial


def _table(rows):
    """dict of columns from the event rows, sorted by onset"""
    rows = sorted(rows, key=lambda r: r["onset"])
    names = ["onset", "label", *_COLUMNS]
    for r in rows:
        for k in r:
            if k not in names:
                names.append(k)
    out = {}
    for name in names:
        values = [r.get(name) for r in rows]
        if all(isinstance(v, (int, float, np.integer, np.floating)) or v is None for v in values):
            out[name] = np.array([np.nan if v is None else float(v) for v in values])
        else:
            out[name] = np.array(["" if v is None else str(v) for v in values])
    for name in ("trial", "block"):
        out[name] = out[name].astype(int) if len(rows) else np.zeros(0, dtype=int)
    return out


# -- helpers: display -----------------------------------------------------------
def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _block_text(block):
    if "rest" in block:
        return f"rest {_fmt_time(block['rest'])}"
    parts = []
    if block["n"] is not None:
        parts.append(f"{block['n']} trial{'s' if block['n'] != 1 else ''}")
    if block["duration"] is not None:
        parts.append(f"for {_fmt_time(block['duration'])}")
    if block["soa"] is not None:
        parts.append(f"SOA {_fmt_time(block['soa'])}")
    if block["isi"] is not None:
        parts.append(f"ISI {_fmt_time(block['isi'])}")
    if block["conditions"]:
        total = sum(block["conditions"].values())
        parts.append(" / ".join(f"{c} {p / total:.0%}" for c, p in block["conditions"].items()))
    return ", ".join(parts)


def _event_text(e):
    out = f"{e['kind']} {e['label']!r} ({e['modality']})"
    out += (f" after {e['after']!r}" if e["after"] is not None else "")
    if e["kind"] == "response" and e["after"] is not None:
        out += f", rt {_fmt_time(e['at'])}"
    elif parse_dist(e["at"]).mean != 0 or parse_dist(e["at"]).kind != "fixed":
        out += f", at {_fmt_time(e['at'])}"
    if e["duration"] is not None:
        out += f", {e['duration'] if e['duration'] == UNTIL_RESPONSE else _fmt_time(e['duration'])}"
    if e["p"] != 1.0:
        out += f", p {_fmt_p(e['p'])}"
    if e["luminance"] is not None:
        out += f", luminance {e['luminance']:g}"
    if e["meta"]:
        out += f", meta {', '.join(e['meta'])}"
    return out


def _layout(block):
    """nominal onsets of the events of a trial: label -> (lo, mean, hi, end)"""
    pos = {}
    for e in block["events"]:
        ref = (0.0, 0.0, 0.0) if e["after"] is None else pos[e["after"]][:3]
        d = parse_dist(e["at"])
        lo, mean, hi = ref[0] + d.quantile(0.1), ref[1] + d.mean, ref[2] + d.quantile(0.9)
        pos[e["label"]] = (lo, mean, hi, mean)
    for e in block["events"]:
        lo, mean, hi, _ = pos[e["label"]]
        if e["duration"] == UNTIL_RESPONSE:
            resp = [pos[f["label"]][1] for f in block["events"]
                    if f["kind"] == "response" and f["after"] == e["label"]]
            end = resp[0] if resp else mean + parse_dist(e["timeout"]).mean
        elif e["duration"] is not None:
            end = mean + parse_dist(e["duration"]).mean
        else:
            end = mean
        pos[e["label"]] = (lo, mean, hi, end)
    return pos


def _extent(block, pos):
    lo = min(0.0, *(p[0] for p in pos.values()))
    hi = max(0.0, *(max(p[2], p[3]) for p in pos.values()))
    if block["soa"] is not None:
        hi = max(hi, parse_dist(block["soa"]).mean)
    if hi <= lo:
        hi = lo + 1000.0
    return lo, hi + 0.05 * (hi - lo)


def _plot_block(ax, block, title):
    events = block["events"]
    pos = _layout(block)
    lo, hi = _extent(block, pos)
    for k, e in enumerate(events):
        y = len(events) - 1 - k
        elo, mean, ehi, end = pos[e["label"]]
        c = _color(e["modality"])
        if ehi > elo:
            ax.fill_between([elo, ehi], y - 0.3, y + 0.3, color=c, alpha=0.2, lw=0)
        if e["kind"] == "stimulus" and end > mean:
            ax.fill_between([mean, end], y - 0.25, y + 0.25, color=c, alpha=0.8, lw=0,
                            hatch="//" if e["duration"] == UNTIL_RESPONSE else None)
        ax.plot([mean], [y], marker="|" if e["kind"] == "stimulus" else "v", color=c,
                markersize=14 if e["kind"] == "stimulus" else 9, mew=2)
        note = f"{e['label']} ({e['modality']})"
        if e["p"] != 1.0:
            note += f"  p {_fmt_p(e['p'])}"
        ax.text(hi, y, "  " + note, va="center", fontsize=8)
    if block["soa"] is not None:
        nxt = parse_dist(block["soa"]).mean
        ax.axvline(nxt, color="0.5", ls="--", lw=1)
        ax.text(nxt, len(events) - 0.4, " next trial", fontsize=8, color="0.4")
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, len(events) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("time in the trial (ms)")
    ax.set_title(title, fontsize=9, loc="left")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def _block_svg(block, width=520, lane=18):
    events = block["events"]
    pos = _layout(block)
    lo, hi = _extent(block, pos)
    label_w, pad = 150, 8
    plot_w = width - label_w - pad
    h = lane * len(events) + 22

    def x(t):
        return pad + (t - lo) / (hi - lo) * plot_w

    out = [f'<svg width="{width}" height="{h}" xmlns="http://www.w3.org/2000/svg" '
           f'style="display:block; font-size:11px; font-family:system-ui, sans-serif;">']
    out.append(f'<line x1="{x(0):.1f}" y1="0" x2="{x(0):.1f}" y2="{h - 18}" stroke="#bbb"/>')
    if block["soa"] is not None:
        xs = x(parse_dist(block["soa"]).mean)
        out.append(f'<line x1="{xs:.1f}" y1="0" x2="{xs:.1f}" y2="{h - 18}" stroke="#888" '
                   f'stroke-dasharray="4,3"/>')
    for k, e in enumerate(events):
        y = k * lane + 4
        elo, mean, ehi, end = pos[e["label"]]
        c = _color(e["modality"])
        if ehi > elo:
            out.append(f'<rect x="{x(elo):.1f}" y="{y}" width="{x(ehi) - x(elo):.1f}" '
                       f'height="{lane - 6}" fill="{c}" opacity="0.2"/>')
        if e["kind"] == "stimulus" and end > mean:
            dash = ' stroke="#333" stroke-dasharray="2,2"' if e["duration"] == UNTIL_RESPONSE else ""
            out.append(f'<rect x="{x(mean):.1f}" y="{y + 2}" width="{max(x(end) - x(mean), 1):.1f}" '
                       f'height="{lane - 10}" fill="{c}" opacity="0.85"{dash}/>')
        if e["kind"] == "response":
            xm = x(mean)
            out.append(f'<polygon points="{xm - 5:.1f},{y + 1} {xm + 5:.1f},{y + 1} {xm:.1f},'
                       f'{y + lane - 7}" fill="{c}"/>')
        else:
            out.append(f'<line x1="{x(mean):.1f}" y1="{y}" x2="{x(mean):.1f}" '
                       f'y2="{y + lane - 6}" stroke="{c}" stroke-width="2"/>')
        note = _esc(e["label"]) + (f" · p {_esc(_fmt_p(e['p']))}" if e["p"] != 1.0 else "")
        out.append(f'<text x="{pad + plot_w + 6}" y="{y + lane - 8}" fill="#333">{note}</text>')
    out.append(f'<text x="{x(lo):.1f}" y="{h - 4}" fill="#666">{lo:.0f} ms</text>')
    out.append(f'<text x="{x(hi):.1f}" y="{h - 4}" fill="#666" text-anchor="end">'
               f'{hi:.0f} ms</text>')
    out.append("</svg>")
    return "".join(out)
