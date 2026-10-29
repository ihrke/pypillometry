"""
quality.py
==========

Data-quality report for eye-tracking data.

The report is created by :meth:`~pypillometry.GenericEyeData.quality_report`
and is a :class:`QualityReport`. It collects quality measures (each a
:class:`QualityMetric`) in sections:

- ``info``: duration, sampling rate etc. (no labels)
- ``pupil``: missing data, dropouts and noise for each eye with pupil data
- ``gaze``: missing data, off-screen samples and precision for each eye with
  gaze data
- ``binocular``: agreement between the eyes (if both eyes are present)
- ``blinks``: blink rate and duration (if blinks were detected)
- ``artifacts``: masked and interpolated data (if any preprocessing was done)
- ``calibration``: validation errors (if calibration data is available)

The measures in ``pupil``, ``gaze`` and ``binocular`` use only the stored
values (missing samples are NaN) and do not assume any preprocessing. Where
it makes sense, a measure is labeled as "good", "ok", "poor" or "bad" based
on thresholds (see :data:`DEFAULT_QUALITY_THRESHOLDS`).
"""

import html
import os
import sys
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

__all__ = ["QualityReport", "QualityMetric"]

#: Quality labels, from best to worst.
LABELS = ("good", "ok", "poor", "bad")

#: Default thresholds for the quality labels.
#:
#: Each entry maps ``"<section>.<metric>"`` to three bounds
#: ``(good, ok, poor)``. If the bounds are increasing, lower values are better
#: (``value <= good`` is "good", ``value <= ok`` is "ok", ``value <= poor`` is
#: "poor" and anything above is "bad"). If they are decreasing, higher values
#: are better (``value >= good`` is "good", etc.). `None` means that the
#: measure is reported without a label.
DEFAULT_QUALITY_THRESHOLDS = {
    # pupil (raw data)
    "pupil.missing_pct": (5, 15, 30),
    "pupil.dropout_rate": (30, 45, 60),
    "pupil.longest_dropout": (1, 3, 10),
    "pupil.noise": (0.5, 1, 2),
    "pupil.noise_sd": None,
    # gaze (raw data)
    "gaze.missing_pct": (5, 15, 30),
    "gaze.dropout_rate": (30, 45, 60),
    "gaze.longest_dropout": (1, 3, 10),
    "gaze.offscreen_pct": (5, 15, 30),
    "gaze.precision_deg": (0.05, 0.1, 0.2),
    "gaze.precision_px": None,
    # both eyes
    "binocular.pupil_correlation": (0.9, 0.8, 0.6),
    "binocular.gaze_disparity_deg": (1, 2, 3),
    "binocular.gaze_disparity_px": None,
    # blinks (after blink detection)
    "blinks.n": None,
    "blinks.rate": (30, 45, 60),
    "blinks.mean_duration": None,
    "blinks.time_pct": (5, 10, 20),
    "blinks.n_long": None,
    # masks and interpolation (after preprocessing)
    "artifacts.pupil_masked_pct": (10, 20, 35),
    "artifacts.gaze_masked_pct": (10, 20, 35),
    "artifacts.interpolated_pct": (10, 20, 30),
    "artifacts.other_pct": None,
    # calibration/validation (thresholds as used by Eyelink)
    "calibration.mean_error": (0.5, 1.0, 1.5),
    "calibration.max_error": (1.0, 1.5, 2.0),
    "calibration.n_points": None,
    "calibration.model": None,
}

#: Blinks longer than this (in ms) are counted as "long blinks".
LONG_BLINK_MS = 500

SECTION_TITLES = {
    "info": "Recording",
    "pupil": "Pupil",
    "gaze": "Gaze",
    "binocular": "Binocular",
    "blinks": "Blinks",
    "artifacts": "Artifacts",
    "calibration": "Calibration",
}

# colors for the labels (HTML and ANSI escape codes)
_HTML_COLORS = {"good": "#2e7d32", "ok": "#9a7d0a", "poor": "#e65100", "bad": "#c62828"}
_ANSI_COLORS = {"good": "\033[32m", "ok": "\033[33m", "poor": "\033[38;5;208m",
                "bad": "\033[1;31m"}
_ANSI_RESET = "\033[0m"


def _label(value, bounds) -> Optional[str]:
    """Return the label for `value` given `bounds` (see DEFAULT_QUALITY_THRESHOLDS)."""
    if bounds is None or value is None or isinstance(value, str):
        return None
    if not np.isfinite(value):
        return None
    lower_is_better = bounds[0] <= bounds[-1]
    for lab, bound in zip(LABELS, bounds):
        if (value <= bound) if lower_is_better else (value >= bound):
            return lab
    return LABELS[-1]


def _bounds_text(bounds) -> str:
    """Describe the thresholds in words."""
    if bounds is None:
        return "no label"
    op = "<=" if bounds[0] <= bounds[-1] else ">="
    return ", ".join(f"{lab} {op} {b:g}" for lab, b in zip(LABELS, bounds)) + ", else bad"


def _worst(labels) -> Optional[str]:
    """Return the worst of `labels` (ignoring None)."""
    labels = [lab for lab in labels if lab is not None]
    if not labels:
        return None
    return max(labels, key=LABELS.index)


def _ansi_label(label, width=0, color=True) -> str:
    """Return `label` padded to `width`, colored with ANSI codes if `color`."""
    if label is None:
        return " " * width
    txt = label.ljust(width)
    return _ANSI_COLORS[label] + txt + _ANSI_RESET if color else txt


def _html_badge(label, big=False) -> str:
    """Return `label` as a colored HTML badge."""
    if label is None:
        return ""
    pad = "3px 10px" if big else "1px 6px"
    size = "1em" if big else "0.8em"
    return (f'<span style="background:{_HTML_COLORS[label]}; color:#fff; '
            f'border-radius:3px; padding:{pad}; font-size:{size}; '
            f'font-weight:600; text-transform:uppercase;">{label}</span>')


_HTML_CELL = "padding:3px 8px; border-bottom:1px solid rgba(128,128,128,0.25);"


def _supports_color(stream=None) -> bool:
    """Whether colored (ANSI) output is likely to be shown correctly."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if "ipykernel" in sys.modules:  # Jupyter shows ANSI colors in outputs
        return True
    stream = sys.stdout if stream is None else stream
    try:
        return stream.isatty()
    except Exception:
        return False


@dataclass
class QualityMetric:
    """
    A single quality measure.

    Attributes
    ----------
    name : str
        short name (e.g., "missing_pct")
    title : str
        human-readable name (e.g., "missing data")
    value : float, int, str or None
        the value (NaN or None if it could not be computed)
    unit : str
        unit of `value` (e.g., "%", "s", "deg")
    label : str or None
        "good", "ok", "poor", "bad", or None if the measure has no label
    description : str
        what is measured
    thresholds : tuple or None
        the thresholds used for the label
    """
    name: str
    title: str
    value: Union[float, int, str, None]
    unit: str = ""
    label: Optional[str] = None
    description: str = ""
    thresholds: Optional[tuple] = None

    def format_value(self) -> str:
        """Return the value with unit as a short string."""
        v = self.value
        if v is None:
            return "n/a"
        if isinstance(v, str):
            return v
        if isinstance(v, (int, np.integer)):
            s = str(int(v))
        elif not np.isfinite(v):
            return "n/a"
        elif self.unit == "%":
            s = f"{v:.1f}" if abs(v) >= 1 or v == 0 else f"{v:.2g}"
        elif abs(v) >= 100:
            s = f"{v:.0f}"
        else:
            s = f"{v:.3g}"
        if not self.unit:
            return s
        sep = "" if self.unit in ("%", "°") else " "
        return s + sep + self.unit

    def to_dict(self) -> dict:
        """Return the metric as a dict (value, unit, label, description)."""
        v = self.value
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        return dict(value=v, unit=self.unit, label=self.label,
                    description=self.description)


class QualityReport:
    """
    Data-quality report for an eyedata object.

    Created by :meth:`~pypillometry.GenericEyeData.quality_report`. The
    report can be

    - shown as text (``print(report)``; labels are colored if the terminal
      supports it),
    - shown as HTML in a Jupyter notebook (just display the object),
    - converted to a dict (:meth:`to_dict`), e.g., to collect the reports
      of several datasets in a :class:`pandas.DataFrame`.

    Individual measures can be accessed like a dict:
    ``report["pupil"]["left"]["missing_pct"]`` is a :class:`QualityMetric`.

    Attributes
    ----------
    name : str
        name of the dataset
    class_name : str
        class of the dataset (e.g., "EyeData")
    sections : dict
        ``{section: {group: {metric_name: QualityMetric}}}``, where the group
        is usually the eye (e.g., "left")
    thresholds : dict
        thresholds used for the labels
    overall : str or None
        worst label of all measures ("good", "ok", "poor" or "bad")
    """

    def __init__(self, name: str, class_name: str, sections: dict, thresholds: dict):
        self.name = name
        self.class_name = class_name
        self.sections = sections
        self.thresholds = thresholds

    def __getitem__(self, section):
        return self.sections[section]

    def __contains__(self, section):
        return section in self.sections

    def keys(self):
        """Return the available sections."""
        return self.sections.keys()

    def metrics(self):
        """Iterate over all measures as ``(section, group, QualityMetric)``."""
        for section, groups in self.sections.items():
            for group, metrics in groups.items():
                for metric in metrics.values():
                    yield section, group, metric

    @property
    def overall(self) -> Optional[str]:
        """Worst label of all measures (None if no measure has a label)."""
        return _worst(m.label for _, _, m in self.metrics())

    def labels(self) -> dict:
        """
        Return the worst label for each section and group.

        Returns
        -------
        dict
            ``{section: {group: label}}`` (sections without labels are left out)
        """
        res = {}
        for section, groups in self.sections.items():
            labs = {g: _worst(m.label for m in metrics.values())
                    for g, metrics in groups.items()}
            labs = {g: lab for g, lab in labs.items() if lab is not None}
            if labs:
                res[section] = labs
        return res

    def issues(self, min_label: str = "poor") -> list:
        """
        Return the measures with a label of `min_label` or worse.

        Parameters
        ----------
        min_label : str
            one of "ok", "poor" or "bad" (default "poor")

        Returns
        -------
        list of tuple
            ``(section, group, QualityMetric)``, worst first
        """
        if min_label not in LABELS:
            raise ValueError(f"min_label must be one of {LABELS}")
        imin = LABELS.index(min_label)
        res = [(s, g, m) for s, g, m in self.metrics()
               if m.label is not None and LABELS.index(m.label) >= imin]
        return sorted(res, key=lambda x: -LABELS.index(x[2].label))

    def to_dict(self, flat: bool = False) -> dict:
        """
        Return the report as a dictionary.

        Parameters
        ----------
        flat : bool
            If False (default), return a nested dict
            ``{"name", "class", "overall", <section>: {<group>: {<metric>: {...}}}}``,
            where each metric is a dict with value, unit, label and description.
            If True, return a flat dict with keys like ``"pupil.left.missing_pct"``
            (the value) and ``"pupil.left.missing_pct.label"`` (the label);
            convenient for creating a :class:`pandas.DataFrame` from the
            reports of several datasets.

        Returns
        -------
        dict
        """
        if flat:
            res = {"name": self.name, "overall": self.overall}
            for section, group, m in self.metrics():
                key = f"{section}.{group}.{m.name}"
                res[key] = m.to_dict()["value"]
                if m.label is not None:
                    res[key + ".label"] = m.label
            return res
        res = {"name": self.name, "class": self.class_name, "overall": self.overall}
        for section, groups in self.sections.items():
            res[section] = {g: {name: m.to_dict() for name, m in metrics.items()}
                            for g, metrics in groups.items()}
        return res

    # ------------------------------------------------------------------
    # Text representation
    # ------------------------------------------------------------------
    def to_text(self, color: Optional[bool] = None) -> str:
        """
        Return the report as a (multi-line) text.

        Parameters
        ----------
        color : bool or None
            color the labels with ANSI escape codes; if None (default),
            colors are used if the output supports it (terminal or
            Jupyter; disable by setting the environment variable
            ``NO_COLOR``)

        Returns
        -------
        str
        """
        if color is None:
            color = _supports_color()

        def lab(label, width=0):
            return _ansi_label(label, width, color)

        overall = self.overall
        lines = [f"Quality report: {self.name} ({self.class_name})"]
        if overall is None:
            lines.append("Overall: n/a")
        elif color:
            lines.append("Overall: " + _ANSI_COLORS[overall] + overall.upper() + _ANSI_RESET)
        else:
            lines.append("Overall: " + overall.upper())

        for section, groups in self.sections.items():
            if not groups:
                continue
            lines.append("")
            group_names = list(groups.keys())
            metric_names = list(dict.fromkeys(n for g in groups.values() for n in g))
            titles = {}
            for metrics in groups.values():
                for n, m in metrics.items():
                    titles.setdefault(n, m.title)
            name_w = max(len(t) for t in titles.values()) + 2
            cells = {(g, n): groups[g][n] for g in group_names for n in groups[g]}
            val_w = {g: max([len(g)] + [len(m.format_value()) for (gg, _), m in cells.items()
                                        if gg == g]) for g in group_names}
            has_label = {g: any(m.label for (gg, _), m in cells.items() if gg == g)
                         for g in group_names}
            col_w = {g: val_w[g] + (5 if has_label[g] else 0) for g in group_names}
            title = SECTION_TITLES.get(section, section)
            header = title.ljust(name_w + 2)
            if section != "info":
                header += "  ".join(g.rjust(val_w[g]).ljust(col_w[g]) for g in group_names)
            lines.append(header.rstrip())
            for n in metric_names:
                row = "  " + titles[n].ljust(name_w)
                for g in group_names:
                    m = cells.get((g, n))
                    if m is None:
                        row += " " * col_w[g] + "  "
                        continue
                    row += m.format_value().rjust(val_w[g])
                    if has_label[g]:
                        row += " " + lab(m.label, 4)
                    row += "  "
                lines.append(row.rstrip())

        issues = self.issues()
        if issues:
            lines.append("")
            lines.append("Issues:")
            for section, group, m in issues:
                lines.append(f"  {lab(m.label, 4)} {section}/{group}: {m.title} = "
                             f"{m.format_value()}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return self.to_text()

    # ------------------------------------------------------------------
    # HTML representation
    # ------------------------------------------------------------------
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        esc = html.escape

        badge = _html_badge
        cell = _HTML_CELL
        out = ['<div style="font-family:system-ui, sans-serif; '
               'border:1px solid rgba(128,128,128,0.4); border-radius:6px; '
               'padding:12px; max-width:720px;">']
        out.append('<div style="display:flex; align-items:center; gap:12px; '
                   'margin-bottom:8px;">'
                   f'<h3 style="margin:0;">Quality report — {esc(self.name)} '
                   f'<span style="opacity:0.6; font-weight:normal;">'
                   f'({esc(self.class_name)})</span></h3>'
                   f'{badge(self.overall, big=True)}</div>')

        for section, groups in self.sections.items():
            if not groups:
                continue
            group_names = list(groups.keys())
            metric_names = list(dict.fromkeys(n for g in groups.values() for n in g))
            out.append(f'<div style="font-weight:600; margin-top:10px;">'
                       f'{esc(SECTION_TITLES.get(section, section))}</div>')
            out.append('<table style="border-collapse:collapse; margin-top:4px; '
                       'min-width:60%;">')
            if section != "info":
                out.append("<thead><tr>" + f'<th style="{cell} text-align:left;"></th>' +
                           "".join(f'<th style="{cell} text-align:right;">{esc(g)}</th>'
                                   for g in group_names) + "</tr></thead>")
            out.append("<tbody>")
            for n in metric_names:
                first = next(groups[g][n] for g in group_names if n in groups[g])
                out.append(f'<tr><td style="{cell} text-align:left;" '
                           f'title="{esc(first.description)}">{esc(first.title)}</td>')
                for g in group_names:
                    m = groups[g].get(n)
                    if m is None:
                        out.append(f'<td style="{cell}"></td>')
                        continue
                    tip = esc(m.description + " (" + _bounds_text(m.thresholds) + ")")
                    out.append(f'<td style="{cell} text-align:right; white-space:nowrap;" '
                               f'title="{tip}">{esc(m.format_value())} {badge(m.label)}</td>')
                out.append("</tr>")
            out.append("</tbody></table>")

        issues = self.issues()
        if issues:
            out.append('<div style="font-weight:600; margin-top:10px;">Issues</div><ul '
                       'style="margin:4px 0;">')
            for section, group, m in issues:
                out.append(f"<li>{badge(m.label)} {esc(section)}/{esc(group)}: "
                           f"{esc(m.title)} = {esc(m.format_value())}</li>")
            out.append("</ul>")

        out.append('<details style="margin-top:8px;"><summary style="cursor:pointer; '
                   'opacity:0.7;">Thresholds</summary><table style="border-collapse:'
                   'collapse; font-size:0.85em; margin-top:4px;">')
        for key, bounds in self.thresholds.items():
            if bounds is not None:
                out.append(f'<tr><td style="{cell}">{esc(key)}</td>'
                           f'<td style="{cell}">{esc(_bounds_text(bounds))}</td></tr>')
        out.append("</table><div style=\"font-size:0.85em; opacity:0.7; margin-top:4px;\">"
                   "The overall label is the worst label of all measures.</div>"
                   "</details></div>")
        return "\n".join(out)


# ----------------------------------------------------------------------
# Summary of several reports
# ----------------------------------------------------------------------
def _summary_rows(reports):
    """
    Return the columns and rows of a summary table for several reports.

    Each row has the duration, the overall label, the worst label of each
    section and the number of issues (measures labeled poor or bad).
    """
    sections = [s for s in SECTION_TITLES if s != "info"
                and any(s in r.labels() for r in reports)]
    rows = []
    for r in reports:
        labels = r.labels()
        try:
            duration = r["info"]["all"]["duration"].format_value()
        except KeyError:
            duration = "n/a"
        rows.append(dict(duration=duration, overall=r.overall,
                         sections={s: _worst(labels.get(s, {}).values()) for s in sections},
                         issues=len(r.issues())))
    return sections, rows


def _summary_text(names, reports, color: Optional[bool] = None) -> str:
    """Summary table of several reports as text (one row per report)."""
    if color is None:
        color = _supports_color()
    sections, rows = _summary_rows(reports)
    titles = ["Duration", "Overall"] + [SECTION_TITLES[s] for s in sections] + ["Issues"]
    name_w = max(len(n) for n in list(names) + ["File"])
    dur_w = max(len(r["duration"]) for r in rows + [dict(duration="duration")])
    widths = [dur_w] + [max(len(t), 4) for t in titles[1:]]
    lines = [f"Summary of {len(reports)} files",
             "  ".join(["File".ljust(name_w)] + [t.ljust(w) for t, w in zip(titles, widths)]).rstrip()]
    for name, row in zip(names, rows):
        labs = [row["overall"]] + [row["sections"][s] for s in sections]
        cells = [name.ljust(name_w), row["duration"].rjust(dur_w)]
        cells += [_ansi_label(lab, w, color) if lab else "-".ljust(w)
                  for lab, w in zip(labs, widths[1:-1])]
        cells.append(str(row["issues"]).rjust(widths[-1]))
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def _summary_html(names, reports, anchors=None, tooltips=None) -> str:
    """
    Summary table of several reports as HTML.

    Names link to `anchors` and show `tooltips` on hover if given.
    """
    esc = html.escape
    sections, rows = _summary_rows(reports)
    cell = _HTML_CELL
    out = ['<div style="overflow-x:auto;"><table style="border-collapse:collapse;">',
           "<thead><tr>"]
    for t in ["File", "Duration", "Overall"] + [SECTION_TITLES[s] for s in sections] + ["Issues"]:
        align = "left" if t == "File" else "center"
        out.append(f'<th style="{cell} text-align:{align};">{esc(t)}</th>')
    out.append("</tr></thead><tbody>")
    for i, (name, row) in enumerate(zip(names, rows)):
        label = esc(name)
        if anchors is not None:
            label = f'<a href="#{esc(anchors[i])}">{label}</a>'
        tip = f' title="{esc(tooltips[i])}"' if tooltips is not None else ""
        out.append(f'<tr><td style="{cell}"{tip}>{label}</td>'
                   f'<td style="{cell} text-align:right; white-space:nowrap;">'
                   f'{esc(row["duration"])}</td>')
        for lab in [row["overall"]] + [row["sections"][s] for s in sections]:
            out.append(f'<td style="{cell} text-align:center;">{_html_badge(lab) or "–"}</td>')
        out.append(f'<td style="{cell} text-align:right;">{row["issues"]}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


# ----------------------------------------------------------------------
# Computation
# ----------------------------------------------------------------------
def _merge_thresholds(thresholds: Optional[dict]) -> dict:
    """Merge user thresholds into the defaults (and check them)."""
    res = dict(DEFAULT_QUALITY_THRESHOLDS)
    for key, bounds in (thresholds or {}).items():
        if key not in res:
            raise ValueError(f"Unknown quality measure '{key}'. "
                             f"Available: {', '.join(res)}")
        if bounds is not None:
            bounds = tuple(float(b) for b in bounds)
            if len(bounds) != 3:
                raise ValueError(f"Thresholds for '{key}' must have 3 values "
                                 f"(good, ok, poor), got {bounds}")
            if not (bounds[0] <= bounds[1] <= bounds[2] or bounds[0] >= bounds[1] >= bounds[2]):
                raise ValueError(f"Thresholds for '{key}' must be monotonic, got {bounds}")
        res[key] = bounds
    return res


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive runs of True in a boolean array."""
    if mask.size == 0 or not mask.any():
        return np.array([], dtype=int)
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def _diff_noise(v: np.ndarray) -> float:
    """
    Robust estimate of the SD of high-frequency noise in `v`.

    Uses the median absolute deviation of sample-to-sample differences, which
    is insensitive to the (slow) signal, to blinks and to other jumps. For
    strongly quantized data (MAD of 0), a trimmed SD is used instead.
    """
    d = np.diff(v)
    d = d[np.isfinite(d)]
    if d.size < 10:
        return np.nan
    sd = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    if sd == 0:
        lo, hi = np.percentile(d, [5, 95])
        sd = np.std(d[(d >= lo) & (d <= hi)]) / np.sqrt(2)
    return float(sd)


def _pct(n, total) -> float:
    return 100.0 * n / total if total > 0 else np.nan


class _Builder:
    """Collects the metrics for the report."""

    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.sections = {}

    def add(self, section, group, name, title, value, unit="", description=""):
        bounds = self.thresholds.get(f"{section}.{name}")
        m = QualityMetric(name=name, title=title, value=value, unit=unit,
                          label=_label(value, bounds), description=description,
                          thresholds=bounds)
        self.sections.setdefault(section, {}).setdefault(group, {})[name] = m


def _degrees(setup, x, y):
    """Convert gaze in pixels to degrees, or return None if not possible."""
    if setup is None:
        return None
    try:
        if not (setup.has_screen_info() and setup.has_eye_distance()):
            return None
        return setup.pixels_to_degrees(x, y)
    except Exception:
        return None


def _compute_quality_report(obj, thresholds: Optional[dict] = None, eyes=[]) -> QualityReport:
    """
    Compute a :class:`QualityReport` for an eyedata object.

    See :meth:`~pypillometry.GenericEyeData.quality_report`.
    """
    thresholds = _merge_thresholds(thresholds)
    b = _Builder(thresholds)
    n = len(obj)
    fs = float(obj.fs)
    dur_min = n / fs / 60.0
    raw = obj.data.data
    masks = obj.data.mask

    if isinstance(eyes, str):
        eyes = [eyes]
    pupil_eyes = obj.data.get_available_eyes("pupil")
    gaze_eyes = [e for e in obj.data.get_available_eyes("x") if f"{e}_y" in raw]
    if eyes:
        pupil_eyes = [e for e in pupil_eyes if e in eyes]
        gaze_eyes = [e for e in gaze_eyes if e in eyes]

    # --- recording info -------------------------------------------------
    b.add("info", "all", "duration", "duration", dur_min, "min")
    b.add("info", "all", "sampling_rate", "sampling rate", fs, "Hz")
    b.add("info", "all", "n_samples", "samples", int(n))
    b.add("info", "all", "n_events", "events", int(obj.nevents()))
    if n > 1:
        gaps = int(np.sum(np.diff(obj.tx) > 1.5 * 1000.0 / fs))
        b.add("info", "all", "time_gaps", "time gaps", gaps, "",
              "number of gaps in the time vector (missing samples)")

    def dropouts(section, eye, missing):
        runs = _run_lengths(missing)
        b.add(section, eye, "dropout_rate", "dropouts",
              len(runs) / dur_min if dur_min > 0 else np.nan, "/min",
              "number of periods with missing data per minute (incl. blinks)")
        b.add(section, eye, "longest_dropout", "longest dropout",
              runs.max() / fs if runs.size else 0.0, "s",
              "duration of the longest period with missing data")

    # --- pupil ----------------------------------------------------------
    for eye in pupil_eyes:
        v = raw[f"{eye}_pupil"]
        missing = ~np.isfinite(v)
        b.add("pupil", eye, "missing_pct", "missing data", _pct(missing.sum(), n), "%",
              "percentage of samples without valid pupil data")
        dropouts("pupil", eye, missing)
        sd = _diff_noise(v)
        med = np.nanmedian(v) if (~missing).any() else np.nan
        rel = 100 * sd / abs(med) if np.isfinite(sd) and np.isfinite(med) and med != 0 else np.nan
        b.add("pupil", eye, "noise", "noise", rel, "%",
              "high-frequency noise (robust SD from sample-to-sample differences) "
              "relative to the median pupil size")
        b.add("pupil", eye, "noise_sd", "noise (SD)", sd, "",
              "high-frequency noise in units of the pupil data")

    # --- gaze -----------------------------------------------------------
    setup = getattr(obj, "experimental_setup", None)
    for eye in gaze_eyes:
        x, y = raw[f"{eye}_x"], raw[f"{eye}_y"]
        missing = ~(np.isfinite(x) & np.isfinite(y))
        b.add("gaze", eye, "missing_pct", "missing data", _pct(missing.sum(), n), "%",
              "percentage of samples without valid gaze data")
        if eye not in pupil_eyes:
            dropouts("gaze", eye, missing)
        valid = ~missing
        if setup is not None and setup.screen_resolution is not None and valid.any():
            w, h = setup.screen_resolution
            off = valid & ((x < 0) | (x > w) | (y < 0) | (y > h))
            b.add("gaze", eye, "offscreen_pct", "off-screen", _pct(off.sum(), valid.sum()),
                  "%", "percentage of valid gaze samples outside the screen")
        deg = _degrees(setup, x, y)
        if deg is not None:
            s2s = np.hypot(np.diff(deg[0]), np.diff(deg[1]))
            s2s = s2s[np.isfinite(s2s)]
            b.add("gaze", eye, "precision_deg", "precision",
                  float(np.median(s2s)) if s2s.size else np.nan, "deg",
                  "median sample-to-sample distance of gaze in degrees")
        else:
            s2s = np.hypot(np.diff(x), np.diff(y))
            s2s = s2s[np.isfinite(s2s)]
            b.add("gaze", eye, "precision_px", "precision",
                  float(np.median(s2s)) if s2s.size else np.nan, "px",
                  "median sample-to-sample distance of gaze in pixels "
                  "(set screen size and eye distance to get degrees and a label)")

    # --- binocular ------------------------------------------------------
    if "left" in pupil_eyes and "right" in pupil_eyes:
        lp, rp = raw["left_pupil"], raw["right_pupil"]
        both = np.isfinite(lp) & np.isfinite(rp)
        r = np.nan
        if both.sum() > 10 and np.std(lp[both]) > 0 and np.std(rp[both]) > 0:
            r = float(np.corrcoef(lp[both], rp[both])[0, 1])
        b.add("binocular", "left/right", "pupil_correlation", "pupil correlation", r, "",
              "correlation of left and right pupil (samples valid in both eyes)")
    if "left" in gaze_eyes and "right" in gaze_eyes:
        lx, ly, rx, ry = raw["left_x"], raw["left_y"], raw["right_x"], raw["right_y"]
        ldeg, rdeg = _degrees(setup, lx, ly), _degrees(setup, rx, ry)
        if ldeg is not None and rdeg is not None:
            dist = np.hypot(ldeg[0] - rdeg[0], ldeg[1] - rdeg[1])
            name, unit = "gaze_disparity_deg", "deg"
        else:
            dist = np.hypot(lx - rx, ly - ry)
            name, unit = "gaze_disparity_px", "px"
        dist = dist[np.isfinite(dist)]
        b.add("binocular", "left/right", name, "gaze disparity",
              float(np.median(dist)) if dist.size else np.nan, unit,
              "median distance between left and right gaze position")

    # --- blinks (only if blink detection was run) ------------------------
    blinks_mask = {}
    for eye in list(dict.fromkeys(pupil_eyes + gaze_eyes)):
        var = "pupil" if eye in pupil_eyes else "x"
        if obj._blinks.get(f"{eye}_{var}") is None:
            continue
        bl = obj.get_blinks(eye, var)
        blinks_mask[eye] = bl.to_mask(n).astype(bool)
        dur_ms = np.array([(e - s) for s, e in bl.intervals], dtype=float) / fs * 1000
        b.add("blinks", eye, "n", "number", int(len(bl)), "", "number of detected blinks")
        b.add("blinks", eye, "rate", "blink rate",
              len(bl) / dur_min if dur_min > 0 else np.nan, "/min",
              "number of detected blinks per minute")
        b.add("blinks", eye, "mean_duration", "mean duration",
              float(dur_ms.mean()) if dur_ms.size else np.nan, "ms",
              "mean duration of the detected blinks")
        b.add("blinks", eye, "time_pct", "time in blinks", _pct(blinks_mask[eye].sum(), n),
              "%", "percentage of the recording covered by blinks")
        b.add("blinks", eye, "n_long", f"long (>{LONG_BLINK_MS} ms)",
              int(np.sum(dur_ms > LONG_BLINK_MS)), "",
              f"number of blinks longer than {LONG_BLINK_MS} ms")

    # --- artifacts (only if data was masked/interpolated) ---------------
    for eye in list(dict.fromkeys(pupil_eyes + gaze_eyes)):
        metrics = []
        processed = eye in blinks_mask
        if eye in pupil_eyes:
            key = f"{eye}_pupil"
            m = masks[key].astype(bool)
            nan = ~np.isfinite(raw[key])
            processed |= bool(np.any(m != nan))
            metrics.append(("pupil_masked_pct", "pupil masked", _pct(m.sum(), n),
                            "percentage of masked pupil samples (all reasons)"))
            other = m & ~nan & ~blinks_mask.get(eye, np.zeros(n, bool))
            if obj._interpolated_blinks.get(key) is not None:
                processed = True
                interp = obj.get_interpolated_blinks(eye, "pupil").to_mask(n).sum()
                metrics.append(("interpolated_pct", "interpolated", _pct(interp, n),
                                "percentage of interpolated pupil samples"))
        if eye in gaze_eyes:
            m = masks[f"{eye}_x"].astype(bool) | masks[f"{eye}_y"].astype(bool)
            nan = ~(np.isfinite(raw[f"{eye}_x"]) & np.isfinite(raw[f"{eye}_y"]))
            processed |= bool(np.any(m != nan))
            metrics.append(("gaze_masked_pct", "gaze masked", _pct(m.sum(), n),
                            "percentage of masked gaze samples (all reasons)"))
            if eye not in pupil_eyes:
                other = m & ~nan & ~blinks_mask.get(eye, np.zeros(n, bool))
        if not processed:
            continue
        for name, title, value, desc in metrics:
            b.add("artifacts", eye, name, title, value, "%", desc)
        b.add("artifacts", eye, "other_pct", "other artifacts", _pct(other.sum(), n), "%",
              "percentage of samples masked for other reasons than missing data or blinks "
              "(e.g., off-screen gaze, eye divergence)")

    # --- calibration ----------------------------------------------------
    calibration = getattr(obj, "calibration", None)
    if calibration:
        for eye, cal in calibration.items():
            if eyes and eye not in eyes:
                continue
            try:
                stats = cal.get_stats()
            except Exception:
                continue
            b.add("calibration", eye, "model", "model", str(cal.model))
            b.add("calibration", eye, "n_points", "points", int(stats["n"]))
            b.add("calibration", eye, "mean_error", "mean error", stats["mean"], "deg",
                  "mean validation error in degrees of visual angle")
            b.add("calibration", eye, "max_error", "max error", stats["max"], "deg",
                  "maximum validation error in degrees of visual angle")

    return QualityReport(name=obj.name, class_name=type(obj).__name__,
                         sections=b.sections, thresholds=thresholds)
