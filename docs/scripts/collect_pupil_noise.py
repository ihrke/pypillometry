#!/usr/bin/env python
"""Collect the high-frequency pupil noise of many recordings in a CSV table.

Used to find plausible thresholds for the ``pupil.noise`` measure of the
quality report (see docs/user_guide/quality.rst). The noise is taken from
:meth:`~pypillometry.GenericEyeData.quality_report`, so it is computed exactly
as in the report.

Each file (EDF or pypillometry pickle, as for ``pypilinfo``) gives one row
per eye with pupil data. Besides the noise, the table contains what may
explain differences in noise: the eye tracker and camera, the pupil units
(area or diameter; noise in percent of the area is about twice the noise in
percent of the diameter), the pupil-fitting method, the tracker's filter
levels and the sampling rate. These are read from the EDF header and
messages and are empty if not available (e.g., for data not read from EDF).

Rows are appended to the CSV file, so you can run the script on several
datasets with different notes and collect everything in one table.

Usage:
    python docs/scripts/collect_pupil_noise.py -o noise.csv \\
        --note "lab A, EyeLink 1000, chin rest" data/*.edf
    python docs/scripts/collect_pupil_noise.py -o noise.csv --note "study B" study_b/

Directories are searched (recursively) for .edf, .pypil and .pkl files.
Only use pickle files you trust: unpickling can run arbitrary code.
"""

import argparse
import csv
import importlib.metadata
import os
import re
import sys
from pathlib import Path

import numpy as np

import pypillometry as pp
from pypillometry.cli import _read_file

EXTENSIONS = (".edf", ".pypil", ".pkl")

COLUMNS = [
    "file", "path", "format", "name", "note",
    "tracker", "camera", "serial", "pupil_units", "pupil_fit",
    "record_mode", "file_filter", "link_filter", "tracked_eyes",
    "sampling_rate", "duration_min", "eye",
    "missing_pct", "pupil_median", "noise_sd", "noise_pct", "noise_label",
    "pypillometry_version",
]


def find_files(sources):
    """Expand directories into the data files they contain."""
    for src in sources:
        p = Path(src)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in EXTENSIONS and f.is_file():
                    yield str(f)
        else:
            yield src


def recording_info(d):
    """Eye-tracker information from the EDF header and messages, if any."""
    info = {}
    el = (getattr(d, "info", None) or {}).get("eyelink_info", {}) or {}
    info["tracker"] = el.get("version", "")
    info["camera"] = el.get("camera", "")
    info["serial"] = el.get("serial", "")
    info["pupil_units"] = el.get("ps_units", "")
    info["tracked_eyes"] = el.get("eye", "")
    labels = [str(lab) for lab in getattr(d, "event_labels", [])]
    for lab in labels:
        # e.g., "RECCFG CR 500 2 1 LR": mode, rate, file filter, link filter, eyes
        m = re.match(r"\s*RECCFG\s+(\S+)\s+\S+\s+(\d)\s+(\d)", lab)
        if m and "record_mode" not in info:
            info["record_mode"], info["file_filter"], info["link_filter"] = m.groups()
        m = re.match(r"\s*ELCL_PROC\s+(.+)", lab)
        if m and "pupil_fit" not in info:
            info["pupil_fit"] = m.group(1).strip()
    return info


def rows_for(source, note):
    d, fmt, _ = _read_file(source, keep_triggers=True)  # keep RECCFG etc.
    report = d.quality_report()
    base = {
        "file": os.path.basename(source), "path": source, "format": fmt,
        "name": getattr(d, "name", ""), "note": note,
        "sampling_rate": d.fs,
        "duration_min": round(report["info"]["all"]["duration"].value
                              if "duration" in report["info"]["all"] else np.nan, 2),
        "pypillometry_version": importlib.metadata.version("pypillometry"),
    }
    base.update(recording_info(d))
    if "pupil" not in report:
        return []
    rows = []
    for eye, metrics in report["pupil"].items():
        noise, sd = metrics["noise"].value, metrics["noise_sd"].value
        # the report gives noise in percent of the median, so the median follows
        median = 100 * sd / noise if noise and np.isfinite(noise) else np.nan
        rows.append({**base, "eye": eye,
                     "missing_pct": metrics["missing_pct"].value,
                     "pupil_median": median,
                     "noise_sd": sd, "noise_pct": noise,
                     "noise_label": metrics["noise"].label or ""})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect the pupil noise of eye-tracking files in a CSV table.")
    parser.add_argument("sources", nargs="+",
                        help="EDF or pickle files, or directories to search")
    parser.add_argument("-o", "--output", required=True,
                        help="CSV file (rows are appended if it exists)")
    parser.add_argument("-n", "--note", default="",
                        help="note stored in every row (e.g., lab, setup, population)")
    args = parser.parse_args(argv)
    pp.logging_set_level("ERROR")
    os.environ["TQDM_DISABLE"] = "1"

    new = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    failed = 0
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        for source in find_files(args.sources):
            try:
                rows = rows_for(source, args.note)
            except Exception as e:  # keep going with the other files
                print(f"{source}: skipped ({e})", file=sys.stderr)
                failed += 1
                continue
            if not rows:
                print(f"{source}: skipped (no pupil data)", file=sys.stderr)
                continue
            writer.writerows(rows)
            f.flush()
            print(source + ": " + ", ".join(
                f"{r['eye']} {r['noise_pct']:.3g}%" for r in rows))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
