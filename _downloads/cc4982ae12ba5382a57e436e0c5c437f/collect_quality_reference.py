#!/usr/bin/env python
"""Collect all quality measures of many recordings in a CSV table.

Used to build reference distributions for the measures of the quality report
(see docs/user_guide/quality.rst), e.g., to choose thresholds. The values
come from :meth:`~pypillometry.GenericEyeData.quality_report`, so they are
computed exactly as in the report.

Each file (EDF or pypillometry pickle, as for ``pypilinfo``) gives one row
per measure, in long format: the measure is identified by ``section``,
``group`` (eye, "left/right" or "all") and ``metric``, followed by its value,
unit and current label. Every row also has the file, the note and the
eye-tracker information that may explain differences between datasets: the
eye tracker and camera, the pupil units (area or diameter; noise in percent
of the area is about twice the noise in percent of the diameter), the
pupil-fitting method, the tracker's filter levels and the sampling rate.
These are read from the EDF header and messages and are empty if not
available (e.g., for data not read from EDF). The table can be filtered and
grouped directly, e.g., with pandas::

    df = pd.read_csv("quality.csv")
    df[df.metric == "noise"].groupby("note").value.describe()

By default, blinks are detected first (``pupil_blinks_detect`` with its
default settings and ``apply_mask=True``), so that the blink and artifact
measures are included as well. This only masks the blinks and does not
change the raw values, so the other measures are the same as for the raw
data. Use --raw-only to skip it (e.g., for data that were already
preprocessed, whose own blinks and masks are then reported).

Rows are appended to the CSV file, so you can run the script on several
datasets with different notes and collect everything in one table.

Usage:
    python docs/scripts/collect_quality_reference.py -o quality.csv \\
        --note "lab A, EyeLink 1000, chin rest" data/*.edf

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

import pypillometry as pp
from pypillometry.cli import _read_file

EXTENSIONS = (".edf", ".pypil", ".pkl")

INFO_COLUMNS = [
    "file", "path", "format", "name", "note",
    "tracker", "camera", "serial", "pupil_units", "pupil_fit",
    "record_mode", "file_filter", "link_filter", "tracked_eyes",
    "sampling_rate", "blinks_detected", "pypillometry_version",
]
COLUMNS = INFO_COLUMNS + ["section", "group", "metric", "title", "value", "unit",
                          "label"]


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


def rows_for(source, note, raw_only=False):
    d, fmt, _ = _read_file(source, keep_triggers=True)  # keep RECCFG etc.
    detect = not raw_only and hasattr(d, "pupil_blinks_detect")
    if detect:
        d.pupil_blinks_detect(apply_mask=True)
    report = d.quality_report()
    base = {
        "file": os.path.basename(source), "path": source, "format": fmt,
        "name": getattr(d, "name", ""), "note": note,
        "sampling_rate": d.fs, "blinks_detected": detect,
        "pypillometry_version": importlib.metadata.version("pypillometry"),
    }
    base.update(recording_info(d))
    return [{**base, "section": section, "group": group, "metric": m.name,
             "title": m.title, "value": m.value, "unit": m.unit,
             "label": m.label or ""}
            for section, group, m in report.metrics()]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect all quality measures of eye-tracking files in a CSV table.")
    parser.add_argument("sources", nargs="+",
                        help="EDF or pickle files, or directories to search")
    parser.add_argument("-o", "--output", required=True,
                        help="CSV file (rows are appended if it exists)")
    parser.add_argument("-n", "--note", default="",
                        help="note stored in every row (e.g., lab, setup, population)")
    parser.add_argument("--raw-only", action="store_true",
                        help="do not detect blinks before the report")
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
                rows = rows_for(source, args.note, args.raw_only)
            except Exception as e:  # keep going with the other files
                print(f"{source}: skipped ({e})", file=sys.stderr)
                failed += 1
                continue
            writer.writerows(rows)
            f.flush()
            print(f"{source}: {len(rows)} measures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
