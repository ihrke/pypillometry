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
eye-tracker information (tracker, camera, pupil units, filter levels, ...;
see collect_pupil_noise.py), so the table can be filtered and grouped
directly, e.g., with pandas::

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
import sys

import pypillometry as pp
from pypillometry.cli import _read_file

# shared with the noise script (in the same folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_pupil_noise import find_files, recording_info  # noqa: E402

INFO_COLUMNS = [
    "file", "path", "format", "name", "note",
    "tracker", "camera", "serial", "pupil_units", "pupil_fit",
    "record_mode", "file_filter", "link_filter", "tracked_eyes",
    "sampling_rate", "blinks_detected", "pypillometry_version",
]
COLUMNS = INFO_COLUMNS + ["section", "group", "metric", "title", "value", "unit",
                          "label"]


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
