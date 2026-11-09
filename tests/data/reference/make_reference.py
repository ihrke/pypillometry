"""
Script that wrote the reference outputs in this folder.

Run it manually from the root of the repository::

    python tests/data/reference/make_reference.py

It runs every case in ``tests/test_reference_outputs.py`` (`CASES`) with the
installed pypillometry and writes

- ``<case id>.npz``: the arrays of the case (compressed)
- ``reference.json``: per case its description, parameters and scalar
  results, plus the pypillometry commit and versions it was run with

Do not re-run it just to make a failing test pass: regenerate the
reference only for an intended change of a value, after checking that the
new values are right, and in the same commit as the change.
"""
import datetime
import glob
import json
import os
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from tests.test_reference_outputs import CASES, compute, split_results  # noqa: E402


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main():
    for f in glob.glob(os.path.join(HERE, "*.npz")):
        os.remove(f)
    cases = {}
    for case_id in sorted(CASES):
        arrays, scalars = split_results(compute(case_id))
        np.savez_compressed(os.path.join(HERE, f"{case_id}.npz"), **arrays)
        c = CASES[case_id]
        cases[case_id] = {"description": c["description"], "params": c["params"],
                          "arrays": sorted(arrays), "scalars": scalars}
        print(f"{case_id}: {len(arrays)} arrays, {len(scalars)} scalars")
    ref = {
        "description": "reference outputs of pypillometry "
                       "(see tests/test_reference_outputs.py)",
        "pypillometry_commit": git("rev-parse", "HEAD"),
        "package_modified": bool(git("status", "--porcelain", "--", "pypillometry")),
        "pypillometry_version": version("pypillometry"),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "python_version": sys.version.split()[0],
        "date": datetime.date.today().isoformat(),
        "cases": cases,
    }
    with open(os.path.join(HERE, "reference.json"), "w") as f:
        json.dump(ref, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
