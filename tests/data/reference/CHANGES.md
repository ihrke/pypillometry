# Intended changes of the reference outputs

The files in this folder record what pypillometry computed before the 2.2
refactor (commit and versions in `reference.json`, written by
`make_reference.py`). `tests/test_reference_outputs.py` recomputes every case
with the current code and compares it with them.

The reference values themselves are not regenerated. When a step of the
refactor changes a result on purpose, the test is adapted for that case and
the change gets an entry here, in the same commit. So every number that
differs from the reference is a deliberate decision, and this list doubles as
the changelog of the computed results for 2.2.

Each entry gives:

- **Case**: the case id(s) (e.g. `e.segments.rlmw.interp`) and the result
  keys affected
- **Change**: what changed (old vs. new value or behaviour)
- **Why**: the reason
- **Step**: the refactor step / branch that made the change

Changes of the API calls alone (same numbers) need no entry.

## Entries

(none yet)
