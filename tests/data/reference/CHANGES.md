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

### `fs` after `downsample` is the actual rate

- **Case**: none of the reference cases (they use integer factors,
  `g.downsample.rlmw.250` and `g.downsample.rlmw.100`, which are unchanged)
- **Change**: `downsample(fsd)` decimates by `int(fs/fsd)` and now stores
  `fs/factor` as the new sampling rate instead of the requested `fsd`
  (1000 Hz, `fsd=300`: `fs` is 333.3, was 300; `fsd=400`: 500, was 400;
  `dsfac=True` with factor 2.5: see the argument checks). Everything that
  converts between ms and samples afterwards (segment lengths, time axes,
  filter design, blink durations) uses the new value.
- **Why**: the stored rate did not match the time vector (finding 3.1).
- **Step**: 3, `refactor/03-timebase`
