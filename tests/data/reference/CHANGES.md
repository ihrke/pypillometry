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

### Default time vector at exactly `1000/fs` ms

- **Case**: none (all reference cases pass a time vector)
- **Change**: objects built without `time` get `tx = arange(n) * 1000/fs`
  instead of `linspace(0, n/fs*1000, n)`, whose steps were
  `n/(n-1) * 1000/fs` ms (25% too long for 5 samples, 0.005% for 20000).
- **Why**: the time vector must match the sampling rate (`_set_timebase`
  now checks it).
- **Step**: 3, `refactor/03-timebase`

### Clock-aligned state after `reset_time` and repeated `remove_intervals`

- **Case**: none (the reference case `g.remove_intervals.rlmw` is a single
  call and unchanged)
- **Change**: `reset_time` also shifts blinks stored in time units
  (finding 4.4) and `params["removed_intervals"]`. The periods of a second
  `remove_intervals` call are recorded in the original time (they were in
  the time after the first call); a period across an earlier join is
  split at the join, and all periods are merged. The shifts between the
  current and the original time are kept in `params["removed_shifts"]`.
- **Why**: blinks and mask disagreed after `reset_time`;
  `params["removed_intervals"]` mixed clocks.
- **Step**: 3, `refactor/03-timebase`

### `FakeEyeData.sim_data` follows the samples

- **Case**: none
- **Change**: `crop`, `remove_intervals`, `downsample` and
  `fill_time_discontinuities` apply to `sim_data` as to the data (it kept
  its original length before, finding 4.11).
- **Why**: the ground truth must stay aligned with the data.
- **Step**: 3, `refactor/03-timebase`
