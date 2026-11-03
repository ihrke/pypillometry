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

### Segment time axis at the sampling rate

- **Case**: `e.segments.{rlmw,synth}.{raw,interp}`, keys `tx.left_pupil`,
  `tx.right_pupil` (the expected value is computed in
  `tests/test_reference_outputs.py`, `_segment_time_axis`)
- **Change**: the time axis of the segments is `w0 + arange(n) * 1000/fs`
  instead of `linspace(w0, w1, n)`: for the window (-200, 1000) ms at 500 Hz
  (n = 600), the steps are 2 ms (were 2.0033 ms) and the last time point is
  998 ms (was 1000 ms). The segment data, masks and the baseline-corrected
  values are unchanged (`baseline_correct((-200, 0))` picks the same
  samples).
- **Why**: the time of each point must be the time of its sample relative
  to the event (finding 2.3); `get_intervals(event, window)` now returns
  `Epochs`, and `get_segments` takes the axis from their window.
- **Step**: 4, `refactor/04-epochs`

### Event-locked windows are not clipped

- **Case**: none (the reference windows are in ms and within the data;
  `e.segments.*` data and masks and `e.stat_per_event.*` are unchanged)
- **Change**: `get_intervals(event, window)` returns `Epochs` (onsets and
  window in ms) instead of Intervals. With a window in samples near the
  start of the recording, the segments are no longer shifted: the event
  stays at t = 0 and the samples before the recording are masked (they
  were clipped to sample 0, and the segment started there; finding 2.2).
  Between two events in samples, the intervals are no longer clipped to
  `(0, n)` (finding 2.6). `Intervals.pad` no longer clips to the data (or
  to sample 0), and `to_units("samples", d)` no longer clips the event
  onsets to the samples of the data. A window in samples is converted to
  ms with the sampling rate; each segment starts at the first sample at or
  after `onset + window[0]` (before: the sample of the event plus the
  window in samples; the same sample unless the time stamps are irregular).
- **Why**: spans are never clipped silently; the parts outside the data
  are handled where the data are read (masked in segments, ignored in
  masks and statistics).
- **Step**: 4, `refactor/04-epochs`

### Segments from Intervals of different lengths raise

- **Case**: none
- **Change**: `get_segments(Intervals)` raises a ValueError unless all
  intervals have the same length (between two events, it returned
  segments with 0 time points, finding 2.1). Intervals of the same length
  give segments with the time axis from 0 at their starts (before: the
  absolute times of the first interval).
- **Why**: a segment needs one time axis; event-locked segments come from
  `Epochs`.
- **Step**: 4, `refactor/04-epochs`

### `baseline_correct` with a window outside the segments raises

- **Case**: none (`e.segments.*` use `(-200, 0)`, within the segments)
- **Change**: a baseline window (or time point) that is not within the
  time axis of the segments raises a ValueError (before: a tuple outside
  masked everything, a time point outside used the first or last sample,
  a window partly outside used the part inside; finding 2.4).
- **Step**: 4, `refactor/04-epochs`

### `stat_per_event` with windows outside the data

- **Case**: none (the reference windows are within the data)
- **Change**: a window that lies entirely outside the data gives NaN
  (before: the value of the first or last sample); a window that runs past
  the last sample includes the last sample (it was left out).
- **Why**: windows are no longer clipped in `get_intervals`; the data
  within a window are used.
- **Step**: 4, `refactor/04-epochs`

### Interpolated blinks are a layer; `downsample` keeps a layer where it covers more than half of a window

- **Case**: `g.downsample.rlmw.{250,100}`, keys `interpolated.*`, `mask.*`
  (pupil) and the scalars `masked.*` (the expected values are computed in
  `tests/test_reference_outputs.py`, `_downsampled_runs` and
  `_downsampled_mask`, from the full-rate intervals of case
  `b.blinks_interpolate.rlmw`)
- **Change**: the interpolated samples are stored as the `interpolated`
  mask layer (not as Intervals). After `downsample`, a new sample has the
  layer if the layer covers more than half of its window (before: the
  interpolated blinks in samples were rescaled, start rounded down and end
  up, so every window that touched an interpolated blink was included).
  `get_interpolated_blinks()` returns the runs of the layer. The mask is
  set where it was set before (all samples of the window masked) and
  where the `interpolated` layer covers more than half of the window
  (rlmw at 100 Hz: 407 instead of 398 masked samples, left eye). The
  values (window means of the unmasked samples) are unchanged.
- **Why**: one store for "interpolated" (finding D1); decision 4 of the
  refactor plan (">½ of the window").
- **Step**: 5, `refactor/05-mask-layers`

### Interpolated blinks belong to the variable that holds the interpolated data

- **Case**: none (the reference cases use `store_as="pupil"`)
- **Change**: `pupil_blinks_interpolate(store_as="pupilinterp")` sets the
  `interpolated` layer on `<eye>_pupilinterp`, so
  `get_interpolated_blinks(eye, "pupilinterp")` returns them (before they
  were stored for `<eye>_pupil`, whose data were not interpolated). The new
  variable gets the masks and layers of the pupil (its blinks, too).
  Calling `pupil_blinks_interpolate` again adds to the layer (before, the
  interpolated blinks of the last call replaced the earlier ones; the
  earlier samples stayed masked).
- **Why**: the layer describes the samples it is stored with.
- **Step**: 5, `refactor/05-mask-layers`

### `interpolate_intervals` sets the `interpolated` layer

- **Case**: none
- **Change**: the samples interpolated by `interpolate_intervals` get the
  `interpolated` layer, so `get_interpolated_blinks()` (and the
  "interpolated" percentage of the quality report) include them (before:
  only `pupil_blinks_interpolate` was recorded). With `store_as_suffix`,
  the new variable gets the masks and layers of the original one.
- **Why**: one layer for reconstructed samples (refactor plan 2.2).
- **Step**: 5, `refactor/05-mask-layers`

### Blinks are a layer: runs of the layer, `downsample` and `set_blinks`

- **Case**: `g.downsample.rlmw.{250,100}`, keys `blinks.*` and, with the
  blinks of case b being applied as mask, `mask.*` and `masked.*` (pupil)
  (expected values: `_downsampled_runs`, `_downsampled_mask`)
- **Change**: blinks are stored as the `blink` layer of their variable;
  `get_blinks()` returns the runs of the layer, so blinks that overlap or
  touch are one blink (a blink inside another one is gone, finding 4.3:
  `blinks_merge_close` no longer shrinks it), and blinks set in time units
  come back as sample runs converted to the requested units. After
  `downsample`, a new sample is in a blink if blinks cover more than half
  of its window (before: start rounded down, end up); the mask of applied
  blinks follows the same rule. After `fill_time_discontinuities`, a blink
  across a gap is two blinks (the inserted samples are not blinks).
  `set_blinks` replaces the blinks *and their mask*: `set_blinks(None)`
  unmasks the blinks that were set with `apply_mask=True` (before, the
  mask stayed, finding 4.2), and so does setting other blinks; NaN samples
  and other masks stay masked. Blinks removed by `exclude_intervals`
  (touching an excluded period) keep their mask, as before. The quality
  report has the blink section for all eyes once blinks were detected
  for any eye (0 blinks for eyes without detection).
- **Why**: one store for blinks (finding D1); the mask of the blinks can
  be undone.
- **Step**: 5, `refactor/05-mask-layers`

### Re-detecting blinks replaces them

- **Case**: none (the reference cases detect blinks once)
- **Change**: `pupil_blinks_detect` removes the blinks detected before
  (and their mask) before it detects blinks, so they do not mask the data
  it detects blinks in, and the new blinks replace them. Re-running it
  with other parameters gives the same blinks and mask as running it
  once (before: the old mask stayed and the new detection skipped the old
  blinks, finding 4.2).
- **Why**: detectors replace their own layer (refactor plan 2.2).
- **Step**: 5, `refactor/05-mask-layers`

### Blinks detected without `apply_mask` are stored in the object

- **Case**: none (`a.blinks_detect.*` detect in place)
- **Change**: `pupil_blinks_detect(apply_mask=False)` (the default)
  returns the blinks and stores them in the object it is called on, also
  with `inplace=False` (before: in a copy that was thrown away, finding
  4.1). The data and the mask are not changed. The call is still not
  recorded in the history (the method returns the Intervals; step 9 makes
  detectors return the object).
- **Why**: the detected blinks were lost.
- **Step**: 5, `refactor/05-mask-layers`
