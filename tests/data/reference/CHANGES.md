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
  (touching an excluded period) keep their mask, as before (see "Blinks
  next to an excluded period keep their layers" below). The quality
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

### `mask_intervals` and masked-array assignment set the `user` layer

- **Case**: none (the masks are the same; `d.offscreen.*` and
  `d.divergence.*` are unchanged)
- **Change**: `mask_intervals` (and so, until they get their own layers,
  `mask_offscreen_coords` and `mask_eye_divergences`) and
  `d[key] = np.ma.array(...)` record the masked samples in the `user`
  layer. After `downsample`, these samples are masked where they cover
  more than half of a window (before: only where the whole window was
  masked). `clear_layer("user")` removes them.
- **Why**: masks can be undone; decision 4 (">½ of the window").
- **Step**: 5, `refactor/05-mask-layers`

### `pupil_blinks_detect` returns the stored blinks

- **Case**: none (in `a.blinks_detect.*` no detected blinks touch, so the
  returned and stored blinks are the same as before)
- **Change**: the returned Intervals are the stored blinks (runs of the
  `blink` layer): blinks that touch (the end of one is the start of the
  next, e.g. found by the velocity and the zero strategy) are one blink.
  In a 55-min 1000-Hz recording, 6 of 328 detected blinks of one eye
  touched the next one, so 322 blinks are returned and stored (the masked
  samples are the same).
- **Why**: the returned blinks are the blinks stored in the object.
- **Step**: 5, `refactor/05-mask-layers`

### Excluded periods are a recording-wide layer

- **Case**: none (`g.exclude_intervals.rlmw` is not downsampled; its masks
  and summary are unchanged)
- **Change**: `exclude_intervals`/`keep_intervals` set the recording-wide
  `excluded` layer instead of writing the periods into the mask of every
  variable. Variables added later (e.g. `d["left", "baseline"] = ...`) are
  therefore masked in the excluded periods, too (before, they were not).
  After `downsample`, a new sample is excluded if the excluded periods
  cover more than half of its window (before: if its time was within an
  excluded period). Excluded samples stay masked under every policy with
  the layer, also where they are interpolated (`unless` does not apply to
  recording-wide layers).
- **Why**: one store for excluded periods (finding D1); refactor plan 2.2.
- **Step**: 5, `refactor/05-mask-layers`

### `get_excluded` returns the runs of the excluded layer

- **Case**: none (`g.exclude_intervals.rlmw`: the periods start and end at
  samples, so the runs are the same)
- **Change**: the excluded periods are no longer stored as times
  (`_excluded`); `get_excluded()` and `get_included()` return the runs of
  the `excluded` layer, from the time of the first sample of a run to the
  time of the sample after its last one. A period that does not start or
  end at a sample is returned from the first sample at or after its
  start. Periods outside the data are gone after `crop` (before, they
  were kept as times). After `downsample`, the runs are at the times of
  the new samples (the window means: 1000 ms excluded at 500 Hz, then
  `downsample(100)`, gives `(7004, 8004)` instead of `(7000, 8000)`).
  `clear_layer("excluded")` removes the excluded periods.
- **Why**: no interval stores next to the samples (refactor plan 2.4:
  nothing else to remap).
- **Step**: 5, `refactor/05-mask-layers`

### Blinks next to an excluded period keep their layers

- **Case**: none (`g.exclude_intervals.rlmw`: same blinks and masks)
- **Change**: `exclude_intervals` no longer removes the blinks that overlap
  or touch an excluded period from the `blink` layer. They keep their
  layer, their mask and their interpolation; `get_blinks()` (and so
  `blinks`, `nblinks`, `blink_stats`, the quality report, the blink plots
  and `pupil_blinks_interpolate`) leaves them out, as before.
  `clear_layer("excluded")` brings them back. Before, the blink was
  removed but its mask and its interpolated blink were kept (step-0 note
  1: in the `eyedata` fixture, the blink (2600, 2620) was gone while the
  interpolated blink (2595, 2635) stayed). Differences: blinks detected
  without `apply_mask` stay masked under `d.mask()` (default policy) next
  to the excluded period; `blinks_merge_close` merges them, too (a merged
  blink that touches an excluded period is left out by `get_blinks`).
- **Why**: layers are independent annotations; `exclude_intervals` sets
  only its own layer (refactor plan 2.3), and the excluded layer
  overrides the others in the masks.
- **Step**: 5, `refactor/05-mask-layers`

### `mask_offscreen_coords` sets the `offscreen` layer and replaces it

- **Case**: none (`d.offscreen.*` run once: same masks)
- **Change**: the off-screen samples are stored as the `offscreen` layer
  of x and y (before: the `user` layer, via `mask_intervals`). Running the
  method again replaces the layer of the checked eyes (before, the masks
  only grew, and with `ignore_existing_mask=False` the samples masked by
  the first run were not checked again). `clear_layer("offscreen")`
  removes the mask.
- **Why**: detectors replace their own layer (refactor plan 2.2).
- **Step**: 5, `refactor/05-mask-layers`

### `mask_eye_divergences` sets the `divergence` layer and replaces it

- **Case**: none (`d.divergence.*` run once: same masks)
- **Change**: the divergent samples are stored as the `divergence` layer
  of x and y of both eyes (before: the `user` layer, via
  `mask_intervals`); running the method again replaces the layer, and
  with `ignore_existing_mask=False` the layer of an earlier run does not
  count as an existing mask. `clear_layer("divergence")` removes it.
- **Why**: detectors replace their own layer (refactor plan 2.2).
- **Step**: 5, `refactor/05-mask-layers`

### `mask_eye_divergences(store_as=...)` stores the distance for both eyes

- **Case**: none (the reference cases do not store the distance)
- **Change**: `store_as="dist"` stores the distance as `left_dist` and
  `right_dist` (the same values) instead of under the key `"dist"`, which
  was taken for an eye (`d.eyes` was `[..., "dist"]`; `crop` and
  `downsample` raised, finding 5.3). The stored distance is masked where
  one of the coordinates is masked (other layers) and has the
  `divergence` layer where the eyes diverge (before: all of it in one
  mask). Its values are as before.
- **Why**: every channel is `<eye>_<variable>`; the distance belongs to
  the pair of eyes, and storing it with each eye keeps it a variable.
- **Step**: 5, `refactor/05-mask-layers`

### No stored mask: the mask is derived from the layers

- **Case**: none (all masks and values are the same: `data.mask` is
  computed from the layers and equals the old mask)
- **Change**: the separate boolean mask per variable is gone. What it held
  is now a layer: masks set with `set_mask`/`add_mask` (and
  `set_with_mask(mask=...)`, e.g. `merge_masks`, `pupil_merge_eyes`) are
  in the `user` layer, so `clear_layer("user")` removes them; NaN values
  at assignment and the samples inserted by `fill_time_discontinuities`
  are in the internal layer `_nan` (they stay masked when
  `pupil_lowpass_filter` fills them with values, as before); after
  `downsample`, a window whose samples were all masked stays masked
  through the layers of its samples (not `interpolated`) when no layer
  covers more than half of it. The data state has no `mask` key any more;
  states with a `mask` (format 2 written before) are translated: where no
  layer explains a masked sample, NaN values get `_nan`, others `user`
  (before the blinks, interpolated blinks and excluded periods of the old
  file are taken out of it).
- **Why**: one store for masks (finding D1); NaN is read from the values
  (refactor plan 2.2) except where a filter filled it (step 7).
- **Step**: 5, `refactor/05-mask-layers`

### Quality report: masked time by layer; interpolation is not "other artifacts"

- **Case**: `f.quality.{rlmw,synth}.interp`, keys
  `artifacts.{left,right}.other_pct` (expected value 0 is computed in
  `tests/test_reference_outputs.py`, `_quality_other`: in case b every
  masked pupil sample is interpolated): rlmw 2.50 -> 0 (left), 1.62 -> 0
  (right); synth 0.93 -> 0, 0.69 -> 0. All other numbers are unchanged.
- **Change**: `other_pct` counts the masked samples that are not missing
  (NaN, or NaN before a filter filled them), not in a blink and not
  interpolated. Before, the interpolated samples around a blink (the
  margins of `pupil_blinks_interpolate`) were counted as other artifacts
  (finding 5.8), and so were NaN filled by `pupil_lowpass_filter` (so a
  report after filtering alone no longer has the artifacts section). The
  artifacts section has new metrics for the masked time per layer, shown
  when the layer exists: `offscreen_pct`, `divergence_pct` and
  `user_pct` (samples where the layer is set for any variable of the eye).
  `pupil_masked_pct`/`gaze_masked_pct` are unchanged (the mask that the
  analyses use until step 7: interpolated samples count as masked).
- **Why**: the layers say why a sample is masked, so the report does not
  have to guess (finding D1).
- **Step**: 5, `refactor/05-mask-layers`

### `downsample` after off-screen and divergence masking: per layer

- **Case**: none (no reference case downsamples gaze masks)
- **Change**: with off-screen gaze and eye divergences in two layers, the
  ">1/2 of the window" rule of `downsample` applies to each layer: a
  window that is 30 % off-screen and 30 % divergent is no longer masked
  (it was, while both were in the `user` layer earlier in step 5; it was
  not before step 5, where only fully masked windows stayed masked).
  Real data (pavlov 10, 15 min at 1000 Hz, `mask_offscreen_coords` +
  `mask_eye_divergences`, then `downsample(50)`): 119 of 25735 masked
  samples of the left gaze are unmasked (9 of 25402 right). Fully masked
  windows stay masked.
- **Why**: decision 3 of the refactor plan (each layer bit set if it
  covers more than half of the window).
- **Step**: 5, `refactor/05-mask-layers`
