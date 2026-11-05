# Changelog

## 2.2.0 (unreleased)

A new data model behind the same kind of analyses: one data class, mask
layers instead of a single mask, pipelines, and the pupil, gaze and plot
methods in namespaces. Most scripts need changes; old method names raise an
`AttributeError` that names the new one. See
[Upgrading from 2.1](../user_guide/upgrading.rst) for an overview and the
tables below for every changed call.

### Breaking changes

- One data class, `EyeData`: `PupilData`, `GazeData`, `to_pupildata()` and
  `to_gazedata()` were removed (files with these classes load as `EyeData`;
  `to_pupildata()` -> `drop_data(variables=["x", "y"], inplace=False)`)
- Pupil methods are in `d.pupil`, gaze methods in `d.gaze`, plots only in
  `d.plot`, without the prefixes (`d.pupil_blinks_detect()` ->
  `d.pupil.blinks_detect()`, `d.pupil_plot()` -> `d.plot.pupil_plot()`); the
  plotter classes (`EyePlotter`, `PupilPlotter`, `GazePlotter`) were removed
- Mask layers: detectors (`pupil.blinks_detect`, `pupil.blinks_merge_close`,
  `gaze.mask_offscreen`, `gaze.mask_divergences`) set their layer, replace
  the results of an earlier run and return the object; `apply_mask` was
  removed (setting the layer is masking); read the results with
  `get_blinks()`, `get_layer_intervals()` etc.
- Mask policies: interpolated samples are *used* by default in all
  analyses (segments, `stat_per_event`, `scale`, `merge_eyes`, `downsample`,
  quality report); `policy="strict"` leaves them out. Several results change
  (see "Changed results")
- Plain numbers are ms everywhere: windows (`get_intervals`), durations,
  margins; sample indices need `units="samples"`. In plots, plain numbers in
  `plot_range` and `interv` are in `units=` (as in 2.1).
- Results per channel are dicts keyed like the data (`"left_pupil"`):
  `get_blinks`, `get_interpolated_blinks`, `mask_as_intervals`, `nblinks`,
  `blink_stats`; `nblinks()` includes channels without blinks (0); the
  parameters of `pupil.estimate_response()` are in
  `params["response"]["left_pupil"]` (saved files are converted)
- `eyes`/`variables`: `None` or `[]` means all existing channels; unknown
  names raise a `KeyError` that lists the channels
- Every transforming method takes `inplace` (default: in place, see
  `set_default_inplace()`); this is new for `drop_data`,
  `fill_time_discontinuities` and `SegmentedEyeData.baseline_correct`, which
  used to return a copy. `set_*` methods are always in place
  (`set_events` lost `inplace`)
- Events: one `Events` object in `d.events`; `d.event_onsets`,
  `d.event_labels`, `d.event_meta`, `d.get_events()` and
  `d.set_event_onsets()` were removed
- `get_intervals(event, window)` returns the new `Epochs` (event onsets and
  one window) instead of `Intervals`; windows are not clipped at the ends
  of the recording; segment time axes are `window[0] + k * 1000/fs`, and
  segment parts outside the data are NaN and masked. `Intervals` lost
  `interval_window`, `event_indices` and `data_time_range`;
  `get_segments()` with `Intervals` needs intervals of equal length
- The merged eye of `pupil.merge_eyes()` is always `"mean"` (`mean_pupil`);
  choose another name with `name=`
- `gaze.mask_divergences(store_as="dist")` stores the eye distance once,
  as `both_dist` (eye `"both"`)
- `Intervals.as_pandas()`: the columns `event_label`/`event_onset` are
  `label`/`onset` (as in `Epochs.as_pandas()` and `Events.to_table()`)
- `pp.downsample` is the pipeline step for `d.downsample()`; the array
  function is `pp.signal.downsample`
- The functions for numpy arrays are only in `pp.signal` (e.g.,
  `pp.signal.smooth_window`, `pp.signal.pupil_kernel`); `pp.smooth_window`
  raises an error that names the new place
- `downsample()` stores the actual sampling rate (`fs/factor`)
- Gaze plots (heatmap, scanpath) draw y downward (screen pixels)
- The time axis of all plots is in seconds by default (`units="sec"`;
  was ms or min depending on the plot)
- Saved files have format 3; files of earlier versions are upgraded when
  they are read (masks whose origin is unknown become the `"user"` layer;
  stored blinks always mask the data)

### Added

- Namespaces `d.pupil`, `d.gaze` and `d.plot`
- Pipelines: `d | pp.pupil.blinks_detect() | pp.pupil.lowpass_filter(4)`;
  `pp.pupil.*`, `pp.gaze.*` and the generic steps (`pp.crop`,
  `pp.downsample`, `pp.scale`, ...) make `pp.Step`s, combined into a
  `pp.Pipeline` with `|`
- Mask layers per channel (`blink`, `interpolated`, `offscreen`,
  `divergence`, `user`) and a recording-wide `excluded` layer;
  `d.clear_layer()`, `d.get_layer_intervals()`, `d.mask()`, `d.values()`
- `pp.MaskPolicy` and the policies `"default"`, `"strict"` and `"raw"`
  (`policy=` in all processing and analysis methods)
- `pp.Epochs` (event-locked windows)
- `pypil-upgrade`: converts saved files to the current format
- Experimental setup and calibration are kept and saved for every object
- `d.plot.plot_masked(policy=...)`, e.g. `policy="strict"` to show the
  interpolated stretches
- `pytest` also runs the examples in the docstrings (doctests)

### Changed results

Every number that differs from 2.1 is listed with case, reason and old and
new values in
[tests/data/reference/CHANGES.md](https://github.com/ihrke/pypillometry/blob/master/tests/data/reference/CHANGES.md).
The main ones:

- Interpolated samples are used: segments, `stat_per_event`, `scale`,
  `merge_eyes`, `downsample` and the quality report change wherever blinks
  were interpolated
- Low-pass filter, smoothing and baseline estimation bridge unusable
  samples linearly (the low-pass filter started from the global mean) and
  change only usable samples
- `downsample` keeps a mask layer where it covers more than half of a
  window
- Segment time axes at the sampling rate (e.g., the last time point of
  the window (-200, 1000) ms at 500 Hz is 998 ms, was 1000 ms)
- Quality report: masked time is split by layer; interpolation is no
  longer counted as "other artifacts"
- Files saved after a `downsample` to a rate that does not divide the
  original one get a corrected `fs` when they are read
- Blinks next to excluded periods are blinks: `get_blinks()` (and blink
  counts, interpolation, plots) no longer leaves out blinks that overlap or
  touch an excluded period; `pupil.blinks_detect()` cuts blinks at excluded
  periods instead of dropping them. A blink without usable data at its
  anchor points (e.g., one that ends in a break) is not interpolated

### Fixed

- `fs` after `downsample` did not match the time vector
- `reset_time` did not shift blinks stored in time units
- `FakeEyeData.sim_data` was not cut or downsampled with the data
- Masks only grew: re-running a detector added to the old mask
- Experimental setup and calibration were lost when saving some objects
- `plot_heatmap(show_masked=True)` with NaN samples
- A `plot_range` with start after end raises an error
- The sample at the end of `plot_range` was not plotted
- `mask_intervals` and `interpolate_intervals` accept a list of
  `(start, end)` tuples in ms, like `exclude_intervals`
- `Events.filter(trial="1")` on a numeric metadata column (or a number on
  a column of strings) matched nothing silently; it raises a `TypeError`
- `scale()` twice (or for different variables) then `unscale()` left the
  data scaled or raised: the stored parameters are combined per channel,
  and `unscale()` removes those of the channels it restores
- `pypilview --eyes right FILE` took the file for an eye
- `CircularROI.from_degrees`/`RectangularROI.from_degrees` read plain
  numbers as radians; they are degrees now (as in
  `ExperimentalSetup.degrees_to_pixels`)
- `load_study_local`/`load_study_osf`: `subjects="001"` loaded subject
  `"1"` (the string was iterated); unknown subjects raise a `ValueError`
  (they were skipped)
- `ExperimentalSetup`: a negative `eye_to_screen_center` lost its sign;
  non-positive distances and resolutions that are not two whole numbers
  raise a `ValueError`
- `pypillometry.viewer.tweak` (the function) was replaced by its submodule
  once the tweak viewer had been used; the submodule is now
  `pypillometry.viewer.tweak_canvas`
- `pupil.estimate_response(eyes=...)` removed the stored parameters of the
  other eyes
- `pupil.estimate_response()` on data without events raises a `ValueError`
  (it returned the start parameters)
- `pp.nologging()` could leave logging disabled after the `with` block
- Labels of `get_intervals(("F_", "R_"))`: `"F_ to R_"` (was `"F__to_R_"`),
  per interval e.g. `"F_1 to R_1"`
- `summary()`, `Intervals.stats()` and the segment summary return Python
  numbers (not `np.float64`); `summary()["params"]` is the dict of
  parameters (was the string `'{...}'`)
- `pupil.blinks_detect()` with an unknown strategy (e.g. `"zeros"`) raises a
  `ValueError` (it only logged a warning and found no blinks)
- `crop()` with start after end or outside the data raises a `ValueError`
  (it returned an empty object)
- Re-running `pupil.blinks_detect()` after `pupil.blinks_interpolate()`
  found no blinks (it read the interpolated values) and cleared the blink
  layer; eyes with interpolated samples are now skipped with a warning
- The source distribution contains all test data (`MANIFEST.in`)
- Docstring examples run as written (several used undefined names or
  outdated calls)

### Upgrading code: old -> new

Methods (`pp.<namespace>.<method>(...)` is the pipeline step):

| 2.1 | 2.2 method | pipeline step |
|---|---|---|
| `d.pupil_lowpass_filter(...)` | `d.pupil.lowpass_filter(...)` | `pp.pupil.lowpass_filter(...)` |
| `d.pupil_smooth_window(...)` | `d.pupil.smooth_window(...)` | `pp.pupil.smooth_window(...)` |
| `d.pupil_merge_eyes(...)` | `d.pupil.merge_eyes(...)` | `pp.pupil.merge_eyes(...)` |
| `d.pupil_blinks_detect(...)` | `d.pupil.blinks_detect(...)` | `pp.pupil.blinks_detect(...)` |
| `d.blinks_merge_close(eyes, variables, ...)` | `d.pupil.blinks_merge_close(eyes, ...)` (pupil only) | `pp.pupil.blinks_merge_close(...)` |
| `d.pupil_blinks_interpolate(...)` | `d.pupil.blinks_interpolate(...)` | `pp.pupil.blinks_interpolate(...)` |
| `d.pupil_estimate_baseline(...)` | `d.pupil.estimate_baseline(...)` | `pp.pupil.estimate_baseline(...)` |
| `d.pupil_estimate_response(...)` | `d.pupil.estimate_response(...)` | `pp.pupil.estimate_response(...)` |
| `d.correct_pupil_foreshortening(...)` | `d.pupil.correct_foreshortening(...)` | `pp.pupil.correct_foreshortening(...)` |
| `d.fit_foreshortening(...)` | `d.pupil.fit_foreshortening(...)` | (returns a calibration) |
| `d.mask_offscreen_coords(...)` | `d.gaze.mask_offscreen(...)` | `pp.gaze.mask_offscreen(...)` |
| `d.mask_eye_divergences(...)` | `d.gaze.mask_divergences(...)` | `pp.gaze.mask_divergences(...)` |
| `d.crop(...)`, `d.downsample(...)`, `d.scale(...)`, ... | unchanged | `pp.crop(...)`, `pp.downsample(...)`, `pp.scale(...)`, ... |

The generic methods that are also steps: `crop`, `downsample`, `scale`,
`unscale`, `merge_eyes`, `reset_time`, `mask_intervals`,
`interpolate_intervals`, `exclude_intervals`, `keep_intervals`,
`remove_intervals`, `merge_masks`, `clear_layer`, `set_blinks`,
`set_events`, `add_events`, `set_experimental_setup`, `drop_data`,
`drop_original`, `fill_time_discontinuities`. The blink getters and setters
(`get_blinks`, `set_blinks`, `nblinks`, `blink_stats`,
`get_interpolated_blinks`) stay on `d`.

Plots: `d.<plot>(...)` and the plotter classes -> `d.plot.<plot>(...)`, with
the same names: `plot_timeseries`, `plot_timeseries_segments`,
`plot_intervals`, `plot_masked`, `plot_experimental_setup`, `pupil_plot`,
`pupil_plot_segments`, `plot_blinks`,
`plot_pupil_foreshortening_error_surface`, `plot_heatmap`, `plot_scanpath`,
`plot_calibration`.

Calls whose arguments or results changed:

| 2.1 | 2.2 |
|---|---|
| `d.pupil_blinks_detect(apply_mask=True)` | `d.pupil.blinks_detect()` |
| `bl = d.pupil_blinks_detect()` (returned the blinks) | `d.pupil.blinks_detect(); bl = d.get_blinks()` |
| `merged = d.blinks_merge_close(...)` (dict) | `d.pupil.blinks_merge_close(..., inplace=False).get_blinks()` |
| `d.mask_offscreen_coords(apply_mask=False)` | `d.gaze.mask_offscreen(inplace=False).get_layer_intervals("offscreen")` |
| `d.mask_eye_divergences(apply_mask=False)` | `d.gaze.mask_divergences(inplace=False).get_layer_intervals("divergence")` |
| `d.set_blinks(iv, apply_mask=False)` | `d.set_blinks(iv)` (blinks always mask) |
| `d.get_blinks("left", "pupil")` (Intervals, samples) | `d.get_blinks("left", "pupil")["left_pupil"]` (ms; `units="samples"` for indices) |
| `d.mask_as_intervals("left", "pupil")` | `d.mask_as_intervals("left", "pupil")["left_pupil"]` (ms) |
| `d.blink_stats()["left"]` (None without blinks) | `d.blink_stats()["left_pupil"]` (n=0 without blinks) |
| `d.get_intervals("S_", (-100, 400))` (samples) | `d.get_intervals("S_", (-100, 400), units="samples")`; plain numbers are ms |
| `d.get_duration()` (min) | `d.get_duration("min")`; default ms (also `excluded_duration`) |
| `d.pupil_plot(plot_range=(0, 20))` (sec) | `d.plot.pupil_plot(plot_range=(0, 20) * u.sec)` |
| `d.plot_timeseries_segments(interv=1)` (min) | `d.plot.plot_timeseries_segments(interv="1 min")` |
| `d2 = d.drop_data(...)` (always a copy) | `d2 = d.drop_data(..., inplace=False)` |
| `seg.baseline_correct(w)` (always a copy) | `seg.baseline_correct(w, inplace=False)` |
| `d.set_events(ev, inplace=False)` | `d.copy().set_events(ev)` |
| `d.pupil_merge_eyes()` -> `merged_pupil` | `d.pupil.merge_eyes()` -> `mean_pupil` (`name=` to choose) |
| `d.event_onsets`, `d.event_labels`, `d.event_meta` | `d.events.onsets`, `d.events.labels`, `d.events.meta` |
| `d.get_events(units)` | `d.events.to_units(units)` |
| `d.set_event_onsets(on, lab, meta)` | `d.set_events(pp.Events(on, lab, meta=meta))` |
| `pp.PupilData(...)`, `pp.GazeData(...)` | `pp.EyeData(...)` |
| `d.to_pupildata()` | `d.drop_data(variables=["x", "y"], inplace=False)` |
| `d.to_gazedata()` | `d.drop_data(variables="pupil", inplace=False)` |
| `pp.downsample(x, factor)` (array) | `pp.signal.downsample(x, factor)` |
| `pp.smooth_window(...)`, `pp.pupil_kernel(...)`, ... (array functions) | `pp.signal.smooth_window(...)`, `pp.signal.pupil_kernel(...)`, ... |
| `d.params["response"]["left"]` | `d.params["response"]["left_pupil"]` |
| `iv.as_pandas()["event_label"]`, `["event_onset"]` | `iv.as_pandas()["label"]`, `["onset"]` |
| `CircularROI.from_degrees((0.087, 0), 0.035, s)` (radians) | `CircularROI.from_degrees((5, 0), 2, s)` (degrees) |
| history `pupil_blinks_detect(..., inplace=True)` | history `pupil.blinks_detect(...)` |

## 2.1.0 (2026-11-02)

### Breaking changes

- `EyeData.get_pupildata()` was removed, use `to_pupildata()` 
- `Events` now need time units, sample indices are `units="samples"`
- Converting `Intervals` between sample indices and time now requires the data
- `sub_slice()` was renamed to `crop()`; `sub_slice()` is a deprecated alias
- `get_intervals()`, `stat_per_event()`: `interval` was renamed to `window`

### Added

- `EyeData.to_pupildata()` and `to_gazedata()`
- `quality_report()`: data-quality measures 
- Default quality thresholds based on reference data from six datasets
- `pypilinfo --quality/--html/--descriptions`: quality reports
- HDF5 files: `write_file()`/`write_pickle()` write HDF5 if filename ends `.h5`
- Excluded periods (e.g., breaks, calibration): `exclude_intervals()`
- `remove_intervals()`: remove periods (e.g., breaks)
- `get_intervals()`: `"<start>"` and `"<end>"` 
- Events from behaviour logs: `Events.from_table()` 
- `sync_clocks()` (fits offset and drift between clocks)
- Event metadata (e.g., trial, condition, reaction time): 
- `get_intervals()` passes the metadata 

### Changed

- `pupil_estimate_response()` does not use excluded periods and NaN
  samples for the fit; events with less than `min_valid` (default 0.5) of
  their response outside them are left out (coefficient NaN)
- `get_intervals()` with two events pairs each first event with the next
  second event (e.g., `("End Block", "Start Block")` gives the breaks)
- Saved files (pickle and HDF5) contain a versioned state of the object.
  Files from earlier versions are upgraded when they are read.

### Fixed

- `pupil_estimate_response()` uses more efficient solver
- `downsample()` respects masked samples
- `set_events()` stored onsets of `Events` with `units=None` as ms
- Segments of index intervals that are not event-locked wrong time
- `plot_intervals()` with index intervals mixed sample indices and ms

## 2.0.0 (2026-10-28)

### Breaking changes

- Python 3.11 or newer is required.
- `inplace=None` now follows the object setting. 
- `fill_time_discontinuities()` fills gaps with NaN (was 0), so filled-in
  samples are masked for all variables, not only pupil.
- `apply_history()` was removed
- `ExperimentalSetup.degrees_to_mm()` take plain numbers as degrees
- Positive y is up in all mm coordinates
- new module `pp.study` hosting all study-related io functions
- Internal helpers no longer exported at the top level
- `blinks_merge_close()` defaults to `apply_mask=False` (returns the merged
  blinks), like `pupil_blinks_detect()`
- `nprange`, `p_asym_laplac`, `p_asym_laplac_kappa` and `trans_logistic_vec`
  are no longer exported at the top level (still in `pp.convenience`)
- `stat_event_interval` is no longer exported at the top level (use
  `stat_per_event()`)
- `pint` is no longer used, replaced by `units.py`

### Added

- files are now saved with compression by default
- `pypilinfo` command prints a summary of EDF or pickle files
- Times with units (`units=` can be omitted)
- `pp.parse_time()` converts times with units
- `pp.set_default_inplace()` and `pp.get_default_inplace()`: package-level
  default for `inplace` (initially `True`).
- `detect_blinks_zero()` accepts `blink_val=np.nan` to detect runs of NaN.
- Reading from SSH URLs (`ssh://[user@]host[:port]/path`)
- `pypilview`: interactive viewer EDF file stored pypil object

### Changed

- `read_pickle()` saves HTTP(S) downloads to a file (temporary by default)
- Packaging moved from `setup.py`/`requirements.txt` to `pyproject.toml`.
- Top-level imports are static, so IDEs can complete and resolve `pp.*`.
- `detect_blinks_velocity()`: velocity runs with exactly `min_onset_len`/
  `min_offset_len` samples are accepted (was: must be longer)
- `interpolate_intervals(margin=...)` uses the sign of `margin` (relative to
  start/end of the interval); `margin[0] > 0` or `margin[1] < 0` raise an error
- `get_example_data()` caches downloaded datasets in `~/.cache/pypillometry`
  (or `cache_dir`) instead of downloading them on every call

### Fixed

- `pp.signal` and `pp.logging` were `scipy.signal` and the standard-library
  `logging` instead of the package's own submodules.
- `fill_time_discontinuities()`: samples between two gaps
  were dropped, so time and data had different lengths
- `pupil_blinks_detect()`: the "zero" strategy never detected blinks
- `stat_per_event()` gave wrong results for intervals not in ms 
- `group_segments()`/`GroupLevelSegmentedData.from_segments()` dropped masks
- `stat_per_event()` ignored masks
- `SegmentedEyeData.plot()`/`GroupLevelSegmentedData.plot()` warn if `meanfct`
  or `varfct` ignore masks (e.g., `np.median`)
- `meanfct`/`varfct` without an `axis` argument raise an error
- `scale()` computed mean and SD including masked samples
- `fit_foreshortening()` used masked samples for fitting
- `GazeData.summary()` was missing the `eyes` entry
- `ExperimentalSetup`: parse string (e.g. `"30 deg"`)
- `test_selector()` raised a `NameError` fixed
- `sub_slice()` dropped all masks and did not shift stored blinks
- `downsample()` dropped all masks  and did not rescale stored blinks
- `fill_time_discontinuities()` called after creation dropped all masks 
- History: internal calls no longer recorded, setters are now recorded
- `get_intervals()` between two events fixed 
- `set_experimental_setup()` lost the eye-to-screen distance when updating
  other parameters
- `plot_heatmap()`/`plot_scanpath()` with a single eye as string 
- `get_example_data("rlmw_010_edf")` failed (loader was missing)
- `detect_blinks_zero()` missed runs at the start/end of the signal
- `pupil_estimate_baseline()` raises a clear error for data with NaN
- `EyeDataDict.copy()` kept only masks for NaN values
- `fit_foreshortening(intervals=...)` treated all intervals as ms
- `view()` of raw arrays without `time`
- `baseline_pupil_model()` used a 4 ms instead of a 4 s response kernel
- `pupil_blinks_interpolate()` left blinks NaN when the anchor points fell
  into a neighbouring blink; such blinks are now merged
- `merge_eyes()`/`pupil_merge_eyes()`: samples masked in all eyes lost their
  values (0 or NaN, e.g., interpolated blinks)
- `blink_onsets_mahot()` (used by `pupil_blinks_interpolate()`) searched the
  offset too late for blinks near the start of the data
- `PupilData.from_eyelink()`/`GazeData.from_eyelink()` raised a `TypeError`
- `Intervals.plot_highlight(zorder=...)` raised a `TypeError`
- `Intervals.stats()` raised a `ValueError` for empty intervals
- `Intervals` repr showed "None" as unit for index intervals
- `Intervals.from_mask(units=...)` labelled sample indices with time units
  without converting them
- `view(highlight=...)`: index intervals failed for raw arrays and were
  converted although the x-axis is in samples; the passed Intervals were
  modified
- `plot_intervals(show_mask=True)` showed only the mask of the last eye/variable
- `plot_masked(units=None, show_events=True)` drew events in ms on an index axis
- `plot_timeseries_segments(ylim=...)`/`pupil_plot_segments(ylim=...)` set
  the limits only for the last subplot
- `scale()` rejected integers and `unscale()` rejected numbers for `mean`/`sd`
- `ExperimentalSetup.mm_to_pixels()` without screen size raised a `TypeError`
  instead of a clear `ValueError`
- `ExperimentalSetup.from_dict()` warned about plain numbers for stored values
- `detect_blinks_velocity()` raised a `ValueError` if no sample was valid
- `detect_blinks_velocity()`: blinks without offset ended one past the last
  sample (other offsets are inclusive)
- `fake_gaze_fixations()` left trailing samples at (0, 0) when the fixations
  were shorter than `duration`
- `pupil_kernel()` was sampled slightly slower than `fs` (spacing
  `duration/(n-1)`)
- `interpolate_intervals()` used masked samples (e.g., other blinks) as anchor
  points
- `get_available_eyes()`/`get_available_variables()` returned eyes/variables
  in random order; now in the order they were added
- `read_pickle()`/`read_eyelink()` of URLs without `cache_file`/`cache_dir`
  left the temporary download directory behind
- `load_study_osf()` printed to stdout instead of logging and showed the
  download size only with `force_download=True`
- Example datasets: missing spaces in descriptions; "rlmw_002" was named
  "test short"
- `pupil_response()` with free `npar` and `tmax` passed the unsupported `disp`
  option to scipy (`OptimizeWarning`); progress is now logged
- Stan baseline: the model was set up on every call and could not be
  compiled in a read-only install; results were not reproducible (new `seed`
  argument)
- `tweak()`: "?" did not open the help
- `nblinks()` was annotated as returning `int` (returns a dict)
- `get_segments()`: the number of samples per segment depended on the timing
  of the first event, so `group_segments()` could fail
- `ExperimentalSetup.screen_point_to_eye_frame()`: tilting the screen moved
  its center towards the eye (z was `d*cos(pitch)*cos(yaw)` instead of `d`)
- `pupil_blinks_interpolate()` skipped blinks without a velocity drop
  before them, leaving NaNs in the data
- `SegmentedEyeData.baseline_correct()`, `Events.filter()` and
  `plot_timeseries_segments(interv=...)` failed with units
- `plot_experimental_setup(calibration=...)`, `set_experimental_setup()`
  (updating an existing setup) and `fit_foreshortening()` gave spurious
  warnings
- `tweak()` showed a wrong time axis until the view was zoomed
- `repr()` of `FakeEyeData` was garbled; summaries and `pypilinfo` print
  rounded numbers
- API signatures showed `units=<object object>`

### Documentation

- Updated website to a more modern feel.
- Updated docstrings across the package (eyedata classes, signal, plotting,
  intervals, io, viewer, experimental setup).
- Fixed broken links, references and markup in the docs and notebooks.
- Docs workflow pinned to Python 3.13.
