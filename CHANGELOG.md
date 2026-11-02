# Changelog

## Unreleased

### Breaking changes

- `EyeData.get_pupildata()` was removed, use `to_pupildata()` 
- `Events` need time units; `units=None` (sample indices) raises an error
- Sample indices are `units="samples"`; `units=None` raises an error
  (`get_intervals`, `get_blinks`, `mask_as_intervals`, `Intervals`, plots)
- Converting `Intervals` between sample indices and time needs the data
  object: `to_units("ms", d)`, `to_mask(d)`, `Intervals.from_mask(mask, d)`;
  `Intervals` no longer have `sampling_rate` and `time_offset`
- `Intervals.merge(distance=...)`: plain numbers are in the units of the
  intervals (were ms)
- `plot_timeseries()`, `plot_scanpath()`, `sub_slice()`: default `units="ms"`
  (was None)
- `sub_slice()` was renamed to `crop()`; `sub_slice()` is a deprecated alias
- `get_intervals()`, `stat_per_event()`: `interval` was renamed to `window`;
  it has no default for single events (was `(-200, 200)` in
  `get_intervals()`), and `(0, 0)` between two events

### Added

- `EyeData.to_pupildata()` and `to_gazedata()`
- `quality_report()`: data-quality measures 
- Default quality thresholds based on reference data from six datasets
- `pypilinfo --quality/--html/--descriptions`: quality reports
- HDF5 files: `write_file()`/`write_pickle()` write HDF5 if filename ends `.h5`
- Excluded periods (e.g., breaks, calibration): `exclude_intervals()`,
  `get_excluded()`, `excluded_duration()`; all variables are masked there,
  blink detection ignores them, and they are saved with the object;
  `quality_report()` only uses the included time; `get_included()`,
  `split_excluded()` (one object per included period)
- `remove_intervals()`: remove periods (e.g., breaks) and join the rest
  (time and events after them are shifted back)
- `keep_intervals()`: exclude everything outside the given periods;
  `Intervals.complement()`: the periods not covered
- `get_intervals()`: `"<start>"` and `"<end>"` select the start and end of
  the recording; `exact=True` (also `Events.filter()`) matches labels
  equal to the string
- Events from behaviour logs: `Events.from_table()` (one column per kind of
  event), `sync_clocks()` (fits offset and drift between the clocks from
  paired times, shows the residuals), `add_events()`; `event_onset()`
  returns the onset of the one event with a label
- Event metadata (e.g., trial, condition, reaction time): `Events.meta`
  (one numpy array per column), stored in the data (`event_meta`) and in
  saved files; from trigger labels (`Events.parse_labels()`) or from a
  behaviour log (`Events.from_table(meta=...)`); `filter()` selects by
  metadata (e.g., `filter("S_", cond="easy", rt=lambda rt: rt < 800)`);
  `set_meta()`, `fill_meta()` (copy to all events of a trial),
  `add_latency()` (e.g., reaction times), `row()`, `to_table()`, `+`
- `get_intervals()` passes the metadata of the events on to the intervals
  (`Intervals.event_meta`, also in `as_pandas()`)
- `pupil_estimate_response()` stores the share of each event's response
  that is valid (`valid`) and that is real, not interpolated, data
  (`observed`) in `params["response"][eye]`; `pupil_response(valid=...)`,
  `pupil_response_coverage()`

### Changed

- `pupil_estimate_response()` does not use excluded periods and NaN
  samples for the fit; events with less than `min_valid` (default 0.5) of
  their response outside them are left out (coefficient NaN)
- `get_intervals()` with two events pairs each first event with the next
  second event (e.g., `("End Block", "Start Block")` gives the breaks);
  events without a partner are skipped with a warning (before: pairs by
  order, error if the numbers differ)
- Saved files (pickle and HDF5) contain a versioned state of the object.
  Files from earlier versions are upgraded when they are read.
- Conversions between sample indices and time use the time vector of the
  data (a time maps to the first sample at or after it), so recordings that
  do not start at 0 or have gaps are handled correctly
- `fit_foreshortening(intervals=...)`: intervals are end-exclusive

### Fixed

- `downsample()` set windows in which all samples are masked to NaN, so
  interpolated blinks were lost; they now hold the mean of the stored values
  (and stay masked)
- Baseline estimation: the weights of the troughs were given to the sample
  before each trough (0-based indices passed to Stan)
- HTML representation of eyedata objects escapes names, labels and history
- `set_events()` stored onsets of `Events` with `units=None` as ms
- Segments of index intervals that are not event-locked got a time axis
  that ignored the start time of the recording
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
