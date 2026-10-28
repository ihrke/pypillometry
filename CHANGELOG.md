# Changelog

## Unreleased

### Breaking changes

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
- `ExperimentalSetup.screen_point_to_eye_frame()`: tilting the screen moved
  its center towards the eye (z was `d*cos(pitch)*cos(yaw)` instead of `d`)

### Documentation

- Updated docstrings across the package (eyedata classes, signal, plotting,
  intervals, io, viewer, experimental setup).
- Fixed broken links, references and markup in the docs and notebooks.
- Docs workflow pinned to Python 3.13.
