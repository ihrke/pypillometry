# Changelog

## Unreleased

- `HeadMountedEyeData`: data from eye-tracking glasses (gaze in the scene video, motion sensors and other channels in `d.aux`)
- `HeadMountedEyeData.from_tobii_prolab()`, `read_tobii_prolab()`: read data exports of Tobii Pro Lab (Tobii Pro Glasses 3)
- `d.scene`: `luminance()` of the scene video, viewing `distance()`, `head_movement()`, `frame()`
- `d.plot.plot_scene_scanpath()`: scanpath on a frame of the scene video

## 2.2.1 (2026-11-10)

- `Events.merge_table()` and `Epochs.merge_table()`: add the columns of a behaviour log 
- figshare support 
- Example dataset `nback_001` + case study
- All plotting functions follow the same conventions 

### Breaking changes

- `pupil_plot()` is `plot_pupil()`, `pupil_plot_segments()` is `plot_pupil_segments()`
- Plotting homogenized


## 2.2.0 (2026-11-10)

This version is a major refactor of the package without changing functionality (many bugs fixed, though). Scripts will need update to the new syntax.

### Breaking changes

- One data class, `EyeData` instead of per modality, pupil methods are in `d.pupil`, gaze methods in `d.gaze`
- similarly,  plotter classes (`EyePlotter`, `PupilPlotter`, `GazePlotter`) were removed
- Mask layers: multiple layers for reasons data may be masked
- Plain numbers are ms everywhere
- `get_intervals()` and `get_epochs(event, window)`
- `GenericEyeData` removed
- `SegmentedEyeData.intervals` is `SegmentedEyeData.epochs` 
- `SegmentedEyeData` holds several channels
- `GroupLevelSegmentedData` was removed
- `scale()`/`unscale()`: `center=`/`scale=`
- The functions for numpy arrays are only in `pp.signal`
- `merge_intervals()` is `concat_intervals()` 
- `Intervals.plot_highlight()` was removed; the plots shade intervals with `highlight=`
- `downsample()` stores the actual sampling rate (`fs/factor`)
- Gaze plots (heatmap, scanpath) draw y downward (screen pixels)
- The time axis of all plots is in seconds by default 
- `nblinks()` was removed (use `blink_stats()` or `layer_stats()`)

### Added

- `scale(method=...)` for common transformations
- `seg.pupil.baseline_correct(method=...)`
- faster `import pypillometry`
- Namespaces `d.pupil`, `d.gaze` and `d.plot`
- Segments of several channels share their epochs
- `seg.gaze` for regions of interest: `roi_hits()`
- User guide page "Regions of interest" 
- `seg.average(by=...)` averages all trials or groups of them - `concat_segments()` 
- Pipelines with pipe operator: `d | pp.pupil.blinks_detect()`
- Mask layers per channel 
- `pp.MaskPolicy` to decide which samples to mask
- `pp.Epochs` (event-locked windows)
- Warning for events on another clock
- `Events.onset(label)` find onset of a special event 
- `highlight=` and `highlight_color=` in `pupil_plot`
- `pypil-upgrade`: converts saved files to the current format
- `d.plot.plot_masked(policy=...)` to show the
  interpolated stretches
- `"<start>"` and `"<end>"` for interval selectors
- `d.layer_stats()`: stats for each mask layer per channel
- `pp.Table` and `pp.RecordDict`
- `seg.plot_carpet()`: all segments of a channel as an image

### Fixed

- `from_eyelink()` with non-English day and month names
- Download progress bars hidden in `pp.nologging()` 
- `fs` after `downsample` did not match the time vector
- `reset_time` did not shift blinks stored in time units
- `FakeEyeData.sim_data` was not cut or downsampled 
- Masks only grew: re-running a detector added to the old mask
- Experimental setup and calibration were sometimes lost
- `plot_heatmap(show_masked=True)` with NaN samples
- A `plot_range` with start after end raises an error
- The sample at the end of `plot_range` was not plotted
- `scale()` twice then `unscale()` left the data scaled
- `pypilview --eyes right FILE` took the file for an eye
- `CircularROI.from_degrees`/`RectangularROI.from_degrees` 
- `pypillometry.viewer.tweak` (the function) was replaced by its submodule
- `pupil.estimate_response(eyes=...)` removed parameters
- `pupil.estimate_response()` on data without events raises a `ValueError`
- `pp.nologging()` could leave logging disabled
- `summary()`, `Intervals.stats()` return Python
  numbers
- `pupil.blinks_detect()` with an unknown strategy 
- `crop()` with start after end or outside the data
- Re-running `pupil.blinks_detect()` after `pupil.blinks_interpolate()`
  found no blinks (it read the interpolated values) and cleared the blink
  layer

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
