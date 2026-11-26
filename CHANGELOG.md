# Changelog

## unreleased

- `EyeData.from_gazepoint()`, `read_gazepoint()`: read recordings of GazePoint eye trackers (Open Gaze API columns); blinks hidden as repeated values are masked
- fix: `fill_time_discontinuities()` left out one sample in some gaps (rounding)
- fix: `pp.signal.velocity_savgol(direction="backward"/"forward")` evaluated the derivative half a window ahead (backward) or behind (forward), so the one-sided velocities of `pupil.blinks_detect()` (velocity strategy) and of the `"mahot"` interpolation looked into the blink; the one-sided velocities of both are now the slope of a line fitted to the window (a quadratic overshoots at the end of its window, e.g. after the reopening). Detected blink onsets come slightly later, offsets slightly earlier (by up to half the window)
- `pp.TimescaleModel` (experimental): state-space model of the pupil size with a fast component and slow components at fixed time scales; slow baseline with uncertainty, checks, AIC/BIC and cross-validation
- `TimescaleModel.interpolate()`, `.predict()`, `.store()`: reconstruction of missing data of the whole recording and the slow part at the original samples
- `pp.TimescaleModel(d, binocular=True)`: one model for both eyes
- `TimescaleModel.fit(n_jobs=, progress=)`: likelihood evaluations of the gradient and the Hessian in parallel threads (about 2.5x faster for one eye, 4x for both); progress bar
- `pp.TimescaleGroup` (experimental): group-level fit of several recordings with shrinkage; `g.prior` for new recordings
- `d.pupil.spectrum()`, `d.plot.plot_pupil_spectrum()`: power spectrum without interpolating the gaps
- `pp.signal.spectrum()`, `pp.signal.block_mean()`, `pp.signal.coarsen()`
- `pp.signal.semivariogram(segments=)`, `Variogram.at_log_lags()`
- `pp.signal.gaussmarkov`: `TimescaleBank`, `fit_bank()`, `fit_group()`, `general_prior()`, `Constant` component, noise variance per sample, segments and diffuse states in the Kalman filter
- `pp.sim` (experimental): simulated data with ground truth, built by chaining steps (`pp.sim.Simulation("5 min").pupil().gaze().foreshortening().units(800).noise(0.2).run()`); `SimEyeData` with `d.truth` (true time series, follow cropping and downsampling), `d.simulation` (the recipe and seed) and the plots `plot_truth()` and `plot_components()`
- `TimescaleModel.simulate()`, `.process()`: new recordings from a fitted model
- `pp.sim.Task`: task designs built by chaining (`.block()`, `.stimulus()`, `.response()`, `.rest()`, `.repeat()`; times as distributions like `"uniform(2 s, 10 s)"`), whose events go into simulated recordings (`Simulation.task()`) with their metadata; ready-made tasks `pp.sim.tasks.pvt()`, `sart()`, `ftrsgt()`, `nback()`; `task.plot()`, a schematic in notebooks and `d.plot.plot_task_events()`

### Breaking changes

- `FakeEyeData` and `pp.signal.fake` (`fake_pupil_baseline()`, `fake_gaze_fixations()`, `add_measurement_noise()`, `generate_foreshortening_data()`) were removed, use `pp.sim`; saved `FakeEyeData` objects can no longer be read

### Changed behaviour

- `pupil.blinks_detect()`: the default velocity thresholds are relative to the noise of the velocity (`vel_onset="25sd"`, `vel_offset="25sd"`: 25 robust standard deviations), instead of +-5 pupil units per ms, so that they work in any units (mm, pixels, arbitrary units) and with noisy trackers; the window of the velocity is 45 ms (was 11 ms). The detected blinks usually cover more of the slow transients. In a simulation study with real blinks and tracker noise of six datasets, this transferred best between recordings (vs. raw units, percentiles or thresholds relative to the pupil size). Numbers (units per ms) work as before; the percentile form (`"5%"`) was removed.
- `pupil.blinks_interpolate(method="mahot")`: default thresholds `"2sd"` and window 30 ms (were +-5 units per ms and 25 ms); on real blinks transplanted into clean data, the error of the interpolation drops by about a third. `blinks_optimize_parameters()` and `tweak_blink_interpolation()` start from these, converted to units per ms.
- the windows of the velocity (detection and interpolation) are rounded to the nearest odd number of samples and have at least 5 samples (at low sampling rates, e.g. 60 Hz, the window was 1-3 samples); the minimal durations of `blinks_detect()` are rounded to the nearest sample, and a blink has at least 2 samples

## 2.2.4 (2026-11-24)

- `d.pupil.blinks_optimize_parameters()`: autotune the parameters of `blinks_interpolate()` 
- `pp.tweak_blink_interpolation()`: tweak the parameters of `blinks_interpolate()` 
- `d.plot.plot_blink_clusters()`: the detected blinks in groups of similar shape, to spot bad detections
- `pp.view(..., overlay=)`: support other EyeData objects 
- `pp.view()`, `pp.tweak()`: also work with PySide6 
- `pp.signal.transplant_blinks()` and related functions

### Changed behaviour

- `blinks_interpolate(method="kalman")`: replaces a `margin` around the core of each blink 
- `blinks_detect()`: `max_duration` (now 2 s) applies to all blinks
- `blinks_interpolate()` (Mahot): blinks are only merged if they overlap
- `blinks_interpolate()` (Mahot): stretches longer than `max_duration` (2 s) are skipped
- dependency: PySide6 instead of PyQt5 

## 2.2.3 (2026-11-19)

- `blinks_interpolate(method="kalman")`: new interpolation method based on Kalman filter
- `blinks_interpolate()`: change parameter handling to allow multiple methods
- `blinks_detect()`: `vel_onset_min_duration` and `vel_offset_min_duration` default to 10 ms (was 5 ms)
- `blinks_detect()`: a velocity onset without its own offset no longer takes the offset of the next blink
- `blinks_detect(margin=...)`: extend the detected blinks on each side, e.g. to include the transient
- `blinks_interpolate()`: the Mahot method now searches for transient onsets in the whole window

## 2.2.2 (2026-11-11)

- `HeadMountedEyeData`: data from eye-tracking glasses 
- `HeadMountedEyeData.from_tobii_prolab()`, `read_tobii_prolab()`: read data exports of Tobii Pro Lab (Tobii Pro Glasses 3)
- `d.scene` functions to analyze scene from video
- `d.plot.plot_scene_scanpath()`: scanpath on a frame of the scene video
- `d.scene.render_video()`: write the scene video with the gaze position drawn on it
- `d.plot.plot_timeseries()` of `HeadMountedEyeData` also plots channels in `d.aux` (e.g. `variables=["pupil", "luminance", "gyro"]`)
- `d.plot.plot_pupil_covariate()`: pupil size against a channel in `d.aux`
- `pp.view()` and `pp.tweak()` start much faster

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
