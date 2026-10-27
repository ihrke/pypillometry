# Changelog

## Unreleased

### Breaking changes

- Pupil values of 0 are no longer converted to NaN automatically.
  `EyeDataDict` now stores data as given and only masks NaN values. Before,
  any 0 in a pupil variable was turned into a masked NaN.
- `inplace=None` now follows the object setting. 
  Methods called with `inplace=None` (the default) use the `inplace` value 
  given to the constructor, then the new package default.
- `fill_time_discontinuities()` fills gaps with NaN (was 0), so filled-in
  samples are masked for all variables, not only pupil.
- `apply_history()` was removed
- `ExperimentalSetup.degrees_to_mm()`/`degrees_to_pixels()` and
  `plot_experimental_setup(viewing_angle=...)` take plain numbers as degrees
  (were radians); `degrees_to_*()` also accept plain arrays.
- Positive y is up in all mm coordinates: `ExperimentalSetup.pixels_to_mm()`
  and `mm_to_pixels()` flip the y-axis (`centered=False` is relative to the
  lower-left corner), and `fit_foreshortening()` uses them. Camera angle
  `phi=-90 deg` now means "camera below the eye" for fitting as well;
  calibrations fitted before have `phi` mirrored.
- new module `pp.study` hosting all study-related io functions

### Added

- `pp.set_default_inplace()` and `pp.get_default_inplace()`: package-level
  default for `inplace` (initially `True`).
- `detect_blinks_zero()` accepts `blink_val=np.nan` to detect runs of NaN.
- Reading from SSH URLs (`ssh://[user@]host[:port]/path`)

### Changed

- `read_pickle()` saves HTTP(S) downloads to a file (temporary by default)
  instead of reading them into memory.
- Packaging moved from `setup.py`/`requirements.txt` to `pyproject.toml`.
  The obsolete `typing` backport and the test tools are no longer installed
  as dependencies.

### Fixed

- `fill_time_discontinuities()` run by constructor: samples between two gaps
  were dropped, so time and data had different lengths
- `pupil_blinks_detect()`: the "zero" strategy never detected blinks
- `stat_per_event()` gave wrong results for intervals not in ms 
- `group_segments()`/`GroupLevelSegmentedData.from_segments()` dropped masks,
  so blinks and padding entered group means; now warns if `meanfct` ignores
  masks (e.g., `np.median`)
- `stat_per_event()` ignored masks (masked samples entered the statistic);
  windows with only masked samples now give NaN
- `SegmentedEyeData.plot()`/`GroupLevelSegmentedData.plot()` warn if `meanfct`
  or `varfct` ignore masks (e.g., `np.median`)
- `meanfct`/`varfct` without an `axis` argument raise a clear error in
  `group_segments()` and the segment plots
- `scale()` computed mean and SD including masked samples
- `fit_foreshortening()` used masked samples for fitting
- `GazeData.summary()` was missing the `eyes` entry
- `ExperimentalSetup`: viewing angles given as strings (e.g., `"30 deg"`) are
  parsed before plotting.
- `test_selector()` raised a `NameError` on every call
- `sub_slice()` dropped all masks and did not shift stored blinks
- `downsample()` dropped all masks (masked samples entered the window means)
  and did not rescale stored blinks
- `fill_time_discontinuities()` called after creation dropped all masks and
  did not move stored blinks
- History: internal calls no longer recorded, setters are now recorded
- `get_intervals()` between two events fixed 
- `set_experimental_setup()` lost the eye-to-screen distance when updating
  other parameters
- `plot_heatmap()`/`plot_scanpath()` with a single eye as string (e.g.,
  `eyes="left"`) plotted one panel per letter
- `get_example_data("rlmw_010_edf")` failed (loader was missing)
- `detect_blinks_zero()` missed runs at the start/end of the signal;
  `min_duration` is now the minimum number of samples (was one more)
- `pupil_estimate_baseline()` raises a clear error for data with NaN (was
  an obscure `zero-size array` error)
- `EyeDataDict.copy()` kept only masks for NaN values
- `fit_foreshortening(intervals=...)` treated all intervals as ms (now
  index-based or any time units)
- `view()` of raw arrays without `time`: selections were labelled as
  seconds although they are sample indices (now `units=None`)
- `baseline_pupil_model()` used a 4 ms instead of a 4 s response kernel
- `pupil_blinks_interpolate()` left blinks NaN when the anchor points fell
  into a neighbouring blink; such blinks are now merged
- `merge_eyes()`/`pupil_merge_eyes()`: samples masked in all eyes lost their
  values (0 or NaN, e.g., interpolated blinks); they now hold the merged
  values (still masked). `merge_eyes()` docstring: masked where *all* eyes
  are masked
- `blink_onsets_mahot()` (used by `pupil_blinks_interpolate()`) searched the
  offset too late for blinks near the start of the data, so they were skipped

### Documentation

- Updated docstrings across the package (eyedata classes, signal, plotting,
  intervals, io, viewer, experimental setup).
- Fixed broken links, references and markup in the docs and notebooks.
- Docs workflow pinned to Python 3.13.
