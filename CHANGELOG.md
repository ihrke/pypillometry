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

### Added

- `pp.set_default_inplace()` and `pp.get_default_inplace()`: package-level
  default for `inplace` (initially `True`).
- `detect_blinks_zero()` accepts `blink_val=np.nan` to detect runs of NaN.

### Fixed

- `fill_time_discontinuities()` run by constructor: samples between two gaps
  were dropped, so time and data had different lengths
- `pupil_blinks_detect()`: the "zero" strategy never detected blinks
- `GazeData.summary()` was missing the `eyes` entry
- `ExperimentalSetup`: viewing angles given as strings (e.g., `"30 deg"`) are
  parsed before plotting.

### Documentation

- Updated docstrings across the package (eyedata classes, signal, plotting,
  intervals, io, viewer, experimental setup).
- Fixed broken links, references and markup in the docs and notebooks.
- Docs workflow pinned to Python 3.13.
