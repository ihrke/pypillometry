## TODOs

- [ ] https://github.com/psychoinformatics-de/remodnav 
- [ ] `FakePupilData` (and `FakeGazeData`?)


- [ ] merge_mask should take eye= and variables=
- [ ] make blink-detection better by also using x-y data (implement an EyeData.blinks_detect() function that uses both gaze and pupil)

## Status of implementation 2025-03-06

Porting all things from old `PupilData` to either `GenericEyeData` (for functions that operate on all variables) or `PupilData` for pupil-specific functions.

- [x] `scale()` and `unscale()` are now in `GenericEyeData` and can be called on any variable
   - they use dicts to store separate scaling parameters for each variable/eye
- [x] `get_intervals()` is now in `GenericEyeData` (only depends on events)
- [x] `sub_slice()` is now in `GenericEyeData` (operates on all variables simultaneously)
- [x] `pupil_lowpass_filter()` is now in `PupilData`
- [x] `pupil_smooth_window()`
- [x] `downsample()`  - implement in `GenericEyeData`
- [x] `merge_eyes()` - implement in `GenericEyeData` to combine left/right eyes into a single variable
  - [ ] perhaps implement regression merging?
- [x] blinks
  - [x] make the `blinks` field a property so that the blinkmask can be kept in sync
  - [x] `pupil_blinks_detect()` - or can this be also for gaze? Will definitely need different algorithms. So should stick with different names
  - [x] `blinks_merge()` - for is a `PupilData` method but if there will be blinks in the gaze data at some point, the whole `blink` functionality should be moved to `GenericEyeData`
  - [x] make a "mask" field for the `EyeDataDict` that is a dict of boolean arrays that can be used to mask out parts of the data for blinks/interpolation?
  - [x] `pupil_blinks_interpolate()` - merge with Mahot function and make one an option for the other
- [x] baseline/response estimation
  - [x] `pupil_estimate_baseline()`
  - [x] `pupil_estimate_response()`
- [x] statistics per events
- [x] some more general interval processing
- [x] plotting
- [ ] rework ERPD class: make it use the EyeDataDict but so that each entry is a 2D np.array that stores all the windows for a given event (have to match all dims)



## Thoughts and TODO after meeting 2024-06-18

- multiple inheritance scheme:
  - `GenericEyeData` implements bookkeeping, history, etc like now
  - `PupilData` inherits from `GenericEyeData` and implements pupil-specific methods working on the `EyeDataDict` fields `left_pupil` and `right_pupil` etc
  - `GazeData` inherits from `GenericEyeData` and implements gaze-specific methods working on the `EyeDataDict` fields `left_x` and `right_y` etc
  - `EyeData` inherits from `PupilData` and `GazeData` and implements methods that work on both pupil and gaze data (e.g., the correction of the pupil by foreshortening)
  - the beauty of it is that they all work on the `self.data` field which is `EyeDataDict`, just assuming different fields are present
  - the plotting could mirror that approach: separate `GazePlotter` and `PupilPlotter` that are then merged in a `EyePlotter` class that inherits both
  - then it the interface would simply be `d.plotting.plot_xx()` for all three classes
  - what about the events? can they go into the `GenericEyeData` class?
- Problem with the scheme: When `EyeData` inherits from both classes, it is not clear whether a given function belongs to the gaze- or the pupil-data (this is even worse for the plotting functions)
  - one solution would be consistent naming of the methods but that kind of defeats the whole purpose of the inheritance
  - another solution would be to have `EyeData` not inherit but keep copies of the `PupilData` and `GazeData` objects and delegate the calls to them. But that is even worse.
  - consistent naming having "pupil" or "gaze" go first so that TAB-completion works:
    - scale()/unscale() - scale can be moved to generic, featuring an "eyes" and "variables" argument that specifies which timeseries should be scaled; 
    - pupil_lowpass_filter()
    - pupil_smooth_window()
    - pupil_downsample()  - or can this be for everything? - yeah, think so
    - pupil_estimate_baseline() 
    - pupil_estimate_response()
    - pupil_blinks_detect() - or can this be also for gaze? Will definitely need different algorithms. So should stick with different names
    - pupil_blinks_merge()
    - pupil_blinks_interpolate() - merge with Mahot function and make one an option for the other; remove plotting from Mahot function but add a function to the `PupilPlotter` that can visualize this
    - pupil_stat_per_event() - should this be based on the `get_intervals()` function instead? yeah! Then it can be `get_stat_per_interval()` and it can accept `eye` and `variable` as arguments to determine which timeseries should be used; but then it has to be implemented in both child classes; perhaps implement at the `GenericEyeData` level and then add a thin wrapper on the child classes that calls the generic function preventing it from using the wrong arguments
    - get_erpd() should also be reworked: based on the `get_intervals()` function and allow to select variables and eyes; has to be called something else like `EyeDataSegments` or so? Then get_segments() can return one segment-object per eye and variable for further processing
      - or I build another class that works like `EyeDataDict` but with the segments
- what about the `FakePupilData`? The difference is that it has additional timeseries corresponding to the simulated components of the signal; I can inherit from `PupilData` and implement those on top pretty easily; but I need also a `FakePupilPlotter` that can visualize those components
  - as long as I am clever about implementing the other methods, I can just send a "sim" variable into as 'variable' name so that I don't need to completely reimplement them


## Notes meeting 2024-06-18

- `pint` package: for unit conversion
- `logging` module for logging and potentially for the history?
- use `pytest` instead of `unittest`
- use a separate `PlotHandler` class, member of PupilData?
  - that would enable to use `d.plotting.plot_blinks()` and everything could be implemented in `plotting` which would be a `PlotHandler`
  - the `self` could be stored in the `obj.plotting` object in the constructor (`self.plotting.obj = self`)
  - or `obj.plotting.plot()` could use `super()` to get at the `self` of the `PupilData` object
- it seems best to drop the old `PupilData` class and implement everything as `EyeData` with the `EyeDataDict` for the data 
  - what happens if someone only has x/y data or only pupil data? 
  - would be nice if those would be somehow separate and only implement a subset of methods

## Notes from Radovan 2024-04-19

- Migrate from Travis CI to GitHub Actions to run tests
- Tests exist but are possible out of date.
- Is efficiency/ code optimization an issue at all?
- One could add type annotations
- Installation instructions: move the `pip install pypillometry` to the top.
  This is what most users will want.
- We could help publishing on conda-forge.
- Consider using logging instead of writing own logging functions for verbose prints.
- Code duplication (example: `def objective`)
- Possible discussion point: functional programming vs. object-oriented programming (many methods could be just functions and the class could be a Python dataclass)
  - pros of using methods: implicit type checking
  - pros of using functions: many methods are without side-effects but bundling them with attributes makes them stateful and maybe more difficult to test
- the history attribute is created in a surprising and hard to read way
- keephistory decorator is used for debugging/logging
- let's discuss what you mean with extend class structure and we can discuss strategies and pros and cons
- why is PupilArray needed instead of a standard data structure?
- lots of scrolling is needed to get an understanding of the class/functions 


