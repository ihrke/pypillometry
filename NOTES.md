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
- multiple inheritance scheme:
  - `GenericEyeData` implements bookkeeping, history, etc like now
  - `PupilData` inherits from `GenericEyeData` and implements pupil-specific methods working on the `EyeDataDict` fields `left_pupil` and `right_pupil` etc
  - `GazeData` inherits from `GenericEyeData` and implements gaze-specific methods working on the `EyeDataDict` fields `left_x` and `right_y` etc
  - `EyeData` inherits from `PupilData` and `GazeData` and implements methods that work on both pupil and gaze data (e.g., the correction of the pupil by foreshortening)
  - the beauty of it is that they all work on the `self.data` field which is `EyeDataDict`, just assuming different fields are present
  - the plotting could mirror that approach: separate `GazePlotter` and `PupilPlotter` that are then merged in a `EyePlotter` class that inherits both
  - then it the interface would simply be `d.plotting.plot_xx()` for all three classes
  - what about the events? can they go into the `GenericEyeData` class?

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


