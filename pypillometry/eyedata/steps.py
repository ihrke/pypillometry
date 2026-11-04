"""
steps.py
========

The pipeline steps available at the top level of the package:

- ``pp.pupil`` and ``pp.gaze``: the pupil and gaze methods
  (e.g., ``pp.pupil.blinks_detect()``),
- ``pp.crop``, ``pp.downsample``, ...: the generic methods.

Each step factory has the signature and docstring of the method it calls
(e.g., ``pp.crop`` calls :meth:`~pypillometry.GenericEyeData.crop`) and
returns a :class:`~pypillometry.Step`; see :mod:`pypillometry.eyedata.pipes`.
"""

from .generic import GenericEyeData
from .pipes import generic_step
from .pupil import PupilNamespace
from .gaze import GazeNamespace

__all__ = ["pupil", "gaze",
           "crop", "downsample", "scale", "unscale", "merge_eyes", "reset_time",
           "mask_intervals", "interpolate_intervals", "exclude_intervals",
           "keep_intervals", "remove_intervals", "merge_masks", "clear_layer",
           "set_blinks", "set_events", "add_events", "set_experimental_setup",
           "drop_data", "drop_original", "fill_time_discontinuities"]

#: pipeline steps for pupil data (``d | pp.pupil.blinks_detect()``)
pupil = PupilNamespace()
#: pipeline steps for gaze data (``d | pp.gaze.mask_offscreen()``)
gaze = GazeNamespace()

crop = generic_step(GenericEyeData.crop)
downsample = generic_step(GenericEyeData.downsample)
scale = generic_step(GenericEyeData.scale)
unscale = generic_step(GenericEyeData.unscale)
merge_eyes = generic_step(GenericEyeData.merge_eyes)
reset_time = generic_step(GenericEyeData.reset_time)
mask_intervals = generic_step(GenericEyeData.mask_intervals)
interpolate_intervals = generic_step(GenericEyeData.interpolate_intervals)
exclude_intervals = generic_step(GenericEyeData.exclude_intervals)
keep_intervals = generic_step(GenericEyeData.keep_intervals)
remove_intervals = generic_step(GenericEyeData.remove_intervals)
merge_masks = generic_step(GenericEyeData.merge_masks)
clear_layer = generic_step(GenericEyeData.clear_layer)
set_blinks = generic_step(GenericEyeData.set_blinks)
set_events = generic_step(GenericEyeData.set_events)
add_events = generic_step(GenericEyeData.add_events)
set_experimental_setup = generic_step(GenericEyeData.set_experimental_setup)
drop_data = generic_step(GenericEyeData.drop_data)
drop_original = generic_step(GenericEyeData.drop_original)
fill_time_discontinuities = generic_step(GenericEyeData.fill_time_discontinuities)
