"""
steps.py
========

The pipeline steps available at the top level of the package:

- ``pp.pupil`` and ``pp.gaze``: the pupil and gaze methods
  (e.g., ``pp.pupil.blinks_detect()``),
- ``pp.scene``: the methods for head-mounted eye trackers
  (:class:`~pypillometry.HeadMountedEyeData`),
- ``pp.crop``, ``pp.downsample``, ...: the generic methods.

Each step factory has the signature and docstring of the method it calls
(e.g., ``pp.crop`` calls :meth:`~pypillometry.EyeData.crop`) and
returns a :class:`~pypillometry.Step`; see :mod:`pypillometry.eyedata.pipes`.
"""

from .eyedata import EyeData
from .pipes import generic_step
from .pupil import PupilNamespace
from .gaze import GazeNamespace
from .head_mounted import SceneNamespace

__all__ = ["pupil", "gaze", "scene",
           "crop", "downsample", "scale", "unscale", "merge_eyes", "reset_time",
           "mask_intervals", "interpolate_intervals", "exclude_intervals",
           "keep_intervals", "remove_intervals", "merge_masks", "clear_layer",
           "set_blinks", "set_events", "add_events", "set_experimental_setup",
           "drop_data", "drop_original", "fill_time_discontinuities"]

#: pipeline steps for pupil data (``d | pp.pupil.blinks_detect()``)
pupil = PupilNamespace()
#: pipeline steps for gaze data (``d | pp.gaze.mask_offscreen()``)
gaze = GazeNamespace()
#: pipeline steps for head-mounted eye trackers (``d | pp.scene.luminance()``)
scene = SceneNamespace()

crop = generic_step(EyeData.crop)
downsample = generic_step(EyeData.downsample)
scale = generic_step(EyeData.scale)
unscale = generic_step(EyeData.unscale)
merge_eyes = generic_step(EyeData.merge_eyes)
reset_time = generic_step(EyeData.reset_time)
mask_intervals = generic_step(EyeData.mask_intervals)
interpolate_intervals = generic_step(EyeData.interpolate_intervals)
exclude_intervals = generic_step(EyeData.exclude_intervals)
keep_intervals = generic_step(EyeData.keep_intervals)
remove_intervals = generic_step(EyeData.remove_intervals)
merge_masks = generic_step(EyeData.merge_masks)
clear_layer = generic_step(EyeData.clear_layer)
set_blinks = generic_step(EyeData.set_blinks)
set_events = generic_step(EyeData.set_events)
add_events = generic_step(EyeData.add_events)
set_experimental_setup = generic_step(EyeData.set_experimental_setup)
drop_data = generic_step(EyeData.drop_data)
drop_original = generic_step(EyeData.drop_original)
fill_time_discontinuities = generic_step(EyeData.fill_time_discontinuities)
