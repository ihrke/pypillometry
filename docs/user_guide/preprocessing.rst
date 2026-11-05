.. currentmodule:: pypillometry

Preprocessing
=============

Raw pupil data usually contain blinks, other artifacts and high-frequency
noise. This page lists the tools for dealing with them. All methods
process all eyes by default (use ``eyes=`` to select) and can be combined
into a pipeline with ``|`` (see :doc:`data_model`).

Visual checks are an important part of preprocessing. Look at the data
before and after each step (see :doc:`plotting`). To tune parameters, use
:func:`tweak` (see :doc:`interactive`): it shows the result while you
change them.


Data quality
------------

Before and after preprocessing, :meth:`~GenericEyeData.quality_report`
summarizes the quality of a dataset (missing data, noise, blinks, artifacts,
calibration) and labels each measure "good", "ok", "poor" or "bad". See
:doc:`quality`.

.. code-block:: python

    d.quality_report()


Blinks
------

During a blink, the eye tracker loses the pupil. Pupil size usually drops
to zero or is missing, and the samples just before and after the blink are
distorted as the eyelid covers the pupil.

.. autosummary::
   :nosignatures:

   EyeData.pupil.blinks_detect
   EyeData.pupil.blinks_merge_close
   EyeData.pupil.blinks_interpolate
   GenericEyeData.get_blinks
   GenericEyeData.get_interpolated_blinks
   GenericEyeData.blink_stats
   EyeData.plot.plot_blinks

A typical sequence is

.. code-block:: python

    d = d | (pp.pupil.blinks_detect()
             | pp.pupil.blinks_merge_close(distance=100 * u.ms)
             | pp.pupil.blinks_interpolate())

:meth:`~EyeData.pupil.blinks_detect` finds missing samples and sharp
drops in the pupil's velocity profile. The detected blinks are stored in the
object (the ``"blink"`` layer, which masks them) and the object is
returned (``d.pupil.blinks_detect()`` works in place). Check them with
:meth:`~GenericEyeData.get_blinks` (a dict with one :class:`Intervals`
per channel, in ms); ``d.clear_layer("blink")`` removes them again.
:meth:`~EyeData.pupil.blinks_interpolate` replaces the pupil size during
blinks by a smooth curve (Mathôt, 2013). Interpolated samples are used by
all later steps (filters, segments, statistics); they keep the
``"interpolated"`` layer, so they can be left out with
``policy="strict"`` (see :doc:`missing_data`).

Blinks differ a lot between participants, and checking the result for
each participant is worth the effort. The tutorial :doc:`/notebooks/blinks`
shows how.


Other artifacts
---------------

For gaze data, samples off the screen and samples where the two eyes point
to very different places are usually artifacts:

.. autosummary::
   :nosignatures:

   EyeData.gaze.mask_offscreen
   EyeData.gaze.mask_divergences

:meth:`~EyeData.gaze.mask_offscreen` needs the screen size (see
:meth:`~EyeData.set_experimental_setup`).

Any stretch of data can be masked or interpolated from an
:class:`Intervals` object, e.g., periods you marked as bad by hand:

.. autosummary::
   :nosignatures:

   GenericEyeData.mask_intervals
   GenericEyeData.interpolate_intervals
   GenericEyeData.mask_as_intervals
   GenericEyeData.merge_masks


Filtering and downsampling
--------------------------

Pupil size changes slowly. A low-pass filter at about 4 Hz removes noise
without distorting the signal (Jackson & Sirois, 2009). Downsampling
reduces the amount of data, e.g., for modeling.

.. autosummary::
   :nosignatures:

   EyeData.pupil.lowpass_filter
   EyeData.pupil.smooth_window
   GenericEyeData.downsample

The filters only use the usable samples: masked samples (e.g., blinks
that were not interpolated, or zero-coded blinks) are bridged linearly
for the filter and keep their values, so they do not leak into their
neighbours. Interpolating blinks before filtering gives a better bridge
than a straight line.

:meth:`~GenericEyeData.downsample` averages the samples within each new
sampling interval (masked samples are left out), so filter before
downsampling to a rate close to the frequencies of interest.


Scaling
-------

Pupil size is measured in arbitrary units that depend on the eye tracker
and the setup. To compare participants, the data are often z-scored:

.. autosummary::
   :nosignatures:

   GenericEyeData.scale
   GenericEyeData.unscale

The mean and standard deviation that were used are stored with the
object, and :meth:`~GenericEyeData.unscale` undoes the scaling.


Combining the eyes
------------------

With binocular data, the two eyes can be combined into one signal that is
less noisy and has fewer gaps:

.. autosummary::
   :nosignatures:

   EyeData.pupil.merge_eyes
   GenericEyeData.merge_eyes

:meth:`~EyeData.pupil.merge_eyes` corrects for a (slowly changing)
difference in pupil size between the eyes and fills samples missing in one
eye from the other (recorded samples come before interpolated ones). The
result is stored as ``mean_pupil`` (eye ``"mean"``; another name with
``name=``); it keeps the layers that both eyes have in common, e.g.,
``"blink"`` and ``"interpolated"`` where both eyes blinked.
:meth:`~GenericEyeData.merge_eyes` simply averages any variable. See
:doc:`/notebooks/eyes_and_variables`.


Selecting and cleaning up data
------------------------------

.. autosummary::
   :nosignatures:

   GenericEyeData.crop
   GenericEyeData.remove_intervals
   GenericEyeData.reset_time
   GenericEyeData.drop_data
   GenericEyeData.drop_original
   GenericEyeData.fill_time_discontinuities
   GenericEyeData.copy

:meth:`~GenericEyeData.crop` cuts out a time range, e.g., one block
of the experiment: ``d.crop(3 * u.min, 5 * u.min)`` (``sub_slice`` is a
deprecated alias). :meth:`~GenericEyeData.remove_intervals` removes periods
in the middle, e.g., breaks, and joins the rest: the time vector and the
events after a removed period are shifted back, so that there are no gaps.
:meth:`~GenericEyeData.reset_time` makes the time vector start at zero
(or at any other time).
:meth:`~GenericEyeData.drop_data` removes eyes or variables, e.g.,
``d.drop_data(variables=["x", "y"])`` keeps only the pupil data; masks,
blinks, events, the experimental setup and all other data are kept. Like
all transforming methods, it works in place unless ``inplace=False`` is
given (or set as the default, see :doc:`settings`).


References
----------

- Jackson, I., & Sirois, S. (2009). Infant cognition: going full factorial
  with pupil dilation. *Developmental Science*, 12(4), 670–679.
  https://doi.org/10.1111/j.1467-7687.2008.00805.x
- Mathôt, S. (2013). A simple way to reconstruct pupil size during eye
  blinks. figshare. https://figshare.com/articles/A_simple_way_to_reconstruct_pupil_size_during_eye_blinks/688001
