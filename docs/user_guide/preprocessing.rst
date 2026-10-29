.. currentmodule:: pypillometry

Preprocessing
=============

Raw pupil data usually contain blinks, other artifacts and high-frequency
noise. This page lists the tools for dealing with them. All methods
process all eyes by default (use ``eyes=`` to select) and can be chained
into a pipeline (see :doc:`data_model`).

Visual checks are an important part of preprocessing. Look at the data
before and after each step (see :doc:`plotting`). To tune parameters, use
:func:`tweak` (see :doc:`interactive`): it shows the result while you
change them.


Data quality
------------

:meth:`~GenericEyeData.quality_report` summarizes the quality of a
dataset: missing data, dropouts and noise of the pupil, missing and
off-screen samples and precision of the gaze, and agreement between the
eyes. These measures need no preprocessing. After preprocessing, blinks and
masked/interpolated data are reported as well, and calibration errors if
they are available (e.g., from an Eyelink file).

Each measure is labeled "good", "ok", "poor" or "bad", and the report has
an overall label (the worst one). The thresholds can be changed with the
``thresholds`` argument (see
:data:`pypillometry.quality.DEFAULT_QUALITY_THRESHOLDS`).

.. code-block:: python

    report = d.quality_report()
    print(report)            # text (colored labels in a terminal)
    print(report.to_text(descriptions=True))  # explain each measure
    report                   # HTML table in a notebook (hover for descriptions)
    report.to_dict(flat=True)  # e.g., for a DataFrame over participants
    d.quality_report(thresholds={"pupil.missing_pct": (2, 10, 20)})


Blinks
------

During a blink, the eye tracker loses the pupil. Pupil size usually drops
to zero or is missing, and the samples just before and after the blink are
distorted as the eyelid covers the pupil.

.. autosummary::
   :nosignatures:

   PupilData.pupil_blinks_detect
   GenericEyeData.blinks_merge_close
   PupilData.pupil_blinks_interpolate
   GenericEyeData.get_blinks
   GenericEyeData.get_interpolated_blinks
   GenericEyeData.blink_stats
   PupilPlotter.plot_blinks

A typical sequence is

.. code-block:: python

    d.pupil_blinks_detect(apply_mask=True)\
     .blinks_merge_close(distance=100 * u.ms, apply_mask=True)\
     .pupil_blinks_interpolate()

:meth:`~PupilData.pupil_blinks_detect` finds missing samples and sharp
drops in the pupil's velocity profile. The detected blinks are stored in the
object. With ``apply_mask=False`` (the default), they are also returned as
:class:`Intervals` so that you can check them first.
:meth:`~PupilData.pupil_blinks_interpolate` replaces the pupil size during
blinks by a smooth curve (Mathôt, 2013). Interpolated samples remain masked.

Blinks differ a lot between participants, and checking the result for
each participant is worth the effort. The tutorial :doc:`/notebooks/blinks`
shows how.


Other artifacts
---------------

For gaze data, samples off the screen and samples where the two eyes point
to very different places are usually artifacts:

.. autosummary::
   :nosignatures:

   GazeData.mask_offscreen_coords
   GazeData.mask_eye_divergences

:meth:`~GazeData.mask_offscreen_coords` needs the screen size (see
:meth:`~GazeData.set_experimental_setup`).

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

   PupilData.pupil_lowpass_filter
   PupilData.pupil_smooth_window
   GenericEyeData.downsample

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

   PupilData.pupil_merge_eyes
   GenericEyeData.merge_eyes

:meth:`~PupilData.pupil_merge_eyes` corrects for a (slowly changing)
difference in pupil size between the eyes and fills samples missing in one
eye from the other. The result is stored as ``merged_pupil``.
:meth:`~GenericEyeData.merge_eyes` simply averages any variable. See
:doc:`/notebooks/eyes_and_variables`.


Selecting and cleaning up data
------------------------------

.. autosummary::
   :nosignatures:

   GenericEyeData.sub_slice
   GenericEyeData.reset_time
   GenericEyeData.drop_data
   GenericEyeData.drop_original
   GenericEyeData.fill_time_discontinuities
   GenericEyeData.copy

:meth:`~GenericEyeData.sub_slice` cuts out a time range, e.g., one block
of the experiment: ``d.sub_slice(3 * u.min, 5 * u.min)``.
:meth:`~GenericEyeData.reset_time` makes the time vector start at zero
(or at any other time).


References
----------

- Jackson, I., & Sirois, S. (2009). Infant cognition: going full factorial
  with pupil dilation. *Developmental Science*, 12(4), 670–679.
  https://doi.org/10.1111/j.1467-7687.2008.00805.x
- Mathôt, S. (2013). A simple way to reconstruct pupil size during eye
  blinks. figshare. https://figshare.com/articles/A_simple_way_to_reconstruct_pupil_size_during_eye_blinks/688001
