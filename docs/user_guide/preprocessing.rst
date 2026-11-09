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

Before and after preprocessing, :meth:`~EyeData.quality_report`
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
   EyeData.get_blinks
   EyeData.get_interpolated
   EyeData.blink_stats
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
:meth:`~EyeData.get_blinks` (a dict with one :class:`Intervals`
per channel, in ms); ``d.clear_layer("blink")`` removes them again.
:meth:`~EyeData.pupil.blinks_interpolate` replaces the pupil size during
blinks by a smooth curve (Mathôt, 2013). Interpolated samples are used by
all later steps (filters, segments, statistics); they keep the
``"interpolated"`` layer, so they can be left out with
``policy="strict"`` (see :doc:`missing_data`).

One blink, before and after: the pupil size drops and is then missing
while the eye is closed; the detected blink (red) is extended by a margin
on each side and replaced by a smooth curve (green):

.. plot::
   :include-source: False

   from pypillometry import u

   raw = pp.get_example_data("rlmw_002_short")
   d = raw | pp.pupil.blinks_detect() | pp.pupil.blinks_interpolate()
   t0, t1 = 21000, 23000
   sel = (raw.tx >= t0) & (raw.tx < t1)
   t = raw.tx[sel] / 1000

   fig, ax = plt.subplots(figsize=(8, 3.4))
   for a, b in d.get_interpolated("left", "pupil")["left_pupil"].intervals:
       if a < t1 and b > t0:
           ax.axvspan(a / 1000, b / 1000, color="C2", alpha=0.12, lw=0)
   for a, b in d.get_blinks("left", "pupil")["left_pupil"].intervals:
       if a < t1 and b > t0:
           ax.axvspan(a / 1000, b / 1000, color="C3", alpha=0.25, lw=0)
   ax.plot(t, raw.data["left_pupil"][sel], color="0.5", lw=2.5, label="recorded")
   ax.plot(t, d.data["left_pupil"][sel], color="C2", lw=1.2, label="after blinks_interpolate()")
   ax.plot([], [], color="C3", alpha=0.25, lw=8, label="blink (detected)")
   ax.plot([], [], color="C2", alpha=0.12, lw=8, label="interpolated (blink + margins)")
   ax.set_xlabel("time (s)")
   ax.set_ylabel("left pupil")
   ax.set_xlim(t[0], t[-1])
   ax.legend(fontsize=8, loc="lower left", frameon=False)
   fig.tight_layout()

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
:meth:`~EyeData.set_experimental_setup` and :doc:`gaze`).

Any stretch of data can be masked or interpolated from an
:class:`Intervals` object, e.g., periods you marked as bad by hand:

.. autosummary::
   :nosignatures:

   EyeData.mask_intervals
   EyeData.interpolate_intervals
   EyeData.mask_as_intervals
   EyeData.merge_masks


Filtering and downsampling
--------------------------

Pupil size changes slowly. A low-pass filter at about 4 Hz removes noise
without distorting the signal (Jackson & Sirois, 2009). Downsampling
reduces the amount of data, e.g., for modeling.

.. autosummary::
   :nosignatures:

   EyeData.pupil.lowpass_filter
   EyeData.pupil.smooth_window
   EyeData.downsample

The filters only use the usable samples: masked samples (e.g., blinks
that were not interpolated, or zero-coded blinks) are bridged linearly
for the filter and keep their values, so they do not leak into their
neighbours. Interpolating blinks before filtering gives a better bridge
than a straight line.

:meth:`~EyeData.downsample` averages the samples within each new
sampling interval (masked samples are left out), so filter before
downsampling to a rate close to the frequencies of interest.

A 4 Hz low-pass filter removes the fast noise:

.. plot::

   d = pp.get_example_data("rlmw_002_short") | pp.pupil.blinks_detect() | pp.pupil.blinks_interpolate()
   filtered = d.pupil.lowpass_filter(cutoff=4, inplace=False)

   sel = (d.tx >= 12000) & (d.tx < 17000)
   fig, ax = plt.subplots(figsize=(8, 3))
   ax.plot(d.tx[sel] / 1000, d["left_pupil"][sel], color="0.6", lw=1, label="before")
   ax.plot(d.tx[sel] / 1000, filtered["left_pupil"][sel], color="C0", lw=1.5,
           label="lowpass_filter(cutoff=4)")
   ax.set(xlabel="time (s)", ylabel="left pupil")
   ax.legend(frameon=False, fontsize=8)
   fig.tight_layout()

Downsampling to 50 Hz keeps the slow changes of the pupil:

.. plot::

   d = pp.get_example_data("rlmw_002_short") | pp.pupil.blinks_detect() | pp.pupil.blinks_interpolate()
   low = d.downsample(50, inplace=False)          # 500 Hz -> 50 Hz

   fig, ax = plt.subplots(figsize=(8, 3))
   for obj, style, label in [(d, dict(color="0.6", lw=1), "500 Hz"),
                             (low, dict(color="C1", marker="o", ms=3, lw=1), "downsample(50): 50 Hz")]:
       sel = (obj.tx >= 12000) & (obj.tx < 14000)
       ax.plot(obj.tx[sel] / 1000, obj["left_pupil"][sel], label=label, **style)
   ax.set(xlabel="time (s)", ylabel="left pupil")
   ax.legend(frameon=False, fontsize=8)
   fig.tight_layout()


Scaling
-------

Pupil size is measured in arbitrary units that depend on the eye tracker
and the setup. To compare participants, the data are often scaled:

.. autosummary::
   :nosignatures:

   EyeData.scale
   EyeData.unscale

:meth:`~EyeData.scale` subtracts a center and divides by a scale, both
computed from the usable samples of each channel. ``method`` chooses them:

.. code-block:: python

    d.scale("pupil")                       # z-scores (mean and SD)
    d.scale("pupil", method="robust")      # median and MAD: units of the MAD
    d.scale("pupil", method="minmax")      # between 0 and 1
    d.scale("pupil", method="percent")     # % change from the mean

The robust scaling uses the raw median absolute deviation (not multiplied
by 1.4826), so artefacts that were not removed have little influence. The
values that were used are stored with the object (``d.params["scale"]``),
and :meth:`~EyeData.unscale` undoes the scaling. For a percent change
relative to a baseline before each event, see
``seg.pupil.baseline_correct(window, method="percent")``
(:doc:`events_intervals`).

All methods keep the shape of the signal and change only its units (the
y-axes):

.. plot::

   d = pp.get_example_data("rlmw_002_short") | pp.pupil.blinks_detect() | pp.pupil.blinks_interpolate()

   fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharex=True)
   for ax, method in zip(axes, ["z", "robust", "minmax", "percent"]):
       s = d.scale("pupil", method=method, inplace=False)
       ax.plot(s.tx / 1000, s["left_pupil"], lw=0.7)
       ax.set_title(f'method="{method}"', fontsize=10)
       ax.set_xlabel("time (s)")
   axes[0].set_ylabel("scaled left pupil")
   fig.tight_layout()


Combining the eyes
------------------

With binocular data, the two eyes can be combined into one signal that is
less noisy and has fewer gaps:

.. autosummary::
   :nosignatures:

   EyeData.pupil.merge_eyes
   EyeData.merge_eyes

:meth:`~EyeData.pupil.merge_eyes` corrects for a (slowly changing)
difference in pupil size between the eyes and fills samples missing in one
eye from the other (recorded samples come before interpolated ones). The
result is stored as ``mean_pupil`` (eye ``"mean"``; another name with
``name=``); it keeps the layers that both eyes have in common, e.g.,
``"blink"`` and ``"interpolated"`` where both eyes blinked.
:meth:`~EyeData.merge_eyes` simply averages any variable. See
:doc:`/notebooks/eyes_and_variables`.

Where the left eye is masked during a blink, the merged pupil continues
from the right eye, shifted by the difference between the eyes, so it
has no jump and no gap:

.. plot::

   d = pp.get_example_data("rlmw_002_short") | pp.pupil.blinks_detect()
   d.pupil.merge_eyes()                     # adds "mean_pupil"

   sel = (d.tx >= 9800) & (d.tx < 12200)
   t = d.tx[sel] / 1000
   fig, ax = plt.subplots(figsize=(8, 3))
   ax.plot(t, d["left_pupil"][sel], color="C0", lw=1, label="left (masked: blink)")
   ax.plot(t, d["right_pupil"][sel], color="C1", lw=1, label="right")
   ax.plot(t, d["mean_pupil"][sel], color="k", lw=1.5, ls="--", label="mean_pupil")
   ax.set(xlabel="time (s)", ylabel="pupil")
   ax.legend(frameon=False, fontsize=8, loc="lower left")
   fig.tight_layout()


Selecting and cleaning up data
------------------------------

.. autosummary::
   :nosignatures:

   EyeData.crop
   EyeData.remove_intervals
   EyeData.reset_time
   EyeData.drop_data
   EyeData.drop_original
   EyeData.fill_time_discontinuities
   EyeData.copy

:meth:`~EyeData.crop` cuts out a time range, e.g., one block
of the experiment: ``d.crop(3 * u.min, 5 * u.min)``.
:meth:`~EyeData.remove_intervals` removes periods
in the middle, e.g., breaks, and joins the rest: the time vector and the
events after a removed period are shifted back, so that there are no gaps.
:meth:`~EyeData.reset_time` makes the time vector start at zero
(or at any other time).
:meth:`~EyeData.drop_data` removes eyes or variables, e.g.,
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
