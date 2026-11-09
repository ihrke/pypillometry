.. currentmodule:: pypillometry

Gaze data and the experimental setup
====================================

Besides pupil size, an :class:`EyeData` object can hold the gaze position
of each eye: the channels ``left_x``, ``left_y``, ``right_x`` and
``right_y``. This page shows what is stored, how to describe the screen
and the camera, and the tools for gaze data. The methods for gaze data are
in the namespace ``d.gaze``; the generic methods (cropping, masks,
segments, ...) work on gaze channels as on any other channel.

The examples use the short example dataset, which has gaze and pupil data
of both eyes:

.. code-block:: python

    import numpy as np
    import pypillometry as pp
    from pypillometry import u
    d = pp.get_example_data("rlmw_002_short")


What is stored
--------------

The gaze position is stored in **screen pixels**, as the eye tracker
reports it: x from left to right, y from top to bottom (the origin is the
upper left corner of the screen; this is the EyeLink convention). Samples
in which the tracker lost the eye are NaN. The gaze plots draw the screen
as it is seen, so y increases downward (see :doc:`plotting`).

.. plot::
   :include-source: False

   import matplotlib.patches as patches

   W, H = 1280, 1024
   fig, ax = plt.subplots(figsize=(6.4, 5.2))
   ax.add_patch(patches.Rectangle((0, 0), W, H, fill=False, lw=2))
   # pixel coordinates: origin top left, x to the right, y down
   arrow = dict(arrowstyle="->", lw=2, color="C0")
   ax.annotate("", xy=(330, 0), xytext=(0, 0), arrowprops=arrow)
   ax.annotate("", xy=(0, 330), xytext=(0, 0), arrowprops=arrow)
   ax.text(345, 30, "x (px)", color="C0", va="center", fontsize=11)
   ax.text(25, 360, "y (px)", color="C0", ha="left", va="top", fontsize=11)
   for (x, y), ha, va in [((0, 0), "left", "bottom"), ((W, 0), "right", "bottom"),
                          ((0, H), "left", "top"), ((W, H), "right", "top")]:
       ax.text(x, y + (-12 if va == "bottom" else 12), f"({x}, {y})", ha=ha, va=va,
               color="C0", fontsize=10)
   # mm and degrees (ROIs, setup): origin at the center, y up
   grey = dict(arrowstyle="->", lw=1.5, color="0.45")
   ax.annotate("", xy=(W / 2 + 230, H / 2), xytext=(W / 2, H / 2), arrowprops=grey)
   ax.annotate("", xy=(W / 2, H / 2 - 230), xytext=(W / 2, H / 2), arrowprops=grey)
   ax.plot(W / 2, H / 2, "o", color="0.45")
   ax.text(W / 2 + 15, H / 2 + 25, f"center ({W // 2}, {H // 2}) px\n= (0, 0) mm or deg",
           color="0.45", va="top", fontsize=10)
   ax.text(W / 2 + 240, H / 2, "x (mm, deg)", color="0.45", va="center", fontsize=10)
   ax.text(W / 2, H / 2 - 245, "y (mm, deg)", color="0.45", ha="center", fontsize=10)
   ax.set_xlim(-60, W + 60)
   ax.set_ylim(H + 70, -70)   # y downward, as on the screen
   ax.set_aspect("equal")
   ax.axis("off")
   ax.set_title("Screen coordinates (here 1280 x 1024 px)")

Positions in mm or degrees (e.g., for regions of interest or in the
experimental setup) are relative to the screen center, with y upward; they
are converted to pixels with the setup (see below).

Blinks are detected on the pupil only (see :doc:`preprocessing`), so the
gaze channels of a blink are not masked unless you copy the blinks to them
(see below).


The experimental setup
----------------------

Many gaze tools need to know the geometry of the experiment: the screen
size in pixels and in mm, the distance between the eyes and the screen
and, for some corrections, the position of the camera. It is stored as an
:class:`ExperimentalSetup` in ``d.experimental_setup`` and saved with the
object. :meth:`~EyeData.set_experimental_setup` sets (or updates) it;
lengths and angles can be given with units:

.. code-block:: python

    d.set_experimental_setup(
        screen_resolution=(1280, 1024),                # pixels
        physical_screen_size=(30 * u.cm, 20 * u.cm),   # width, height
        eye_to_screen_center=60 * u.cm,                # viewing distance
    )
    d.experimental_setup                               # summary

Which parts are needed depends on what you do:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - parameters
     - needed for
   * - ``screen_resolution``
     - :meth:`~EyeData.gaze.mask_offscreen`, the screen in the gaze plots
   * - ``physical_screen_size``, ``eye_to_screen_center``
     - conversion to mm and visual angle, regions of interest in degrees
   * - ``screen_offset``, ``screen_pitch``, ``screen_yaw``, ``ipd``
     - a screen that is not centered or not perpendicular to the line of
       sight, the position of each eye
   * - ``camera_offset`` or ``camera_spherical``
     - the pupil foreshortening correction (see below)

:meth:`EyeData.from_eyelink` sets the screen resolution from the EDF
file; the other values are not in the file and have to be added. The
example datasets come with the screen size and the viewing distance (no
camera position).


Converting coordinates
----------------------

The setup converts between pixels, mm on the screen and visual angle
(degrees from the screen center, along x and y):

.. code-block:: python

    s = d.experimental_setup
    s.pixels_to_degrees(740, 512)      # (2.24, 0.0): 100 px right of the center
    s.degrees_to_pixels("1 deg", 0)    # (684.7, 512.0)
    s.pixels_to_mm(740, 512)           # (23.4, 0.0) mm from the screen center
    s.mm_to_degrees(10, 0)             # (0.95, 0.0)

All of them take numbers or arrays, e.g., ``s.pixels_to_degrees(d.data["left_x"],
d.data["left_y"])`` for a whole channel. See :doc:`units` for how to give
values with units.


Masking gaze artifacts
----------------------

.. autosummary::
   :nosignatures:

   EyeData.gaze.mask_offscreen
   EyeData.gaze.mask_divergences

- :meth:`~EyeData.gaze.mask_offscreen` masks the samples outside the
  screen (``"offscreen"`` layer of the x and y channels of each eye).
- :meth:`~EyeData.gaze.mask_divergences` masks the samples where the two
  eyes look at very different points (``"divergence"`` layer of all four
  gaze channels). The threshold is a quantile of the distance between the
  eyes (default ``threshold=0.99``, ``thr_type="percentile"``: the 1% most
  divergent samples) or a distance in pixels (``thr_type="pixel"``).
  ``store_as="dist"`` also stores the distance as a channel ``both_dist``
  (eye ``"both"``).

.. code-block:: python

    d.gaze.mask_offscreen()                               # 0.2% per eye here
    d.gaze.mask_divergences(threshold=100, thr_type="pixel", store_as="dist")
    d.get_layer_intervals("offscreen", variables="x")     # the periods, in ms

Like the blink detector, both replace their own layer when they run again,
so trying other thresholds does not add up masks.
:meth:`~EyeData.gaze.mask_divergences` needs the gaze of both eyes.

**Masks of the gaze and the pupil are separate.** Off-screen and divergent
samples are masked on the gaze channels only, blinks on the pupil only.
To use one for the other, copy the periods with
:meth:`~EyeData.mask_intervals` (into the ``"user"`` layer):

.. code-block:: python

    # mask the gaze during blinks
    bl = d.pupil.blinks_detect().get_blinks(variables="pupil")
    d.mask_intervals({"left_x": bl["left_pupil"], "left_y": bl["left_pupil"],
                      "right_x": bl["right_pupil"], "right_y": bl["right_pupil"]})

The opposite direction (the pupil where the gaze is off the screen) is
shown in :doc:`missing_data`, which also explains how the layers are used
by the analyses.

Combining the eyes: :meth:`~EyeData.merge_eyes` with
``variables=["x", "y"]`` averages the gaze of both eyes into ``mean_x``
and ``mean_y`` (using the samples that are usable in each eye).


Plotting gaze data
------------------

.. autosummary::
   :nosignatures:

   EyeData.plot.plot_heatmap
   EyeData.plot.plot_scanpath
   EyeData.plot.plot_timeseries
   EyeData.plot.plot_calibration

:meth:`~EyeData.plot.plot_heatmap` shows where the eyes looked (a 2D
histogram over the screen), :meth:`~EyeData.plot.plot_scanpath` the path
of the gaze over a short time, colored by time, and
``d.plot.plot_timeseries(variables=["x", "y"])`` the coordinates over
time. All of them draw the screen if the setup is known. See
:doc:`plotting` and the tutorial :doc:`/notebooks/plotting`.


Calibration
-----------

:meth:`EyeData.from_eyelink` reads the calibration and validation results
of the EDF file into ``d.calibration`` (one :class:`SpatialCalibration` per
eye: the target points, the measured points and their errors).
:meth:`~EyeData.plot.plot_calibration` shows them, and the quality report
lists the calibration errors (see :doc:`quality`). Here, for a recording with
a 9-point validation (black crosses: targets, blue plus signs: measured
gaze; the background shows the error between the points):

.. plot::

   d = pp.EyeData.from_eyelink("../data/test.edf")   # data folder of the repository
   d.calibration          # {'left': <SpatialCalibration: left eye, HV9, n=9, ...>, ...}
   fig, axes = d.plot.plot_calibration(figsize=(6, 5))


Regions of interest
-------------------

:class:`CircularROI` and :class:`RectangularROI` describe regions of the
screen, in pixels or, with the setup, in mm or degrees. They test which
samples fall into them and can be drawn into the gaze plots (``rois=``):

.. code-block:: python

    roi = pp.CircularROI.from_degrees((0, 0), 2, d.experimental_setup)  # 2 deg around the center
    inside = roi.contains(np.column_stack([d.data["left_x"], d.data["left_y"]]))
    d.plot.plot_heatmap(rois=[roi])

For event-related analyses, ``seg.gaze`` of the segments
(:meth:`~EyeData.get_segments`) uses ROIs: the share of trials in which
the gaze is in an ROI over time, and the dwell time and first entry per
trial. See :doc:`rois` and the tutorial :doc:`/notebooks/rois`.


Pupil foreshortening
--------------------

The pupil looks smaller when the eye looks away from the camera. With the
camera position in the setup, :meth:`d.pupil.fit_foreshortening()
<pypillometry.eyedata.pupil.PupilNamespace.fit_foreshortening>` estimates
the camera angles from the data and :meth:`d.pupil.correct_foreshortening()
<pypillometry.eyedata.pupil.PupilNamespace.correct_foreshortening>`
corrects the pupil size for the gaze position. This is experimental as
well; see :doc:`/experimental/foreshortening`.


What is not included
--------------------

:mod:`pypillometry` does not detect fixations or saccades, and the
saccade and fixation events of EyeLink files are not read. If you need
them, read them with the ``eyelinkio`` package (or your eye tracker's
software) and add them as events (see :doc:`events_intervals`).
