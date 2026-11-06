.. currentmodule:: pypillometry

Missing data
============

Eye-tracking data have gaps and bad samples for many reasons: the eye
tracker loses the pupil during a blink, the gaze leaves the screen, the
two eyes point to different places, or a part of the recording is not part
of the analysis. :mod:`pypillometry` keeps the *reason* for each bad
sample, and a *mask policy* decides which samples the analyses use. This
page explains how that works and how each operation treats bad samples.
The tutorial :doc:`/notebooks/missing_data` goes through the same steps
with a real recording.


Where the information is stored
-------------------------------

There are three places:

- **NaN in the values**: samples that the eye tracker did not record.
  Eyelink codes missing pupil samples as 0; they are converted to NaN when
  an EDF file is read (see :doc:`loading_data`). If you create an object
  from your own arrays, convert such codes to NaN yourself, or let
  :meth:`~EyeData.pupil.blinks_detect` find them (its ``blink_val``
  argument).
- **Mask layers of a channel**: each channel (e.g., ``left_pupil``) has a
  set of named layers. A layer marks the samples that one method found to
  be bad, without changing the values.
- **Recording-wide layers**: the ``"excluded"`` layer marks periods that
  are not part of the analysis, for all channels at once (also channels
  that are added later).

The standard layers:

.. list-table::
   :header-rows: 1
   :widths: 14 22 34 15 15

   * - layer
     - channels
     - set by
     - ``"default"``
     - ``"strict"``
   * - NaN (values)
     - any
     - the eye tracker; :meth:`~EyeData.fill_time_discontinuities`
     - masked
     - masked
   * - ``"blink"``
     - pupil
     - :meth:`~EyeData.pupil.blinks_detect`,
       :meth:`~EyeData.pupil.blinks_merge_close`,
       :meth:`~EyeData.set_blinks`
     - masked
     - masked
   * - ``"offscreen"``
     - x and y of the checked eyes
     - :meth:`~EyeData.gaze.mask_offscreen`
     - masked
     - masked
   * - ``"divergence"``
     - x and y of both eyes
     - :meth:`~EyeData.gaze.mask_divergences`
     - masked
     - masked
   * - ``"user"``
     - the selected channels
     - :meth:`~EyeData.mask_intervals`,
       :meth:`~EyeData.merge_masks`, assigning a masked array
       (``d["left_pupil"] = np.ma.masked_array(...)``),
       :meth:`~EyeData.merge_eyes` (where no other reason is
       common to the eyes)
     - masked
     - masked
   * - ``"interpolated"``
     - the interpolated channel
     - :meth:`~EyeData.pupil.blinks_interpolate`,
       :meth:`~EyeData.interpolate_intervals`
     - **used**
     - masked
   * - ``"excluded"``
     - all (recording-wide)
     - :meth:`~EyeData.exclude_intervals`,
       :meth:`~EyeData.keep_intervals`
     - masked
     - masked

A layer only exists on the channels the method checks. Blinks are
detected on the pupil, so the gaze channels have no ``"blink"`` layer
(they are usually NaN during blinks anyway). Off-screen gaze does not mask
the pupil: whether the pupil size is still valid then depends on the
setup. To mask it, copy the layer (see `Common tasks`_ below).

``d.data.layer_names`` lists the layers that have been set. A channel can
have up to 16 layers, so there is room for your own (e.g.,
``d.data.set_layer("left_pupil", "saccade", mask)``). Your own layers are
not part of the predefined policies below: they mask samples only where
you name them in a policy (e.g., ``policy=["saccade"]``).


Mask policies
-------------

A *mask policy* (:class:`MaskPolicy`) decides which samples are
*unusable*. All methods that read the data take a ``policy=`` argument
with one of the predefined policies:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - policy
     - unusable samples
   * - ``"default"``
     - NaN, ``"blink"``, ``"offscreen"``, ``"divergence"``, ``"user"``,
       ``"excluded"``; but samples in the ``"interpolated"`` layer are used
   * - ``"strict"``
     - the same, and the interpolated samples
   * - ``"raw"``
     - nothing, not even NaN

The exact rule: a sample is unusable if its value is NaN, or if one of
the policy's layers is set for it and none of its ``unless`` layers is.
The default policy has ``unless="interpolated"``, which applies to all
channel layers, but not to the recording-wide ``"excluded"`` layer. Two
consequences:

- **An interpolated sample is always usable under the default policy**,
  even if another layer of the channel is set, and no matter whether that
  layer was set before or after the interpolation. Its value was
  reconstructed, so the reason for masking the recorded value no longer
  applies.
- Excluded periods are always unusable, also where the data were
  interpolated.

A policy can also be a list of layers (each of them makes a sample
unusable, as does NaN), or a :class:`MaskPolicy` of your own:

.. code-block:: python

    d.mask("left_pupil")                          # default policy
    d.mask("left_pupil", policy="strict")         # interpolated samples too
    d.mask("left_pupil", policy=["blink"])        # only blinks (and NaN)
    my_policy = pp.MaskPolicy(["blink", "user", "excluded"], unless="interpolated")


Reading masks and values
------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - call
     - result
   * - ``d.mask(eye, var, policy=)``
     - boolean array, True where unusable
   * - ``d.values(eye, var, policy=)``
     - copy of the values with NaN where unusable
   * - ``d["left_pupil"]``, ``d.data.mask``
     - masked array and 0/1 masks under the default policy
   * - ``d.data["left_pupil"]``
     - the stored values (masked samples keep their values; see below)
   * - :meth:`~EyeData.get_blinks`,
       :meth:`~EyeData.get_interpolated_blinks`,
       :meth:`~EyeData.get_layer_intervals`
     - the runs of a layer as :class:`Intervals` (dict keyed by channel, ms)
   * - :meth:`~EyeData.mask_as_intervals`
     - the runs of unusable samples under a policy
   * - :meth:`~EyeData.get_excluded`,
       :meth:`~EyeData.get_included`
     - the excluded periods and the periods between them
   * - :meth:`~EyeData.plot.plot_masked`
     - plot of the unusable stretches (with ``policy=``)
   * - :meth:`~EyeData.quality_report`
     - missing data and masked time per layer (see :doc:`quality`)

``d.data.get_layer("left_pupil", "blink")`` gives a single layer as a
boolean array (``d.data.get_rec_layer("excluded")`` for the
recording-wide one).


How the operations treat unusable samples
-----------------------------------------

Detecting
~~~~~~~~~

Detectors write only their own layer and **replace** it when they run
again on the same channels, so trying other parameters never adds up
masks from earlier runs. :meth:`~EyeData.clear_layer` removes a
layer (``d.clear_layer("blink")`` undoes blink detection).

- :meth:`~EyeData.pupil.blinks_detect` looks for runs of NaN (and
  ``blink_val``) in the stored values and for blinks in the velocity
  profile of the samples that are usable under the strict policy. Blinks
  are cut at excluded periods (so a break without data is not a
  blink); blinks next to them are kept. It can find blinks
  where no sample is NaN (from the velocity).
- :meth:`~EyeData.gaze.mask_offscreen` and
  :meth:`~EyeData.gaze.mask_divergences` check the gaze coordinates.

Interpolating
~~~~~~~~~~~~~

:meth:`~EyeData.pupil.blinks_interpolate` replaces the pupil values in
each stored blink (extended by a margin) and sets the ``"interpolated"``
layer there; :meth:`~EyeData.interpolate_intervals` does the same
for any intervals. The ``"blink"`` layer stays, so you can always tell
recorded from reconstructed samples. After interpolation, the pupil has no
NaN in the blinks any more.

Each interpolation adds to the ``"interpolated"`` layer, so different kinds
of artifacts can be interpolated one after the other with different
methods (e.g., :meth:`~EyeData.pupil.blinks_interpolate`, then
:meth:`~EyeData.interpolate_intervals` for periods marked by hand),
and the layer holds all of them. ``d.clear_layer("interpolated")`` (also
for selected ``eyes=``/``variables=``) empties it: the values stay as they
were interpolated, but the samples are no longer marked as reconstructed,
so they are unusable again where another layer (e.g., ``"blink"``) masks
them.

Processing
~~~~~~~~~~

All processing methods read the samples that are usable under their
``policy`` argument (by default, the default policy):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - method
     - unusable samples
   * - :meth:`~EyeData.pupil.lowpass_filter`,
       :meth:`~EyeData.pupil.smooth_window`
     - bridged linearly for the filter; only the usable samples are
       changed, unusable samples keep their stored values
   * - :meth:`~EyeData.pupil.estimate_baseline`,
       :meth:`~EyeData.pupil.estimate_response`
     - left out of the fit
   * - :meth:`~EyeData.downsample`
     - left out of the window means; a layer is set for a new sample if it
       covers more than half of its window (counted for each layer
       separately); a window without usable samples stays unusable
   * - :meth:`~EyeData.scale`
     - left out of the mean and the standard deviation
   * - :meth:`~EyeData.pupil.merge_eyes`,
       :meth:`~EyeData.merge_eyes`
     - each eye contributes its usable samples, recorded samples first
       (an interpolated sample only where no eye has a recorded one); the
       merged channel is masked only where no eye is usable, and it gets
       the layers that all eyes used for a sample have in common (e.g.,
       ``"interpolated"`` where both eyes were interpolated, ``"blink"``
       where both blinked; otherwise ``"user"`` for masked samples)
   * - :meth:`~EyeData.crop`,
       :meth:`~EyeData.remove_intervals`
     - values and layers are cut together
   * - :meth:`~EyeData.fill_time_discontinuities`
     - inserted samples are NaN

Analysing
~~~~~~~~~

- :meth:`~EyeData.get_segments` masks the unusable samples in each
  segment, and the parts of a window outside the recording.
- :meth:`~EyeData.stat_per_event` computes each statistic from the
  usable samples.
- :meth:`~EyeData.quality_report` counts missing data from the
  values (NaN), shows the masked time per layer in its "Artifacts"
  section, and leaves the excluded periods out of all measures.

Saving
~~~~~~

The layers are saved with the object and restored when it is read.
Files from pypillometry 2.1 and earlier are converted when they are
read: blinks, interpolated blinks and excluded periods become their
layers, and all other masks become ``"user"`` (the reason is not stored in
those files).


Changing masks by hand
----------------------

- :meth:`~EyeData.mask_intervals` adds intervals to the
  ``"user"`` layer (existing masks are kept).
- ``d.data.add_mask(key, where)`` masks samples by index (``"user"``).
- ``d.data.set_mask(key, mask)`` makes the default mask of a channel equal
  to ``mask``: it removes the layers from samples that should be usable
  and puts the newly masked samples into ``"user"`` (clearing their
  ``"interpolated"`` mark, which would make them usable). NaN samples and
  excluded periods stay masked.
- :meth:`~EyeData.merge_masks` masks every channel where any
  channel is masked (``"user"`` layer).
- ``d.clear_layer(name)`` removes a layer from the selected channels.

.. _missing-data-changing-values:

Changing values
~~~~~~~~~~~~~~~

Writing new values into an existing channel can change its layers, too:

- ``d["left_pupil"] = array`` (a plain numpy array) **removes all layers
  of the channel**: blinks, interpolated stretches, off-screen periods,
  ... are gone; only NaN samples and excluded periods stay masked.
- ``d["left_pupil"] = masked_array`` makes the default mask of the
  channel equal to the mask of the array (like ``d.data.set_mask``). This
  also changes layers that are not part of the mask: ``d["left_pupil"] =
  d["left_pupil"] * 2`` after :meth:`~EyeData.pupil.blinks_interpolate`
  drops the ``"blink"`` layer, because interpolated blinks are usable and
  so not masked in ``d["left_pupil"]``.
- ``d.data.set_with_mask("left_pupil", values, preserve_mask=True)``
  replaces the values and keeps all layers. Use it for your own
  processing of a channel:

.. code-block:: python

    v = d.data["left_pupil"]                    # the stored values (numpy array)
    d.data.set_with_mask("left_pupil", my_filter(v), preserve_mask=True)

A new channel (``d["left", "baseline"] = array``) starts without layers;
excluded periods mask it as well.


Common tasks
------------

**Mask the pupil where the gaze is off the screen.** Copy the off-screen
periods of each eye to its pupil:

.. code-block:: python

    off = d.get_layer_intervals("offscreen", variables="x")
    d.mask_intervals({"left_pupil": off["left_x"], "right_pupil": off["right_x"]})

Where these periods overlap an interpolated blink, the samples stay
usable under the default policy (see `Mask policies`_); use
``policy="strict"`` to leave them out as well.

**Leave out interpolated samples.** Pass ``policy="strict"``, e.g. to
:meth:`~EyeData.get_segments` or
:meth:`~EyeData.stat_per_event`. To drop trials that consist
mostly of reconstructed data, compute the share of interpolated samples
per trial and select the epochs with :meth:`Epochs.keep`:

.. code-block:: python

    stim = d.get_epochs("S_", window=(-500, 2500) * u.ms)
    seg = d.get_segments(stim, "left_pupil", policy="strict")
    recon = np.ma.getmaskarray(seg.data).mean(axis=0)   # one value per trial
    good = stim.keep(recon < 0.25)

**Look at one reason only.** :meth:`~EyeData.get_layer_intervals`
gives the runs of one layer, e.g. ``d.get_layer_intervals("offscreen")``.
A list of layers as policy selects these layers and NaN, e.g.
``d.mask("left_x", policy=["offscreen"])``.

**Exclude the parts that are not part of the experiment**, e.g. breaks or
the calibration (see :ref:`data-model-masks`):

.. code-block:: python

    d.exclude_intervals(d.get_intervals(start="break_start", end="break_end"))
    d.keep_intervals([(first_onset - 1000, last_onset + 3000)])   # in ms

**Undo a step**: ``d.clear_layer("blink")``, ``d.clear_layer("offscreen")``,
``d.clear_layer("excluded")``. Interpolated values cannot be restored from
the object; keep a copy (``d | step`` returns a processed copy, see
:ref:`data-model-pipelines`).


Things to watch for
-------------------

- **Masked samples keep their values.** ``d.data["left_pupil"]`` holds
  the stored values, including those in blinks (e.g., the unfiltered
  values after :meth:`~EyeData.pupil.lowpass_filter`). Use
  ``d.values(...)`` or ``d["left_pupil"]`` to get the usable samples only.
- **Merged eyes keep only the common layers.** Where one eye was
  interpolated and the other recorded, ``mean_pupil`` holds the recorded
  value and no ``"interpolated"`` layer; where a sample is masked in the
  eyes for different reasons (e.g., a blink in one, a user mask in the
  other), the merged sample has the ``"user"`` layer.
- **Assigning values removes layers.** ``d["left_pupil"] = array`` clears
  the layers of the channel (see `Changing values`_); use
  ``d.data.set_with_mask(..., preserve_mask=True)``.
- **Detecting blinks after interpolating** is skipped (with a warning)
  for eyes with interpolated samples: their blinks are kept. Interpolating
  twice makes the interpolated stretches longer by the margins. Detect and
  interpolate once, on the raw data.
- **Blinks and excluded periods**: :meth:`~EyeData.get_blinks`
  returns all blinks of the layer, also those detected before a period
  was excluded around them. Exclude periods before detecting blinks, so
  that blinks are cut at them. A blink that ends in an excluded period
  is not interpolated (there is no usable data after it).
- **Downsampling counts each layer separately.** A window that is 40%
  off-screen, 40% divergent and 20% usable is not more than half covered
  by either layer, so the new sample is usable (it is the mean of the
  usable 20%).
- **The** ``"raw"`` **policy includes NaN**: values read with
  ``policy="raw"`` can be NaN.
