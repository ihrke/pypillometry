:html_theme.sidebar_secondary.remove:

.. currentmodule:: pypillometry

.. title:: pypillometry

.. raw:: html

   <div class="pp-hero">
     <img class="pp-hero-logo" src="_static/logo.png" alt="pypillometry logo">
     <h1>pypillometry</h1>
     <p class="pp-tagline">Preprocessing, analysis and visualization of
     pupillometric and eye-tracking data in Python.</p>

.. code-block:: bash

   pip install pypillometry

.. raw:: html

   </div>

**pypillometry** reads recordings from eye trackers (e.g., Eyelink EDF
files), cleans them (blinks, artifacts, filtering), cuts them into
event-related segments and plots the results. Pupil size and gaze position
of one or both eyes are kept together in a single object, and every
processing step is recorded in the object's history.

.. code-block:: python

   import pypillometry as pp
   from pypillometry import u

   d = pp.EyeData.from_eyelink("recording.edf")   # read an Eyelink file
   d.pupil_blinks_detect(apply_mask=True)           # detect and mask blinks
   d.pupil_blinks_interpolate()                     # interpolate them
   d.pupil_lowpass_filter(cutoff=4)                 # 4 Hz low-pass filter

   # average pupil response from 200 ms before to 1 s after each stimulus
   intervals = d.get_intervals("stim", (-200, 1000) * u.ms)
   erpd = d.get_segments(intervals, "left_pupil").baseline_correct((-200, 0))
   erpd.plot()                                          # event-related pupil dilation


.. grid:: 1 2 2 3
   :gutter: 3
   :margin: 4 4 0 0

   .. grid-item-card:: :fas:`rocket` Getting started
      :link: getting_started/index
      :link-type: doc

      Install the package and work through a ten-minute example from raw
      data to an event-related pupil response.

   .. grid-item-card:: :fas:`book-open` User guide
      :link: user_guide/index
      :link-type: doc

      The concepts behind the package: data objects, eyes and variables,
      units, events and intervals, preprocessing and plotting.

   .. grid-item-card:: :fas:`graduation-cap` Tutorials
      :link: tutorials/index
      :link-type: doc

      Worked examples as Jupyter notebooks that you can run in your
      browser.

   .. grid-item-card:: :fas:`code` API reference
      :link: api/index
      :link-type: doc

      Every class and function, with parameters, return values and
      examples.

   .. grid-item-card:: :fas:`flask` Experimental features
      :link: experimental/index
      :link-type: doc

      Correction of pupil foreshortening, spatial calibration and regions
      of interest. Usable, but the API may change.

   .. grid-item-card:: :fas:`users` Development
      :link: development/index
      :link-type: doc

      Report bugs, contribute code and read the changelog.


Features
--------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      - **Reading data** from Eyelink EDF and ASC files, saved pickle files,
        and whole studies from a local folder or `OSF <https://osf.io>`_
      - **Binocular data**: pupil size and gaze of both eyes in one object
      - **Blinks**: detection, merging and interpolation (Mathôt, 2013)
      - **Filtering**: low-pass filtering, smoothing, downsampling, scaling

   .. grid-item::

      - **Events and intervals** with flexible selection by label, function
        or time
      - **Event-related pupil dilation** for single subjects and groups
      - **Units** everywhere: ``3 * u.min``, ``"200 ms"``, ``60 * u.cm``
      - **Interactive viewer** for long recordings and command-line tools
      - **Modeling** of tonic baseline and phasic pupil responses
        (Mittner & Groot, 2025)


Citing pypillometry
-------------------

If you use **pypillometry** in your research, please cite

   Mittner, M. (2020). pypillometry: A Python package for pupillometric
   analyses. *Journal of Open Source Software*, 5(51), 2348.
   https://doi.org/10.21105/joss.02348

If you use the estimation of tonic and phasic pupil components
(:doc:`/user_guide/modeling`), please also cite

   Mittner, M., & Groot, J. M. (2025). A novel method for modeling tonic and
   phasic pupil dynamics in humans. *Behavior Research Methods*, 57(9), 233.
   https://doi.org/10.3758/s13428-025-02755-7


.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   user_guide/index
   tutorials/index
   api/index
   experimental/index
   development/index
