User guide
==========

The user guide explains the main ideas behind :mod:`pypillometry`. Each
page gives an overview of a topic and links to the
:doc:`tutorials </tutorials/index>` for worked examples and to the
:doc:`API reference </api/index>` for all parameters.

If you have not used the package before, start with the
:doc:`/getting_started/quickstart`.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :fas:`cubes` Data objects
      :link: data_model
      :link-type: doc

      ``EyeData``, ``PupilData`` and ``GazeData``; eyes and variables;
      masks; pipelines, in-place operations and history.

   .. grid-item-card:: :fas:`file-import` Loading and saving data
      :link: loading_data
      :link-type: doc

      Eyelink EDF and ASC files, your own arrays, example datasets and
      saved objects.

   .. grid-item-card:: :fas:`ruler` Units
      :link: units
      :link-type: doc

      Give times, distances and angles with units: ``3 * u.min``,
      ``"200 ms"``, ``60 * u.cm``.

   .. grid-item-card:: :fas:`broom` Preprocessing
      :link: preprocessing
      :link-type: doc

      Blinks, gaze artifacts, filtering, downsampling, scaling and
      combining the eyes.

   .. grid-item-card:: :fas:`clipboard-check` Data quality
      :link: quality
      :link-type: doc

      A quality report with labeled measures (missing data, noise,
      blinks, calibration) and how the thresholds were chosen.

   .. grid-item-card:: :fas:`stopwatch` Events, intervals and segments
      :link: events_intervals
      :link-type: doc

      Select events, build time windows around them and compute
      event-related pupil dilations.

   .. grid-item-card:: :fas:`chart-line` Plotting
      :link: plotting
      :link-type: doc

      Time series, blinks, intervals, heatmaps and scanpaths.

   .. grid-item-card:: :fas:`display` Interactive tools
      :link: interactive
      :link-type: doc

      A fast viewer for long recordings and a tool for tuning parameters
      while you watch the result.

   .. grid-item-card:: :fas:`wave-square` Modeling the pupil signal
      :link: modeling
      :link-type: doc

      Separate the signal into a slow baseline and event-related
      responses.

   .. grid-item-card:: :fas:`folder-tree` Studies with many participants
      :link: studies
      :link-type: doc

      Load a whole study from a local folder or from OSF with a
      configuration file.

   .. grid-item-card:: :fas:`sliders` Settings and logging
      :link: settings
      :link-type: doc

      Package-wide defaults and how much the package tells you about what
      it does.

.. toctree::
   :hidden:
   :maxdepth: 2

   data_model
   loading_data
   units
   preprocessing
   quality
   events_intervals
   plotting
   interactive
   modeling
   studies
   settings
