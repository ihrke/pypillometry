API reference
=============

.. module:: pypillometry


This is the complete list of classes and functions. Everything listed here
is available directly from the package, e.g. ``pp.EyeData`` after
``import pypillometry as pp``. For an introduction to how they fit
together, see the :doc:`/user_guide/index`.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Data classes
      :link: data
      :link-type: doc

      ``EyeData``, ``PupilData``, ``GazeData`` and their shared base class.

   .. grid-item-card:: Events and intervals
      :link: events_intervals
      :link-type: doc

      ``Events``, ``Intervals`` and functions to combine them.

   .. grid-item-card:: Segmented data
      :link: segments
      :link-type: doc

      Event-related segments for single participants and groups.

   .. grid-item-card:: Plotting
      :link: plotting
      :link-type: doc

      The plotting functions available under ``d.plot``.

   .. grid-item-card:: Reading and writing data
      :link: io
      :link-type: doc

      Files, URLs, OSF, studies and example datasets.

   .. grid-item-card:: Signal processing
      :link: signal
      :link-type: doc

      Low-level functions on numpy arrays: filtering, blinks, baseline and
      response models, simulated data.

   .. grid-item-card:: Geometry and gaze (experimental)
      :link: geometry
      :link-type: doc

      Experimental setup, calibration, foreshortening and regions of
      interest.

   .. grid-item-card:: Units
      :link: units
      :link-type: doc

      Times, distances and angles with units.

   .. grid-item-card:: Viewer, settings and logging
      :link: misc
      :link-type: doc

      Interactive tools, package-wide settings and logging.

.. toctree::
   :hidden:

   data
   events_intervals
   segments
   plotting
   io
   signal
   geometry
   units
   misc
