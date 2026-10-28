Tutorials
=========

The tutorials are Jupyter notebooks that show how to do specific tasks
from start to finish. Each page has a link to the notebook on GitHub and a
Binder link |binder| that opens it in your browser, so you can run it
without installing anything.

.. |binder| image:: https://mybinder.org/badge_logo.svg

If you are new to :mod:`pypillometry`, start with the
:doc:`/getting_started/quickstart`.


Reading data
------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Eyelink EDF files
      :link: /notebooks/importdata_edf
      :link-type: doc

      Read an EDF file, including events and calibration data.

   .. grid-item-card:: Text files (ASC)
      :link: /notebooks/importdata_asc
      :link-type: doc

      Read text files with pandas and build a data object from arrays.

   .. grid-item-card:: Example datasets
      :link: /notebooks/example_datasets
      :link-type: doc

      The datasets that come with the package.


Working with data objects
-------------------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Pipelines and history
      :link: /notebooks/pipes
      :link-type: doc

      Chain processing steps, keep a record of them, and decide between
      in-place changes and copies.

   .. grid-item-card:: Eyes and variables
      :link: /notebooks/eyes_and_variables
      :link-type: doc

      Monocular and binocular data, selecting and combining eyes, adding
      variables.

   .. grid-item-card:: Summarizing data
      :link: /notebooks/summary
      :link-type: doc

      Summaries of one or many datasets as tables.


Preprocessing and plotting
--------------------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Blinks
      :link: /notebooks/blinks
      :link-type: doc

      Detect, check and interpolate blinks.

   .. grid-item-card:: Plotting
      :link: /notebooks/plotting
      :link-type: doc

      Plot pupil and gaze data for exploration and quality checks.


Events and event-related analyses
---------------------------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Events
      :link: /notebooks/events
      :link-type: doc

      Select, filter, convert and change the events of a recording.

   .. grid-item-card:: Intervals
      :link: /notebooks/intervals
      :link-type: doc

      Time windows around events, blinks and artifacts: create, combine
      and use them.

   .. grid-item-card:: Event-related pupil dilation
      :link: /notebooks/erpds
      :link-type: doc

      Average pupil responses per condition and across participants.


Modeling and complete analyses
------------------------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Modeling the pupil signal
      :link: /notebooks/modeling
      :link-type: doc

      Separate the signal into tonic baseline and phasic responses.

   .. grid-item-card:: A complete study
      :link: /notebooks/preproc_example_pavlov
      :link-type: doc

      Preprocess a full study with many participants.

More examples for the experimental features are in
:doc:`/experimental/index`.

.. toctree::
   :hidden:
   :caption: Reading data

   /notebooks/importdata_edf
   /notebooks/importdata_asc
   /notebooks/example_datasets

.. toctree::
   :hidden:
   :caption: Working with data objects

   /notebooks/pipes
   /notebooks/eyes_and_variables
   /notebooks/summary

.. toctree::
   :hidden:
   :caption: Preprocessing and plotting

   /notebooks/blinks
   /notebooks/plotting

.. toctree::
   :hidden:
   :caption: Events and event-related analyses

   /notebooks/events
   /notebooks/intervals
   /notebooks/erpds

.. toctree::
   :hidden:
   :caption: Modeling and complete analyses

   /notebooks/modeling
   /notebooks/preproc_example_pavlov
