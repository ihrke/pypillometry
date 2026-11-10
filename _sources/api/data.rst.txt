.. currentmodule:: pypillometry

Data classes
============

See :doc:`/user_guide/data_model` for an introduction. The methods for
pupil and gaze data are in the namespaces ``d.pupil`` (:doc:`pupil`) and
``d.gaze`` (:doc:`gaze`), the plots in ``d.plot`` (:doc:`plotting`).

Main class
----------

:class:`EyeData` holds the data. Its methods for any channel (crop,
downsample, masks and layers, events, intervals, segments, saving, ...) are
listed on its page.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   EyeData

Channels, layers and mask policies
----------------------------------

The time series (``d["left_pupil"]``) and their mask layers are stored in
an :class:`EyeDataDict` (``d.data``). A :class:`MaskPolicy` says which
layers make a sample unusable (``"default"``, ``"strict"``, ``"raw"``).

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   EyeDataDict
   MaskPolicy

Simulated data and history
--------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   FakeEyeData

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   keephistory

Quality report
--------------

See :meth:`EyeData.quality_report`.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   QualityReport
   QualityMetric

Tables
------

Some results are dicts that are displayed as a table
(:class:`pandas.DataFrame`):

- results with one row per event or trial (e.g., :meth:`Events.to_table`,
  :meth:`SegmentedEyeData.gaze.dwell_time
  <pypillometry.segmented.SegmentedGazeNamespace.dwell_time>`) are a
  :class:`Table`, a dict of numpy columns
- results per channel (e.g., :meth:`EyeData.layer_stats`: channel ->
  layer -> statistics) and the list of example datasets are a
  :class:`RecordDict`, a dict of records

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Table
   RecordDict
