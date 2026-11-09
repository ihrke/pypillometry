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

Results with one row per channel, event or trial (e.g.,
:meth:`EyeData.layer_stats`, :meth:`Events.to_table`,
:meth:`SegmentedEyeData.gaze.dwell_time
<pypillometry.segmented.SegmentedGazeNamespace.dwell_time>`) are tables:
dicts of numpy columns that are displayed as a :class:`pandas.DataFrame`.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Table
