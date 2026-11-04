.. currentmodule:: pypillometry

Data classes
============

See :doc:`/user_guide/data_model` for an introduction.

Main classes
------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   EyeData

Pupil and gaze methods
----------------------

The methods for pupil data are used as ``d.pupil.<method>(...)``, those
for gaze data as ``d.gaze.<method>(...)``. As pipeline steps, the same
methods are ``pp.pupil.<method>(...)`` and ``pp.gaze.<method>(...)`` (see
:doc:`pipelines`).

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   eyedata.pupil.PupilNamespace
   eyedata.gaze.GazeNamespace

Base class and containers
-------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   GenericEyeData
   EyeDataDict
   MaskPolicy
   FakeEyeData

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   keephistory

Quality report
--------------

See :meth:`GenericEyeData.quality_report`.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   QualityReport
   QualityMetric
