.. currentmodule:: pypillometry

Segmented data
==============

Data cut into segments around events (see
:meth:`EyeData.get_segments` and the tutorial
:doc:`/notebooks/erpds`).

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   SegmentedEyeData
   GroupLevelSegmentedData

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   group_segments

Methods for pupil segments (``seg.pupil``) and for gaze segments and
regions of interest (``seg.gaze``):

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   segmented.SegmentedPupilNamespace
   segmented.SegmentedGazeNamespace
