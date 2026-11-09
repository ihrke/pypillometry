.. currentmodule:: pypillometry

Segmented data
==============

Data cut into segments around events (see
:meth:`EyeData.get_segments`, the user guide pages
:doc:`/user_guide/events_intervals` and :doc:`/user_guide/rois`, and the
tutorials :doc:`/notebooks/erpds` and :doc:`/notebooks/rois`).

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   SegmentedEyeData

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   concat_segments
   group_segments

Methods for pupil segments (``seg.pupil``) and for gaze segments and
regions of interest (``seg.gaze``):

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   segmented.SegmentedPupilNamespace
   segmented.SegmentedGazeNamespace
