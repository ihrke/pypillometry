.. currentmodule:: pypillometry

Geometry and gaze
=================

.. admonition:: Experimental
   :class: experimental

   Most classes on this page belong to the
   :doc:`experimental features </experimental/index>`. Their API may
   change. :class:`ExperimentalSetup` is also used by stable functions,
   e.g., :meth:`EyeData.gaze.mask_offscreen`.

Experimental setup and calibration
----------------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ExperimentalSetup
   SpatialCalibration
   ForeshorteningCalibration

Regions of interest
-------------------

ROIs are used with the segments around events by ``seg.gaze`` (see
:doc:`/user_guide/rois`).

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ROI
   CircularROI
   RectangularROI
