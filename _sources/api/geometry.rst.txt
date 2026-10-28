.. currentmodule:: pypillometry

Geometry and gaze
=================

.. admonition:: Experimental
   :class: experimental

   Most classes on this page belong to the
   :doc:`experimental features </experimental/index>`. Their API may
   change. :class:`ExperimentalSetup` is also used by stable functions,
   e.g., :meth:`GazeData.mask_offscreen_coords`.

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

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ROI
   CircularROI
   RectangularROI
