.. currentmodule:: pypillometry

Gaze methods (``d.gaze``)
=========================

The methods for gaze data are used as ``d.gaze.<method>(...)``: masking
gaze samples outside the screen and samples where the eyes diverge. They
set their mask layer (``"offscreen"``, ``"divergence"``), change the
object in place unless ``inplace=False`` is given, and return it. Read the
results with :meth:`~GenericEyeData.get_layer_intervals`.

As pipeline steps, the same methods are ``pp.gaze.<method>(...)`` with the
same arguments (see :doc:`pipelines`). The experimental setup, calibration,
foreshortening and regions of interest are on :doc:`geometry`.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   eyedata.gaze.GazeNamespace
