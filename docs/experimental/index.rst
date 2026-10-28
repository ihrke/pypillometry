.. currentmodule:: pypillometry

Experimental features
=====================

.. admonition:: Work in progress
   :class: experimental

   The features on these pages can be used, but they are still being
   developed. Their API may change in future versions without a
   deprecation period, and some parts have not yet been validated on real
   data. Feedback and bug reports on the `issue tracker
   <https://github.com/ihrke/pypillometry/issues>`_ are very welcome.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: :fas:`eye` Pupil foreshortening
      :link: foreshortening
      :link-type: doc

      Correct pupil size for the angle at which the camera sees the eye,
      by fitting the camera position from the data.

   .. grid-item-card:: :fas:`crosshairs` Setup and regions of interest
      :link: gaze_rois
      :link-type: doc

      Describe the screen and camera geometry, convert between pixels, mm
      and degrees, and mark regions of the screen.

   .. grid-item-card:: :fas:`bullseye` Spatial calibration
      :link: spatial_calibration
      :link-type: doc

      Read and inspect the calibration and validation results of the eye
      tracker.

The related classes are listed in the API reference under
:doc:`/api/geometry`.

.. toctree::
   :hidden:

   foreshortening
   foreshortening_correction_algorithm
   gaze_rois
   spatial_calibration
