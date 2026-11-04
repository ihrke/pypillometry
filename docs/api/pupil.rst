.. currentmodule:: pypillometry

Pupil methods (``d.pupil``)
===========================

The methods for pupil data are used as ``d.pupil.<method>(...)``: blink
detection, merging and interpolation, filters, merging the eyes, baseline
and response models, and the foreshortening correction. They raise an
error if the object has no pupil data, change the object in place unless
``inplace=False`` is given, and return it (except
:meth:`~eyedata.pupil.PupilNamespace.fit_foreshortening`, which returns
the fitted calibration).

As pipeline steps, the same methods are ``pp.pupil.<method>(...)`` with
the same arguments (see :doc:`pipelines`):

.. code-block:: python

    d.pupil.blinks_detect().pupil.lowpass_filter(cutoff=4)    # methods
    d = d | (pp.pupil.blinks_detect() | pp.pupil.lowpass_filter(cutoff=4))  # steps

The blink getters and setters (:meth:`~GenericEyeData.get_blinks`,
:meth:`~GenericEyeData.set_blinks`, :meth:`~GenericEyeData.nblinks`, ...)
work on mask layers of any variable and are methods of the data object
itself.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   eyedata.pupil.PupilNamespace
