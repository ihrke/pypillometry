.. currentmodule:: pypillometry

Modeling the pupil signal
=========================

The pupil signal can be seen as the sum of a slowly changing (*tonic*)
baseline and fast (*phasic*) responses to events. :mod:`pypillometry`
implements methods to separate the two:

.. autosummary::
   :nosignatures:

   PupilData.pupil_estimate_baseline
   PupilData.pupil_estimate_response

:meth:`~PupilData.pupil_estimate_baseline` estimates the baseline as the
lower envelope of the signal, fitted with B-splines through the troughs
between events. :meth:`~PupilData.pupil_estimate_response`
then fits a pupil response function (Hoeks & Levelt, 1993) to every event
in the signal that remains. Both store their results as new variables
(``baseline`` and ``response``) next to the pupil data.

The data must not contain missing samples, so blinks have to be
interpolated first. The baseline is fitted with a model written in Stan
and needs CmdStan (see :doc:`/getting_started/installation`). The
response model is fitted with non-negative least squares.

The underlying functions work on plain arrays and are listed in the API
reference under :doc:`/api/signal`.

See the tutorial :doc:`/notebooks/modeling`.

References
----------

- Hoeks, B., & Levelt, W. J. (1993). Pupillary dilation as a measure of
  attention: A quantitative system analysis. *Behavior Research Methods,
  Instruments, & Computers*, 25(1), 16–26.
  https://doi.org/10.3758/BF03204445
