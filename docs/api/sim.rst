.. currentmodule:: pypillometry.sim

Simulated data: ``pp.sim``
==========================

Simulated recordings with their ground truth: a :class:`Simulation` is
built by chaining steps, :meth:`Simulation.run` returns a
:class:`SimEyeData` object. See :doc:`/user_guide/simulation` for an
introduction and :doc:`/notebooks/simulation` for examples.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Simulation
   SimEyeData
   SimPlotNamespace

Tasks and events
----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Task

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   parse_dist

Ready-made tasks: ``pp.sim.tasks``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pypillometry.sim.tasks

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   pvt
   sart
   ftrsgt
   nback

.. currentmodule:: pypillometry.sim

Steps
-----

The steps run in a fixed order of stages: events of the task, latent
pupil size (log-%), gaze, foreshortening, units of the tracker, noise.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   TimescalePupil
   Fixations
   Foreshortening
   Units
   WhiteNoise

From a fitted model
-------------------

.. currentmodule:: pypillometry

.. autosummary::
   :nosignatures:

   TimescaleModel.process
   TimescaleModel.simulate
