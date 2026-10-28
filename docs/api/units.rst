.. currentmodule:: pypillometry

Units
=====

See :doc:`/user_guide/units`.

.. data:: u

   Namespace of units, e.g. ``u.ms``, ``u.sec``, ``u.min``, ``u.cm``,
   ``u.deg``. Multiplying a number, tuple or array by a unit gives a
   :class:`Quantity`. All aliases work as attributes (e.g. ``u.seconds``,
   ``u.degrees``). ``u.m`` is metres (use ``u.min`` for minutes).

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Quantity
   Unit

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   parse_time
   parse_distance
   parse_angle
