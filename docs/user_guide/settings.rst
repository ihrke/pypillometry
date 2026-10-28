.. currentmodule:: pypillometry

Settings and logging
====================

In-place operations
-------------------

By default, methods change the object they are called on (see
:doc:`data_model`). The default can be changed for the whole package:

.. code-block:: python

    pp.set_default_inplace(False)   # methods return changed copies
    pp.get_default_inplace()        # False

A value given to an object's constructor (``inplace=...``) or to a single
method call takes precedence.

.. autosummary::
   :nosignatures:

   set_default_inplace
   get_default_inplace


Logging
-------

:mod:`pypillometry` reports what it does with `loguru
<https://loguru.readthedocs.io/>`_. By default, the log level is ``INFO``,
which gives a moderate amount of information. Use ``WARNING`` to see only
problems, or ``DEBUG`` to see everything:

.. code-block:: python

    pp.logging_set_level("WARNING")
    pp.logging_disable()            # no messages at all
    pp.logging_enable()

    with pp.loglevel("DEBUG"):      # temporarily, for a block of code
        d.pupil_blinks_detect()

    with pp.nologging():            # temporarily silent
        d.pupil_blinks_detect()

To see detailed messages from one part of the package only:

.. code-block:: python

    pp.logging_disable()
    pp.logger.enable("pypillometry.eyedata")

.. autosummary::
   :nosignatures:

   logging_set_level
   logging_get_level
   logging_disable
   logging_enable
   loglevel
   nologging
