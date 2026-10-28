.. currentmodule:: pypillometry

Interactive tools
=================

:mod:`pypillometry` has two interactive tools that open in their own
window. They use `VisPy <https://vispy.org>`_ for fast OpenGL rendering, so
even long recordings (60+ minutes at 1000 Hz) can be browsed smoothly.

:func:`view`
    Explore an :class:`EyeData` object or numpy arrays: zoom, scroll,
    show masked regions and events, overlay several signals and select
    regions. Also available from the command line as ``pypilview``.

:func:`tweak`
    Tune the parameters of a function (e.g., a filter or blink detection)
    while you see the result on the data.

Both tools share their keyboard shortcuts. Press :kbd:`H` or :kbd:`?` in
the window to see them.

.. note::

   The tools need a display. They do not work on headless servers or in
   Binder sessions.

.. toctree::
   :maxdepth: 1

   viewer
   tweak
