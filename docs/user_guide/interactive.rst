.. currentmodule:: pypillometry

Interactive tools
=================

:mod:`pypillometry` has two interactive tools that open in their own
window. They use `VisPy <https://vispy.org>`_ for fast OpenGL rendering, so
even long recordings (60+ minutes at 1000 Hz) can be browsed smoothly.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: ``view()``: browse recordings
      :link: viewer
      :link-type: doc
      :img-bottom: ../images/viewer_eyedata.png

      Explore an :class:`EyeData` object or numpy arrays: zoom, scroll,
      show masked regions and events, overlay several signals and select
      regions. Also available from the command line as ``pypilview``.

   .. grid-item-card:: ``tweak()``: tune parameters
      :link: tweak
      :link-type: doc
      :img-bottom: ../images/tweak_intervals.png

      Tune the parameters of a function (e.g., a filter or blink
      detection) while you see the result on the data.

Both tools share their keyboard shortcuts. Press :kbd:`H` or :kbd:`?` in
the window to see them.

.. note::

   The tools need a display. They do not work on headless servers or in
   Binder sessions.

.. toctree::
   :maxdepth: 1

   viewer
   tweak
