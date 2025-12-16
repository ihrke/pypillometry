.. currentmodule:: pypillometry

Graphical Tools
===============

pypillometry provides GPU-accelerated graphical tools for interactive exploration and parameter tuning of eye-tracking data. These tools use VisPy for fast OpenGL rendering, enabling smooth navigation even with long recordings (60+ minutes at 1000 Hz).

.. toctree::
   :maxdepth: 2
   :caption: Available Tools:
   
   viewer
   tweak


Overview
--------

:func:`view`
    Interactive viewer for exploring EyeData objects or numpy arrays. Supports multiple subplots, overlays, masked region highlighting, and interactive region selection.

:func:`tweak`  
    Interactive parameter tuning tool. Displays original data with function results overlaid, allowing real-time adjustment of function parameters through a GUI panel.

Both tools share common keyboard shortcuts - press ``H`` or ``?`` while in the viewer to see the full list.

