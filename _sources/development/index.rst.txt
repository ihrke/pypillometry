.. currentmodule:: pypillometry

Development
===========

:mod:`pypillometry` is developed and maintained by the `Cognitive
Neuroscience Research Group <https://uit.no/research/cognitive-neuroscience>`_
at UiT The Arctic University of Norway. Everyone is welcome to contribute,
by `reporting bugs <https://github.com/ihrke/pypillometry/issues>`_,
suggesting features or `sending pull requests
<https://github.com/ihrke/pypillometry/pulls>`_.

.. toctree::
   :maxdepth: 1

   changelog


How to contribute
-----------------

Development happens on GitHub: https://github.com/ihrke/pypillometry

- Use the `issue tracker <https://github.com/ihrke/pypillometry/issues>`_
  to report bugs or propose enhancements. A small example that shows the
  problem (e.g., with one of the :doc:`example datasets
  </notebooks/example_datasets>`) helps a lot.
- Use `pull requests <https://github.com/ihrke/pypillometry/pulls>`_ to
  contribute code or documentation.

If you are new to GitHub, see `About pull requests
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_.

For development, install the package in editable mode with the test
dependencies:

.. code-block:: bash

    git clone https://github.com/ihrke/pypillometry.git
    cd pypillometry
    pip install -e ".[test]"


Package architecture
--------------------

Most functionality is available as methods of the data class
:class:`EyeData` (``eyedata/eyedata.py``). Its data are an ``EyeDataDict``
(``eyedata/eyedatadict.py``), which also does the operations on all
channels at once (cutting, inserting and decimating samples, merging eyes);
the decorator that records calls in the history is in
``eyedata/history.py``. The generic
methods (crop, downsample, intervals, masks, events, I/O, ...) are methods of
the object itself. The methods for pupil and gaze data are kept in separate
modules, as *namespaces*: ``eyedata/pupil.py`` (``PupilNamespace``, used as
``d.pupil.blinks_detect()``) and ``eyedata/gaze.py`` (``GazeNamespace``,
``d.gaze.mask_offscreen()``). A namespace (``eyedata/namespace.py``) is a
view on one data object: it holds only that object (``self._d``), keeps no
state, and is created anew on each access (``accessor`` descriptor), so it
does not affect copying or saving. Its methods work on ``self._d`` and
return the data object.

The plots use the same mechanism: ``d.plot`` is a
:class:`~pypillometry.plot.PlotNamespace`, combined from one class per
domain in its own module (``plot/generic.py``: ``GenericPlots``,
``plot/pupil.py``: ``PupilPlots``, ``plot/gaze.py``: ``GazePlots``). These
classes hold no state; a new plot goes into the module of its domain and
uses ``self._d`` (and ``self._d._require(...)`` for the channels it needs).
There is no forwarding of unknown attributes: ``d.plot_pupil()`` does not
exist, only ``d.plot.plot_pupil()``.

All plotting functions (also those outside ``d.plot``) are named
``plot_*`` (or ``plot``) and follow the conventions in
``plot/_axes.py``: one-panel plots take ``ax=None`` (``get_ax()``), draw
with ``ax.*`` (not with ``plt.plot()`` etc.) and return the axes; multi-panel
plots take ``axes=None`` and ``figsize=None`` and return the figure; plots
with several figures return the list of figures. A ``plot_range`` is the
first argument (so that ``plot_pupil((0, 20) * u.sec)`` works). All take
``title=``, and
none calls ``plt.show()``. ``tests/test_plotting_conventions.py`` checks
each function.

Every transforming method can also be used as a pipeline step
(``eyedata/pipes.py``): decorated with ``@step``, the same method runs
when it is called on data (``d.crop(...)``, ``d.pupil.blinks_detect()``)
and returns a :class:`Step` when it is called without data
(``pp.pupil.blinks_detect()``, or the generic steps ``pp.crop(...)`` made
in ``eyedata/steps.py``). ``d | step`` copies ``d`` once and runs the steps
in place on the copy (see :class:`Pipeline`).

The computations themselves are implemented as stateless functions on
:mod:`numpy` arrays in ``pypillometry.signal`` (see
:doc:`/api/signal`). To add a new processing step, first write a function
that works on arrays, then add a thin method to the appropriate class or
namespace that calls it with the right data and parameters. Decorate the
method with :func:`keephistory` so that it is recorded in the object's
history, with ``@step`` (outermost) if it transforms the data, add a
generic step to ``eyedata/steps.py`` if it is a generic method, and
follow the API conventions below.


API conventions
---------------

All methods of the data classes follow five rules. New methods must
follow them too; ``tests/test_api_conventions.py`` checks them.

1. **Every transforming method returns the object and takes** ``inplace``
   (``None`` by default: the object-level setting from the constructor,
   else the package setting, see :func:`set_default_inplace`). With
   ``inplace=False`` it changes and returns a copy. Only ``set_*`` methods
   (:meth:`~EyeData.set_events`, :meth:`~EyeData.set_blinks`,
   :meth:`~EyeData.set_experimental_setup`) always work in place; they
   return the object as well. Methods that change the object are decorated
   with :func:`keephistory`, which records the call in the returned object.

2. **Detectors set their layer and return the object.** Blink detection,
   :meth:`~EyeData.pupil.blinks_merge_close`,
   :meth:`~EyeData.gaze.mask_offscreen` and
   :meth:`~EyeData.gaze.mask_divergences` replace their mask layer
   (``"blink"``, ``"offscreen"``, ``"divergence"``) for the channels they
   process. Setting a layer *is* masking: there is no ``apply_mask``
   argument. The results are read with getters
   (:meth:`~EyeData.get_blinks`,
   :meth:`~EyeData.get_layer_intervals`, ...) and removed with
   :meth:`~EyeData.clear_layer`. To compute without storing, use the
   functions in ``pypillometry.signal``.

3. **Getters with per-channel results always return a dict** keyed by
   the channel name ``"<eye>_<variable>"`` (e.g., ``"left_pupil"``), also
   for a single channel, and with an entry for every selected channel (e.g.,
   :meth:`~EyeData.get_blinks` gives empty Intervals for a channel without
   blinks). The keys are the same strings as in ``d["left_pupil"]`` and
   ``d.data``. Such results can be a :class:`RecordDict` (e.g.,
   :meth:`~EyeData.layer_stats`: channel -> layer -> statistics), a
   nested dict that is displayed as a table. Results with one row per
   event or trial (e.g., :meth:`Events.to_table`) are a :class:`Table`, a
   dict of numpy columns that is displayed as a table.

4. **Plain numbers are ms**: windows, durations, margins, and the default
   units of returned times and intervals
   (:meth:`~EyeData.get_blinks`, :meth:`~EyeData.get_duration`,
   ...). Other units are given with the values (``(-200, 800) * u.ms``,
   ``"2 min"``). Sample indices only with an explicit ``units="samples"``.
   Exception: in plotting functions, plain numbers in ``plot_range`` and
   ``interv`` are in ``units`` (the unit of the time axis).

5. **eyes and variables**: ``None`` or ``[]`` select all existing
   channels; a string or a list is accepted; unknown names raise a
   ``KeyError`` that lists the available channels. Use
   ``self.data.select(eyes, variables)`` (existing ``(eye, variable)``
   pairs) or ``self._eyes_with(eyes, variables)`` (eyes that have all
   `variables`) instead of looping over eyes and variables.

New eyes created from others are named ``"mean"`` by default
(:meth:`~EyeData.merge_eyes`, :meth:`~EyeData.pupil.merge_eyes`;
both take ``name=``).


Saved files and their state
---------------------------

The content of eyedata and segmented objects is defined by their *state*:
a dict with only numbers, strings, lists, dicts and numpy arrays. Each
class lists the attributes that belong to it in ``_STATE_FIELDS``
(objects inside it, such as :class:`Intervals`, become nested states; see
``pypillometry/state.py``). Pickle files (via
``__getstate__``/``__setstate__``) and HDF5 files
(``pypillometry/hdf5.py``) only store this state, together with its
``format_version``. A test (``tests/test_state.py``) fails if an object
has an attribute that is not part of its state.

If you add an attribute to a class (or change what an attribute
contains), the saved files have to follow:

1. add it to ``_STATE_FIELDS`` of the class (or to ``_STATE_EXCLUDE`` if
   it is derived from other attributes and need not be saved),
2. increase ``FORMAT_VERSION`` in ``pypillometry/state.py`` and add a
   function to ``_MIGRATIONS`` that upgrades a state of the previous
   version (e.g., sets a default for the new entry); old migrations are
   never changed,
3. keep the fixture files of the older versions in ``tests/data/state/``
   (they must keep loading) and add files written by the new version
   (e.g., ``eyedata_v2.h5``).

Files written before the state was versioned (pypillometry 2.0) count as
version 0.


Documenting code
----------------

Docstrings follow the `NumPy style
<https://numpydoc.readthedocs.io/en/latest/format.html>`_. Every public
class and function should have a docstring that explains what it does,
its parameters and its return value, ideally with a short example. The
:doc:`API reference </api/index>` is generated from them.


Tests
-----

Tests are in the ``tests/`` folder and run with `pytest
<https://docs.pytest.org/>`_. The examples in the docstrings are run as
well (doctests):

.. code-block:: bash

    pytest                # tests/ and docstring examples
    pytest tests/         # tests only
    pytest pypillometry/  # docstring examples only

Examples should run as written (use the bundled
``pp.get_example_data("rlmw_002_short")`` for data). Mark single lines
with ``# doctest: +SKIP`` only if they need data that are not available
(e.g., a user's file) or give output that varies between runs. Run pytest
from the repository root without other paths; the root contains scripts
that are not tests. The tests run automatically for every push and pull
request on GitHub. New functions should come with tests: at a minimum, call them
with reasonable arguments; better, compare the result to the expected
output.


Building the documentation
--------------------------

The documentation (this website) is built with `Sphinx
<https://www.sphinx-doc.org/>`_ and the `PyData Sphinx theme
<https://pydata-sphinx-theme.readthedocs.io/>`_. Its sources are in the
``docs/`` folder:

.. list-table::
   :widths: 30 70

   * - ``index.rst``
     - landing page
   * - ``getting_started/``
     - installation, quickstart (a notebook), command-line tools
   * - ``user_guide/``
     - concepts, one page per topic
   * - ``notebooks/``
     - tutorials (Jupyter notebooks)
   * - ``experimental/``
     - pages and notebooks for experimental features
   * - ``api/``
     - API reference; the pages in ``api/generated/`` are created by
       autosummary during the build and are not under version control
   * - ``conf.py``, ``_static/``, ``_templates/``
     - configuration, CSS and autosummary templates

Install the requirements and build the HTML pages from the repository
root:

.. code-block:: bash

    pip install -r docs-requirements.txt
    make html                      # output in _build/html
    make livehtml                  # rebuild automatically while editing

``make clean`` removes the build directory ``_build``.

Notebooks
^^^^^^^^^

Notebooks are stored **with their outputs** and are **not executed**
during the build (``nbsphinx_execute = "never"``). This keeps the build
fast and independent of data downloads. After changing a notebook, or when
the package changes in a way that affects its output, run it again and
commit the result:

.. code-block:: bash

    cd docs/notebooks
    jupyter nbconvert --to notebook --execute --inplace blinks.ipynb

Use ``pp.logging_set_level("WARNING")`` at the top of a notebook to keep
log messages out of the rendered pages. Cells can be hidden from the
website by setting their metadata to ``{"nbsphinx": "hidden"}``.

The documentation is built and published to GitHub Pages automatically
for every push to ``master``.


Logging
-------

The package logs with `loguru <https://loguru.readthedocs.io/>`_. During
development, it can help to see the debug messages of a single module only:

.. code-block:: python

    import pypillometry as pp
    pp.logging_disable()
    pp.logger.enable("pypillometry.eyedata.eyedata")
    pp.logging_set_level("DEBUG")


Releases
--------

Version numbers follow the scheme ``x.y.z``: increment ``z`` for bug
fixes, ``y`` for new features and ``x`` for changes that break backward
compatibility.

To release a version:

1. On ``master``, set the file ``VERSION`` to the new version (e.g.,
   ``v2.2.0``; it must follow `PEP 440
   <https://peps.python.org/pep-0440/#version-scheme>`_) and give the
   version and the date as heading of its section in ``CHANGELOG.md``
   (``## 2.2.0 (2026-11-10)``); commit and push.
2. Create a release with the same tag (``v2.2.0``) on `GitHub releases
   <https://github.com/ihrke/pypillometry/releases>`_. A GitHub action
   checks that the tag matches ``VERSION`` and uploads the release to
   `PyPI <https://pypi.org/project/pypillometry/>`_.

The documentation is built and published by another action whenever
``master`` is pushed.


Major versions
--------------

Version 2 (2025) is a rewrite that breaks the API of version 1. Version 1
is kept on the `v1 branch <https://github.com/ihrke/pypillometry/tree/v1>`_
and its documentation at https://ihrke.github.io/pypillometry/v1/html/.
