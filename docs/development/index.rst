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

Most functionality is available as methods of the data classes
(:class:`GenericEyeData`, :class:`PupilData`, :class:`GazeData`,
:class:`EyeData`). This makes it easy to find functions and to chain them
into pipelines (see :doc:`/user_guide/data_model`).

The computations themselves are implemented as stateless functions on
:mod:`numpy` arrays in ``pypillometry.signal`` (see
:doc:`/api/signal`). To add a new processing step, first write a function
that works on arrays, then add a thin method to the appropriate class that
calls it with the right data and parameters. Decorate the method with
:func:`keephistory` so that it is recorded in the object's history, and
support ``eyes=``, ``variables=`` and ``inplace=`` like the existing
methods.


Saved files and their state
---------------------------

The content of eyedata and segmented objects is defined by their *state*:
a dict with only numbers, strings, lists, dicts and numpy arrays, written
by ``_to_state()`` and read by ``_set_state()`` (nested objects such as
:class:`Intervals` have ``_to_state()``/``_from_state()``). Pickle files
(via ``__getstate__``/``__setstate__``) and HDF5 files
(``pypillometry/hdf5.py``) only store this state, together with its
``format_version``.

If you add an attribute to a class (or change what an attribute
contains), the saved files have to follow:

1. add it to ``_to_state()`` and ``_set_state()`` of the class,
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
<https://docs.pytest.org/>`_:

.. code-block:: bash

    pytest tests/

Run pytest on ``tests/`` only; the repository root contains scripts that
are not tests. The tests run automatically for every push and pull request
on GitHub. New functions should come with tests: at a minimum, call them
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
    pp.logger.enable("pypillometry.eyedata.generic")
    pp.logging_set_level("DEBUG")


Releases
--------

Version numbers follow the scheme ``x.y.z``: increment ``z`` for bug
fixes, ``y`` for new features and ``x`` for changes that break backward
compatibility.

Releases are created on `GitHub releases
<https://github.com/ihrke/pypillometry/releases>`_. A GitHub action then
writes the tag into the file ``VERSION`` (it must follow `PEP 440
<https://peps.python.org/pep-0440/#version-scheme>`_) and uploads the
release to `PyPI <https://pypi.org/project/pypillometry/>`_.


Major versions
--------------

Version 2 (2025) is a rewrite that breaks the API of version 1. Version 1
is kept on the `v1 branch <https://github.com/ihrke/pypillometry/tree/v1>`_
and its documentation at https://ihrke.github.io/pypillometry/v1/html/.
