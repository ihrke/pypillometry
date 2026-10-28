.. currentmodule:: pypillometry

Installation
============

:mod:`pypillometry` requires Python 3.11 or newer. Released versions are on
`PyPI <https://pypi.org/project/pypillometry/>`_:

.. code-block:: bash

    pip install pypillometry

The dependencies are installed automatically.

.. tab-set::

   .. tab-item:: Development version

      The latest version from GitHub:

      .. code-block:: bash

          pip install git+https://github.com/ihrke/pypillometry.git

   .. tab-item:: Editable install

      To work on the code yourself (see :doc:`/development/index`), clone
      the repository and install it in editable mode, together with the
      test dependencies:

      .. code-block:: bash

          git clone https://github.com/ihrke/pypillometry.git
          cd pypillometry
          pip install -e ".[test]"

   .. tab-item:: Conda environment

      It is a good idea to install the package into its own environment,
      e.g., with `conda <https://docs.conda.io/en/latest/>`_:

      .. code-block:: bash

          conda create -n pypil python=3.13
          conda activate pypil
          pip install pypillometry

      Python's built-in `virtual environments
      <https://docs.python.org/3/tutorial/venv.html>`_ work as well.

To check that everything works, load one of the bundled example datasets:

.. code-block:: python

    import pypillometry as pp
    d = pp.get_example_data("rlmw_002_short")
    print(d)


Dependencies
------------

All of these are installed with the package (see ``pyproject.toml``):

- :mod:`numpy`, :mod:`scipy`, :mod:`pandas` and :mod:`matplotlib` for
  numerical work and plotting
- `eyelinkio <https://github.com/scott-huberty/eyelinkio>`_ for reading
  Eyelink EDF files
- :mod:`PyQt5` and :mod:`vispy` for the interactive :doc:`viewer
  </user_guide/viewer>`
- :mod:`cmdstanpy` for the baseline model (see below)
- `loguru <https://loguru.readthedocs.io/>`_ for logging

Saved files are compressed with zstd, which is part of the standard library
from Python 3.14. On older Python versions, the `backports.zstd
<https://pypi.org/project/backports.zstd/>`_ package is installed instead.

Jupyter is not required, but most examples in this documentation are
notebooks and the package works well in them (objects display as tables,
plots appear inline).


CmdStan (optional)
------------------

The baseline estimation (:meth:`~PupilData.pupil_estimate_baseline`, see
:doc:`/user_guide/modeling`) uses the probabilistic programming language
`Stan <https://mc-stan.org/>`_ through :mod:`cmdstanpy`. Apart from the Python
package, it needs the CmdStan command-line tools, which are not installed
automatically. If you need them, install CmdStan as described in the
`CmdStanPy documentation
<https://mc-stan.org/cmdstanpy/installation.html>`_, e.g.:

.. code-block:: python

    import cmdstanpy
    cmdstanpy.install_cmdstan()

All other parts of :mod:`pypillometry` work without CmdStan.


Upgrading from version 1
------------------------

Version 2 is a major rewrite and scripts written for version 1 will not run
without changes. The main differences:

- The classes are now :class:`EyeData`, :class:`PupilData` and
  :class:`GazeData` (see :doc:`/user_guide/data_model`). They hold both eyes
  and several variables (pupil, x, y) at the same time.
- Most methods were renamed, e.g. ``blinks_detect()`` is now
  :meth:`~PupilData.pupil_blinks_detect` and ``lowpass_filter()`` is
  :meth:`~PupilData.pupil_lowpass_filter`.
- ERPDs are created with :meth:`~GenericEyeData.get_intervals` and
  :meth:`~GenericEyeData.get_segments` (see :doc:`/notebooks/erpds`).

Version 1 is still available on the `v1 branch
<https://github.com/ihrke/pypillometry/tree/v1>`_ and its documentation is
`here <https://ihrke.github.io/pypillometry/v1/html/index.html>`_. You can
install it with ``pip install "pypillometry<2"``.
