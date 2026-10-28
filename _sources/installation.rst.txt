Installation
============

Installing :mod:`pypillometry` and its dependencies is automated and can be done by running the following lines (on Mac OS X or Linux). 

.. code-block:: bash

    $ git clone https://github.com/ihrke/pypillometry.git
    $ cd pypillometry
    $ pip install .

:mod:`pypillometry` is on `PyPI <https://pypi.org/>`_ and released versions can be installed with `pip` (this will also install the dependencies automatically):

.. code-block:: bash

    $ pip install pypillometry

(`link to the PyPI project page <https://pypi.org/project/pypillometry/>`_).

It is also possible to install the developer's version directly from github using `pip`

.. code-block:: bash

    $ pip install git+https://github.com/ihrke/pypillometry.git


Requirements
------------

:mod:`pypillometry` requires Python 3.10 or newer and a range of standard numerical computing packages (all of which are listed in `pyproject.toml` and installed automatically)

- :mod:`numpy`, :mod:`scipy` and :mod:`matplotlib`
- :mod:`cmdstanpy` 

Eyelink's .edf files are read with the `eyelinkio <https://github.com/scott-huberty/eyelinkio>`_ package, which is installed as well.

Saved eyedata files are compressed with zstd, which is part of Python's standard library from Python 3.14 (on older versions, the `backports.zstd <https://pypi.org/project/backports.zstd/>`_ package is installed).

The interactive viewer uses :mod:`PyQt5` and :mod:`vispy`, which are installed by default.

For development, ``pip install -e ".[test]"`` installs the test dependencies as well.

It can be useful to access :mod:`pypillometry` through Jupyter or Jupyter Notebook, so installing those packages is also useful but not necessary.

Virtual environments
--------------------

It can sometimes be useful to install a new package in a new virtual environment using either `Python's virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ or `conda <https://docs.conda.io/en/latest/>`_. 


.. code-block:: bash

    $ conda create -n pypil python=3
    $ conda activate pypil
    $ conda install anaconda 
    $ pip install pypillometry

The ``anaconda`` package contains all the requirements except :mod:`cmdstanpy` which can be installed from `conda-forge <https://anaconda.org/conda-forge/pystan>`_

.. code-block:: bash

    $ conda install -c conda-forge cmdstanpy


CmdStanPy
---------

:mod:`pypillometry` uses :mod:`cmdstanpy` to interface with the `Stan <https://mc-stan.org/>`_ probabilistic programming language. :mod:`cmdstanpy` is a Python interface to the `CmdStan <https://mc-stan.org/users/interfaces/cmdstan>`_ command-line interface to Stan. 

Please refer to the `CmdStanPy documentation <https://mc-stan.org/cmdstanpy/installation.html>`_ for more information on how to install and use it.