Installation
============

Installing :mod:`pypillometry` and its dependencies is automated and can be done by running the following lines (on Mac OS X or Linux). 

.. code-block:: bash

    $ git clone https://github.com/ihrke/pypillometry.git
    $ cd pypillometry
    $ pip install -r requirements.txt
    $ python setup.py install

:mod:`pypillometry` is on `PyPI <https://pypi.org/>`_ and released versions can be installed with `pip` (this will also install the dependencies automatically):

.. code-block:: bash

    $ pip install pypillometry

(`link to the PyPI project page <https://pypi.org/project/pypillometry/>`_).

It is also possible to install the developer's version directly from github using `pip`

.. code-block:: bash

    $ pip install git+https://github.com/ihrke/pypillometry.git


Requirements
------------

:mod:`pypillometry` requires Python3 and a range of standard numerical computing packages (all of which listed in the file `requirements.txt`)

- :mod:`numpy`, :mod:`scipy` and :mod:`matplotlib`
- :mod:`cmdstanpy` 

It is useful to access :mod:`pypillometry` through Jupyter or Jupyter Notebook, so installing those packages is also useful but not necessary.

All requirements can be installed by running `pip install -r requirements.txt`.

Virtual environments
--------------------

It can sometimes be useful to install a new package in a new virtual environment using either `Python's virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ or `conda <https://docs.conda.io/en/latest/>`_. 


.. code-block:: bash

    $ conda create -n pypil python=3
    $ conda activate pypil
    $ conda install anaconda 

The ``anaconda`` package contains all the requirements except :mod:`cmdstanpy` which can be installed from `conda-forge <https://anaconda.org/conda-forge/pystan>`_

.. code-block:: bash

    $ conda install -c conda-forge cmdstanpy


CmdStanPy
---------

:mod:`pypillometry` uses :mod:`cmdstanpy` to interface with the `Stan <https://mc-stan.org/>`_ probabilistic programming language. :mod:`cmdstanpy` is a Python interface to the `CmdStan <https://mc-stan.org/users/interfaces/cmdstan>`_ command-line interface to Stan. 

Please refer to the `CmdStanPy documentation <https://mc-stan.org/cmdstanpy/installation.html>`_ for more information on how to install and use it.