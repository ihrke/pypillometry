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
- :mod:`pystan` 

It is useful to access :mod:`pypillometry` through Jupyter or Jupyter Notebook, so installing those packages is also useful but not necessary.

All requirements can be installed by running `pip install -r requirements.txt`.

Virtual environments
--------------------

It can sometimes be useful to install a new package in a new virtual environment using either `Python's virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ or `conda <https://docs.conda.io/en/latest/>`_. 


.. code-block:: bash

    $ conda create -n pypil python=3
    $ conda activate pypil
    $ conda install anaconda 

The ``anaconda`` package contains all the requirements except :mod:`pystan` which can be installed from `conda-forge <https://anaconda.org/conda-forge/pystan>`_

.. code-block:: bash

    $ conda install -c conda-forge pystan


Pystan 
------

Note that the installation of :mod:`pystan` may cause trouble on Windows-systems (you may need to install a compiler). Please follow the instructions on `the Pystan-webpage <https://pystan.readthedocs.io/en/latest/getting_started.html>`_ should you encounter any trouble.


Notes/Potential Problems
-------------------------

Under Linux, I encountered a problem where :mod:`pystan` crashed the `Jupyter kernel <https://jupyter.org/>`_.
To circumvent this issue, I needed to install :mod:`pystan` using

.. code-block:: bash

    $ pip install pystan
    $ conda install gcc_linux-64
    $ conda install gxx_linux-64

otherwise, there were random crashes of the jupyter kernel for some reason. 

On Mac OS X, I had some trouble getting the compiler to work with PyStan. See `this issue <https://github.com/stan-dev/pystan/issues/622#issuecomment-518825883>`_ for a solution that worked for me.

To enable interactive plotting widgets in jupyter notebook and jupyter lab, widgets need to be enabled in the notebook.

.. code-block:: bash

    $ conda install ipywidgets nodejs
    $ jupyter nbextension enable --py widgetsnbextension
    $ jupyter labextension install @jupyter-widgets/jupyterlab-manager

