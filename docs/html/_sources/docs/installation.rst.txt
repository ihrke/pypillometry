Installation
============

.. code-block:: bash

    git clone https://github.com/ihrke/pypillometry.git
    cd pypillometry
    pip install -r requirements.txt
    python setup.py install


Requirements
------------

:mod:`pypillometry` requires Python3 and a range of standard numerical computing packages (all of which listed in the file `requirements.txt`)

- :mod:`numpy`, :mod:`scipy` and :mod:`matplotlib`
- :mod:`pystan` 

It is useful to access :mod:`pypillometry` through Jupyter or Jupyter Notebook, so installing those packages is also useful but not necessary.

All requirements can be installed by running `pip install -r requirements.txt`.

Notes
^^^^^


To install the requirements, either use `pip` as described above, manually install all the packages, or use ``conda``.

.. code-block:: bash

    $ conda create -n pypil python=3
    $ conda activate pypil
    $ conda install anaconda 

The ``anaconda`` package contains all the requirements except :mod:`pystan`.

To install :mod:`pystan`, I needed to do

.. code-block:: bash

    pip install pystan
    conda install gcc_linux-64
    conda install gxx_linux-64

otherwise, there were random crashes of the jupyter kernel for some reason. 

To enable interactive plotting widgets in jupyter notebook and jupyter lab, widgets need to be enabled in the notebook.

.. code-block:: bash

    $ conda install ipywidgets nodejs
    $ jupyter nbextension enable --py widgetsnbextension
    $ jupyter labextension install @jupyter-widgets/jupyterlab-manager

