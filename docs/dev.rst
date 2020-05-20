Developers
==========

Contibuting
-----------

Development of this package happens on GitHub: https://github.com/ihrke/pypillometry

Issue-tracker: https://github.com/ihrke/pypillometry/issues
PR's welcome: https://github.com/ihrke/pypillometry/pulls

Building docs
-------------

A couple of non-standard sphinx extensions are being used:

.. code-block:: python

    ['sphinx_math_dollar',  "m2r",'sphinx_autodoc_typehints', 'nbsphinx']


Installation: 


.. code-block:: bash

    # jupyter notebooks
    conda install -c conda-forge nbsphinx 
    # math
    conda install -c conda-forge sphinx-math-dollar
    # markdown 2 rst (for README)
    pip install m2r
    # for the typehints
    conda install -c conda-forge sphinx-autodoc-typehints

.. warning::

    don't run `make clean`, it will delete the source-files from `docs`