Developers
==========

Building docs
-------------

A couple of non-standard sphinx extensions are being used:

~~~
['sphinx_math_dollar',  "m2r",'sphinx_autodoc_typehints', 'nbsphinx']
~~~

Installation: 

~~~
# jupyter notebooks
conda install -c conda-forge nbsphinx 
# math
conda install -c conda-forge sphinx-math-dollar
# markdown 2 rst (for README)
pip install m2r
# for the typehints
conda install -c conda-forge sphinx-autodoc-typehints
~~~

.. note::

    don't run `make clean`, it will delete the source-files from `docs`