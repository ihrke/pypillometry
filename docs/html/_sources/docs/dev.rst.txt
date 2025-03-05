.. currentmodule:: pypillometry.pupildata

Developers
==========

.. image:: https://travis-ci.com/ihrke/pypillometry.svg?branch=master
    :target: https://travis-ci.com/ihrke/pypillometry

This package is developed and maintained by the `Cognitive Neuroscience Research Group <http://uit.no/research/cognitive-neuroscience>`_ at the `University of Tromsø <http://uit.no>`_. We encourage everyone to become a member of the team and to contribute to the package's development, either by `reporting bugs  <https://github.com/ihrke/pypillometry/issues>`_, `providing enhancements <https://github.com/ihrke/pypillometry/pulls>`_ or otherwise.

Major versions
--------------

Pypillometry switched to version 2 in March 2025. This introduces breaking changes to the API. 


How to contribute
-----------------

Development of this package happens on GitHub: https://github.com/ihrke/pypillometry

In order to contribute, please use

- `github's issue tracker <https://github.com/ihrke/pypillometry/issues>`_ for reporting bugs or proposing enhancements
- `github's pull-request (PR) feature <https://github.com/ihrke/pypillometry/pulls>`_ for contributing to the package and/or documentation

If you are unfamiliar with Github's system, you can read up about `collaborating with issues and pull-requests on Github <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_.


Coding style/package architecture
---------------------------------

We chose to put most functionality directly below :mod:`pypillometry`'s main class :class:`PupilData`. This is mostly for convenience,
as users can easily find and chain commands together (see the following page more on this: :ref:`Pipeline-based processing in pypillometry </docs/pipes.ipynb>`).

The implementation of the functionality, however, is coded in submodules using "stateless" functions that take :mod:`numpy`-arrays as arguments. So, in order to create a new function that changes a :class:`PupilData`-object, you will first want to implement a function working purely on :mod:`numpy`-arrays and then create a thin wrapper function below :class:`PupilData` that calls this low-level function with the correct arguments.

Unit-testing
------------

Every newly developed function should be provided with `Unit-testing functions <https://en.wikipedia.org/wiki/Unit_testing>`_. As a minimum, the function should be called with some reasonable arguments to ensure that the function does not crash. Better yet, the results of the functions should be tested against the desired output within these test functions. All unit-tests are located in `/pypillometry/tests/` and are run automatically for every commit to Github using `Travis CI <https://travis-ci.com/ihrke/pypillometry>`_. 

Building the packages documentation
-----------------------------------

This is only necessary when extending `pypillometry`'s documentation (i.e., updating the website at https://ihrke.github.io/pypillometry).

The package uses the `Sphinx documentation generator <https://www.sphinx-doc.org/>`_ to create this website. All source-files are located under `/docs/` (both `.rst` and `.ipynb` files are being used). In addition, the API-documentation is placed within each function or classes' docstring.

To compile the documentation, sphinx must be installed. In addition, a couple of non-standard sphinx extensions are being used:

.. code-block:: python

    ['sphinx_math_dollar',  "sphinx-mdinclude",'sphinx_autodoc_typehints', 'nbsphinx']


These can be installed by using the following commands:


.. code-block:: bash

    # jupyter notebooks
    conda install -c conda-forge nbsphinx 
    # math
    conda install -c conda-forge sphinx-math-dollar
    # markdown 2 rst (for README)
    pip install sphinx-mdinclude
    # for the typehints
    conda install -c conda-forge sphinx-autodoc-typehints

Finally, the documentation can be created by running

.. code-block:: bash

    make html
    
in the packages' root-directory. 

.. warning::

    don't run `make clean`, it will delete the source-files from `docs`


Creating Releases
-----------------


A release should be created whenever crucial bugs have been fixed or new functionality has been added.
:mod:`pypillometry` goes with version numbers `x.y.z`. Increment `z` for bug-fixes, `y` for new features and
`x` for releases that break backward-compatibility.

.. note::

    the process of uploading to PyPI has been automatized using Github actions; when a
    release is created on `Github-releases <https://github.com/ihrke/pypillometry/releases>`_, the file 
    `VERSION` is updated with the most recent tag
    (must be `PEP-440 <https://www.python.org/dev/peps/pep-0440/#version-scheme>`_-compliant)
    and a `PyPI <https://pypi.org/>`_ release is being issued

   

