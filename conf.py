# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'pypillometry'
copyright = '2020, Matthias Mittner'
author = 'Matthias Mittner'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions=['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 
            'sphinx_math_dollar', 'sphinx.ext.mathjax', 'sphinx_autodoc_typehints', "m2r",
            'sphinx.ext.intersphinx', 'sphinx.ext.autosummary', "nbsphinx"]

nbsphinx_epilog = """

.. raw:: html

    <div class="admonition note">
    This file was created from the following Jupyter-notebook: <a href="https://github.com/ihrke/pypillometry/tree/master/{{ env.doc2path(env.docname, base=None) }}">{{ env.doc2path(env.docname, base=None) }}</a>
    <br>
    Interactive version:
    <a href="https://mybinder.org/v2/gh/ihrke/pypillometry/master?filepath={{ env.doc2path(env.docname, base=None)|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>
    </div>

"""

nbsphinx_prolog=nbsphinx_epilog

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', 'notebooks', "**nbsphinx", "docs/html"]


# Add mappings
intersphinx_mapping = {
    'urllib3': ('http://urllib3.readthedocs.org/en/latest', None),
    'python': ('http://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('http://matplotlib.sourceforge.net', None),
    'pystan': ('https://pystan.readthedocs.io/en/latest/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------
pygments_style = 'sphinx'
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'nature'
#alabaster'

html_logo = "logo/pypillometry_logo_200x200.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
