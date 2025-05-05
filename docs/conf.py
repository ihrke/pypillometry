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
from pathlib import Path

#sys.path.insert(0, str(Path('.').resolve()))


# -- Project information -----------------------------------------------------

project = 'pypillometry'
copyright = '2020, Matthias Mittner'
author = 'Matthias Mittner'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions=['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 
            "sphinx.ext.doctest", 'sphinx.ext.viewcode', ## enable doctests
            "sphinxcontrib.mermaid", ## enable mermaid diagrams
            'sphinx_math_dollar', 'sphinx.ext.mathjax', 'sphinx_autodoc_typehints', 
            #"sphinx_mdinclude", ## include markdown files
            "myst_parser", ## enable myst-parser
            'sphinx.ext.intersphinx', 'sphinx.ext.autosummary', 
            "nbsphinx", ## enable jupyter notebooks
            "numpydoc" ## enable numpy-style docstrings
            ]

myst_enable_extensions = ["colon_fence"]

nbsphinx_epilog = """

.. raw:: html

    <div class="admonition note">
    This file was created from the following Jupyter-notebook: <a href="https://github.com/ihrke/pypillometry/tree/master/{{ env.doc2path(env.docname, base=None) }}">{{ env.doc2path(env.docname, base=None) }}</a>
    <br>
    Interactive version:
    <a href="https://mybinder.org/v2/gh/ihrke/pypillometry/master?filepath={{ env.doc2path(env.docname, base=None)|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>
    </div>

"""

# do not create TOC entries for each function/class
toc_object_entries = False

nbsphinx_prolog=nbsphinx_epilog

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', 'notebooks', "**nbsphinx", "docs/html", "src", "docs/v1"]


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

html_logo = "../logo/pypillometry_logo_200x200.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


_GITHUB_ADMONITIONS = {
    "> [!NOTE]": "note",
    "> [!TIP]": "tip",
    "> [!IMPORTANT]": "important",
    "> [!WARNING]": "warning",
    "> [!CAUTION]": "caution",
}

def run_convert_github_admonitions_to_rst(app, relative_path, parent_docname, lines):
    print("HOOK:", relative_path, parent_docname)
    # loop through lines, replace github admonitions
    for i, orig_line in enumerate(lines):
        orig_line_splits = orig_line.split("\n")
        replacing = False
        for j, line in enumerate(orig_line_splits):
            # look for admonition key
            for admonition_key in _GITHUB_ADMONITIONS:
                if admonition_key in line:
                    line = line.replace(admonition_key, ":::{" + _GITHUB_ADMONITIONS[admonition_key] + "}\n")
                    # start replacing quotes in subsequent lines
                    replacing = True
                    break
            else:
                # replace indent to match directive
                if replacing and "> " in line:
                    line = line.replace("> ", "  ")
                elif replacing:
                    # missing "> ", so stop replacing and terminate directive
                    line = f"\n:::\n{line}"
                    replacing = False
            # swap line back in splits
            orig_line_splits[j] = line
        # swap line back in original
        lines[i] = "\n".join(orig_line_splits)

def setup(app):
    app.connect('include-read', run_convert_github_admonitions_to_rst)