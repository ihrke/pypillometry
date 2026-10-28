# Configuration file for the Sphinx documentation builder.
#
# Build from the repository root with `make html` (output in _build/html).
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pypillometry  # noqa: E402

# -- Project information -----------------------------------------------------

project = "pypillometry"
copyright = "2020-2026, Matthias Mittner"
author = "Matthias Mittner"
release = (project_root / "VERSION").read_text().strip()
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",  # writes .nojekyll (needed for _static)
    "matplotlib.sphinxext.plot_directive",  # plots in docstrings and pages
    "sphinx_autodoc_typehints",
    "sphinx_math_dollar",
    "sphinxcontrib.mermaid",
    "sphinx_design",       # cards, grids, tabs
    "sphinx_copybutton",   # copy button on code blocks
    "myst_parser",         # markdown pages
    "nbsphinx",            # jupyter notebooks
    "numpydoc",            # numpy-style docstrings
]

myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints",
    "**nbsphinx", "html", "src", "v1",
]

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = True
autodoc_default_options = {
    "member-order": "groupwise",
}
# members are documented by autodoc (:members:); numpydoc's additional
# "Methods"/"Attributes" tables duplicate them and cause "more than one
# target found" warnings
numpydoc_show_class_members = False
numpydoc_validation_checks = set()

# one entry per method in the page's "On this page" sidebar
toc_object_entries = True
toc_object_entries_show_parents = "hide"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Plots -------------------------------------------------------------------

# `.. plot::` blocks (e.g., in the Examples of the plotting functions) are
# run during the build; they use the example data shipped with the package
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 100)]
plot_pre_code = """
import numpy as np
import matplotlib.pyplot as plt
import pypillometry as pp
pp.logging_set_level("WARNING")
"""

# -- Notebooks ---------------------------------------------------------------

# notebooks are committed with their outputs and never executed during the
# build (see the development section of the docs)
nbsphinx_execute = "never"

_nb_path = "docs/{{ env.doc2path(env.docname, base=None) }}"
nbsphinx_prolog = f"""
.. raw:: html

    <div class="pp-notebook-banner">
      <span>This page was generated from a Jupyter notebook.</span>
      <a href="https://github.com/ihrke/pypillometry/blob/master/{_nb_path}">View on GitHub</a>
      <a href="https://mybinder.org/v2/gh/ihrke/pypillometry/master?filepath={_nb_path}">
        <img alt="Binder badge" src="https://mybinder.org/badge_logo.svg"
             style="vertical-align:text-bottom">
      </a>
    </div>
"""

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"pypillometry {release}"
html_logo = "../logo/pypillometry_logo_200x200.png"
html_favicon = "../logo/pypillometry_logo_200x200.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_context = {
    "github_user": "ihrke",
    "github_repo": "pypillometry",
    "github_version": "master",
    "doc_path": "docs",
    "default_mode": "light",
}

html_theme_options = {
    "logo": {"text": "pypillometry"},
    "github_url": "https://github.com/ihrke/pypillometry",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pypillometry/",
            "icon": "fa-brands fa-python",
        },
    ],
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_prev_next": True,
    "use_edit_page_button": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "announcement": (
        "pypillometry v2 is a major rewrite and not compatible with v1. "
        "The <a href='https://ihrke.github.io/pypillometry/v1/html/index.html'>"
        "v1 documentation</a> is still available."
    ),
}

# the landing page has no sidebar
html_sidebars = {
    "index": [],
}

pygments_style = "sphinx"
pygments_dark_style = "monokai"


# -- Cross-references to internal module paths --------------------------------
#
# Docstrings often link to the module where an object is defined, e.g.
# :func:`pypillometry.signal.pupil.pupil_response`, but the API reference
# documents everything under its public name (pypillometry.pupil_response).
# This hook redirects such links (and bare method names) to the public name.

_PUBLIC_CLASSES = ["GenericEyeData", "PupilData", "GazeData", "EyeData"]


def _resolve_public_name(app, env, node, contnode):
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    parts = target.split(".")
    candidates = []
    if target.startswith("pypillometry."):
        candidates += [f"pypillometry.{parts[-1]}",
                       f"pypillometry.{'.'.join(parts[-2:])}"]
    elif len(parts) == 1:
        cls = node.get("py:class")
        if cls:
            candidates.append(f"pypillometry.{cls.split('.')[-1]}.{target}")
        candidates += [f"pypillometry.{target}"]
        candidates += [f"pypillometry.{c}.{target}" for c in _PUBLIC_CLASSES]
    py = env.get_domain("py")
    for cand in candidates:
        if cand in py.objects:
            return py.resolve_xref(env, node["refdoc"], app.builder,
                                   node["reftype"], cand, node, contnode)
    return None


def setup(app):
    app.connect("missing-reference", _resolve_public_name)
