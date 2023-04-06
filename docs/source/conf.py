# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'AlphaTrade'
copyright = '2023, Kang'
author = 'Kang'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "sphinx.ext.napoleon",
    "myst_parser",
    "myst_parser.sphinx_",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

autosummary_generate = True
autosummary_imported_members = True








"""Sphinx documentation configuration."""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import os
# from importlib import metadata

# project = "gym_exchange"
# copyright = "2019-2022, Center for Human-Compatible AI"  # noqa: A001
# author = "Center for Human-Compatible AI"

# # The full version, including alpha/beta/rc tags
# version = metadata.version("gym_exchange")


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_github_changelog",
    "sphinx.ext.doctest",
    "myst_nb",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autosummary_generate = True

nb_execution_mode = os.getenv("NB_EXECUTION_MODE", "cache")
nb_execution_timeout = 120
nb_merge_streams = True
nb_output_stderr = "remove"

# skip_doctests is checked in our :skipif: directives in doctest examples
doctest_global_setup = """
import os

skip_doctests = os.getenv("SKIP_DOCTEST")
"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "latest"
html_theme = "furo"
html_title = "AlphaTrade"
html_theme_options = {
    "source_repository": "https://github.com/KangOxford/AlphaTrade",
    "source_branch": "master",
    "source_directory": "docs",
    "light_css_variables": {
        "sidebar-item-font-size": "85%",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/baselines_theme.css"]

# -- Customization -----------------------------------------------------------


def no_namedtuple_attrib_docstring(app, what, name, obj, options, lines):
    """Remove redundant documentation in named tuples.

    Worksaround https://github.com/sphinx-doc/sphinx/issues/7353 -- adapted from
    https://chrisdown.name/2015/09/20/removing-namedtuple-docstrings-from-sphinx.html
    """  # noqa: DAR101
    is_namedtuple_docstring = 1 <= len(lines) <= 2 and lines[0].startswith(
        "Alias for field number",
    )
    if is_namedtuple_docstring:
        # We don't return, so we need to purge in-place
        del lines[:]


def setup(app):
    app.connect(
        "autodoc-process-docstring",
        no_namedtuple_attrib_docstring,
    )

# ==============================================================================
pygments_style = "sphinx"
