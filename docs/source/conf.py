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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "joliGEN"
copyright = "2023, Jolibrain"
author = "Jolibrain"

html_title = "joliGEN"

# The full version, including alpha/beta/rc tags
release = "0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

import os
import sys

sys.path.insert(0, os.path.abspath("."))
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["_static/css/custom.css"]
html_favicon = "https://www.jolibrain.com/static/img/logo.png"
html_logo = "https://raw.githubusercontent.com/jolibrain/joliGEN/57088323771c1bb0a9540e567f9fb55c3c416c4c/imgs/joligen.svg"
html_theme_option = {"logo_only": True, "style_nav_header_background": "#cd0b0b"}
html_context = {
    "display_github": True,  # Integrate Gitlab
    "github_host": "github.com",
    "github_user": "jolibrain",  # Username
    "github_repo": "joliGEN",  # Repo name
    "github_version": "feat_jolidoc",  # Branch
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
}
