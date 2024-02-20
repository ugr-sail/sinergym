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

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'sinergym'
copyright = '2023, J. Jiménez, J. Gómez, M. Molina, A. Manjavacas, A. Campoy'
author = 'J. Jiménez, J. Gómez, M.l Molina, A. Manjavacas, A. Campoy'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
    'sphinxcontrib.spelling',
    'sphinx_multiversion',
    'sphinx_multitoc_numbering',
    'IPython.sphinxext.ipython_console_highlighting',
    'nbsphinx',
    'nbsphinx_link']

autodoc_mock_imports = ['stable_baselines3', 'gym']

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# VERSIONING configuration
# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r'^.*0$'
# smv_tag_whitelist = None

# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = r'main'
# smv_branch_whitelist = None

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = None

# Pattern for released versions
smv_released_pattern = r'^tags/.+?0$'

# Format for versioned output directories inside the build directory
smv_outputdir_format = '{ref.name}'

# Determines whether remote or local git branches/tags are preferred if
# their output dirs conflict
smv_prefer_remote_refs = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Spelling word list white list.
spelling_word_list_filename = 'spelling_wordlist.txt'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = ['doc_theme.css']

# Modify icons
html_logo = '_static/logo-sidebar.png'
html_favicon = '_static/logo-sidebar.png'

# Change sidebar-logo background
html_theme_options = {'style_nav_header_background': '#a5beba',
                      'display_version': True,
                      }

# Enable global sidebar
html_sidebars = {'**': ['globaltoc.html',
                        'relations.html',
                        'sourcelink.html',
                        'searchbox.html', ]}

# disable nbsphinx errors to suppres imports checks not working
nbsphinx_allow_errors = True

# disable nbsphinx nodes execution (it fails to import sinergym)
# if a node is previously executed it will include the output
# but nbsphonx will not execute it if the output is missing.
nbsphinx_execute = 'never'
