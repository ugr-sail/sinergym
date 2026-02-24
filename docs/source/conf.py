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

import mock
from packaging.version import InvalidVersion, Version

sys.path.insert(0, os.path.abspath('./../..'))

# -- sphinx-multiversion compatibility ---------------------------------------
#
# When building older tags with sphinx-multiversion, the codebase and its runtime
# dependencies may not be compatible with the Python version used by the docs
# job (e.g. legacy tags depending on `gym` and `pkg_resources`).
#
# In those cases, importing the package for autosummary/autodoc can fail and
# break the whole multiversion build. We therefore disable API stub generation
# and exclude API reference pages for non-main versions.


def _get_smv_current_version_from_argv() -> str | None:
    # sphinx-multiversion runs sphinx with overrides like:
    #   -D smv_current_version=v1.4.0
    # so we can detect the current ref from sys.argv.
    for i, arg in enumerate(sys.argv):
        if arg == '-D' and i + 1 < len(sys.argv):
            maybe_kv = sys.argv[i + 1]
            if maybe_kv.startswith('smv_current_version='):
                return maybe_kv.split('=', 1)[1]
    return None


smv_current_version = _get_smv_current_version_from_argv() or "main"


def semver_sort(items, reverse: bool = True):
    """Sort sphinx-multiversion Version objects using semantic version order.

    This prevents lexicographic ordering issues like v3.10.0 appearing between
    v3.1.0 and v3.2.0.
    """

    def _key(item):
        name = getattr(item, "name", str(item))
        normalized = name[1:] if name.startswith("v") else name
        try:
            return Version(normalized)
        except InvalidVersion:
            # Put non-semver refs (e.g. 'main') at the end.
            return Version("0")

    return sorted(list(items), key=_key, reverse=reverse)


def _add_jinja_filters(app) -> None:  # pragma: no cover
    # Not all builders expose a Jinja templates environment (e.g. SpellingBuilder).
    templates = getattr(app.builder, "templates", None)
    env = getattr(templates, "environment", None) if templates is not None else None
    if env is None:
        return
    env.filters["semver_sort"] = semver_sort


def setup(app):  # pragma: no cover
    app.connect("builder-inited", _add_jinja_filters)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


# -- Project information -----------------------------------------------------

project = 'Sinergym'
copyright = '2026, J. Jiménez, J. Gómez, M. Molina, A. Manjavacas, A. Campoy'
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
    'nbsphinx_link',
]

autodoc_mock_imports = [
    'stable_baselines3',
    'wandb',
    'gym',
    'opyplus',
    'gcloud',
    'googleapiclient',
    'oauth2client',
    'google',
    'google.cloud',
    'pyenergyplus',
]
for module in ['gymnasium.wrappers.normalize']:
    sys.modules[module] = mock.MagicMock()

nbsphinx_custom_formats = {}

# If we're building a non-main version via sphinx-multiversion, avoid importing
# the package to generate autosummary stubs (it may fail for legacy tags).
autosummary_generate = (
    False if (smv_current_version and smv_current_version != 'main') else True
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# VERSIONING configuration
# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r'^v\d+\.\d+\.0$'
# smv_tag_whitelist = None

# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = r'main'
# smv_branch_whitelist = None

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = None

# Pattern for released versions
smv_released_pattern = r'^tags/v\d+\.\d+\.0$'

# Format for versioned output directories inside the build directory
smv_outputdir_format = '{ref.name}'

# Determines whether remote or local git branches/tags are preferred if
# their output dirs conflict
smv_prefer_remote_refs = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
if smv_current_version and smv_current_version != 'main':
    # Keep narrative docs for legacy versions, but skip API pages that rely on
    # importing the Python package (often incompatible across major versions).
    exclude_patterns.extend(
        [
            "pages/API-reference.rst",
            "pages/modules/**",
        ]
    )

# Spelling word list white list.
spelling_word_list_filename = 'spelling_wordlist.txt'


# -- Options for HTML output -------------------------------------------------

# Warnings
suppress_warnings = ["config.cache"]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = ['doc_theme.css', 'github_style.css']

# Modify icons
html_logo = '_static/logo-sidebar.png'
html_favicon = '_static/logo-sidebar.png'

# Change sidebar-logo background
html_theme_options = {
    'style_nav_header_background': '#a5beba',
}

html_context = {
    "display_github": True,
    "github_user": "ugr-sail",
    "github_repo": "sinergym",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# Enable global sidebar
html_sidebars = {
    '**': [
        'versions.html',
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ],
}

# disable nbsphinx errors to suppress import checks not working
nbsphinx_allow_errors = True

# disable nbsphinx nodes execution (it fails to import sinergym)
# if a node is previously executed it will include the output
# but nbsphinx will not execute it if the output is missing.
nbsphinx_execute = 'never'
