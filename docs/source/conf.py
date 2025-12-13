"""
Sphinx configuration file for tdspy documentation.

This module configures Sphinx documentation generation including:
- Path setup for the package
- Project metadata
- Extensions (autodoc, Napoleon, type hints, LaTeX)
- HTML theme and output settings
- Autodoc and Napoleon options
"""

import os
import sys

from sphinx_gallery.sorting import FileNameSortKey


# -- Path setup --------------------------------------------------------------

PACKAGE_ROOT = os.path.abspath('../..')
SRC_PATH = os.path.join(PACKAGE_ROOT, 'src')
EXAMPLES_PATH = os.path.join(PACKAGE_ROOT, 'examples')
sys.path.insert(0, SRC_PATH) # package root

# -- Project information -----------------------------------------------------

project = 'tdspy'       # Your package name
author = 'Adam Peichl'        # Your name
copyright = '2025, Adam Peichl'  # Copyright info
release = '0.1.0'           # Version of your package


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',          # Automatically document docstrings
    'sphinx.ext.napoleon',         # Google / NumPy style docstrings
    "sphinx.ext.autodoc.typehints",
    # 'sphinx_autodoc_typehints',    # Include type hints in docs
    'sphinx.ext.mathjax',          # Render LaTeX math
    'sphinx.ext.todo',
    'sphinx_gallery.gen_gallery',  # This is for examples
    "sphinxcontrib.bibtex",        # LaTeX like citations
]

templates_path = ['_templates']
exclude_patterns = []

# -- HTML output -------------------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
html_logo = '_static/tdspy_logo.png'  # Path to your logo
html_favicon = '_static/tdspy_favicon.ico'  # Path to your favicon

# -- Autodoc settings --------------------------------------------------------

autodoc_member_order = 'bysource'  # Order members as in the source code
autodoc_typehints = 'description'  # Show type hints in the description rather than signature

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Example Gallery settings ------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": [EXAMPLES_PATH],
    'gallery_dirs': ['auto_examples'],
    "nested_sections": False,
    'filename_pattern': r'.*\.py$', # include all .py files
    'within_subsection_order': FileNameSortKey,
    'plot_gallery': True,           # render plots
    'backreferences_dir': None,     # optional
    'run_stale_examples': False,     # force re-execution set True
    'download_all_examples': False,  # optional, button for download all .zip
    'matplotlib_animations': (True, 'html5'), # (True, 'mp4') - to save .rst size
}

# -- Bibtex settings ---------------------------------------------------------

bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"
bibtex_scan_rst = True
bibtex_overwrite_cache = True