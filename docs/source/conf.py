"""
Sphinx configuration file for tdspy documentation.

This module configures Sphinx documentation generation including:
- Path setup for the package
- Project metadata
- Extensions (autodoc, Napoleon, type hints, LaTeX)
- HTML theme and output settings
- Autodoc and Napoleon options
"""

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../../src')) # package root

# -- Project information -----------------------------------------------------

project = 'tdspy'       # Your package name
author = 'Adam Peichl'        # Your name
release = '0.1.0'           # Version of your package

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',          # Automatically document docstrings
    'sphinx.ext.napoleon',         # Google / NumPy style docstrings
    'sphinx_autodoc_typehints',    # Include type hints in docs
    'sphinx.ext.mathjax',          # Render LaTeX math
    'sphinx.ext.todo',
    'sphinx_gallery.gen_gallery',  # This is for examples
]

templates_path = ['_templates']
exclude_patterns = []

# -- HTML output -------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

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
    'examples_dirs': [os.path.abspath('../../examples')],
    'gallery_dirs': 'auto_examples',
    'filename_pattern': r'.*\.py$',
    # 'filename_pattern': r'^(example01\.py|example02\.py)$',
    # 'ignore_pattern': r'^test_.*\.py$',
    'plot_gallery': True,           # render plots
    'backreferences_dir': None,     # optional
    'run_stale_examples': True,     # force re-execution
}
