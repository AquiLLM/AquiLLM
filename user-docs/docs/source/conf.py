# -*- coding: utf-8 -*-
#
# AquiLLM documentation build configuration file
#

import os
import sys

# -- Path setup --------------------------------------------------------------
# AquiLLM repo layout: user-docs/docs/source -> ../../../aquillm
AQUILLM_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'aquillm'))
sys.path.insert(0, AQUILLM_PACKAGE_ROOT)
sys.path.insert(0, os.path.abspath('.'))

# Importing lib.skills pulls lib.llm.providers; mock LLM SDK deps for autodoc only.
autodoc_mock_imports = [
    'tiktoken',
    'openai',
    'anthropic',
    'google',
    'google.genai',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': False,
}

# -- Project information -----------------------------------------------------
project = 'AquiLLM'
copyright = '2025, AquiLLM Team'
author = 'AquiLLM Team'
release = '1.0.0'
version = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Add any paths that contain templates here
templates_path = ['_templates']

# The suffix(es) of source filenames
source_suffix = '.rst'

# The master toctree document
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# Set in CI so CSS/JS resolve on GitHub Pages project sites (owner.github.io/repo/).
html_baseurl = os.environ.get('SPHINX_HTML_BASEURL', '')

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': True,
    'version_selector': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
}

html_logo = '_static/AquiLLMLogo.png'
html_favicon = '_static/aquila-small-dark.ico'

# Add any paths that contain custom static files
html_static_path = ['_static']

html_css_files = [
    'css/aquillm_custom.css',
]

# Title
html_title = 'AquiLLM Documentation'
html_short_title = 'AquiLLM'

# Sidebar
html_sidebars = {
    '**': [
        'globaltoc.html',
        'searchbox.html',
    ]
}

# Output
htmlhelp_basename = 'AquiLLMdoc'
