# Configuration file for Sphinx documentation builder
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information
project = 'K-talysticFlow (KAST)'
copyright = '2025, Laboratory of Molecular Modeling (LMM-UEFS)'
author = 'Késsia Souza Santos'
release = '1.0.0'

# -- Extensions
extensions = [
    'myst_parser',
    'sphinx.ext.mathjax',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst',
}

# -- MyST
myst_enable_extensions = [
    "colon_fence",
    "tasklist",
    "dollarmath",
    "amsmath",
]

# -- HTML
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': True,
    'navigation_depth': 4,
    'collapse_navigation': True,   # sem setinhas de expandir
    'sticky_navigation': True,
    'prev_next_buttons_location': 'both',
    'style_nav_header_background': "#ffffff",
}

html_logo = '_static/kast_logo.png'
html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Read the Docs
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
language = 'en'