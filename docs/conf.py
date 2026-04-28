# Configuration file for Sphinx documentation builder
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information
project = 'K-talysticFlow (KAST)'
copyright = '2025, Laboratory of Molecular Modeling (LMM-UEFS)'
author = 'Késsia Souza Santos'
release = '1.0.0'

# -- General configuration
extensions = [
    'myst_parser',           # substitui sphinx_md
    'sphinx.ext.mathjax',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst',           # myst processa os .md agora
}

# -- HTML output
html_theme = 'alabaster'
html_theme_options = {
    'logo': '',
    'description': 'Automated Deep Learning Pipeline for Molecular Bioactivity Prediction',
    'github_user': 'kelsouzs',
    'github_repo': 'KAST',
    'github_button': True,
    'github_type': 'star',
    'sidebar_width': '250px',
}

html_static_path = []
html_logo = None

# -- Read the Docs
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- MyST Markdown extensions
myst_enable_extensions = [
    "colon_fence",
    "tasklist",
    "dollarmath",
    "amsmath",
]

# -- Language
language = 'en'
