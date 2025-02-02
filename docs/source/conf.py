import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

project   = 'DynaMapp'
copyright = '2025, Wissem CHIHA'
author    = 'Wissem CHIHA'
release   = '1.0.0'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'myst_parser',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo'
]

myst_enable_extensions = [
    "dollarmath",  
    "deflist",     
]
autodoc_mock_imports = ['jax', 'jax.numpy','seaborn', 'seaborn', 'numpy', 'matplotlib', 'pandas', 'scipy']
autodoc_default_flags = ['members', 'private-members']
autodoc_member_order = 'bysource'
templates_path = ['build/html/_templates']
exclude_patterns = ['Thumbs.db', '.DS_Store']
language = 'python'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
todo_include_todos = True
