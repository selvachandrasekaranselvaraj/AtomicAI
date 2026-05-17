import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'AtomicAI'
copyright = '2024, Selva Chandrasekaran Selvaraj'
author = 'Selva Chandrasekaran Selvaraj'
release = '0.4.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_mock_imports = [
    'numba', 'dscribe', 'pymatgen', 'mpi4py',
    'AtomicAI.data.descriptor_cutoff',
    'AtomicAI.descriptors.force_descriptor_functions',
    'AtomicAI.descriptors.MultiSplit2b3b_index_ss',
    'AtomicAI.descriptors.prepare_vforce',
    'AtomicAI.descriptors.get_parameter',
    'AtomicAI.mlff.select_data_from_trajectory',
    'AtomicAI.mlff.LassoLarCV',
    'AtomicAI.mlff.plot_vt_r2',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
}
