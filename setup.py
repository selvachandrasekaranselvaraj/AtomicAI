from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
        'pip   >= 22.1.2',
        'wheel >= 0.37.1',
        #'twine >= 4.0.1',
        #'ase   >= 3.22.1',
        ]

setup(
    name='AtomicAI',
    version='0.3.0',
    token='selva',
    author='Selva Chandrasekaran Selvaraj',
    author_email='selvachandrasekar.s@gmail.com',
    description='Processing and visualization of atomic coordinates \n Featurizing atomic structures',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://sites.google.com/view/selvas',

    project_urls = {
        "Bug Tracker": "https://github.com/SelvaGith/AtomicAI/issues",
        "Source": "https://github.com/SelvaGith/AtomicAI",
        "Twitter": "https://twitter.com/selva_odc",
        "LinkedIn" : "https://www.linkedin.com/in/SelvaCS/?original_referer=https%3A%2F%2Fwww.google.com%2F",
        "Scopus": "https://www.scopus.com/authid/detail.uri?authorId=57225319817",
        "Scholar" : "https://scholar.google.com/citations?user=vNozeNYAAAAJ&hl=en",

    },
    license='MIT',


    #packages=['AtomicAI'], #find_packages(),
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",

    ],
    python_requires='>=3.9',
    install_requires= install_requires, #open('requirements.txt').read(), 
    entry_points={'console_scripts': [
        'cq2vasp=AtomicAI.tools.cq2vasp:cq2vasp',
        'xyz2vasp=AtomicAI.tools.xyz2vasp:xyz2vasp',
        'vasp2cif=AtomicAI.tools.vasp2cif:vasp2cif',
        'vasp2xyz=AtomicAI.tools.vasp2xyz:vasp2xyz',
        'vasp2cq=AtomicAI.tools.vasp2cq:vasp2cq',
        'cif2cq=AtomicAI.tools.cif2cq:cif2cq',
        'cq2cif=AtomicAI.tools.cq2cif:cq2cif',

        'rdf=AtomicAI.tools.rdf:RDF',
        'structure_analysis=AtomicAI.tools.structure_analysis:structure_analysis',

        'plot_rdf_data=AtomicAI.plots.plot_rdf_data:read_RDF_data',
        'plot_md_stats=AtomicAI.plots.plot_md_stats:plot_md_stats',
        'plot_clusters=AtomicAI.plots.plot_clusters:plot_clusters',

        'laaf=AtomicAI.descriptors.calculate_laaf_des:calculate_laaf_des',
        'generate_descriptors=AtomicAI.descriptors.calculate_descriptors:calculate_descriptors',

        'pca=AtomicAI.dim_reduction.pca:pca',
        'lpp=AtomicAI.dim_reduction.lpp:lpp',
        'dim_reduction=AtomicAI.dim_reduction.dim_reduction:dim_reduction',
        'dim_reduction_mpi=AtomicAI.dim_reduction.dim_reduction_mpi:dim_reduction_mpi',
        'optimize_tslpp_hyperparameters_without_prediction=AtomicAI.dim_reduction.perform_tslpp_for_hyperparameters:perform_tslpp_hyperparameters',
        'optimize_tslpp_hyperparameters_with_prediction=AtomicAI.dim_reduction.perform_tslpp_for_hyperparameters_with_prediction:perform_tslpp_hyperparameters',
        'predict_tslpp=AtomicAI.dim_reduction.perform_tslpp_predict:perform_tslpp_predict'



                                    ]},
)
