# AtomicAI

AtomicAI is a Python-based package for building machine learning models in the field of computational materials science. It can be used for machine learning force field generation, atomic structure classifications based on temperature, defects, phase, and more.

## Features

### Analysis
- Processing and visualization of atomic coordinates in various file formats (CIF, VASP, XYZ, Conquest, etc.)
- Utilizes 'ase' library for reading and writing structure files

### Data Processing
- Classification of atoms in trajectory files
- Structure analysis tools

### Machine Learning
- Featurization of atomic structures
- Dimension reduction techniques (PCA, LPP, TsLPP, TsNE, UMAP)
- Machine Learning Force Field (MLFF) generation

### Clustering and Visualization
- Various clustering methods
- Data plotting and visualization tools

## Installation

Install AtomicAI using pip:

```
pip install AtomicAI
```

## Requirements

AtomicAI requires Python 3.7 or higher. Other dependencies will be automatically installed during the pip installation process.

## Usage

AtomicAI provides several command-line tools for various tasks. Here are some examples:

### File Conversion
- `cq2vasp`: Convert Conquest files to VASP format
- `xyz2vasp`: Convert XYZ files to VASP format
- `vasp2cif`: Convert VASP files to CIF format
- `vasp2xyz`: Convert VASP files to XYZ format
- `vasp2lmp_data`: Convert VASP files to LAMMPS data format

### Structure Manipulation
- `supercell`: Create supercell structures
- `build_interface`: Build interfaces between materials

### Analysis
- `rdf`: Calculate Radial Distribution Function
- `structure_analysis`: Perform various structural analyses

### Visualization
- `plot_rdf_data`: Plot RDF data
- `plot_md_stats`: Plot molecular dynamics statistics
- `plot_vasp_md`: Plot VASP molecular dynamics results
- `plot_lammps_md`: Plot LAMMPS molecular dynamics results
- `plot_clusters`: Visualize clustering results

### Machine Learning
- `generate_descriptors`: Calculate atomic environment descriptors
- `generate_force_descriptors`: Generate force-related descriptors
- `dim_reduction`: Perform dimension reduction
- `dim_reduction_mpi`: Perform parallel dimension reduction using MPI
- `optimize_tslpp_hyperparameters_without_prediction`: Optimize TsLPP hyperparameters
- `predict_tslpp`: Predict using optimized TsLPP model

For more details on each tool, use the `--help` flag after the command.

## Examples

(Add some basic examples of how to use the main features of AtomicAI)

## Documentation

(Add a link to the full documentation when available)

## Contributing

We welcome contributions to AtomicAI! Please see our contributing guidelines for more information.

## License

AtomicAI is released under the MIT License. See the LICENSE.md file for details.

## Authors

- Selva Chandrasekaran Selvaraj

## Contact

- Email: selvachandrasekar.s@gmail.com
- Website: https://sites.google.com/view/selvas
- Twitter: https://twitter.com/selva_odc
- LinkedIn: https://www.linkedin.com/in/SelvaCS/
- Google Scholar: https://scholar.google.com/citations?user=vNozeNYAAAAJ&hl=en

## Cite

If you use AtomicAI in your research, please cite: (Add citation information when available)