# AtomicAI

AtomicAI is a Python-based package for building machine learning models in the field of computational materials science. It can be used for machine learning force field generation, atomic structure classifications based on temperature, defects, phase, and more.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Examples](#examples)
6. [Documentation](#documentation)
7. [Contributing](#contributing)
8. [License](#license)
9. [Authors](#authors)
10. [Contact](#contact)
11. [Cite](#cite)

## Features

AtomicAI offers a wide range of features for computational materials science:

### Analysis
- Processing and visualization of atomic coordinates in various file formats (CIF, VASP, XYZ, Conquest, etc.)
- Utilizes 'ase' library for reading and writing structure files
- Radial Distribution Function (RDF) calculation and analysis
- Structure analysis tools

### Data Processing
- Classification of atoms in trajectory files
- File format conversion between different atomic structure formats

### Machine Learning
- Featurization of atomic structures
- Dimension reduction techniques (PCA, LPP, TsLPP, TsNE, UMAP)
- Machine Learning Force Field (MLFF) generation
- Descriptor generation for atomic environments and forces

### Clustering and Visualization
- Various clustering methods
- Data plotting and visualization tools for RDF, molecular dynamics statistics, and clustering results

## Installation

Install AtomicAI using pip:

```bash
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
- `vasp2cq`: Convert VASP files to Conquest format
- `lmp2vasp`: Convert LAMMPS trajectory to VASP format
- `cif2cq`: Convert CIF files to Conquest format
- `cq2cif`: Convert Conquest files to CIF format
- `ase_traj2xyz_traj`: Convert ASE trajectory to XYZ trajectory

### Structure Manipulation
- `supercell`: Create supercell structures
- `build_interface`: Build interfaces between materials
- `wrap2unwrap`: Convert wrapped coordinates to unwrapped

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

#### Atomic Descriptors — `generate_descriptors`

Computes locally-averaged atomic fingerprints (LAAF) for all element pairs in a trajectory.

```bash
generate_descriptors trajectory.xyz [--descriptor TYPE [TYPE ...]] [--n-eta N]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--descriptor` / `-d` | `ACSF_G2 ACSF_G2G4 SOAP` | One or more descriptor types (see table below) |
| `--n-eta` / `-n` | `50` | Number of eta decay functions |

**Available descriptor types:**

| Type | Description |
|------|-------------|
| `ACSF_G2` | Radial symmetry functions (Behler-Parrinello G2) |
| `ACSF_G3` | Cosine basis functions parameterised by kappa |
| `ACSF_G4` | Angular symmetry functions including the r_jk cutoff (Behler-Parrinello G4) |
| `ACSF_G5` | Angular symmetry functions without r_jk cutoff (G5 variant) |
| `ACSF_G2G4` | Combined radial (G2) + angular (G4) descriptor |
| `ACSF_G2G4G5` | Combined radial (G2) + angular G4 + angular G5 descriptor |
| `SOAP` | Smooth Overlap of Atomic Positions (via DScribe) |
| `MBSF` | Many-body symmetry functions (radial gr + angular ga) |

**Examples:**

```bash
# G2 radial only, 60 eta functions
generate_descriptors traj.xyz --descriptor ACSF_G2 --n-eta 60

# Multiple descriptor types in one run
generate_descriptors traj.xyz --descriptor ACSF_G2G4 ACSF_G2G4G5 MBSF

# SOAP descriptor
generate_descriptors traj.xyz --descriptor SOAP
```

Output files are written to `./descriptors/` with naming convention `<TYPE>_<rc>_<ra>_<elem1>_<elem2>.dat`.

#### Force Descriptors — `generate_force_descriptors`

Generates force-projected descriptors for machine learning force field training.

```bash
generate_force_descriptors trajectory.xyz [--fp-type TYPE] [--rc R] [--n2b N] [--n3b N]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--fp-type` / `-f` | `Split2b3b_ss` | Fingerprint type: `BP2b` or `Split2b3b_ss` |
| `--rc` / `-r` | `10.5` | Cutoff radius in Angstrom |
| `--n2b` | `20` | Number of 2-body eta functions |
| `--n3b` | `10` | Number of 3-body eta functions |

**Available fingerprint types:**

| Type | Description |
|------|-------------|
| `BP2b` | 2-body Behler-Parrinello fingerprint |
| `Split2b3b_ss` | Split 2-body + 3-body fingerprint (same species) |

**Examples:**

```bash
# Default: Split2b3b_ss with 10.5 Å cutoff
generate_force_descriptors traj.xyz

# BP2b with shorter cutoff and more eta functions
generate_force_descriptors traj.xyz --fp-type BP2b --rc 8.0 --n2b 30

# Split2b3b with custom eta counts
generate_force_descriptors traj.xyz --fp-type Split2b3b_ss --rc 9.0 --n2b 25 --n3b 12
```

Output is written to `./descriptors/force_descriptors.dat`.

#### Other ML Tools
- `laaf`: Calculate LAAF descriptors
- `dim_reduction`: Perform dimension reduction
- `dim_reduction_mpi`: Perform parallel dimension reduction using MPI
- `pca`: Perform Principal Component Analysis
- `lpp`: Perform Locality Preserving Projection
- `optimize_tslpp_hyperparameters_without_prediction`: Optimize TsLPP hyperparameters
- `optimize_tslpp_hyperparameters_with_prediction`: Optimize TsLPP hyperparameters with prediction
- `predict_tslpp`: Predict using optimized TsLPP model

### Structure Building
- `build_multilayers`: Stack multiple VASP structures into commensurate multilayer systems
- `build_constrained_multilayers`: Build multilayers with lattice constraints
- `surfaces`: Generate surface slabs for common Miller indices from a bulk structure

### LAMMPS Input Generation
- `lammps_npt_inputs`: Generate LAMMPS NPT input files
- `lammps_nvt_inputs`: Generate LAMMPS NVT input files

For more details on any tool, run it with the `--help` flag.

## Examples

(This section should be filled with basic examples of how to use the main features of AtomicAI. As the context doesn't provide specific examples, I'll leave this section for you to fill in with relevant use cases.)

## Documentation

(Add a link to the full documentation when available. The context doesn't provide this information, so you may want to add it when documentation is ready.)

## Contributing

We welcome contributions to AtomicAI! Please see our contributing guidelines for more information. (You may want to add a CONTRIBUTING.md file to your repository with detailed guidelines.)

## License

AtomicAI is released under the MIT License. See the `LICENSE.md` file for details.

## Authors

- Selva Chandrasekaran Selvaraj

## Contact

- Email: selvachandrasekar.s@gmail.com
- Website: https://sites.google.com/view/selvas
- Twitter: https://twitter.com/selva_odc
- LinkedIn: https://www.linkedin.com/in/selvachandrasekaranselvaraj/
- Google Scholar: https://scholar.google.com/citations?user=vNozeNYAAAAJ&hl=en
- Scopus: https://www.scopus.com/authid/detail.uri?authorId=57225319817
- ResearchGate: https://www.researchgate.net/profile/Selva-Chandrasekaran-Selvaraj
- PyPI: https://pypi.org/project/AtomicAI/
- Documentation: https://atomicai.readthedocs.io/en/latest/
- GitHub: https://github.com/SelvaGith/AtomicAI


## Cite

If you use AtomicAI in your research, please cite: 


