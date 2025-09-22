#!/usr/bin/env python3
"""
Multilayer Structure Generator for VASP Calculations

This script performs the following key functions:
1. Reads multiple VASP structure files (POSCAR/CONTCAR format)
2. Analyzes lattice vectors to find optimal supercell multiples
3. Generates commensurate supercells with minimal mismatch
4. Stacks structures vertically to create multilayer systems
5. Outputs combined structure in VASP format with comprehensive logging

Key Features:
- Automatic lattice vector matching with user-defined tolerance
- Smart atomic position adjustment during stacking
- Detailed logging of all operations
- Support for both bilayer and multilayer systems
- Comprehensive error handling
"""

import itertools
import random
import ast
import sys
import os
import logging
import numpy as np
import pandas as pd
from itertools import product, combinations, combinations_with_replacement
from math import gcd
from functools import reduce
from ase import Atoms
from ase.io import read, write
from ase.data import chemical_symbols
from ase.build import make_supercell


# Global logger variable
logger = None

def setup_logging():
    """Initialize logging to both console and fixed file"""
    files = sys.argv[1:]
    names = '_'.join([os.path.splitext(f)[0] for f in files])
    log_filename = f"{names}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def read_vasp_files(filenames):
    """
    Read multiple VASP files with error handling
    
    Args:
        filenames (list): List of VASP structure files
        
    Returns:
        tuple: (list of ASE Atoms objects, list of structure names)
    """
    try:
        structures = [read(f) for f in filenames]
        names = [os.path.splitext(f)[0] for f in filenames]
        logger.info(f"Successfully read {len(structures)} structures: {', '.join(names)}")
        return structures, names
    except Exception as e:
        if logger:
            logger.error(f"Error reading VASP files: {str(e)}")
        else:
            print(f"Error reading VASP files: {str(e)}", file=sys.stderr)
        raise

def lcm(a, b):
    """Calculate least common multiple using math.gcd"""
    return a * b // gcd(a, b)

def print_results(structures, results, all_lengths, threshold, structure_names):
    """
    Print formatted results based on number of structures
    
    Args:
        structures (list): List of ASE Atoms objects
        results (list): Analysis results
        all_lengths (ndarray): Vector lengths for all structures
        threshold (float): Mismatch threshold used
        structure_names (list): Names of input structures
    """
    logger.info("\nDetailed Vector Lengths (Å):")
    logger.info("Structure " + " ".join(f"{v:>8}" for v in ['a', 'b', 'c']))
    for i, lengths in enumerate(all_lengths, 1):
        logger.info(f"{i:<9}" + " ".join(f"{l:8.3f}" for l in lengths))

    if len(structures) == 2:
        logger.info(f"\nVector Comparison Results (<{threshold}% mismatch):")
        for struct_results in results:
            s1, s2 = struct_results[0]['structures']
            logger.info(f"\nStructures {s1}-{s2}:")
            logger.info("Vectors  Multiples   Length1   Length2   Mismatch%  Sum")
            logger.info("------------------------------------------------------")
            for res in struct_results:
                if res['multiples']:
                    line = (f"{res['vectors'][0]}{s1}-{res['vectors'][1]}{s2}  "
                           f"{res['multiples'][0]}:{res['multiples'][1]:<7}  "
                           f"{res['scaled_lengths'][0]:8.3f}  {res['scaled_lengths'][1]:8.3f}  "
                           f"{res['mismatch%']:6.2f}%  {res['sum']:3d}")
                    if res['mismatch%'] < threshold:
                        logger.info(line)
                    else:
                        logger.warning(line)
                else:
                    logger.info(f"{res['vectors'][0]}{s1}-{res['vectors'][1]}{s2}  {'-':<7}  {'-':<8}  {'-':<8}  {'No match':<8}  {'-':<3}")
    else:
        logger.info(f"\nVector Analysis Across All Structures (Threshold = {threshold}%):")
        logger.info("Vector  MinLength(Å)  MaxLength(Å)  Mismatch%  Multiples")
        logger.info("-------------------------------------------------------")
        for res in results:
            if res['multiples']:
                if res['mismatch%'] < threshold:
                    logger.info(f"{res['vector']:<6}  {res['min_length']:11.3f}  {res['max_length']:11.3f}  {res['mismatch%']:6.2f}%  {':'.join(map(str, res['multiples']))}")
                else:
                    logger.warning(f"{res['vector']:<6}  {res['min_length']:11.3f}  {res['max_length']:11.3f}  {res['mismatch%']:6.2f}%  {':'.join(map(str, res['multiples']))}")
            else:
                logger.info(f"{res['vector']:<6}  {'-':11}  {'-':11}  {'No match':<8}  {'-':<9}")

def supercell(data, nx, ny, nz, name, ith):
    """
    Generate supercell from base structure
    
    Args:
        data (ASE Atoms): Input structure
        nx, ny, nz (int): Supercell multiples
        name (str): Structure identifier
        
    Returns:
        ASE Atoms: Supercell structure
    """
    multiplier = np.identity(3) * [nx, ny, nz]
    data = make_supercell(data, multiplier)
    if ith == 1:
       logger.info(f'Number of atoms in {name} supercell: {len(data.positions)}')

    # Sort atoms by element
    elements, positions = list(data.symbols), data.positions.T
    array = np.array([elements, positions[0], positions[1], positions[2]])
    df = pd.DataFrame(array.T, columns=['Sy', 'x', 'y', 'z'])
    df.index = elements
    df = df.sort_index()
    data.symbols = df['Sy']
    array = np.array([list(df['x']), list(df['y']), list(df['z'])]).T
    data.positions = array
    
    return data

def find_positionsB(atomsA, atomsB):
    """
    Calculate optimal z-position for stacking structure B on structure A
    
    Args:
        atomsA (ASE Atoms): Bottom structure
        atomsB (ASE Atoms): Top structure
        
    Returns:
        ndarray: Adjusted positions for structure B
    """
    positionsA = atomsA.positions
    positionsB = atomsB.positions
    heightA = max(positionsA[:, 2])
    heightB = max(positionsB[:, 2])
    gap = (atomsA.cell[2][2] + atomsB.cell[2][2]) - (heightA + heightB)
    z_offset = gap * 0.5
    positionsB = atomsB.positions
    positionsB[:, 2] += (heightA + z_offset)
    
    return positionsB

def updatepositions(input_data, n_cells, structure_names, ith):
    """
    Generate multilayer structure from component structures
    
    Args:
        input_data (list): List of ASE Atoms objects
        supercell_details (list): Supercell multiples
        structure_names (list): Names of structures
        
    Returns:
        ASE Atoms: Combined multilayer structure
    """
    out_file = f"{'_'.join(structure_names)}.vasp"
    atoms_all = []
    cells = []
    m_cell = np.mean([c.cell for c in input_data], axis=0)
    
    if ith == 1:
        logger.info("\nBuilding multilayer structure:")
    for i, (atoms,n,  name) in enumerate(zip(input_data, n_cells, structure_names)):
        m_cell[2] = atoms.cell[2]
        atoms.set_cell(m_cell, scale_atoms=True)
        atoms = supercell(atoms, 1, 1, n, name, ith) 
        cell = atoms.cell
        cells.append(cell)
        if i == 0:
            atoms.positions[:, 2] += 0.0
            mean_cell = cell
        else:
            atoms.positions = find_positionsB(combined_atoms, atoms)
            mean_cell = np.mean(cells, axis=0)

        atoms_all.extend(atoms)
        combined_atoms = Atoms(atoms_all)
        mean_cell[2][2] = sum(c[2][2] for c in cells)
        combined_atoms.cell = mean_cell

    # Sort atoms by atomic number
    sorted_indices = sorted(range(len(combined_atoms)), 
                          key=lambda k: chemical_symbols.index(combined_atoms[k].symbol))
    combined_atoms = combined_atoms[sorted_indices]

    if ith == 1:
       logger.info(f'\nFinal multilayer structure:')
       logger.info(f'Total atoms: {len(combined_atoms.positions)}')
       # Correct way to calculate norms of cell vectors
       cell_norms = [f"{np.linalg.norm(v):3.3f}" for v in combined_atoms.cell]
       logger.info(f'Cell vectors (Å):{cell_norms}')

    # Write output files
    write(out_file, combined_atoms, vasp5=True, sort=True)
    logger.info(f'Structure written to {out_file}')
    
    return combined_atoms

import itertools

def main():
    global logger
    logger = setup_logging()

    try:
        if len(sys.argv) < 2:
            print(f"Usage: {sys.argv[0]} POSCAR1 POSCAR2 [POSCAR3 ...]", file=sys.stderr)
            sys.exit(1)

        files = sys.argv[1:]

        all_structures, all_names = read_vasp_files(files)
        structure_dict = dict(zip(all_names, all_structures))

        generated_set = set()
        counter = 1

        logger.info("Generating unique multilayer structures (excluding mirrored permutations)...\n")

        for combo in itertools.permutations(all_names):
            name_key = "_".join(combo)
            reverse_key = "_".join(reversed(combo))

            # Skip if this or its reverse has already been generated
            if name_key in generated_set or reverse_key in generated_set:
                continue

            generated_set.add(name_key)
            selected_structures = [structure_dict[name] for name in combo]
            n_cells = [1] * len(selected_structures)

            updatepositions(selected_structures, n_cells, list(combo), counter)
            counter += 1

        logger.info(f"\nSummary: {counter - 1} unique multilayer structures generated.")
        logger.info("Multilayer generation completed successfully\n")

    except Exception as e:
        if logger:
            logger.error(f"Error in multilayer generation: {str(e)}", exc_info=True)
        else:
            print(f"Error in multilayer generation: {str(e)}", file=sys.stderr)
        sys.exit(1)


#if __name__ == "__main__":
#    main()
