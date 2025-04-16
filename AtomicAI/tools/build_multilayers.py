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

def setup_logging(structure_names):
    """Initialize logging to both console and file with dynamic naming"""
    log_filename = f"{'_'.join(structure_names)}.log"
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
        global logger
        logger = setup_logging(names)
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

def analyze_vectors_2(structures, threshold, max_multiples):
    """
    Find minimal multiples with mismatch < threshold for structure pairs
    
    Args:
        structures (list): List of ASE Atoms objects
        threshold (float): Maximum allowed mismatch percentage
        max_multiples (int): Maximum supercell multiples to consider
        
    Returns:
        tuple: (analysis results, array of vector lengths)
    """
    vec_names = ['a', 'b', 'c']
    all_norms = np.array([[np.linalg.norm(v) for v in s.get_cell()] for s in structures])
    results = []

    for (s1, s2) in combinations(range(len(structures)), 2):
        struct_results = []
        for (v1, v2) in combinations_with_replacement(range(3), 2):
            best = None
            for total in range(2, 2 * max_multiples + 1):
                for i1 in range(1, min(total, max_multiples + 1)):
                    i2 = total - i1
                    if not (1 <= i2 <= max_multiples):
                        continue

                    len1, len2 = i1 * all_norms[s1,v1], i2 * all_norms[s2,v2]
                    mismatch = abs(len1 - len2) / min(len1, len2) * 100 * 0.5
                    mean_len = np.mean([len1, len2]) 
                    if s1 == 0 and s2 == 1 and v1 ==0 and v2 ==0:
                        mean_len_a = np.mean([len1, len2]) 

                    if mismatch < threshold:
                        best = {
                            'structures': (s1+1, s2+1),
                            'vectors': (vec_names[v1], vec_names[v2]),
                            'lengths': (all_norms[s1,v1], all_norms[s2,v2]),
                            'multiples': (i1, i2),
                            'scaled_lengths': (len1, len2),
                            'mismatch%': mismatch,
                            'mean_length': mean_len
                        }
                        break
                if best:
                    break
            struct_results.append(best or {
                'structures': (s1+1, s2+1),
                'vectors': (vec_names[v1], vec_names[v2]),
                'lengths': (all_norms[s1,v1], all_norms[s2,v2]),
                'multiples': None,
                'scaled_lengths': None,
                'mismatch%': None,
                'mean_length': None
            })
        results.append(struct_results)
    r = results[0]
    results = [[r[0], r[3], r[5], r[1], r[2], r[4]]]
    return results, all_norms, mean_len_a

def analyze_vectors_3(structures, threshold, n_multiples):
    """
    Analyze vectors across all structures and find minimal multiples
    
    Args:
        structures (list): List of ASE Atoms objects
        threshold (float): Maximum allowed mismatch percentage
        n_multiples (int): Maximum supercell multiples to consider
        
    Returns:
        tuple: (analysis results, array of vector lengths)
    """
    vec_names = ['a', 'b', 'c']
    all_lengths = np.array([[np.linalg.norm(v) for v in s.get_cell()] for s in structures])
    n_structures = len(structures)
    results = []

    for vec_idx in range(3):
        multiples = np.arange(1, n_multiples+1)[:, None] * all_lengths[:, vec_idx]
        best = None
        min_sum = float('inf')

        for indices in product(range(n_multiples), repeat=n_structures):
            current_multiples = [i+1 for i in indices]
            current_lengths = [multiples[i,j] for j, i in enumerate(indices)]
            min_len, max_len = min(current_lengths), max(current_lengths)
            mismatch = (max_len - min_len)/min_len * 100 * 0.5
            mean_len = np.mean([max_len, min_len])

            if mismatch < threshold and sum(current_multiples) < min_sum:
                min_sum = sum(current_multiples)
                best = {
                    'vector': vec_names[vec_idx],
                    'multiples': current_multiples,
                    'lengths': current_lengths,
                    'mismatch%': mismatch,
                    'min_length': min_len,
                    'max_length': max_len,
                    'mean_length': mean_len
                }
                if vec_idx == 0:
                     mean_len_a = mean_len #np.mean([max_len, min_len])
                if min_sum == n_structures:  # Early exit for perfect match
                    break
        results.append(best or {
            'vector': vec_names[vec_idx],
            'multiples': None,
            'lengths': None,
            'mismatch%': None,
            'min_length': None,
            'max_length': None,
            'mean_length': None
        })
    return results, all_lengths, mean_len_a

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
            logger.info("Vectors  Multiples   Length1   Length2   Mismatch%  Mean_Length")
            logger.info("----------------------------------------------------------------")
            for res in struct_results:
                if res['multiples']:
                    line = (f"{res['vectors'][0]}{s1}-{res['vectors'][1]}{s2}  "
                           f"{res['multiples'][0]}:{res['multiples'][1]:<7}  "
                           f"{res['scaled_lengths'][0]:8.3f}  {res['scaled_lengths'][1]:8.3f}  "
                           f"{res['mismatch%']:6.2f}%  {res['mean_length']:8.3f}")
                    if res['mismatch%'] < threshold:
                        logger.info(line)
                    else:
                        logger.warning(line)
                else:
                    logger.info(f"{res['vectors'][0]}{s1}-{res['vectors'][1]}{s2}  {'-':<7}  {'-':<8}  {'-':<8}  {'No match':<8}  {'-':<3}")
    else:
        logger.info(f"\nVector Analysis Across All Structures (Threshold = {threshold}%):")
        logger.info("Vector  MinLength(Å)  MaxLength(Å) Mean_Length(Å) Mismatch%  Multiples")
        logger.info("-----------------------------------------------------------------------")
        for res in results:
            if res['multiples']:
                if res['mismatch%'] < threshold:
                    logger.info(f"{res['vector']:<6}  {res['min_length']:11.3f}  {res['max_length']:11.3f} {res['mean_length']:12.3f} {res['mismatch%']:10.2f}%  {':'.join(map(str, res['multiples']))} ")
                else:
                    logger.warning(f"{res['vector']:<6}  {res['min_length']:11.3f}  {res['max_length']:11.3f}  {res['mismatch%']:6.2f}%  {':'.join(map(str, res['multiples']))} {res['mean_length']:6.3f}")
            else:
                logger.info(f"{res['vector']:<6}  {'-':11}  {'-':11}  {'No match':<8}  {'-':<9}")

def supercell(data, nx, ny, nz, name):
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
    height = np.linalg.norm(data.cell[2])
    logger.info(f'\nNumber of atoms in {name} supercell: {len(data.positions)}')
    logger.info(f'z-axis height of {name} supercell: {height:3.3f}')
    if height < 15:
        extend = input("Do you want to EXTEND the height [y/n]:", )
    if extend in ['', 'y']:
        nz = int(input("No. of z_units to extand:", ))
        multiplier = np.identity(3) * [1, 1, nz]
        data = make_supercell(data, multiplier)
        height = np.linalg.norm(data.cell[2])
        logger.info(f'Now, number of atoms in {name} supercell: {len(data.positions)}')
        logger.info(f'Now, z-axis height of {name} supercell: {height:3.3f}')
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

def updatepositions(input_data, supercell_details, structure_names, mean_len):
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
    
    logger.info("\nBuilding multilayer structure:")
    for i, (atoms_data, n, name) in enumerate(zip(input_data, supercell_details, structure_names)):
        nx = ny = n
        nz = 1
        atoms = supercell(atoms_data, nx, ny, nz, name)
        cell = atoms.cell
        len_a = np.linalg.norm(cell[0])
        mismatch = abs(len_a - mean_len)/min(len_a, mean_len) * 100 
        logger.info(f'Mismatch of {name} structure:{mismatch:6.2f}%')
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

    logger.info(f'\nFinal multilayer structure:')
    logger.info(f'Total atoms: {len(combined_atoms.positions)}')
    # Correct way to calculate norms of cell vectors
    cell_norms = [f"{np.linalg.norm(v):3.3f}" for v in combined_atoms.cell]
    logger.info(f'Cell vectors (Å):{cell_norms}')
    #logger.info(f'Cell vectors (Å):\n{combined_atoms.cell}')

    # Write output files
    write(out_file, combined_atoms, vasp5=True, sort=True)
    logger.info(f'Structure written to {out_file}')
    
    return combined_atoms


def build_multilayers():
    """Main execution function"""
    try:
        if len(sys.argv) < 2:
            print(f"Usage: {sys.argv[0]} POSCAR1 POSCAR2 [POSCAR3 ...]", file=sys.stderr)
            sys.exit(1)

        structures, structure_names = read_vasp_files(sys.argv[1:])
        
        max_multiples = 20
        threshold = 11.4
        
        if len(structures) == 2:
            results, all_lengths, mean_len = analyze_vectors_2(structures, threshold, max_multiples)
        else:
            results, all_lengths, mean_len = analyze_vectors_3(structures, threshold, max_multiples)

        print_results(structures, results, all_lengths, threshold, structure_names)
        updatepositions(structures, results[0][0]['multiples'] if len(structures) == 2 else results[0]['multiples'], 
                      structure_names, mean_len)
        logger.info("Multilayer generation completed successfully")
        
    except Exception as e:
        if logger:
            logger.error(f"Error in multilayer generation: {str(e)}", exc_info=True)
        else:
            print(f"Error in multilayer generation: {str(e)}", file=sys.stderr)
        sys.exit(1)

