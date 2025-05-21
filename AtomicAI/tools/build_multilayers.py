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

"""
Multilayer Structure Generator - All Permutations
"""

import sys, math
import os
import logging
import numpy as np
from itertools import permutations, product, combinations, combinations_with_replacement
from math import gcd
from typing import List, Tuple, Dict, Any
from ase import Atoms
from ase.io import read, write
from ase.data import chemical_symbols
from ase.build import make_supercell

# Global logger setup
logger = logging.getLogger(__name__)

def setup_logging():
    """Initialize logging to both console and single file"""
    files = sys.argv[1:]
    names = '_'.join([os.path.splitext(f)[0] for f in files])
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(f"{names}.log", mode='w'),
            logging.StreamHandler()
        ]
    )


def read_vasp_files(filenames: List[str]) -> Tuple[List[Atoms], List[str]]:
    """Read multiple VASP files with error handling"""
    try:
        structures = [read(f) for f in filenames]
        names = [os.path.splitext(f)[0] for f in filenames]
        logger.info(f"Successfully read {len(structures)} structures: {', '.join(names)}")
        return structures, names
    except Exception as e:
        logger.error(f"Error reading VASP files: {str(e)}")
        raise

def analyze_vectors(structures: List[Atoms], threshold: float, max_multiples: int) -> Tuple[List[Dict], np.ndarray, float]:
    """Unified vector analysis function"""
    vec_names = ['a', 'b', 'c']
    all_lengths = np.array([[np.linalg.norm(v) for v in s.get_cell()] for s in structures])
    n_structures = len(structures)
    results = []
    mean_len_a = 0.0

    if n_structures == 2:
        struct_results = []
        for v1, v2 in combinations_with_replacement(range(3), 2):
            best = None
            for total in range(2, 2 * max_multiples + 1):
                for i1 in range(1, min(total, max_multiples + 1)):
                    i2 = total - i1
                    if not (1 <= i2 <= max_multiples):
                        continue

                    len1, len2 = i1 * all_lengths[0, v1], i2 * all_lengths[1, v2]
                    mismatch = abs(len1 - len2) / min(len1, len2) * 50
                    mean_len = np.mean([len1, len2])

                    if v1 == 0 and v2 == 0:
                        mean_len_a = mean_len

                    if mismatch < threshold:
                        best = {
                            'structures': (1, 2),
                            'vectors': (vec_names[v1], vec_names[v2]),
                            'lengths': (all_lengths[0, v1], all_lengths[1, v2]),
                            'multiples': (i1, i2),
                            'scaled_lengths': (len1, len2),
                            'mismatch%': mismatch,
                            'mean_length': mean_len
                        }
                        break
                if best:
                    break
            struct_results.append(best)
        results = [[struct_results[0], struct_results[3], struct_results[5], 
                  struct_results[1], struct_results[2], struct_results[4]]]
    else:
        for vec_idx in range(3):
            multiples = np.arange(1, max_multiples+1)[:, None] * all_lengths[:, vec_idx]
            best = None
            min_sum = float('inf')

            for indices in product(range(max_multiples), repeat=n_structures):
                current_multiples = [i+1 for i in indices]
                current_lengths = [multiples[i,j] for j, i in enumerate(indices)]
                min_len, max_len = min(current_lengths), max(current_lengths)
                mismatch = (max_len - min_len)/min_len * 50
                mean_len = np.mean([max_len, min_len])

                if vec_idx == 0:
                    mean_len_a = mean_len

                if mismatch < threshold and sum(current_multiples) < min_sum:
                    min_sum = sum(current_multiples)
                    best = {
                        'vector': vec_names[vec_idx],
                        'multiples': current_multiples,
                        'lengths': current_lengths,
                        'mismatch%': mismatch,
                        'mean_length': mean_len
                    }
                    if min_sum == n_structures:
                        break
            results.append(best)

    return results, all_lengths, mean_len_a

def print_results(structures: List[Atoms], results: List[Dict], all_lengths: np.ndarray, 
                 threshold: float, structure_names: List[str]) -> None:
    """Print formatted analysis results"""
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
                if res:
                    status = logger.info if res['mismatch%'] < threshold else logger.warning
                    line = (f"{res['vectors'][0]}{s1}-{res['vectors'][1]}{s2}  "
                          f"{res['multiples'][0]}:{res['multiples'][1]:<7}  "
                          f"{res['scaled_lengths'][0]:8.3f}  {res['scaled_lengths'][1]:8.3f}  "
                          f"{res['mismatch%']:6.2f}%  {res['mean_length']:8.3f}")
                    status(line)
    else:
        logger.info(f"\nVector Analysis Across All Structures (Threshold = {threshold}%):")
        logger.info("Vector  MinLength(Å)  MaxLength(Å) Mean_Length(Å) Mismatch%  Multiples")
        logger.info("-----------------------------------------------------------------------")
        for res in results:
            if res:
                status = logger.info if res['mismatch%'] < threshold else logger.warning
                status(f"{res['vector']:<6}  {min(res['lengths']):11.3f}  {max(res['lengths']):11.3f} "
                      f"{res['mean_length']:12.3f} {res['mismatch%']:10.2f}%  {':'.join(map(str, res['multiples']))}")
def create_supercell(atoms: Atoms, nx: int, ny: int, nz: int, name: str, ith_structure: int) -> Atoms:
    """Generate supercell from base structure"""
    multiplier = np.diag([nx, ny, nz])
    supercell = make_supercell(atoms, multiplier)
    height = np.linalg.norm(supercell.cell[2])
    
    if ith_structure == 1:
       logger.info(f'\nNumber of atoms in {name} supercell: {len(supercell.positions)}')
       logger.info(f'z-axis height of {name} supercell: {height:.3f}')

    # Sort atoms by element
    symbols = np.array(supercell.get_chemical_symbols())
    positions = supercell.positions
    sorted_indices = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0], symbols))
    supercell = supercell[sorted_indices]
    
    return supercell

def calculate_stack_position(bottom: Atoms, top: Atoms, z_gap: float) -> np.ndarray:
    """Calculate optimal z-position for stacking with specified gap"""
    max_z_bottom = np.max(bottom.positions[:, 2]) 
    top_positions = top.positions.copy() - np.min(top.positions[:, 2])
    top_positions[:, 2] += (max_z_bottom + z_gap)
    return top_positions

def build_multilayer(structures: List[Atoms], supercell_details: List[int], 
        structure_names: List[str], mean_len: float, z_gap: float, ith_structure: int) -> Atoms:
    """Generate multilayer structure from component structures"""
    out_file = f"{'_'.join(structure_names)}.vasp"
    
    # Create supercells
    supercells = []
    for atoms, n, name in zip(structures, supercell_details, structure_names):
        sc = create_supercell(atoms, n, n, 1, name, ith_structure)
        len_a = np.linalg.norm(sc.cell[0])
        supercells.append(sc)

    original_z = sum([s.cell[2][2] for s in supercells])
    mean_cell = np.mean([s.cell for s in supercells], axis=0)
    mean_len_a = np.linalg.norm(mean_cell[0])
    for sc, name in zip(supercells, structure_names):
        len_a = np.linalg.norm(sc.cell[0])
        mismatch = abs(len_a - mean_len_a)/min(len_a, mean_len_a) * 100 
        if ith_structure == 1:
            logger.info(f'Mismatch of {name} structure: {mismatch:.2f}%')

    # Combine structures
    mean_cell[2] = supercells[0].cell[2]
    supercells[0].set_cell(mean_cell, scale_atoms=True)
    combined = supercells[0].copy()
    for i, top in enumerate(supercells[1:], 1):
        mean_cell[2] = supercells[i].cell[2]
        top.set_cell(mean_cell, scale_atoms=True)
        new_positions = calculate_stack_position(combined, top, z_gap)
        top.positions = new_positions
        combined += top

    # Update combined cell dimensions
    modified_z = np.max(combined.positions[:, 2]) + z_gap
    mean_cell[2, 2] = modified_z
    combined.cell = mean_cell

    # Final sort by atomic number
    atomic_numbers = combined.get_atomic_numbers()
    sorted_indices = np.argsort(atomic_numbers)
    combined = combined[sorted_indices]

    # Log final structure details
    if ith_structure == 1:
        logger.info(f'\nFinal multilayer structure:')
        logger.info(f'Total atoms: {len(combined)}')
        cell_norms = [f"{np.linalg.norm(v):.3f}" for v in combined.cell]
        logger.info(f'Cell vectors (Å): {cell_norms}')

    write(out_file, combined, vasp5=True, sort=True)
    logger.info(f'Structure written to {out_file}')
    
    return combined

def get_multiples_mapping(initial_names: List[str], permuted_names: List[str], multiples: List[int]) -> List[int]:
    """Map multiples to permuted order"""
    name_to_multiple = dict(zip(initial_names, multiples))
    return [name_to_multiple[name] for name in permuted_names]

def get_nonredundant_permutations(files: List[str]) -> List[tuple]:
    """Generate non-redundant permutations considering periodicity"""
    # For N layers, we only need (N-1)! permutations since cyclic permutations are equivalent
    n = len(files)
    if n <= 1:
        return [tuple(files)]
    
    # Fix first element and permute the rest
    fixed = files[0]
    remaining = files[1:]
    perms = [tuple([fixed] + list(p)) for p in permutations(remaining)]
    
    return perms[:math.factorial(n-1)]  # Only need (n-1)! permutations

def build_multilayers():
    """Main execution function - now testing ALL permutations"""
    setup_logging()
    
    try:
        if len(sys.argv) < 2:
            print(f"Usage: {sys.argv[0]} POSCAR1 POSCAR2 [POSCAR3 ...]", file=sys.stderr)
            sys.exit(1)

        max_multiples = 20
        threshold = 9.0
        z_gap = 2.3
        files = sys.argv[1:]
        
        # Initial analysis (done once)
        initial_structures, initial_names = read_vasp_files(files)
        results, lengths, mean_len = analyze_vectors(initial_structures, threshold, max_multiples)
        
        # Get supercell multiples based on initial order
        if len(initial_structures) == 2:
            multiples = results[0][0]['multiples']
        else:
            multiples = results[0]['multiples']
        
        # Print initial analysis results
        print_results(initial_structures, results, lengths, threshold, initial_names)
        
        # Generate non-redundant permutations
        nonred_perms = get_nonredundant_permutations(files)
        total_perms = len(nonred_perms)
        
        logger.info(f"\nExploring {total_perms} non-redundant stacking permutations:")
        
        for i, perm_files in enumerate(nonred_perms, 1):
            logger.info(f"\n=== PERMUTATION {i}/{total_perms} ===")
            logger.info(f"Stacking order: {' → '.join([os.path.splitext(f)[0] for f in perm_files])}")
            
            structures, names = read_vasp_files(perm_files)
            permuted_multiples = get_multiples_mapping(initial_names, names, multiples)
            
            # Generate output filename with permutation number
            out_prefix = f"perm_{i}_of_{total_perms}"
            build_multilayer(structures, permuted_multiples, names, mean_len, z_gap, i)
            
            logger.info(f"Completed permutation {i}/{total_perms}")
        
        logger.info(f"\nFinished exploring all {total_perms} possible stacking orders")
        
    except Exception as e:
        logger.error(f"Error in multilayer generation: {str(e)}", exc_info=True)
        sys.exit(1)

