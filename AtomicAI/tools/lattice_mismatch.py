#!/usr/bin/env python3
import sys
import numpy as np
from ase.io import read
from itertools import product, combinations, combinations_with_replacement
from math import gcd
from functools import reduce

def read_vasp_files(filenames):
    """Read multiple VASP files with error handling."""
    return [read(f) for f in filenames]  # More concise list comprehension

def lcm(a, b):
    """Least common multiple using math.gcd."""
    return a * b // gcd(a, b)

def find_min_multiples(structures, threshold, max_multiples):
    """Find minimal multiples with mismatch < threshold for structure pairs."""
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
                    mismatch = abs(len1 - len2) / min(len1, len2) * 100

                    if mismatch < threshold:
                        best = {
                            'structures': (s1+1, s2+1),
                            'vectors': (vec_names[v1], vec_names[v2]),
                            'lengths': (all_norms[s1,v1], all_norms[s2,v2]),
                            'multiples': (i1, i2),
                            'scaled_lengths': (len1, len2),
                            'mismatch%': mismatch,
                            'sum': i1 + i2
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
                'sum': None
            })
        results.append(struct_results)
    return results, all_norms

def analyze_vectors(structures, threshold, n_multiples):
    """Analyze vectors across all structures and find minimal multiples."""
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
            mismatch = (max_len - min_len)/min_len * 100

            if mismatch < threshold and sum(current_multiples) < min_sum:
                min_sum = sum(current_multiples)
                best = {
                    'vector': vec_names[vec_idx],
                    'multiples': current_multiples,
                    'lengths': current_lengths,
                    'mismatch%': mismatch,
                    'min_length': min_len,
                    'max_length': max_len
                }
                if min_sum == n_structures:  # Early exit for perfect match
                    break
        results.append(best or {
            'vector': vec_names[vec_idx],
            'multiples': None,
            'lengths': None,
            'mismatch%': None,
            'min_length': None,
            'max_length': None
        })
    return results, all_lengths

def print_results(structures, results, all_lengths, threshold):
    """Print formatted results based on number of structures."""
    if len(structures) == 2:
        print("\nVector Comparison Results (<5.0% mismatch):")
        for struct_results in results:
            s1, s2 = struct_results[0]['structures']
            print(f"\nStructures {s1}-{s2}:")
            print("Vectors  Multiples   Length1   Length2   Mismatch%  Sum")
            print("------------------------------------------------------")
            for res in struct_results:
                if res['multiples']:
                    line = (f"{res['vectors'][0]}{s1}-{res['vectors'][1]}{s2}  "
                           f"{res['multiples'][0]}:{res['multiples'][1]:<7}  "
                           f"{res['scaled_lengths'][0]:8.3f}  {res['scaled_lengths'][1]:8.3f}  "
                           f"{res['mismatch%']:6.2f}%  {res['sum']:3d}")
                    print("\033[92m" + line + "\033[0m" if res['mismatch%'] < threshold else line)
                else:
                    print(f"{res['vectors'][0]}{s1}-{res['vectors'][1]}{s2}  {'-':<7}  {'-':<8}  {'-':<8}  {'No match':<8}  {'-':<3}")
    else:
        print(f"\nVector Analysis Across All Structures (Threshold = {threshold}%):")
        print("Vector  MinLength(Å)  MaxLength(Å)  Mismatch%  Multiples")
        print("-------------------------------------------------------")
        for res in results:
            if res['multiples']:
                color = "\033[92m" if res['mismatch%'] < threshold else "\033[91m"
                print(f"{res['vector']:<6}  {res['min_length']:11.3f}  {res['max_length']:11.3f}  "
                     f"{color}{res['mismatch%']:6.2f}%\033[0m  {':'.join(map(str, res['multiples']))}")
            else:
                print(f"{res['vector']:<6}  {'-':11}  {'-':11}  {'No match':<8}  {'-':<9}")

    print("\nDetailed Vector Lengths (Å):")
    print("Structure " + " ".join(f"{v:>8}" for v in ['a', 'b', 'c']))
    for i, lengths in enumerate(all_lengths, 1):
        print(f"{i:<9}" + " ".join(f"{l:8.3f}" for l in lengths))

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} POSCAR1 POSCAR2 [POSCAR3 ...]")
        sys.exit(1)

    try:
        structures = read_vasp_files(sys.argv[1:])
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)

    max_multiples = 20
    threshold = 7.0
    if len(structures) == 2:
        results, all_lengths = find_min_multiples(structures, threshold, max_multiples)
    else:
        results, all_lengths = analyze_vectors(structures, threshold, max_multiples)
    print_results(structures, results, all_lengths, threshold)

if __name__ == "__main__":
    main()
