#!/usr/bin/env python3
import sys, os, shutil
from glob import glob
from ase.io import read, write
from AtomicAI.tools.vaspDB_inputs_modules import (
    expand_supercell_if_needed, read_poscar_elements, read_poscar_quantities,
    make_output_folder, setup_aimd_folders
)

def sanitize_poscar(filepath):
    """
    Clean a VASP POSCAR file by ensuring:
    - Only 3 values per coordinate line
    - No velocity section
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Make sure we can read basic metadata
    try:
        natoms = sum(map(int, lines[6].split()))
    except Exception as e:
        raise ValueError(f"Unable to parse atom counts on line 7: {lines[6]}\n{e}")

    coord_start = 8
    coord_end = coord_start + natoms

    # Strip any extra tokens from each coordinate line
    for i in range(coord_start, coord_end):
        xyz = lines[i].split()[:3]  # Keep only 3 numbers
        lines[i] = "  ".join(xyz) + "\n"

    # Strip everything after atomic coordinates (e.g., velocities)
    lines = lines[:coord_end]

    # Write to a clean file
    clean_path = filepath
    with open(clean_path, "w") as f:
        f.writelines(lines)

    return clean_path


def vaspDB_vc_inputs():
    input_files = sys.argv[1:] if len(sys.argv) > 1 else glob("*.vasp")
    if not input_files:
        print("[ERROR] No input files given or found (*.vasp).")
        sys.exit(1)

    for input_file in input_files:
        if input_file in ["POSCAR", "CONTCAR"] or input_file.endswith(".vasp"):
            if not os.path.isfile(input_file):
                print(f"[ERROR] {input_file} not found, skipping.")
                continue

            print(f"\n=== [VC SETUP] Processing {input_file} ===")
            sanitized_file = sanitize_poscar(input_file)
            expanded_file, natoms = expand_supercell_if_needed(sanitized_file)
            elements = read_poscar_elements(expanded_file)
            quantities = read_poscar_quantities(expanded_file)
            target_folder = make_output_folder(elements, quantities)
            shutil.move(expanded_file, os.path.join(target_folder, "POSCAR"))
            setup_aimd_folders(target_folder, vc_only=True)
        else:
            print(f"[WARNING] {input_file} is not POSCAR/CONTCAR/.vasp, skipping.")
