#!/usr/bin/env python3
import sys, os, shutil
from glob import glob
from AtomicAI.tools.vaspDB_inputs_modules import (
    expand_supercell_if_needed, read_poscar_elements, read_poscar_quantities,
    make_output_folder, setup_aimd_folders
)

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
            expanded_file, natoms = expand_supercell_if_needed(input_file)
            elements = read_poscar_elements(expanded_file)
            quantities = read_poscar_quantities(expanded_file)
            target_folder = make_output_folder(elements, quantities)
            if target_folder is None:
                if expanded_file:
                    os.remove(expanded_file)
                continue

            shutil.move(expanded_file, os.path.join(target_folder, "POSCAR"))
            setup_aimd_folders(target_folder, vc_only=True)
        else:
            print(f"[WARNING] {input_file} is not POSCAR/CONTCAR/.vasp, skipping.")

