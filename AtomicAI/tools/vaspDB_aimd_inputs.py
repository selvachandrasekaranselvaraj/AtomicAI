#!/usr/bin/env python3
import sys, os, shutil
from glob import glob
from ase.io import read
from AtomicAI.tools.vaspDB_vc_inputs import (
    read_poscar_elements, read_poscar_quantities,
    make_output_folder, setup_aimd_folders
)

def vaspDB_aimd_inputs":
    base_dirs = sys.argv[1:] if len(sys.argv) > 1 else [d for d in os.listdir(".") if os.path.isdir(d)]

    for base_folder in base_dirs:
        vc_contcar = os.path.join(base_folder, "vc", "CONTCAR")
        if not os.path.isfile(vc_contcar):
            print(f"[ERROR] VC CONTCAR not found in {base_folder}/vc, skipping.")
            continue

        print(f"\n=== [FOLLOW-UP SETUP] Using VC output from {vc_contcar} ===")
        shutil.copy(vc_contcar, os.path.join(base_folder, "POSCAR"))
        setup_aimd_folders(base_folder, vc_only=False)

