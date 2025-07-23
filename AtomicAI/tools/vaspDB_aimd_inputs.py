#!/usr/bin/env python3
import sys, os, shutil
from glob import glob
from ase.io import read
from AtomicAI.tools.vaspDB_inputs_modules import (
    read_poscar_elements, read_poscar_quantities,
    make_output_folder, setup_aimd_folders
)

def is_vc_converged(outcar_path):
    """
    Return True if OUTCAR shows VASP run completed successfully.
    """
    if not os.path.isfile(outcar_path):
        print(f"[WARNING] OUTCAR not found at {outcar_path}")
        return False
    with open(outcar_path, "rb") as f:
        # Read only last 20KB for efficiency
        f.seek(-20480, os.SEEK_END)
        tail = f.read().decode(errors="ignore")

    return "reached required accuracy" in tail or "Voluntary context switches" in tail

def vaspDB_aimd_inputs():
    base_folder = './'
    vc_dir = os.path.join(base_folder, "vc")
    vc_contcar = os.path.join(vc_dir, "CONTCAR")
    vc_outcar = os.path.join(vc_dir, "OUTCAR")

    if not os.path.isfile(vc_contcar):
        print(f"[ERROR] VC CONTCAR not found in {vc_dir}, skipping.")
        return

    if not is_vc_converged(vc_outcar):
        print(f"[WARNING] VC calculation in {vc_dir} not converged yet, skipping AIMD setup.")
        return

    print(f"\n=== [FOLLOW-UP SETUP] Using VC output from {vc_contcar} ===")
    shutil.copy(vc_contcar, os.path.join(base_folder, "POSCAR"))
    setup_aimd_folders(base_folder, vc_only=False)

