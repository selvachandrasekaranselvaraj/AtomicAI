import os, sys
from ase.io import read, write
from ase.io import lammpsdata
import numpy as np

def vasp2lmp_data():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"vasp2cif vasp_file_name with .vasp extension\"")
        print()
        exit()

    if input_file == 'POSCAR':
       a_vasp = read(input_file)
       out_file = input_file+'.lmp_data'
    elif input_file == 'CONTCAR':
       a_vasp = read(input_file)
       out_file = input_file+'.lmp_data'      
    elif input_file[-4:] == 'vasp':
       a_vasp = read(input_file)
       out_file = input_file[:-4]+'lmp_data'       
    else:
        print('Input file is not POSCAR or .vasp')
        exit()

    # Sort atoms alphabetically by symbol
    sorted_indices = np.argsort(a_vasp.get_chemical_symbols())
    atoms_sorted = a_vasp[sorted_indices]

    # Update positions if needed (e.g., shift or remove atoms)
    # Here, we assume no specific updates are needed, but you can modify as required.
    lammpsdata.write_lammps_data(out_file, atoms_sorted, masses=True, force_skew=True, atom_style="atomic")
    lammpsdata.write_lammps_data("data.lmp_data", atoms_sorted, masses=True, force_skew=True, atom_style="atomic")
    return

