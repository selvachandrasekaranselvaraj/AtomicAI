import os, sys
from ase.io import read, write
from ase.io import lammpsdata
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

    lammpsdata.write_lammps_data(out_file, a_vasp, masses=True)
    return

