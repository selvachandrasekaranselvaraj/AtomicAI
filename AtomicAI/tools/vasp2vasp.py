#!/usr/bin/env python3.10
from AtomicAI.io.read import read
import os, sys
import ase.io
import numpy as np

def vasp2vasp():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"vasp2vasp vasp_file_name with .vasp extension\"")
        print()
        exit()

    data = ase.io.read(input_file)
    positions = data.positions 
    positions[:, 2] += 2.0
    cell_matrix = data.cell
    cell_matrix_inv = np.linalg.inv(cell_matrix)
    fractional_positions = np.dot(positions, cell_matrix_inv.T)
    fractional_positions[fractional_positions < 0] += 1.0
    data.positions = np.dot(fractional_positions, cell_matrix)
    if input_file[-5:] == '.vasp':
        out_file = input_file[:-5]+'_m.vasp'
    elif input_file in ['POSCAR', 'CONTCAR']:
        out_file = input_file+'.vasp'
    else:
        print("Input file is not correct")
        exit()
    ase.io.write(out_file, data, format='vasp')
    return 

