#!/usr/bin/env python3.10
from AtomicAI.io.read import read
import os, sys
import ase.io
from ase.io import read, write
import ase.io.vasp as ase_vasp
import numpy as np

def vasp2xyz():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"vasp2cif vasp_file_name with .vasp extension\"")
        print()
        exit()

    data = ase.io.read(input_file) 
    out_file = input_file[:-4]+'xyz'
    ase.io.write(out_file, data, format='xyz')

    def updatepositions(atoms):
        cell = atoms.cell
        # print('Updating positions. Cell is:', cell)
        inverse_cell = np.linalg.inv(cell)
        positions = atoms.positions
        fractional_positions = np.dot(positions, inverse_cell)
        convert_positions = fractional_positions % 1.0
        atoms.positions = np.dot(convert_positions, cell)
        # Sort atoms alphabetically by symbol
        sorted_indices = np.argsort(atoms.get_chemical_symbols())
        atoms_sorted = atoms[sorted_indices]
        atoms = atoms_sorted
        return atoms


    if input_file == 'POSCAR' or input_file == 'CONTCAR':
        data = read(input_file)
        data = updatepositions(data)
        out_file = input_file+'.xyz'
        write(out_file, data, format='xyz')
    elif input_file[-4:] == 'vasp':
        data = read(input_file)
        data = updatepositions(data)
        out_file = input_file[:-5]+'.xyz'
        write(out_file, data)
    elif input_file[-4:] == '.xml':
        frames = ase_vasp.read_vasp_xml(filename=input_file, index=slice(None))
        out_file = input_file[:-4]+'.xyz'
        write(out_file, frames)
    else:
        print('No file format matches!!!')
        exit()
    return
