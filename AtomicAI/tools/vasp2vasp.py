#!/usr/bin/env python3.10
import os, sys
from ase.io import read, write
import ase.io.vasp as ase_vasp
import numpy as np

def remove_atoms_below_z(atoms, z_threshold):
    indices_to_remove = [atom.index for atom in atoms if atom.position[2] < z_threshold]
    del atoms[indices_to_remove]
    return atoms

def updatepositions(atoms, x_shift, y_shift, z_shift):
    cell = atoms.cell
    # print('Updating positions. Cell is:', cell)
    inverse_cell = np.linalg.inv(cell)
    positions = atoms.positions
    for i, p in enumerate(positions):
        positions[i] = np.array([p[0] + x_shift, p[1] + y_shift, p[2] + z_shift])#
    atoms.positions = positions #np.dot(convert_positions, cell)
    return atoms

def vasp2vasp():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"vasp2xyz vasp_file_name \"")
        print()
        exit()

    z_threshold = 10.0
    x_shift = 0.0
    y_shift = 0.0
    z_shift = 5.0

    if input_file == 'POSCAR' or input_file == 'CONTCAR':
        data = read(input_file)
        data = updatepositions(data, x_shift, y_shift, z_shift)
        data = remove_atoms_below_z(data, z_threshold)
        out_file = "m_"+input_file+'.vasp'
        write(out_file, data)
    elif input_file[-4:] == 'vasp':
        data = read(input_file)
        data = updatepositions(data, x_shift, y_shift, z_shift)
        data = remove_atoms_below_z(data, z_threshold)
        out_file = "m_"+input_file[:-5]+'.vasp'
        write(out_file, data)
    elif input_file[-4:] == '.xml':
        #frames = ase_vasp.read_vasp_xml(filename='vasprun.xml', index=slice(None))
        frames = ase_vasp.read_vasp_xml(filename=input_file, index=slice(None))
        out_file = input_file[:-4]+'_out.vasp'
        write(out_file, frames)
    else:
        print('No file format matches!!!')
        exit()
    return
vasp2vasp()
