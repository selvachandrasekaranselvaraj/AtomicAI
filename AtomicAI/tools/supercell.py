#!/usr/bin/env python3.10
from AtomicAI.io.read import read
from ase.build import make_supercell
import os, sys
import ase.io
import numpy
import numpy as np
import pandas as pd
def supercell():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"vasp2cif vasp_file_name with .vasp extension\"")
        print()
        exit()

    data = ase.io.read(input_file)
    print("Total number of atoms in this cell are: ", len(data.positions))
    a, b, c = data.cell[0][0], data.cell[1][1], data.cell[2][2]
    print(f'a={a}, b={b}, c={c} Angstrom')
    x_ = int(input("No. of x_units to extand: ", ))
    y_ = int(input("No. of y_units to extand: ", ))
    z_ = int(input("No. of z_units to extand: ", ))

    multiplier = numpy.identity(3) * [x_, y_, z_]

    data = make_supercell(data, multiplier)
    print(f'Number of atoms in the supercell are: {len(data.positions)}')
    name = input_file.split('.')[0] 
    out_file = f'{x_}x{y_}x{z_}_{name}'

    elements, positions = list(data.symbols), data.positions.T
    array = np.array([elements, positions[0], positions[1], positions[2]])
    df = pd.DataFrame(array.T, columns=['Sy', 'x', 'y', 'z'])
    df.index = elements
    df = df.sort_index()
    data.symbols = df['Sy']
    array = np.array([list(df['x']), list(df['y']), list(df['z'])]).T
    data.positions = array
    ase.io.write(out_file+'.vasp', data)
    #ase.io.write(out_file+'.xyz', data)
    return 
