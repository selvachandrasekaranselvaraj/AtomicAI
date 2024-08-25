#!/usr/bin/env python3.10
from AtomicAI.io.read import read
#from AtomicAI.io.write_cq import write_cq_file
import os, sys
import ase.io
import numpy as np
import pandas as pd
def xyz2vasp():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"xyz2vasp xyz_file_name with .xyz extension\"")
        print()
        exit()

    data = ase.io.read(input_file)
    print(np.array(data.cell)) 
    elements, positions = list(data.symbols), data.positions.T
    array = np.array([elements, positions[0], positions[1], positions[2]])
    df = pd.DataFrame(array.T, columns=['Sy', 'x', 'y', 'z'])
    df.index = elements
    df = df.sort_index()
    data.symbols = np.array(df['Sy'])
    array = np.array([list(df['x']), list(df['y']), list(df['z'])]).T
    data.set_positions(array)
    out_file = input_file[:-3]+'vasp'
    ase.io.write(out_file, data, format='vasp')
    return
