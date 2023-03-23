#!/usr/bin/env python3.10
from AtomicAI.io.read import read
import os, sys
import ase.io
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
    return 
