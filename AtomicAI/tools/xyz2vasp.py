#!/usr/bin/env python3.10
from AtomicAI.io.read import read
from AtomicAI.io.write_cq import write_cq_file
import os, sys
import ase.io
def xyz2vasp():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"xyz2vasp xyz_file_name with .xyz extension\"")
        print()
        exit()

    data = ase.io.read(input_file) 
    out_file = input_file[:-3]+'vasp'
    ase.io.write(out_file, data, format='vasp')
    return
