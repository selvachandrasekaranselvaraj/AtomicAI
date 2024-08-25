#!/usr/bin/env python3.10
from AtomicAI.io.read import read
from AtomicAI.io.write_cq import write_cq_file
import os, sys
import ase.io
def vasp2cq():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"vasp2cq vasp_file_name with .vasp extension\"")
        print()
        exit()

    data = ase.io.read(input_file) 
    out_file = input_file[:-4]+'dat'
    write_cq_file(data, out_file[:-4])
    return 
