#!/usr/bin/env python3.10
from AtomicAI.io.read import read
from AtomicAI.io.write_cq import write_cq_file
import os, sys
import ase.io
def cif2cq():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"cif2cq vasp_file_name with .cif extension\"")
        print()
        exit()

    data = ase.io.read(input_file) 
    out_file = input_file[:-3]+'dat'
    write_cq_file(data, out_file[:-4])
    return 
