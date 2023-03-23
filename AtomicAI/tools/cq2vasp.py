#!/usr/bin/env python3.10
from AtomicAI.io.read import read
import os, sys
import ase.io
def cq2vasp():
    try:
        cq_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"cq2vasp cq_file_name_with_.dat_extension\"")
        print()
        exit()

    data = read(input_file=cq_file, file_format='cq', structure_conversion='True')
    vasp_file = cq_file[:-4]+'.vasp'
    ase.io.write(vasp_file, data, format='vasp')
