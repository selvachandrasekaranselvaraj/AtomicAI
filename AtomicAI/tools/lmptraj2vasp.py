
import warnings

# Filter and ignore specific warnings
warnings.filterwarnings('ignore')

#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)

import os, sys, time
from ase.io import read, write
from ase import Atoms
import numpy as np
from ase.io import lammpsdata

def lmptraj2vasp():
 try:
     input_file = sys.argv[1]
 except:
     print("Input error!!!!")
     print("Usage: \"lmp2xyz lammps_dump_file_name \"")
     print()
     exit()

 run_time = time.time()
 print("Reading file...")

 try:
    data = read(input_file, format="lammps-dump-text", index=":", parallel='True')[-1]
 except:
    data = lammpsdata.read_lammps_data(input_file)

 symbols = np.array(list(data.symbols)).reshape(-1, 1)
 positions = data.positions
 print(symbols.shape, positions.shape)
 
 sy_pos = np.concatenate((symbols, positions), axis=1)
 
 # Define a function to get the atomic symbol from a data row
 def get_atomic_symbol(row):
     return row[0]
 sorted_data = np.array(sorted(sy_pos, key=get_atomic_symbol))
 
 data.symbols = (sorted_data[:, 0:1]).flatten()
 data.positions = sorted_data[:, 1:]
 print(f"Excution time: {round(time.time() - run_time, 3)}s")
 
 out_file = input_file.split('.')[0]+'.vasp'
 write(out_file, data, format='vasp')
 out_file = input_file.split('.')[0]+'_data.lmp'
 lammpsdata.write_lammps_data(out_file, data, masses=True)
 return
