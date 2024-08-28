import warnings
<<<<<<< HEAD
=======

# Filter and ignore specific warnings (e.g., DeprecationWarning)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import ase.io
from ase.build import molecule
from ase.geometry.analysis import Analysis
import os, sys
from os import listdir
from os.path import isfile, join

from ase.build import make_supercell
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math

#from scipy import integrate

from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii

# List of cations and anions based on periodic table groups

def pairs(a, b, predict):
    pairs_ = []
    for e_i, element1 in enumerate(a):
        for e_j, element2 in enumerate(b):
            if not predict:
                if e_j >= e_i: # and element1+element2 != 'TiTi':
                    pairs_.append(f'{element1}_{element2}')
            else:
                pairs_.append(f'{element1}_{element2}')
    return pairs_

def predict_pairs(atoms_list):
    cations = np.array(['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Al', 'Ga', 'In', 'Tl',
           'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
           'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'])
    anions = np.array(['N', 'P', 'As', 'Sb', 'Bi', 'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I'])
    atoms_list = list(atoms_list)
    atoms_c = [atom for atom in atoms_list if atom in cations]
    atoms_a = [atom for atom in atoms_list if atom in anions]
    if 'P' in atoms_a and 'S' in atoms_a:
        atoms_c.append('P')
        atoms_a.remove('P')
    print("Cations:", atoms_c)
    print("Anions:", atoms_a)
    print()
    return pairs(atoms_c, atoms_a, predict=True)

def available_pairs(atoms_list):
    return pairs(atoms_list, atoms_list, predict=False)

# Smoothing function (moving average)
def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

def RDF(): 
    # Searching structure file in the persent directory
    filenames = []
    for i in range(1, 10):
        try:
            filenames.append(sys.argv[i])
        except:
            if i == 1:
                print("No structure file is available HERE!!!")
                print("Usage: python rdf.py file_name1 file_name2 file_name3 ...")
                exit()
            else:
                pass
 
    traj_data, symbols, legends = [], [], []
    for trajectory_file in filenames:
         print(f"Reading {trajectory_file}...")
         if 'xyz' in trajectory_file:
             traj = ase.io.read(trajectory_file, ':')
         elif 'lmp' in trajectory_file:
             traj = ase.io.read(trajectory_file, format="lammps-dump-text", index=":")
         all_symbols = [list(tra.symbols) for tra in traj]
         symbols.extend(list(set(np.array(all_symbols).flatten())))
         traj_data.append(traj)
         legends.append(trajectory_file[:-4])
    symbols = set(np.array(symbols).flatten())
    print()

    no_of_subplots = 0
    pairs = []
    for e_i, element1 in enumerate(symbols):
        for e_j, element2 in enumerate(symbols):
            if e_j >= e_i: # and element1+element2 != 'TiTi':
                no_of_subplots += 1
                pairs.append(f'{element1}_{element2}')

    pairs_predicted = predict_pairs(symbols)
    pairs_available = available_pairs(symbols)
    print("Available atomic pairs:", pairs_available)
    print("Predicted atomic pairs:", pairs_predicted)
    print()


    #all_pairs = input("Do you want to use availble pairs(1) or predicted(2):", )
    if len(pairs_available) > 5:
        all_pairs = '2'
    else:
        all_pairs = '1'

    if all_pairs=='1':
        pairs = pairs_available
    else:
        pairs = pairs_predicted

    no_of_subplots = len(pairs)
    print(f"No. of subplots: {no_of_subplots} for pairs {pairs}")

    #no_of_subplots = math.factorial(len(symbols))
    #print("No. of subplots: ", no_of_subplots, symbols)
    fig, axs = plt.subplots(no_of_subplots, sharex=True, figsize=(6, no_of_subplots*2))
    for legend_no, traj in enumerate(traj_data):
        if len(list(traj[0].symbols)) < 600:
            if len(traj) > 5000:
                traj = traj[:5000]
            else:
                traj = traj[:100]

        print()
        print(f"No. of frames {len(traj)} in {legends[legend_no]}")

        rMax = 6.0
        for traj_i in range(len(traj)):
            traj[traj_i] = supercell(traj[traj_i], rMax)
       
        cell = np.linalg.norm(np.array(traj[0].cell), axis=0) #np.linalg.norm(traj[0].cell)
        if (cell > 2*rMax).all():
            rMax = rMax
        else:
            rMax = min(cell) * 0.45

        print("Selected rMax: ", round(rMax, 2))
        nBins = int(rMax*100)
        smoothing_window_size = 10

        ana = Analysis(traj)
        fontsize = 16
        for i, pair in enumerate(pairs):
          all_symbols = [list(set(tra.symbols)) for tra in traj]
          avai_symbols = list(set(np.array(all_symbols).flatten()))
          elements = pair.split('_')
          if elements[0] in avai_symbols and elements[1] in avai_symbols: 

              atomic_number1 = atomic_numbers[elements[0]]
              atomic_number2 = atomic_numbers[elements[1]]
              
              rdf = ana.get_rdf(rMax,
                                nBins,
                                imageIdx=None,
                                elements=(atomic_number1, atomic_number2))[0]
              

              # Smoothing the RDF
              x = (np.arange(nBins) + 0.5) * rMax / nBins
              y = moving_average(rdf, window_size=smoothing_window_size)
              x = x[smoothing_window_size-1:]

              print(pair)
              axs[i].plot(x, y, label = legends[legend_no])
          else:
              x = (np.arange(nBins) + 0.5) * rMax / nBins
              axs[i].plot(x, x*0.0, label = legends[legend_no])
          # Plotting
          axs[i].set_xlabel(r"Distance ($\AA$)", fontsize=fontsize)
          axs[i].set_ylabel(f"RDF ({elements[0]}-{elements[1]})", fontsize=fontsize)
 
          axs[i].tick_params(
              axis='both',
              which='both',
              length=6,
              width=1,
              bottom=True,
              labelbottom=True,
              direction='in',
              size=10,
              labelsize=fontsize,
              #gridOn = 'both',
              tick1On='both')
          axs[i].set_xlim(0.9, rMax)
          #plt.yticks(fontsize=15)
    axs[0].legend(ncol=2, loc='lower left', mode = 'expand', bbox_to_anchor=(0.0, 1.02, 1, 0.2), fontsize=fontsize-3, frameon=False)
    plt.savefig('rdf.png', bbox_inches="tight")
    return


def supercell(data, rMax):
    rMax = rMax*2
    a, b, c = data.cell[0][0], data.cell[1][1], data.cell[2][2]
    #print(f'a={a}, b={b}, c={c} Angstrom')
    x_ = y_ = z_ = 1
    while rMax > a:
        x_ += 1  # Increment x_ by 1
        a *= x_

    while rMax > b:
        y_ += 1  # Increment x_ by 1
        b *= y_

    while rMax > c:
        z_ += 1  # Increment x_ by 1
        c *= z_

    multiplier = np.identity(3) * [x_, y_, z_]

    data = make_supercell(data, multiplier)

    elements, positions = list(data.symbols), data.positions.T
    array = np.array([elements, positions[0], positions[1], positions[2]])
    df = pd.DataFrame(array.T, columns=['Sy', 'x', 'y', 'z'])
    df.index = elements
    df = df.sort_index()
    data.symbols = df['Sy']
    array = np.array([list(df['x']), list(df['y']), list(df['z'])]).T
    data.positions = array
    return  data


'''
import warnings
>>>>>>> 5bcf4f0 (plot_lammps_md added)
warnings.filterwarnings("ignore")
import numpy as np
import ase.io
from ase.build import molecule
from ase.geometry.analysis import Analysis
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
import os, sys
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from lmfit import models

from AtomicAI.io.write_data_in_py import write_data_in_py

def RDF():
    # Searching md.stats file in the persent directory
    traj_files = [f for f in listdir('./') if '.xyz' in f]
    if len(traj_files) > 0:
        print(f"Availabel files are {' '.join(traj_files)}")
    else:
        print("No traj.xyz file is availabel HERE!!!")
        exit()
    out_directory = './rdf/'
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)

    traj_data, symbols, legends = [], [], []
    for trajectory_file in traj_files:
        print(f"Reading {trajectory_file}...")
        traj = ase.io.read(trajectory_file, ':')
        all_symbols = [list(tra.symbols) for tra in traj]
        for sy in list(set(np.array(all_symbols).flatten())):
            symbols.append(sy) #list(set(np.array(all_symbols).flatten())))
        traj_data.append(traj)
        legends.append(trajectory_file[:-4])
    symbols = set(symbols)
    no_of_subplots = 0
    for e_i, element1 in enumerate(symbols):
        for e_j, element2 in enumerate(symbols):
            no_of_subplots += 1
    fig, axs = plt.subplots(no_of_subplots, sharex=True, figsize=(9, 5))
    print()
    for legend_no, traj in enumerate(traj_data):
        py_output_file = out_directory+legends[legend_no]+"_rdf_data.py"
        if os.path.isfile(py_output_file):
            os.remove(py_output_file)

        # Initializing output py file
        data_names = ['distance', 'rdf']
        for data_name in data_names:
            initialize_variables = {}
            initialize_variables['data_name'] = data_name
            initialize_variables['data'] = '{}'
            write_data_in_py(py_output_file, initialize_variables)

        no_of_snapshots = len(traj)
        print(f"{legends[legend_no]}.xyz has {no_of_snapshots} snapshots.")
        if no_of_snapshots == 1:
            no_of_snapshots = slice(0, 1, 1)
        elif no_of_snapshots > 1 and no_of_snapshots <= 100:
            no_of_snapshots = slice(-no_of_snapshots, -1, 1)
        else:
            no_of_snapshots = slice(-100, -1, 1)
            print("Only last 100 snapshots will be used to caculate RDF!!!")

        ana = Analysis(traj)
        rMax = 9.0
        if traj[0].cell[0][0] < 2*rMax:
            rMax = traj[0].cell[0][0] * 0.45
        nBins = 200
 
        fontsize = 12
        i = 0
        for e_i, element1 in enumerate(symbols):
            for e_j, element2 in enumerate(symbols):
                if e_j >= e_i:
                    atomic_number1 = atomic_numbers[element1]
                    atomic_number2 = atomic_numbers[element2]
                    print(f"Calculating RDF for {element1}-{element2} pair in {traj_files[legend_no]}") 
                    rdf = ana.get_rdf(rMax,
                                      nBins,
                                      imageIdx= no_of_snapshots,
                                      elements=(atomic_number1, atomic_number2))[0]
 
                    x = (np.arange(nBins) + 0.5) * rMax / nBins
                    y = rdf
                    # Plotting
                    if no_of_subplots == 1:
                        axs.plot(x, rdf, label = legends[legend_no])
                        axs.set_ylabel(f"RDF[{element1}-{element2}]", fontsize=fontsize)
                     
                        axs.tick_params(axis='both',
                                which='both',
                                direction='in',
                                length = 7,
                                left=True,
                                top=True,
                                right=True,
                                labelleft=True,
                                labelsize = 12,
                                pad = 3)
                        axs.set_xlim(0.5, 9)
                    else:
                        axs[i].plot(x, rdf, label = legends[legend_no])
                        axs[i].set_ylabel(f"RDF[{element1}-{element2}]", fontsize=fontsize)
                     
                        axs[i].tick_params(axis='both',
                                which='both',
                                direction='in',
                                length = 7,
                                left=True,
                                top=True,
                                right=True,
                                labelleft=True,
                                labelsize = 12,
                                pad = 3)
                        axs[i].set_xlim(0.5, 9)
                        #plt.yticks(fontsize=15)
                        i += 1
                    for key, arg in zip([f"distance['{element1}_{element2}']", f"rdf['{element1}_{element2}']"], [list(x), list(rdf)]):
                        output_data = {}
                        output_data['data_name'] = key
                        output_data['data'] = arg
                        write_data_in_py(py_output_file, output_data)
    if no_of_subplots == 1:
        axs.legend(ncol=7, loc = 'best', fontsize=fontsize, frameon=False)
        axs.set_xlabel(r"Distance ($\AA$)", fontsize=fontsize)
    else:
        axs[0].legend(ncol=7, loc = 'best', fontsize=fontsize, frameon=False)
        axs[no_of_subplots-2].set_xlabel(r"Distance ($\AA$)", fontsize=fontsize)

    plt.savefig(out_directory+'rdf.png', bbox_inches='tight')
    print("Done!!!")
    return
<<<<<<< HEAD
=======
'''
>>>>>>> 5bcf4f0 (plot_lammps_md added)
