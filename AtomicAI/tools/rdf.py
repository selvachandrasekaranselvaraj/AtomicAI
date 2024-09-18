import warnings

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