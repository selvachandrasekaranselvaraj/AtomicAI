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


def RDF():  
    # Searching md.stats file in the persent directory
    traj_files = [f for f in listdir('./') if '.xyz' in f]
    if len(traj_files) > 0:
        print(f"Availabel files are {' '.join(traj_files)}.")
    else:
        print("No traj.xyz file is availabel HERE!!!")
        exit()

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
    no_of_subplots = -1
    for e_i, element1 in enumerate(symbols):
        for e_j, element2 in enumerate(symbols):
            no_of_subplots += 1
    print(no_of_subplots, symbols)
    fig, axs = plt.subplots(no_of_subplots, sharex=True, figsize=(9, 5))
    for legend_no, traj in enumerate(traj_data):
        ana = Analysis(traj)
        rMax = 9.0
        nBins = 200
 
        fontsize = 12
        i = 0
        for e_i, element1 in enumerate(symbols):
            for e_j, element2 in enumerate(symbols):
                if e_j >= e_i:
                    atomic_number1 = atomic_numbers[element1]
                    atomic_number2 = atomic_numbers[element2]
 
                    rdf = ana.get_rdf(rMax,
                                      nBins,
                                      imageIdx=None,
                                      elements=(atomic_number1, atomic_number2))[0]
 
                    x = (np.arange(nBins) + 0.5) * rMax / nBins
                    y = rdf
                    # Plotting
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
    axs[0].legend(ncol=7, loc = 'best', fontsize=fontsize, frameon=False)
    axs[no_of_subplots-1].set_xlabel(r"Distance ($\AA$)", fontsize=fontsize)
    plt.savefig('rdf.png')
    return
