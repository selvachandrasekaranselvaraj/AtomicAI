import warnings
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
        py_output_file = out_directory+legends[legend_no]+"_rdf.py"
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
                    for key, arg in zip([f"distance['{element1}_{element2}']", f"rdf['{element1}_{element2}']"], [x, rdf]):
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

    plt.savefig(out_directory+'rdf.png')
    plt.show()
    return
