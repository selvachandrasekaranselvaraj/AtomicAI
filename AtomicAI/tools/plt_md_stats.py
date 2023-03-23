import matplotlib.pyplot as plt
import numpy as np
import sys, os
from os import listdir
from os.path import isfile, join

def plot(axs, data):
    fontsize = 12
    time = data['steps'] #* 0.05
    axs[2].plot(time, data['Temp'])
    axs[1].plot(time, data['PE'], label=data['name'])
    axs[0].plot(time, data['KE'])

    axs[1].legend(ncol=3, loc = 'best', fontsize=fontsize, frameon=False)
    ylabel = ['KE(eV)', 'PE(eV)', 'T(K)']
    axs[2].set_xlabel('No. of MD steps', fontsize=fontsize)
    for i in [0, 1, 2]:
        axs[i].set_xlim(min(time), max(time))
        axs[i].set_ylabel(ylabel[i], fontsize=fontsize)
        axs[i].tick_params(axis='both',
                which='both',
                direction='in',
                length = 7,
                left=True,
                top=True,
                right=True,
                labelleft=True,
                labelsize = 12,
                pad = 1)

    return axs


def plt_md_stats():
    # Searching md.stats file in the persent directory
    md_files = [f for f in listdir('./') if '.stats' in f]
    if len(md_files) > 0:
        print(f"Availabel files are {' '.join(md_files)}.")
    else:
        print("No md.stats file is availabel HERE!!!")
        exit()
    fig, axs = plt.subplots(3, sharex=True, figsize=(9, 5))
    no_of_steps = []
    for md_file in md_files:
        data__ =  np.loadtxt(md_file, skiprows = 1).transpose()
        keys = ['steps', 'PE', 'KE', 'thermostat', 'H_prim', 'Temp', 'P']
        data = {}
        data['name'] = md_file[:-6]
        for key, data_ in zip(keys, data__):
            data[key] = data_
        plot(axs, data)
        no_of_steps.append(max(data['steps']))
    for ax in axs:
        ax.set_xlim(1, min(no_of_steps))

    plt.savefig('md_stats.png', bbox_inches='tight')
    plt.show()
    return 
