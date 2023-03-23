# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")  
import os, sys
import ase.io
import numpy as np

###############################################################
def select_snapshots():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"structure_analysis traj_file_name with .xyz extension\"")
        print()
        exit()
    selected_snapshots = ':'
    trajs = ase.io.read(input_file, selected_snapshots)
    if len(trajs) < 500:
        print('**********************************************')
        print(f'Only {len(trajs)} trajctories are available!!')
        print('**********************************************')
    else:
        trajs = trajs[200:]
    symbols = np.array([list(traj.symbols) for traj in trajs]).flatten()
    if len(trajs) > 1000 and len(symbols) > 500000:
        frames = trajs[-50:]
        symbols = np.array([list(traj.symbols) for traj in trajs]).flatten()
    elif len(trajs) > 1000 and len(symbols) < 500000:
        frames = trajs[-50:]
        symbols = np.array([list(traj.symbols) for traj in trajs]).flatten()
    else:
        frames = trajs
    return frames, symbols
###############################################################
