# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")  
<<<<<<< HEAD
import os, sys
=======
import sys
>>>>>>> 5bcf4f0 (plot_lammps_md added)
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
<<<<<<< HEAD
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
=======

    frames =ase.io.read(input_file, ':')
    total_frames = len(frames)
    print('Total number of frames in this trajectory are: ', total_frames )
    selected_frames = frames
    symbols = np.array([list(frame.symbols) for frame in frames]).flatten()
    total_number_of_data = 20000
    if len(symbols) > total_number_of_data :
        no_of_atoms = len(list(frames[-1].symbols))
        no_of_selected_frames = int(total_number_of_data/no_of_atoms) + 1

        frame_interval = int((total_frames/no_of_selected_frames) * 0.5 )
        if frame_interval == 0:
            frame_interval = 1

        frame_indices = slice(-total_frames+1, -total_frames + 1 + (no_of_selected_frames*frame_interval), frame_interval)
        #np.arange(1, no_of_selected_frames) * frame_interval - total_frames - 1
        print('Selected indices: ', frame_indices)
        selected_frames = frames[frame_indices]
        symbols = np.array([list(frame.symbols) for frame in selected_frames]).flatten()

    return selected_frames, symbols
>>>>>>> 5bcf4f0 (plot_lammps_md added)
###############################################################
