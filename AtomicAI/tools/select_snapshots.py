import warnings
warnings.filterwarnings("ignore")
import os, sys
import ase.io
import numpy as np


def select_snapshots():
    try:
        input_file = sys.argv[1]
    except IndexError:
        print("Input error!!!!")
        print('Usage: "structure_analysis traj_file_name with .xyz extension"')
        sys.exit(1)

    frames = ase.io.read(input_file, ':')
    total_frames = len(frames)
    print('Total number of frames in this trajectory:', total_frames)

    symbols = np.array([list(frame.symbols) for frame in frames]).flatten()
    total_number_of_data = 20000
    if len(symbols) > total_number_of_data:
        no_of_atoms = len(list(frames[-1].symbols))
        no_of_selected_frames = int(total_number_of_data / no_of_atoms) + 1
        frame_interval = max(1, int((total_frames / no_of_selected_frames) * 0.5))
        frame_indices = slice(
            -total_frames + 1,
            -total_frames + 1 + no_of_selected_frames * frame_interval,
            frame_interval,
        )
        print('Selected frame indices:', frame_indices)
        selected_frames = frames[frame_indices]
        symbols = np.array([list(frame.symbols) for frame in selected_frames]).flatten()
    else:
        selected_frames = frames

    return selected_frames, symbols
