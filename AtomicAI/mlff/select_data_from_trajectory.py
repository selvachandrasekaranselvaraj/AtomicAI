import ase.io
import pandas as pd
import numpy as np
from collections import Counter

###############################################################
def select_data_from_trajectory():
    try:
        input_file = 'trajectory.xyz'  #sys.argv[1]
    except:
        print("Input error!!!!")
        print(
            "Usage: \"structure_analysis traj_file_name with .xyz extension\"")
        print()
        exit()

    frames = ase.io.read(input_file, ':')
    #frames = frames[int(len(frames) - len(frames) *
    #                    0.75):len(frames)]  # cut first 25% of frames
    total_no_of_frames = len(frames)

    print('Total number of frames in this trajectory are: ',
          total_no_of_frames)
    symbols_list = np.array([list(frame.symbols)
                             for frame in frames]).flatten()
    print(Counter(symbols_list))

    frames_index_list = np.array([[n] * len(frame)
                                  for n, frame in enumerate(frames)
                                  ]).flatten()
    atoms_index_list = np.array([[i for i in range(len(frame))]
                                 for frame in frames]).flatten()
    df = pd.DataFrame()
    df['Frames indices'] = frames_index_list
    df['Atomic indices'] = atoms_index_list
    df.index = symbols_list

    MLFF = {}

    for symbol in set(symbols_list):
        no_of_data = df.loc[symbol].shape[0]
        if no_of_data >= df.loc[symbol].shape[0]:
            no_of_data = df.loc[symbol].shape[0] - 5
        #df_ = df.loc[symbol].sample(no_of_data)
        df_ = df.loc[symbol].tail(no_of_data)
        df_.index = df_['Frames indices']
        slected_frames_index_list = list(set(df_['Frames indices']))
        slected_atoms_index_list_ = [
            np.array(df_['Atomic indices'].loc[f_i])
            for f_i in slected_frames_index_list
        ]
        slected_atoms_index_list = [
            list(arr) if arr.shape else [int(arr)]
            for arr in slected_atoms_index_list_
        ]
        MLFF[symbol] = {
            'Selected_frames_indices': slected_frames_index_list,
            'Selected_atoms_indices': slected_atoms_index_list,
            'Number of data' : no_of_data
        }
    return frames, MLFF
