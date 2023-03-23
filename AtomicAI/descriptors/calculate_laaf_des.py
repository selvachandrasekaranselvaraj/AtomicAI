from AtomicAI.descriptors.laaf import AverageFingerprintCalculator
#from ConquesTools.analysis.laaf import AverageFingerprintCalculator
from AtomicAI.data.descriptor_cutoff import descriptor_cutoff
import sys, os
import ase.io
import numpy as np
import pandas as pd
def calculate_laaf_des():
    try:
        traj_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"laaf traj_file with .xyz extension\"")
        print()
        exit()

    out_directory = './descriptors/'
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)

    selected_snapshots = ':'
    trajs = ase.io.read(traj_file, selected_snapshots)
    symbols = np.array([list(traj.symbols) for traj in trajs]).flatten()
    if len(trajs) > 1000 and len(symbols) > 500000:
        trajs = trajs[-500:] 
        symbols = np.array([list(traj.symbols) for traj in trajs]).flatten()
        symbols_type = list(set(symbols))
    elif len(trajs) > 1000 and len(symbols) < 500000:
        trajs = trajs[-1000:] 
        symbols = np.array([list(traj.symbols) for traj in trajs]).flatten()
        symbols_type = list(set(symbols))
    else:
        symbols_type = list(set(symbols))

    symbols = np.array(symbols)
    selected_indices = []
    target_elements = {}
    for sy_no, symbol in enumerate(symbols_type):
        target_elements[symbol] = sy_no
        indices = np.where(symbols == symbol)[0]
        #if len(indices) > 5000:
        #    indices = np.random.choice(len(indices), size=5000)
        selected_indices.append(indices)
        #print(selected_indices)


    descriptors_type = ['ACSF_G2', 'ACSF_G2G4', 'SOAP']


    number_of_eta = 50 # number of decay functions
    for des_type in descriptors_type:
        for i in range(len(symbols_type)):  # Target_specie or target_element
            t_specie = symbols_type[i]  # target specie
            for j in range(len(symbols_type)): # target_neighbor_element
                if i >= j:
                    tne = symbols_type[j]  # target_neighbor_element
                    try:
                        d_cutoff = descriptor_cutoff[t_specie+'_'+tne]
                    except:
                        print(f'Descriptor cutoff data is not available for {t_specie}-{tnr} \n in AtomicAI/data/descriptor_cutoff.py file')
                        exit()
                    for d in d_cutoff: 
                        a_cutoff = [1.0] + d_cutoff
                        for a in a_cutoff:
                            r_d = round(float(d),1) # Descriptor cutoff
                            r_a = round(float(a),1) # Averaging cutoff
                            if 'ACSF' in des_type:
                                des_name = des_type.split('_')[1]
                            else:
                                des_name = des_type
                           
                            my_laaf = AverageFingerprintCalculator(
                                cutoff_descriptor=r_d,
                                cutoff_average=r_a,
                                traj_data=trajs,
                                selected_snapshots=selected_snapshots,
                                number_of_eta=number_of_eta,
                                element_conversion=target_elements,
                                descriptor_type=des_type,
                            )
                            laaf_file = f'{out_directory}{des_name}_{r_d}_{r_a}_{t_specie}_{tne}.dat'
                            print(laaf_file)
                            my_laaf.compute_averaged_fingeprints_selection(
                                #append= tne_id > 0,
                                output_file=laaf_file,
                                target_element=target_elements[t_specie],
                                target_neighbor_element = target_elements[tne], 
                                selected_atoms=None #list(selected_atom_indices)
                            )
    return        
