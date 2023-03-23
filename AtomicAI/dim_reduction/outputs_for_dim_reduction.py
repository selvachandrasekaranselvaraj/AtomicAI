import warnings
warnings.filterwarnings("ignore")
import sys, os
import numpy as np
import pandas as pd

from AtomicAI.data import data_lib 

def features_treatment(des_file, count):
    selected_descriptors = select_descriptors_and_manual_labels(des_file, count)
    train_features = selected_descriptors[0]
    train_features_vt = VarianceThreshold(threshold=1e-16).fit_transform(train_features)
    train_features_vt_sds = StandardScaler().fit_transform(train_features_vt)
    return train_features_vt_sds

def outputs_for_dim_reduction():
    '''

    '''
    out_directory = './dim_reduction/'
    rd, ra = data_lib.descriptor_cutoff['Si_Si']
    descriptors = data_lib.descriptors
    dim_reductions = [2]
    dim_reduc_models = data_lib.dim_reduction_models


    print("Concardinating output files...")

    for reduced_dim in dim_reductions:
        for descriptor in descriptors:
            for dim_reduc_model in dim_reduc_models:
                outfile = f'./dim_reduction/2D_{descriptor}_{dim_reduc_model}.csv'
                A = pd.DataFrame()
                files= []
                for a in ra:
                    for d in rd:
                        if dim_reduc_model in ['LPP', 'TsLPP']:
                            for sigma in data_lib.sigmas:
                                data_file = f'{out_directory}{dim_reduc_model}/2D_{dim_reduc_model}_{descriptor}_{a}_{d}_{sigma}.csv'
                                if not os.path.isfile(data_file):
                                    print(data_file, "is NOT availabel.")
                                else:
                                    df = pd.read_csv(data_file, header=0, sep='\t', encoding='utf-8', index_col=0)
                                    A = pd.concat([A, df], axis=1)
                                    files.append(True)
                        else:
                            data_file = f'{out_directory}{dim_reduc_model}/2D_{dim_reduc_model}_{descriptor}_{a}_{d}.csv'
                            if not os.path.isfile(data_file):
                                print(data_file, "is NOT availabel.")
                            else:
                                df = pd.read_csv(data_file, header=0, sep='\t', encoding='utf-8', index_col=0)
                                A = pd.concat([A, df], axis=1)
                                files.append(True)
                if np.sum(np.array(files)) > 0:
                    A.to_csv(outfile, sep='\t', encoding='utf-8')

    return
