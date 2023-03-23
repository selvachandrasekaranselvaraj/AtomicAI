import warnings
warnings.filterwarnings("ignore")
import sys, os
import numpy as np
import pandas as pd
from AtomicAI.data import data_lib 

def inputs_for_dim_reduction():
    '''

    '''
    in_dir = './descriptors/Processed_data/'
    des_files = []
    des_files.extend(sorted([f for f in os.listdir(in_dir) if '.dat' in f]))
    des_files.extend(sorted([f for f in os.listdir(in_dir) if '.csv' in f]))
    if len(des_files) > 0 :
        print(f'Available input files are {des_files}')
    else:
        print('No input descriptor file HERE!!!')
        exit()
    out_directory = './dim_reduction/'
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)
    rd, ra = data_lib.descriptor_cutoff['Si_Si']
    descriptors = data_lib.descriptors 
    dim_reductions = data_lib.final_reduced_dimensions
    dim_reduc_models = data_lib.dim_reduction_models 
    jobs = []
    for reduced_dim in dim_reductions:
        for descriptor in descriptors:
            for a in ra:
                for d in rd:
                   #des_file = f'{in_dir}{descriptor}_{d}_{a}_Si_Si.dat'
                   #if os.path.isfile(des_file):
                   #    variables = [des_file, reduced_dim, descriptor, a, d]
                   #    jobs.append(variables)
                   #else:
                   #    print(f'{descriptor}_{d}_{a}_Si_Si.dat is NOT available')
                    des_file = f'{in_dir}train_{descriptor}_{d}_{a}_Si_Si.csv'
                    if os.path.isfile(des_file):
                        variables = [des_file, reduced_dim, descriptor, a, d]
                        jobs.append(variables)
                    else:
                        print(f'{descriptor}_{d}_{a}_Si_Si.csv is NOT available')
    #results = [job for job in jobs]
    #results = [['./descriptors/3D_TsLPP_SOAP_1.0_7.0_7_25_324.csv', 2, 'SOAP', 1.0, 7.0]]


    input_variables = []
    for job in jobs:
        des_file, reduced_dim, descriptor, a, d = job
        for dim_reduc_model in dim_reduc_models:
            out_directory = f'./dim_reduction/{dim_reduc_model}/'
            if not os.path.isdir(out_directory):
                os.makedirs(out_directory)
            if dim_reduc_model == 'LPP':
                for sigma in data_lib.sigmas:
                    input_variables.append((des_file, reduced_dim, descriptor, a, d, out_directory, dim_reduc_model, sigma, None))
            if dim_reduc_model == 'TsLPP':
                for sigma in data_lib.sigmas:
                    for inter_dim in data_lib.intermediate_reduced_dimensions:
                        if inter_dim > reduced_dim:
                            input_variables.append((des_file, reduced_dim, descriptor, a, d, out_directory, dim_reduc_model, sigma, inter_dim))
            if dim_reduc_model in ['PCA', 'TsNE', 'UMAP']:
                    input_variables.append((des_file, reduced_dim, descriptor, a, d, out_directory, dim_reduc_model, None, None))
    return input_variables

