import warnings
warnings.filterwarnings("ignore")
import sys, os
import numpy as np
import pandas as pd
from AtomicAI.data import data_lib 

from AtomicAI.dim_reduction.ts_lpp import TsLpp
import pickle
from AtomicAI.dim_reduction.select_descriptors import select_descriptors

import time, multiprocessing
from AtomicAI.dim_reduction.outputs_for_dim_reduction import outputs_for_dim_reduction
from AtomicAI.data.data_lib import no_mpi_processors

import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def find_vt_list(df):
    raw_vts = np.sort(np.array(df.var()))
    tot_dim = len(raw_vts)
    selected_vts_indices = [0]
    for i in range(5, 51, 10):  # 5% to 50% of data to remove
        selected_vts_indices.append(int(tot_dim/100*i))
    selected_vts_indices
    return raw_vts[selected_vts_indices]  #selected vt_list


def inputs_for_tslpp():
    '''

    '''
    in_dir = './descriptors/train_data/'
    des_files = []
    des_files.extend(sorted([f for f in os.listdir(in_dir) if '.dat' in f]))
    des_files.extend(sorted([f for f in os.listdir(in_dir) if '.csv' in f]))
    if len(des_files) > 0 :
        print(f'Available input files are {des_files}')
    else:
        print('No input descriptor file HERE!!!')
        exit()
    dim_reduc_model = 'TsLPP'
    out_directory = f'./dim_reduction/{dim_reduc_model}/'
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)
    rd, ra = data_lib.descriptor_cutoff['Si_Si']
    descriptors = data_lib.descriptors 
    final_reduced_dimensions = data_lib.final_reduced_dimensions
    jobs = []
    for final_reduced_dim in final_reduced_dimensions:
        for descriptor in descriptors:
            for a in ra:
                for d in rd:
                    des_file = f'{in_dir}{descriptor}_{d}_{a}_Si_Si.csv'
                    cluster_data_file = f'{out_directory}{final_reduced_dim}D_{descriptor}_{d}_{a}_cluster_score.csv'
                    df = pd.DataFrame(columns = [
                        'sigma', 
                        'Inter_dimensions', 
                        'PseudoF_score', 
                        'Number_of_clusters', 
                        'VT', 
                        'Total dimensions'
                        ])
                    if os.path.isfile(cluster_data_file):
                        os.remove(cluster_data_file)
                        df.to_csv(cluster_data_file, sep='\t', encoding='utf-8', mode='w')
                    else:
                        df.to_csv(cluster_data_file, sep='\t', encoding='utf-8', mode='w')
                    if os.path.isfile(des_file):
                        raw_datas = select_descriptors(des_file, descriptor)
                        df_ = pd.DataFrame(raw_datas[0])
                        vt_list = find_vt_list(df_)
                        for vt in vt_list:
                            tot_dimensions = np.sum(np.array(df_.var() > vt))
                            variables = [des_file, 
                                    cluster_data_file, 
                                    final_reduced_dim, 
                                    tot_dimensions, 
                                    descriptor, 
                                    a, 
                                    d, 
                                    vt]
                            jobs.append(variables)
                    else:
                        print(f'{descriptor}_{d}_{a}_Si_Si.csv is NOT available')

    input_variables = []
    for job in jobs:
        des_file, cluster_data_file, final_reduced_dim, tot_dimensions, descriptor, a, d, vt = job
        for sigma in data_lib.sigmas:
            for inter_dim in data_lib.intermediate_reduced_dimensions:
                if inter_dim > final_reduced_dim and inter_dim  < 75: #int(tot_dimensions*0.6):
                    input_variables.append((
                        des_file, 
                        cluster_data_file, 
                        final_reduced_dim, 
                        descriptor, 
                        a, 
                        d, 
                        out_directory, 
                        dim_reduc_model, 
                        sigma, 
                        inter_dim, 
                        vt
                        ))
    return input_variables




def perform_tslpp(variables):
    descriptor_file, cluster_data_file, reduced_dim, descriptor, a, d, out_directory, reduced_dim_model, sigma, intermediate_dimension, vt = variables
    descriptor_file = variables[0]
    selected_descriptors = select_descriptors(descriptor_file, descriptor)
    raw_data = selected_descriptors[0]
    vt_data = VarianceThreshold(threshold=vt).fit_transform(selected_descriptors[0])

    train_features_vt_sds = pd.DataFrame(StandardScaler().fit_transform(vt_data))
    train_labels = selected_descriptors[3]
    tot_dim = np.shape(train_features_vt_sds)[1]
    label = f'{reduced_dim}D_{reduced_dim_model}_{descriptor}_{d}_{a}_{sigma}_{intermediate_dimension}_{tot_dim}'
    outfile = f'{out_directory}{str(reduced_dim)}D_{reduced_dim_model}_{descriptor}_{d}_{a}_{sigma}_{intermediate_dimension}_{tot_dim}'

    reduced_dimensions_data = None
    outfile_labels = []

    model = TsLpp()
    try:
        reduced_dimensions_data = model.fit(np.array(train_features_vt_sds),
                                            inter_dimension = intermediate_dimension,
                                            final_dimension = reduced_dim,
                                            sigma = sigma,
                                            ).transpose()
    except np.linalg.LinAlgError:
        print(outfile, "EXITS due to error.")
        exit()

    if reduced_dimensions_data is not None:

      # df = pd.DataFrame()
      # for i in range(len(reduced_dimensions_data)):
      #     writing_dim = i + 1
      #     df[label + f'_D{writing_dim}'] = reduced_dimensions_data[i]
      # df['m_labels'] = train_labels
      # vt = '{:.1e}'.format(float(vt))
      # df.to_csv(f'{outfile}_{vt}_{tot_dim}.csv', sep='\t', encoding='utf-8')

        PseudoF_max,best_PseudoF, best_cluster_no = 0.0, [], []
        for c_no in data_lib.cluster_numbers:
            c_model = KMeans(n_clusters=c_no, random_state=10).fit(reduced_dimensions_data.T)
            labels = c_model.labels_
            PseudoF = sklearn.metrics.calinski_harabasz_score(reduced_dimensions_data.T, labels)
            if PseudoF > PseudoF_max:
                best_P = PseudoF
                best_c_no = c_no
        columns = ['sigma', 'Inter_dimensions', 'PseudoF_score', 'Number_of_clusters','VT','Total dimensions']
        c_data = [[sigma, intermediate_dimension, best_P, best_c_no, vt, tot_dim]]
        df_ = pd.DataFrame(c_data, columns=columns)
        df_.to_csv(cluster_data_file, sep='\t', encoding='utf-8', mode='a', header=False)
        print(label, "is DONE.")
    else:
        pass
    return


def perform_tslpp_hyperparameters():
    pool = multiprocessing.Pool(no_mpi_processors)
    jobs = []
    input_variables = inputs_for_tslpp()
    for variables in input_variables:
        #print(variables)
        # variables = des_file, reduced_dim, descriptor, a, d, out_directory, dim_reduc_model, sigma, inter_dim, vt
        jobs.append(pool.apply_async(perform_tslpp, args=(variables,)))
    results = [job.get() for job in jobs]
    print()
    print('All Jobs done')
    print('*************')
    return


