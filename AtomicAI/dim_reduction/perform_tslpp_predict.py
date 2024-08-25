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


def find_vt_list(df, tot_dim_after_vt):
    raw_vts = np.sort(np.array(df.var()))
    dim_list=np.arange(1, len(raw_vts)+1)[::-1]
    vt = raw_vts[dim_list==int(tot_dim_after_vt)]
    return vt[0]



def inputs_for_tslpp():
    '''

    '''
    in_dir = './descriptors/'
    if not os.path.isdir(in_dir):
        print(f"{in_dir} directory is not available here")
        exit()
    train_des_files, test_des_files = [], []
    train_des_files.extend(
        sorted([f for f in os.listdir(in_dir) if 'train' in f]))
    test_des_files.extend(
        sorted([f for f in os.listdir(in_dir) if 'test' in f]))
    if len(train_des_files) == 0 or len(test_des_files) == 0:
        print('No input train and/or test descriptor files in ./descriptors/ directory!!!')
        exit()

    dim_reduction_model = 'TsLPP'
    out_directory = f'./dim_reduction/{dim_reduction_model}/'


#    rd, ra = data_lib.descriptor_cutoff['Si_Si']
#    descriptors = data_lib.descriptors
#    final_reduced_dimensions = data_lib.final_reduced_dimensions

    columns = [
            'final_red_dimensions', 'Descriptor', 'Dim_reduction_model', 'rd',
            'ra', 'opt_vt', 'opt_sigma', 'opt_inter_dimensions',
            'Total dimensions', 'cluster number', 'PseudoF'
        ]

    input_variables = []
    file1 = f'./dim_reduction/TsLPP/train_and_predicted_cluster_score_optimized.csv'
    file2 = f'./dim_reduction/TsLPP/train_cluster_score_optimized.csv'
    optimized_hyperparamters_df = pd.DataFrame()
    for l_, file in zip(['predicted', 'train'], [file1, file2]):
        if os.path.isfile(file):
            df_input = pd.read_csv(file, sep='\t', header=0, dtype='object')
            df_input['train_or_test'] = [l_] * df_input.shape[0]
            optimized_hyperparamters_df = pd.concat([df_input, optimized_hyperparamters_df ], axis=0)
        else:
            print(f'{file} is not availabel.')
            exit()

    for index, row_ in optimized_hyperparamters_df.iterrows():
        row = np.array(row_)
        dim_reduc_model = row[1]
        descriptor = row[2]
        d, a, vt, sigma, final_reduced_dim, inter_dim, tot_dim_avt, tot_dim_bvt, cluster_no = row[3:12]
        l_ = row[-1]
        train_des_file = f'{in_dir}train_{descriptor}_{d}_{a}_Si_Si.csv'
        test_des_file = f'{in_dir}test_{descriptor}_{d}_{a}_Si_Si.csv'
        print(train_des_file)
        if os.path.isfile(train_des_file):
            var_ = [
                train_des_file, test_des_file,
                final_reduced_dim, descriptor,
                a, d, out_directory, dim_reduc_model, sigma, inter_dim, vt, tot_dim_avt, cluster_no, l_
            ]
            input_variables.append(var_)

        else:
            print(f'{descriptor}_{d}_{a}_Si_Si.csv is NOT available')



    return input_variables


def perform_tslpp(variables):
    train_descriptor_file = variables[0]
    test_descriptor_file = variables[1]
    final_reduced_dim = int(variables[2])
    descriptor = variables[3]
    a = variables[4]
    d = variables[5]
    out_directory = variables[6]
    reduced_dim_model = variables[7]
    sigma = int(variables[8])
    intermediate_dimension = int(variables[9])
    vt = variables[10]
    tot_dim_after_vt = int(variables[11])
    cluster_no = int(variables[12])
    l_ = variables[13]


    train_df = pd.read_csv(train_descriptor_file,
                           header=0,
                           sep='\t',
                           encoding='utf-8',
                           index_col=0)
    test_df = pd.read_csv(test_descriptor_file,
                          header=0,
                          sep='\t',
                          encoding='utf-8',
                          index_col=0)

    m_labels = [f'{l}' for l in train_df['m_labels']]
    m_labels += [f'{l}' for l in test_df['m_labels']]
    m_sublabels = [f'{l}' for l in train_df['m_sublabels']]
    m_sublabels += [f'{l}' for l in test_df['m_sublabels']]

    m_labels1 = [f'{l}' for l in train_df['m_labels']]
    m_labels1 += [f'{l}_{l1}' for l, l1 in zip(test_df['m_labels'], test_df['m_sublabels'])]

    train_features = np.array(
        train_df.drop(columns=['m_labels', 'm_sublabels']))

    vt_select = find_vt_list(
        train_df.drop(columns=['m_labels', 'm_sublabels']), tot_dim_after_vt)
    test_features = np.array(test_df.drop(columns=['m_labels', 'm_sublabels']))
    tot_dim_before_vt = test_features.shape[1]
    vt_model = VarianceThreshold(threshold=vt_select)
    vt_model.fit(train_features)
    vt_train_features = vt_model.transform(train_features)
    vt_test_features = vt_model.transform(test_features)

    sds_model = StandardScaler()
    sds_model.fit(vt_train_features)
    vt_sds_train_features = sds_model.transform(vt_train_features)
    vt_sds_test_features = sds_model.transform(vt_test_features)

    l1 = final_reduced_dim
    l2 = reduced_dim_model
    l3 = descriptor
    l4 = d
    l5 = a
    l6 = sigma
    l7 = intermediate_dimension
    l8 = tot_dim_after_vt
    l9 = tot_dim_before_vt
    l10 = str('{:.1e}'.format(float(vt)))

    label = f'{l_}_{l1}D_{l2}_{l3}_{l4}_{l5}_{l6}_{l7}_{l8}_{l9}'
    outfile = f'{out_directory}{label}.csv'

    tslpp_model = TsLpp()
    try:
        train_reduced_dimensions_data = tslpp_model.fit(
            vt_sds_train_features,
            inter_dimension=int(intermediate_dimension),
            final_dimension=int(final_reduced_dim),
            sigma=sigma,
        )
        predicted_reduced_dimensions_data = tslpp_model.transform(
            vt_sds_test_features)
        train_and_predicted_reduced_dimensions_data = np.concatenate((train_reduced_dimensions_data , predicted_reduced_dimensions_data), axis=0)

    except np.linalg.LinAlgError:
        print((outfile, "EXITS due to error in the TS-LPP dimensions reduction part."))
        exit()

    if train_reduced_dimensions_data is not None:
        c_model = KMeans(
            n_clusters=cluster_no,
            random_state=10).fit(train_reduced_dimensions_data)
        train_labels = c_model.labels_
        predicted_labels = c_model.predict(train_and_predicted_reduced_dimensions_data)

        if int(final_reduced_dim) == 2:
            columns_2d = ['D1', 'D2']
            df_2d = pd.DataFrame(train_and_predicted_reduced_dimensions_data,
                                columns=columns_2d)
            t_p = ['train']*len(train_reduced_dimensions_data)
            t_p += ['predicted']*len(predicted_reduced_dimensions_data)
            df_2d['labels'] = [
               f'{m}_{p}' for m, p in zip(t_p, predicted_labels)
            ]
            df_2d['m_lables'] = m_labels
            df_2d['m_lables1'] = m_labels1
            df_2d['m_sublabels'] = m_sublabels
            df_2d.to_csv(outfile, sep='\t', encoding='utf-8')

        print(label, "is DONE.")

    else:
        pass
    return



def perform_tslpp_predict():
    pool = multiprocessing.Pool(no_mpi_processors)
    jobs = []
    input_variables = inputs_for_tslpp()
    for variables in input_variables:
        #print(variables)
        # variables = des_file, reduced_dim, descriptor, a, d, out_directory, dim_reduc_model, sigma, inter_dim, vt
        jobs.append(pool.apply_async(perform_tslpp, args=(variables, )))
    results = [job.get() for job in jobs]
    print()
    print('All Jobs done')
    print('*************')





    return
