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
        selected_vts_indices.append(int(tot_dim / 100 * i))
    selected_tot_dim = tot_dim - 1 - np.array(selected_vts_indices)
    return raw_vts[selected_vts_indices], selected_tot_dim  #selected vt_list


def inputs_for_tslpp():
    '''

    '''
    in_dir1 = './descriptors/train_data/'
    in_dir2 = './descriptors/test_data/'
    train_des_files, test_des_files = [], []
    train_des_files.extend(
        sorted([f for f in os.listdir(in_dir1) if '.dat' in f]))
    train_des_files.extend(
        sorted([f for f in os.listdir(in_dir1) if '.csv' in f]))
    if len(train_des_files) > 0:
        print(f'Available train input files are {train_des_files}')
    else:
        print('No input train descriptor file HERE!!!')
        exit()

    test_des_files.extend(
        sorted([f for f in os.listdir(in_dir2) if '.dat' in f]))
    test_des_files.extend(
        sorted([f for f in os.listdir(in_dir2) if '.csv' in f]))
    if len(test_des_files) > 0:
        print(f'Available test input files are {test_des_files}')
    else:
        print('No input test descriptor file HERE!!!')
        exit()
    print('*********')

    dim_reduc_model = 'TsLPP'
    out_directory = f'./dim_reduction/{dim_reduc_model}/'
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)

    rd, ra = data_lib.descriptor_cutoff['Si_Si']
    descriptors = data_lib.descriptors
    final_reduced_dimensions = data_lib.final_reduced_dimensions

    columns = [
        'Final_red_dimensions', 'Descriptor', 'Dim_reduction_model', 'rd',
        'ra', 'VT', 'Sigma', 'Inter_dimensions',
        'Total dimensions', 'Cluster number', 'PseudoF'
    ]

    input_variables = []

    for final_reduced_dim in final_reduced_dimensions:
        for descriptor in descriptors:
            for a in ra:
                for d in rd:

                    train_des_file = f'{in_dir1}{descriptor}_{d}_{a}_Si_Si.csv'
                    test_des_file = f'{in_dir2}{descriptor}_{d}_{a}_Si_Si.csv'
                    train_cluster_data_file = f'{out_directory}train_cluster_score.csv'
                    predicted_cluster_data_file = f'{out_directory}Predicted_cluster_score.csv'
                    df = pd.DataFrame(columns=columns)

                    df.to_csv(train_cluster_data_file,
                              sep='\t',
                              encoding='utf-8',
                              mode='w')
                    df.to_csv(predicted_cluster_data_file,
                              sep='\t',
                              encoding='utf-8',
                              mode='w')

                    if os.path.isfile(train_des_file):
                        train_df = pd.read_csv(train_des_file,
                                               header=0,
                                               sep='\t',
                                               encoding='utf-8',
                                               index_col=0)
                        vt_list, tot_dim_list = find_vt_list(train_df)
                        for vt, tot_dim in zip(vt_list, tot_dim_list):
                            for sigma in data_lib.sigmas:
                                for inter_dim in data_lib.intermediate_reduced_dimensions:
                                    if inter_dim > final_reduced_dim and inter_dim < int(
                                            tot_dim * 0.75):
                                        variables = [
                                            train_des_file,
                                            test_des_file,
                                            train_cluster_data_file,
                                            predicted_cluster_data_file,
                                            final_reduced_dim, descriptor, a,
                                            d, out_directory,
                                            dim_reduc_model, sigma,
                                            inter_dim, vt, tot_dim
                                        ]
                                        input_variables.append(variables)
                    else:
                        print(
                            f'{descriptor}_{d}_{a}_Si_Si.csv is NOT available')

    return input_variables


def perform_tslpp(variables):
    train_descriptor_file = variables[0]
    test_descriptor_file = variables[1]
    train_cluster_data_file = variables[2]
    predicted_cluster_data_file = variables[3]
    reduced_dim = variables[4]
    descriptor = variables[5]
    a = variables[6]
    d = variables[7]
    out_directory = variables[8]
    reduced_dim_model = variables[9]
    sigma = variables[10]
    intermediate_dimension = variables[11]
    vt = float(variables[12])
    tot_dim = variables[13]

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

    train_labels = np.array(train_df['m_labels'])
    train_features = np.array(
        train_df.drop(columns=['m_labels', 'm_sublabels']))

    vt_list, tot_dim_list = find_vt_list(
        train_df.drop(columns=['m_labels', 'm_sublabels']))
    vt_select = vt_list[np.where(tot_dim_list == tot_dim)]

    test_labels = np.array(test_df['m_labels'])
    test_labels1 = np.array(test_df['m_sublabels'])
    test_features = np.array(test_df.drop(columns=['m_labels', 'm_sublabels']))

    vt_model = VarianceThreshold(threshold=vt_select)
    vt_model.fit(train_features)
    vt_train_features = vt_model.transform(train_features)
    vt_test_features = vt_model.transform(test_features)

    sds_model = StandardScaler()
    sds_model.fit(vt_train_features)
    vt_sds_train_features = sds_model.transform(vt_train_features)
    vt_sds_test_features = sds_model.transform(vt_test_features)

    #tot_dim = np.shape(vt_sds_train_features)[1]
    l1 = reduced_dim
    l2 = reduced_dim_model
    l3 = descriptor
    l4 = d
    l5 = a
    l6 = sigma
    l7 = intermediate_dimension
    l8 = tot_dim
    l9 = str('{:.1e}'.format(float(vt)))

    label = f'{l1}D_{l2}_{l3}_{l4}_{l5}_{l6}_{l7}_{l8}_{l9}'
    outfile = f'{out_directory}{label}.csv'

    reduced_dimensions_data = None
    outfile_labels = []
    tslpp_model = TsLpp()
    try:
        train_reduced_dimensions_data = tslpp_model.fit(
            vt_sds_train_features,
            inter_dimension=int(intermediate_dimension),
            final_dimension=int(reduced_dim),
            sigma=sigma,
        )
        test_reduced_dimensions_data = tslpp_model.transform(
            vt_sds_test_features)

    except np.linalg.LinAlgError:
        print((outfile, "EXITS due to error."))
        exit()

    if train_reduced_dimensions_data is not None:
        train_PseudoF_max, test_PseudoF_max = 0.0, 0.0
        for c_no in data_lib.cluster_numbers:
            c_model = KMeans(
                n_clusters=c_no,
                random_state=10).fit(train_reduced_dimensions_data)
            train_labels = c_model.labels_
            predicted_labels = c_model.predict(test_reduced_dimensions_data)
            train_PseudoF = sklearn.metrics.calinski_harabasz_score(
                train_reduced_dimensions_data, train_labels)
            test_PseudoF = sklearn.metrics.calinski_harabasz_score(
                test_reduced_dimensions_data, predicted_labels)
            if train_PseudoF > train_PseudoF_max:
                train_PseudoF_max = train_PseudoF
                best_train_P = train_PseudoF
                best_train_c_no = c_no
            if test_PseudoF > test_PseudoF_max:
                test_PseudoF_max = test_PseudoF
                best_test_P = test_PseudoF
                best_test_c_no = c_no

       #if int(reduced_dim) == 2:
       #    columns_2d = ['D1', 'D2']
       #    df_2d = pd.DataFrame(test_reduced_dimensions_data,
       #                         columns=columns_2d)
       #    df_2d['p_labels'] = [
       #        f'{l1}_{l2}' for l1, l2 in zip(test_labels1, predicted_labels)
       #    ]
       #    df_2d['m_labels'] = [
       #        f'{l1}_{l2}' for l1, l2 in zip(test_labels, test_labels1)
       #    ]
       #    df_2d.to_csv(outfile, sep='\t', encoding='utf-8')

       #if int(reduced_dim) == 3:
       #    columns_3d = ['D1', 'D2', 'D3']
       #    df_3d = pd.DataFrame(test_reduced_dimensions_data,
       #                         columns=columns_3d)
       #    df_3d['p_labels'] = [
       #        f'{l1}_{l2}' for l1, l2 in zip(test_labels1, predicted_labels)
       #    ]
       #    df_3d['m_labels'] = [
       #        f'{l1}_{l2}' for l1, l2 in zip(test_labels, test_labels1)
       #    ]
       #    df_3d.to_csv(outfile, sep='\t', encoding='utf-8')

        columns = [
                'Final_red_dimensions', 'Descriptor', 'Dim_reduction_model', 'rd',
                'ra', 'VT', 'Sigma', 'Inter_dimensions',
                'Total dimensions', 'Cluster number', 'PseudoF'
        ]

        vt = '{:.1e}'.format(float(vt))
        train_c_data = [[
            reduced_dim, descriptor, reduced_dim_model, d, a, vt, sigma,
            intermediate_dimension, tot_dim, best_train_c_no,
            int(best_train_P)
        ]]
        test_c_data = [[
            reduced_dim, descriptor, reduced_dim_model, d, a, vt, sigma,
            intermediate_dimension, tot_dim, best_test_c_no,
            int(best_test_P)
        ]]
        df_train = pd.DataFrame(train_c_data, columns=columns)
        df_test = pd.DataFrame(test_c_data, columns=columns)
        df_train.to_csv(train_cluster_data_file,
                        sep='\t',
                        encoding='utf-8',
                        mode='a',
                        header=False)
        df_test.to_csv(predicted_cluster_data_file,
                       sep='\t',
                       encoding='utf-8',
                       mode='a',
                       header=False)

        print(label, "is DONE.")

    else:
        pass
    return


def perform_tslpp_hyperparameters():
    pool = multiprocessing.Pool(no_mpi_processors)
    jobs = []
    input_variables = inputs_for_tslpp()
    for variables in input_variables:
        jobs.append(pool.apply_async(perform_tslpp, args=(variables, )))
    results = [job.get() for job in jobs]
    print()
    print('All Jobs done')
    print('*************')
    return
