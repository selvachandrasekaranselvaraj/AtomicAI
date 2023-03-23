import warnings

warnings.filterwarnings("ignore")
import sys, os
import numpy as np
import pandas as pd
from AtomicAI.data import data_lib

from AtomicAI.dim_reduction.ts_lpp import TsLpp
import pickle

import time, multiprocessing
from AtomicAI.data.data_lib import no_mpi_processors

import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


from plotly.subplots import make_subplots
import plotly.graph_objects as go

def find_vt_list(df):
    proposed_no_of_vts = 10
    raw_vts = np.sort(np.array(df.var()))
    tot_dim = len(raw_vts)
    max_selected_tot_dim = tot_dim
    min_selected_tot_dim = int(tot_dim / 100 * 25)

    step = int(len(np.arange(min_selected_tot_dim, max_selected_tot_dim + 1)) / proposed_no_of_vts)
    proposed_tot_dim = np.arange(min_selected_tot_dim, max_selected_tot_dim + 1, step)

    selected_vt, selected_tot_dim = [], []
    for vt in raw_vts[:-1]:
        vt_model = VarianceThreshold(threshold=vt)
        tot_dim_ = vt_model.fit_transform(df).shape[1]
        if tot_dim_ in proposed_tot_dim:
            selected_vt.append(vt)
            selected_tot_dim.append(tot_dim_)
    return selected_vt, selected_tot_dim


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
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)

    rd, ra = data_lib.descriptor_cutoff['Si_Si']
    descriptors = data_lib.descriptors
    final_reduced_dimensions = data_lib.final_reduced_dimensions

    columns = [
        'Dimension reduction model', 'Descriptor',  'rd',
        'ra', 'VT', 'Sigma', 'Final reduced dimensions', 'Intermediate dimensions', 'Total dimensions after VT',
        'Total dimensions before VT', 'Cluster number', 'PseudoF'
    ]

    input_variables = []

    for final_reduced_dim in final_reduced_dimensions:
        for descriptor in descriptors:
            for a in ra:
                for d in rd:
                    train_des_file = f'{in_dir}train_{descriptor}_{d}_{a}_Si_Si.csv'
                    test_des_file = f'{in_dir}test_{descriptor}_{d}_{a}_Si_Si.csv'
                    train_cluster_data_file = f'{out_directory}train_cluster_score.csv'
                    train_and_predicted_cluster_data_file = f'{out_directory}train_and_predicted_cluster_score.csv'
                    df = pd.DataFrame(columns=columns)

                    df.to_csv(train_cluster_data_file,
                              sep='\t',
                              encoding='utf-8',
                              mode='w')
                    df.to_csv(train_and_predicted_cluster_data_file,
                              sep='\t',
                              encoding='utf-8',
                              mode='w')

                    if os.path.isfile(train_des_file):
                        train_df = pd.read_csv(train_des_file,
                                               header=0,
                                               sep='\t',
                                               encoding='utf-8',
                                               index_col=0)

                        vt_list, tot_dim_list = find_vt_list(train_df.drop(columns=['m_labels', 'm_sublabels']))
                        for vt, tot_dim_after_vt in zip(vt_list, tot_dim_list):
                            for sigma in [1, 3, 5, 7, 10, 20, 30, 40, 50]:
                                for inter_dim in [5, 8, 10, 13, 15, 20, 25]:
                                    if inter_dim > final_reduced_dim and inter_dim < int(
                                            tot_dim_after_vt * 0.75):
                                        variables = [
                                            train_des_file,
                                            test_des_file,
                                            train_cluster_data_file,
                                            train_and_predicted_cluster_data_file,
                                            final_reduced_dim, descriptor, a,
                                            d, out_directory,
                                            dim_reduction_model, sigma,
                                            inter_dim, vt, tot_dim_after_vt
                                        ]
                                        input_variables.append(variables)
                    else:
                        print(
                            f'{descriptor}_{d}_{a}_Si_Si.csv is NOT available')

    return input_variables

def plot_hyperparameters(hyperparameter_data_file: str = None):
    if hyperparameter_data_file == None:
        hyperparameter_data_file = f'train_and_predicted_cluster_score.csv'

    df = pd.read_csv(
        hyperparameter_data_file,
        header=0,
        sep='\t',
        encoding='utf-8',
        index_col=0,
    ).reset_index(drop=True)

    optimized_hyperparameter_data_file = hyperparameter_data_file[:-4]+'_optimized.csv'
    optimized_hyperparameters_df = pd.DataFrame(columns=df.columns)
    optimized_hyperparameters_df.to_csv(
        optimized_hyperparameter_data_file,
        sep='\t',
        encoding='utf-8',
        mode='w',
    )
    descriptors = sorted(set(df['Descriptor']))
    ra = sorted(set(df['ra']))
    rd = sorted(set(df['rd']))
    rd_ra = [f'(rd={d} ra={a})' for d in rd for a in ra]
    inter_dims = np.array(sorted(set(df['Intermediate dimensions'])))
    no_of_clusters = sorted(set(df['Cluster number']))
    tot_dim_bvt = sorted(set(df['Total dimensions before VT']))
    final_reduced_dim = sorted(set(df['Final reduced dimensions']))
    #
    df['Intermediate dimensions and Sigma'] = [
        f'{i} & {s} & (rd={d} ra={a})'
        if len(str(s)) == 2 else f'{i} & 0{s} & (rd={d} ra={a})'
        for i, s, d, a in zip(
            df['Intermediate dimensions'],
            df['Sigma'],
            df['rd'],
            df['ra'],
        )
    ]

    df['Descriptor and Total dimensions after VT'] = [
        f'{des} & {i}' for i, des in zip(
            df['Total dimensions after VT'],
            df['Descriptor'],
        )
    ]

    data = df.pivot_table(
        values='PseudoF',
        index=
        'Intermediate dimensions and Sigma',  # VT, Sigma, and Inter_dimensions',
        columns='Descriptor and Total dimensions after VT',
    )

    data_indices = list(data.index)

    new_data_indices = []
    for data_i in data_indices:
        inter_dim_ = int(data_i.split('&')[0])
        new_data_indices.append(
            str(np.where(inter_dims == inter_dim_)[0][0] + 1) + '_' + data_i)

    # data.index = new_data_indices
    data['index'] = new_data_indices
    data = data.sort_values(by=['index'])
    data = data.drop(columns=['index'])

    no_of_columns = len(descriptors)
    no_of_rows = len(rd_ra)
    horizontal_spacing = 0.18
    vertical_spacing = 0.12
    if no_of_columns == 3:
        c_x = [0.21, 0.6, 1.0]
    height = 500 * no_of_rows
    if no_of_rows == 2:
        c_y = [0.8, 0.2]
        length = 0.45

    if no_of_rows == 1:
        c_y = [0.5]
        length = 0.8

    subplot_titles = [f'{des}{rd_ra_}' for rd_ra_ in rd_ra for des in descriptors]
    fig = make_subplots(
        rows=no_of_rows,
        cols=no_of_columns,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        subplot_titles=subplot_titles,
    )

    # data_ra_rd = data

    for r_i, rd_ra_ in enumerate(rd_ra):
        selected_indices = [i for i in data.index if rd_ra_ in i]
        data_rd_ra = data.loc[selected_indices]
        plot_y_tick_labels = [
            f"{i.split('&')[0]} & {i.split('&')[1]}" for i in data_rd_ra.index
        ]
        data_rd_ra.set_index([plot_y_tick_labels], drop=True, inplace=True)
        for c_i, des_dim_bvt in enumerate(zip(descriptors, tot_dim_bvt)):
            data_des = data_rd_ra[[
                c for c in data_rd_ra.columns if f'{des_dim_bvt[0]} &' in c
            ]]
            df_data = data_des.dropna(how='all', axis=0).dropna(how='all', axis=1)
            plot_x_tick_labels = [c.split('&')[1] for c in df_data.columns]
            df_data.columns = [
                c if len(c) == 4 else f'  {c}' for c in plot_x_tick_labels
            ]

            column_wise_max = np.array(df_data.max())
            hyperparameters_ = np.array([
                f'{a} & {b}'
                for a, b in zip(list(df_data.idxmax().index), df_data.idxmax())
            ])
            optimized_hyperparameters = hyperparameters_[column_wise_max == max(
                column_wise_max)][0]

            rd = rd_ra_.split(' ')[0].split('=')[1]
            ra = rd_ra_.split(' ')[1].split('=')[1]
            sigma = optimized_hyperparameters.split('&')[2]
            dim_avt = optimized_hyperparameters.split('&')[0]
            inter_dim = optimized_hyperparameters.split('&')[1]
            oprimized_pseudof = max(column_wise_max)
            optimized_values = list([
                'TS-LPP', des_dim_bvt[0], rd, ra[:-1], 'nan', sigma, final_reduced_dim[0], inter_dim,
                dim_avt, des_dim_bvt[1], no_of_clusters[0], oprimized_pseudof
            ])
            optimized_hyperparameters_df = pd.DataFrame(
                [optimized_values], columns=optimized_hyperparameters_df.columns)
            optimized_hyperparameters_df.to_csv(optimized_hyperparameter_data_file,
                                                sep='\t',
                                                encoding='utf-8',
                                                mode='a',
                                                header=False)

            fig.add_trace(
                go.Heatmap(z=df_data,
                           x=df_data.columns,
                           y=df_data.index,
                           colorbar=dict(
                               x=c_x[c_i],
                               y=c_y[r_i],
                               thickness=10,
                               len=length,
                           )),
                row=r_i + 1,
                col=c_i + 1,
            )

    fig.update_xaxes(title='Total dimentions after features selection',
                     row=no_of_rows,
                     col=2)
    for r_i in range(1, no_of_rows + 1):
        fig.update_yaxes(
            title='Intermediate dimentions and Sigma',
            row=r_i,
            col=1,
        )

    fig.update_layout(
        title_text="PseudoF score",
        title_x=0.5,
        height=height,
    )
    fig.write_html(hyperparameter_data_file[:-4]+'.html')
    fig.write_image(hyperparameter_data_file[:-4]+'.png')
    fig.show()


def perform_tslpp(variables):
    train_descriptor_file = variables[0]
    test_descriptor_file = variables[1]
    train_cluster_data_file = variables[2]
    train_and_predicted_cluster_data_file = variables[3]
    reduced_dim = variables[4]
    descriptor = variables[5]
    a = variables[6]
    d = variables[7]
    out_directory = variables[8]
    reduced_dim_model = variables[9]
    sigma = variables[10]
    intermediate_dimension = variables[11]
    vt = float(variables[12])
    tot_dim_after_vt = variables[13]

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
    m_sublabels = [f'{l}' for l in train_df['m_sublabels']]

    #train_labels = np.array(train_df['m_labels'])
    train_features = np.array(
        train_df.drop(columns=['m_labels', 'm_sublabels']))

    vt_list, tot_dim_list = find_vt_list(train_df.drop(columns=['m_labels', 'm_sublabels']))
    #vt_select = vt_list[np.where(tot_dim_list == tot_dim_after_vt)]
    #print(vt_list, tot_dim_list,tot_dim_after_vt )
    #test_labels = np.array(test_df['m_labels'])
    #test_labels1 = np.array(test_df['m_sublabels'])
    test_features = np.array(test_df.drop(columns=['m_labels', 'm_sublabels']))
    tot_dim_before_vt = test_features.shape[1]

    vt_model = VarianceThreshold(threshold=vt)
    vt_model.fit(train_features)
    vt_train_features = vt_model.transform(train_features)
    vt_test_features = vt_model.transform(test_features)

    sds_model = StandardScaler()
    sds_model.fit(vt_train_features)
    vt_sds_train_features = sds_model.transform(vt_train_features)
    vt_sds_test_features = sds_model.transform(vt_test_features)

    # tot_dim = np.shape(vt_sds_train_features)[1]
    l1 = reduced_dim
    l2 = reduced_dim_model
    l3 = descriptor
    l4 = d
    l5 = a
    l6 = sigma
    l7 = intermediate_dimension
    l8 = tot_dim_after_vt
    l9 = tot_dim_before_vt
    #l10 = str('{:.1e}'.format(float(vt)))

    label = f'{l3}_{l4}_{l5}_{l6}_{l7}_{l8}_{l9}'
    outfile = f'{out_directory}{label}.csv'

    #reduced_dimensions_data = None
    #outfile_labels = []
    tslpp_model = TsLpp()
    try:
        train_reduced_dimensions_data = tslpp_model.fit(
            vt_sds_train_features,
            inter_dimension=int(intermediate_dimension),
            final_dimension=int(reduced_dim),
            sigma=sigma,
        )
        predicted_reduced_dimensions_data = tslpp_model.transform(
            vt_sds_test_features)

    except np.linalg.LinAlgError:
        print((outfile, "EXITS due to error in the TS-LPP dimensions reduction part."))
        exit()

    if train_reduced_dimensions_data is not None:
        train_PseudoF_max, test_PseudoF_max = 0.0, 0.0
        for c_no in data_lib.cluster_numbers:
            c_model = KMeans(
                n_clusters=c_no,
                random_state=10).fit(train_reduced_dimensions_data)
            train_labels = c_model.labels_
            predicted_labels = c_model.predict(predicted_reduced_dimensions_data)
            train_PseudoF = sklearn.metrics.calinski_harabasz_score(
                train_reduced_dimensions_data, train_labels)
            test_PseudoF = sklearn.metrics.calinski_harabasz_score(
                predicted_reduced_dimensions_data, predicted_labels)
            if train_PseudoF > train_PseudoF_max:
                train_PseudoF_max = train_PseudoF
                best_train_P = train_PseudoF
                best_train_c_no = c_no
            if test_PseudoF > test_PseudoF_max:
                test_PseudoF_max = test_PseudoF
                best_test_P = test_PseudoF
                best_test_c_no = c_no

        # if int(reduced_dim) == 2:
        #     columns_2d = ['D1', 'D2']
        #     df_2d = pd.DataFrame(train_reduced_dimensions_data,
        #                          columns=columns_2d)
        #
        #     df_2d['m_labels'] = m_labels
        #     df_2d['m_sublabels'] = m_sublabels
        #     df_2d.to_csv(outfile, sep='\t', encoding='utf-8')

        print(label, "is DONE.")

        # if int(reduced_dim) == 3:
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
            'Dimension reduction model', 'Descriptor', 'rd',
            'ra', 'VT', 'Sigma', 'Final reduced dimensions', 'Intermediate dimensions', 'Total dimensions after VT',
            'Total dimensions before VT', 'Cluster number', 'PseudoF'
        ]

        vt = '{:.1e}'.format(float(vt))
        train_c_data = [[
            reduced_dim_model, descriptor,  d, a, vt, sigma, reduced_dim,
            intermediate_dimension, tot_dim_after_vt, tot_dim_before_vt, best_train_c_no,
            int(best_train_P)
        ]]
        test_c_data = [[
            reduced_dim_model, descriptor,  d, a, vt, sigma, reduced_dim,
            intermediate_dimension, tot_dim_after_vt, tot_dim_before_vt, best_test_c_no,
            int(best_test_P)
        ]]

        df_train = pd.DataFrame(train_c_data, columns=columns)
        df_train_and_predicted = pd.DataFrame(test_c_data, columns=columns)
        df_train.to_csv(train_cluster_data_file,
                        sep='\t',
                        encoding='utf-8',
                        mode='a',
                        header=False)
        df_train_and_predicted.to_csv(train_and_predicted_cluster_data_file,
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
        jobs.append(pool.apply_async(perform_tslpp, args=(variables,)))
    results = [job.get() for job in jobs]
    print()
    print('All Jobs done for hyperparameter optimization')
    print('*************')

    train_cluster_data_file = variables[2]
    train_and_predicted_cluster_data_file = variables[3]
    a = variables[6]
    d = variables[7]

    plot_hyperparameters(hyperparameter_data_file=train_cluster_data_file)
    plot_hyperparameters(hyperparameter_data_file=train_and_predicted_cluster_data_file)
    return
