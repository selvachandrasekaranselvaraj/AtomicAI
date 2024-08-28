"""
    Generic functions for atomic fingerprints.
"""

# -*- coding: utf-8 -*-
import math
import random
import copy
import gc  # Garbage collector
import pandas as pd
import numpy as np
import numba as nb
import pprint
from numpy import pi

import ase.io
from ase.data import atomic_numbers

# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore")
import sys
import ase.io
import numpy as np

from collections import Counter
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def set_param_dict(parameters, fp_flag):
    Rc2b = parameters.get('Rc2b')
    Rc3b = parameters.get('Rc3b')
    #print('Rc2b/Rc3b =', Rc2b, Rc3b)
    param_dict = {"Rc2b": Rc2b}

    G2b_eta_range = list(parameters['2b'][0:2])
    G2b_eta_num = parameters.get('2b')[2]
    G2b_dRs = parameters.get('2b')[3]

    eta_const = list(np.exp(np.array(G2b_eta_range)) * Rc2b)
    G2b_eta = define_eta(eta_const, G2b_eta_num)

    G2b_eta = 0.5 / np.square(np.array(G2b_eta))
    G2b_Rs = np.arange(0, Rc2b, G2b_dRs)
    #print('G2b_eta')
    #print(G2b_eta)
    #print('G2b_Rs')
    #print(G2b_Rs)

    param_dict.update(G2b_eta=G2b_eta)
    param_dict.update(G2b_Rs=G2b_Rs)

    if fp_flag == 'BP2b':
        nfp = len(G2b_eta) * len(G2b_Rs)
        param_dict.update(nfp=int(nfp))
    elif (fp_flag == 'LA2b3b') or (fp_flag == 'DerMBSF2b3b'):
        # parameters for 3body-term
        para_G3b = parameters.get('3b')
        G3b_eta_range = list(para_G3b[0:2])
        G3b_eta_num = para_G3b[2]

        eta_const = list(np.exp(np.array(G3b_eta_range)) * Rc3b)
        G3b_eta = define_eta(eta_const, G3b_eta_num)
        G3b_eta = 0.5 / np.square(np.array(G3b_eta))

        G3b_dRs = parameters.get('3b')[3]
        G3b_zeta_num = parameters.get('3b')[4]
        G3b_theta_num = parameters.get('3b')[5]
        G3b_Rs = np.arange(0, Rc3b, G3b_dRs)

        zeta_lst = [1, 2, 4, 16]
        G3b_zeta = np.array(zeta_lst[0:G3b_zeta_num])

        nx = G3b_theta_num
        dx = pi / (nx - 1)
        G3b_theta = np.array([dx * x for x in range(nx)])

        param_dict.update(Rc3b=Rc3b)
        param_dict.update(G3b_eta=G3b_eta)
        param_dict.update(G3b_Rs=G3b_Rs)
        param_dict.update(G3b_zeta=G3b_zeta)
        param_dict.update(G3b_theta=G3b_theta)
        nfp = len(G2b_eta) * len(G2b_Rs) + len(G3b_eta) * len(G3b_zeta) * len(G3b_theta)
        param_dict.update(nfp=int(nfp))

    elif 'Split2b3b' in fp_flag:
        # parameters for 3body-term

        para_G3b = parameters.get('split3b')
        G3b_eta_range = list(para_G3b[0:2])
        G3b_eta_num = para_G3b[2]

        eta_const = list(np.exp(np.array(G3b_eta_range)) * Rc3b)
        G3b_eta = define_eta(eta_const, G3b_eta_num)
        G3b_eta = 0.5 / np.square(np.array(G3b_eta))

        param_dict.update(Rc3b=Rc3b)
        param_dict.update(G3b_eta=G3b_eta)
        nfp = len(G2b_eta) * len(G2b_Rs) + len(G3b_eta) * (len(G3b_eta) + 1) * (len(G3b_eta) + 2) / 6
        param_dict.update(nfp=int(nfp))
    else:
        print('Error: No such type of fingerprint(Define Parameters)!')
        exit()

    num_G1_d = 2*len(G2b_eta) * len(G2b_Rs)
    num_G2_d = nfp - num_G1_d
    #print('Number of descriptor = %d(G1=%d,G2=%d)' % (num_G1_d + num_G2_d, num_G1_d, num_G2_d))
    return param_dict

def get_parameters():
    parameters = {
    #      cut off for fingerprint
    #        AA
    'Rc2b': 1.5 * 5.0,
    'Rc3b': 1.5 * 5.0,
    'Reta': 1.5 * 5.0,
    #        |    2-body term      |
    #        |    Eta       |  Rs  |
    #        min   max   num| dRs  |
    #        AA    AA    int  AA
    '2b': [-3.0, 1.0, 20, 2.5],
    #      |  3-body term |
    #      |        Eta   | Rs  | zeta | theta |
    #      min   max   num| dRs | num  |  num  |
    #      AA    AA    int|  AA | int  |  int  |
    '3b': [-3.0, 1.0, 10, 10.5, 3, 10],
    #        |split 3-body term|
    #        | min   max   num|
    #        | AA    AA    int|
    'split3b': [-3.0, 1.0, 10]
        }
    fp_flag = 'Split2b3b_ss'
    param_dict, nfp = set_param_dict(parameters, fp_flag)
    return param_dict


###############################################################
def select_data_from_trajectory(no_of_data=25000):
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
        if no_of_data >= df.loc[symbol].shape[0]:
            no_of_data = df.loc[symbol].shape[0] - 5
        df_ = df.loc[symbol].sample(no_of_data)
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

###############################################################
# make random vector for force projection
def prepare_vforce(no_of_data):
    vforce = []

    for _ in range(no_of_data):
        zr = random.uniform(0.0, 1.0) * 2 - 1
        pr = random.uniform(0.0, 1.0) * 2 * pi

        vx = math.sqrt(1 - zr**2) * math.cos(pr)
        vy = math.sqrt(1 - zr**2) * math.sin(pr)
        vz = zr

        vforce.append([vx, vy, vz])

    return vforce

###############################################################
# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore")
from AtomicAI.descriptors.force_descriptor_functions import *
import math, os, random, copy, sys
import numpy as np
from AtomicAI.predictor.split2b3b import MultiSplit2b3b_index_ss


def get_mlff():  # traj_file name
    selected_frames, MLFF = select_data_from_trajectory()
    param_dict = get_parameters()
    for element in MLFF.keys():
        no_of_data = MLFF[element]['Number of data']

        # initialize random seed
        seed_ran = 16835
        seed_rand = seed_ran
        random.seed(seed_rand)
        num_frames = len(selected_frames)
        vforce = prepare_vforce(no_of_data)
        #random_choice_time = time() - before_time

        diter = 1000
        # collect results of fingerprint
        Xv, Fv = [], []
        ID_vector = 0
        diter_local = 0
        vforce_no = 0
        for selected_frame_index, selected_atoms_indices in zip(
                MLFF[element]['Selected_frames_indices'],
                MLFF[element]['Selected_atoms_indices']):
            atoms_local = selected_frames[selected_frame_index]
            cell = atoms_local.cell
            numbers = atoms_local.numbers
            symbols = list(atoms_local.symbols)
            positions = atoms_local.positions
            atomcforces = atoms_local.arrays.get('forces')
            natoms = len(numbers)

            if ID_vector >= diter_local:
                print(element, 'Process : %d' % ID_vector)
                diter_local += diter

            for selected_atom_index in selected_atoms_indices:
                Force_xyz = atomcforces[selected_atom_index]
                Fvtmp = np.dot(
                    Force_xyz, vforce[vforce_no]
                )  # Fv : atomic force projection along the random vector

                # prepare feature: fingerprint value(descriptor value)
                Xvtmp = []

                inputs = (atoms_local, selected_atom_index, vforce[vforce_no], param_dict)
                Xvtmp = MultiSplit2b3b_index_ss(inputs)
                vforce_no += 1
                Xv.append(Xvtmp)
                Fv.append(Fvtmp)
                # update id of force vector
                ID_vector += 1
        print(np.array(Xvtmp).shape)
        descriptors = {'Descriptors': Xv, 'Forces': Fv}
        MLFF[element].update(descriptors)
        MLFF[element].update(param_dict)

    ###############################################################
    for element in MLFF.keys():
        descriptor = np.array(MLFF[element]['Descriptors'])
        labels = np.array(MLFF[element]['Forces'])
        no_of_data = descriptor.shape[0]
        print(f'No. of features in {element} are: {descriptor.shape[1]} and no of data are {no_of_data}')
        data_split_at = int(no_of_data * 0.8)
        train_features = descriptor[0:data_split_at]
        test_features = descriptor[data_split_at:no_of_data]
        train_labels = labels[0:data_split_at]
        test_labels = labels[data_split_at:no_of_data]
        df = pd.DataFrame(train_features)
        variances = list(df.var().sort_values())
        #print(min(variances), max(variances))
        r2, no_of_dimensions_vt = [], []
        vts = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-20, 1e-40, 0.0] # variances[:-2]
        for variance in vts:
            predicted_test_lables, no_of_dimensions_vt_ = LassoLarCV_VT_SDS(
                train_features, test_features, train_labels, variance)
            r2.append(round(r2_score(test_labels, predicted_test_lables), 3))
            no_of_dimensions_vt.append(no_of_dimensions_vt_)
        data_ = {'No of dimensions after VT': no_of_dimensions_vt,
                 'Predictd accuracy for VTs': r2,
                 'Variances' : vts,
                }
        MLFF[element].update(data_)
        no_of_dimensions_vt = MLFF[element]['No of dimensions after VT']
        r2 = MLFF[element]['Predictd accuracy for VTs']
        vts = MLFF[element]['Variances']
        plot_line(vts, no_of_dimensions_vt, r2, element)
    ###############################################################
    for element in MLFF.keys():
        r2 = np.array(MLFF[element]['Predictd accuracy for VTs'])
        vts = np.array(MLFF[element]['Variances'])
        optimized_vt = {'optimized_vt': vts[r2==max(r2)]}
        MLFF[element].update(optimized_vt)
        descriptors = np.array(MLFF[element]['Descriptors'])

        # VT
        vt = VarianceThreshold(threshold=MLFF[element]['optimized_vt'][0])
        descriptos_vt = vt.fit_transform(descriptors)

        # SDS
        sds = StandardScaler()   # with_mean=False)
        descriptos_vt_sds = sds.fit_transform(descriptos_vt)

        # Create regression object
        lasso = LassoLarsCV(fit_intercept=False, max_iter=5000)
        lasso.fit(descriptos_vt_sds, np.array(MLFF[element]['Forces']))

        # Models
        models = {
            'VT_model' : vt,
            'SDS_model': sds,
            'Lasso_model': lasso

        }

        MLFF[element].update(models)

        forces = np.array(MLFF[element]['Forces'])
        lasso_model = MLFF[element]['Lasso_model']
        sds_model =  MLFF[element]['SDS_model']
        vt_model =  MLFF[element]['VT_model']
        descriptors = np.array(MLFF[element]['Descriptors'])
        predicted_forces = lasso_model.predict(sds_model.transform(vt_model.transform(descriptors)))
        print(round(r2_score(forces, predicted_forces), 3))
    return MLFF
###############################################################

def LassoLarCV_VT_SDS(train_features, test_features, train_labels, vt_value):
    # Feature selection
    vt = VarianceThreshold(threshold=vt_value)
    train_features_vt  = vt.fit_transform(train_features)
    test_features_vt = vt.transform(test_features)

    #SDS
    sds = StandardScaler()   # with_mean=False)
    train_features_vt_sds = sds.fit_transform(train_features_vt)
    test_features_vt_sds = sds.transform(test_features_vt)

    # Create regression object
    regr = LassoLarsCV(fit_intercept=False, max_iter=5000)
    regr.fit(train_features_vt_sds, train_labels)

    return  regr.predict(test_features_vt_sds),  len(train_features_vt[0])  # Prediction and no_of_dimensions
###############################################################
import plotly.express as px
def plot_line(vts, dimensions, r2, element):
    fig = px.line(x=dimensions, y=r2, markers=True, text= ["{:.1E}".format(vt_) for vt_ in vts])
    fig.update_layout(
                      width = 1000,
                      height = 400,

                      xaxis_title='VT',
                      yaxis_title=r'$R^2$',
    )
    fig.write_html(f"{element}_VT_optimization.html")
