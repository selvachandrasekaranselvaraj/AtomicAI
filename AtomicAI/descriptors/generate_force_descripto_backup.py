# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
from AtomicAI.descriptors.force_descriptor_functions import *
import math, os, random, copy, sys
from time import time
import numpy as np
from AtomicAI.tools.select_snapshots import select_snapshots



def generate_force_descriptors():
    out_directory = './descriptors/'
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)
    selected_frames, symbols = select_snapshots()
    #print('No. of frame:', len(selected_frames))

    # Calculate force descriptors
    force_descriptor(selected_frames)
    return

def writeout_fp(Xv,Fv,atomic_symbols, nfout):
    if len(Fv)==len(Xv):
        with open(nfout,mode='w') as nf:
            lines = []
            for i in range(len(Fv)):
                tmp_lst = copy.deepcopy(Xv[i])
                tmp_lst.append(Fv[i])
                V_str_list = ["{0: >30.16e}".format(v) for v in tmp_lst]
                lines.append("".join(V_str_list) +' '+atomic_symbols[i]+'\n')
            nf.writelines(lines)
            nf.close()
    else:
        print('Sizes of Fv and Xv are not same!\n')
        exit()
    return None


# make random vector for force projection
def prepare_vforce(data_num_rand):
    vforce=[]

    for _ in range(data_num_rand):
        zr = random.uniform(0.0, 1.0) * 2 - 1
        pr = random.uniform(0.0, 1.0) * 2 * pi

        vx = math.sqrt(1 - zr ** 2) * math.cos(pr)
        vy = math.sqrt(1 - zr ** 2) * math.sin(pr)
        vz = zr

        vforce.append([vx,vy,vz])

    return vforce


def force_descriptor(selected_frames): # traj_file name


    parameters = {
        #      cut off for fingerprint
        #        AA
        'Rc2b': 1.5*7.0,
        'Rc3b': 1.5*7.0,
        'Reta': 1.5*7.0,
        #        |    2-body term      |
        #        |    Eta       |  Rs  |
        #        min   max   num| dRs  |
        #        AA    AA    int  AA
        '2b'  : [-3.0, 1.0,  20,  2.5],
        #      |  3-body term |
        #      |        Eta   | Rs  | zeta | theta |
        #      min   max   num| dRs | num  |  num  |
        #      AA    AA    int|  AA | int  |  int  |
        '3b': [-3.0, 1.0,  10,  10.5,   3,    10],
        #        |split 3-body term|
        #        | min   max   num|
        #        | AA    AA    int|
        'split3b': [-3.0, 1.0,  10]}

    nG3b = parameters.get('split3b')[2]
    verysmall=1e-8
    small=1e-4
    pi=math.pi
    # ##############################################
    seed_ran = 16835
    #data_num = 300000
    flag_angstrom = True
    r_nb = 4.00

    fp_flag_lst = ['BP2b', 'LA2b3b', 'DerMBSF2b3b', 'Split2b3b_ss']
    fpID, regID = 3, 2
    fp_flag = fp_flag_lst[fpID]



    main_start = time()
    nfout = './descriptors/force_descriptors.dat' #nf_all_list[nfi]
    if os.path.isfile(nfout):
        os.remove(nfout)


    # calculate the features with the atomic environment and G1G2 parameters
    #print('Now calculating the features...')
    count_vtime = 0.0
    count_mtime = 0.0

    # deal with input parameters
    param_dict, nfp = set_param_dict(parameters, fp_flag)

    # initialize random seed
    seed_rand = seed_ran
    random.seed(seed_rand)

    # random selected atom matrix with elements of (n_frame, i_atom)
    before_time = time()
    num_frames = len(selected_frames)
    selected_frames = selected_frames
    print("Selected no. of frames are", len(selected_frames))
    data_num = 0
    for frame_i in range(0, num_frames):
        data_num += len(selected_frames[frame_i].numbers)
    data_num += len(selected_frames[frame_i].numbers)

    print ("Total data: ", data_num)
    #choice_random = prepare_choice_normal(data_num, frames)  # get random choices from the setting
    #data_num = 10000
    vforce = prepare_vforce(data_num)
    #random_choice_time = time() - before_time

    diter = 1000
    # collect results of fingerprint
    Xv, Fv, atomic_pairs = [], [], []
    before_time = time()
    ID_vector = 0
    diter_local = 0
    for i_frame in range(0, num_frames):
        atoms_local = selected_frames[i_frame]
        cell = atoms_local.cell
        numbers = atoms_local.numbers
        symbols = atoms_local.symbols
        positions = atoms_local.positions
        atomcforces = atoms_local.arrays.get('forces')
        natoms = len(numbers)

        if ID_vector >= diter_local:
            print('Process : %d' % ID_vector)
            diter_local += diter

        #
        symbols_ = np.array(list(symbols))
        sorted_symbols_ = list(set(symbols_))

        for i in range(natoms):
            before_time = time()
            # prepare force
            Forcex = atomcforces[i]
            v = np.array(vforce[ID_vector])  # get random vector
            Fvtmp = np.dot(Forcex, v)  # Fv : atomic force projection along the random vector

            # prepare feature: fingerprint value(descriptor value)
            before_time = time()
            Xvtmp = []

            for target_atom in sorted_symbols_:
                #base_atom = symbols_[i]
                positions_ = positions[symbols_ == target_atom]
                vij0 = positions_ - positions[i]
                xij0, yij0, zij0 = vij0[:, 0], vij0[:, 1], vij0[:, 2]

                Rc2b = param_dict.get('Rc2b')
                Rc3b = param_dict.get('Rc3b')
                Rc = max(Rc2b, Rc3b)
                m_x, m_y, m_z = define_mirror_cubic(positions[i], cell, Rc)

                if (fp_flag == 'BP2b'):
                    Xvtmp.extend(make_BP2b(xij0,yij0,zij0,m_x,m_y,m_z, v, param_dict))
                elif fp_flag == 'Split2b3b_ss':
                    Xvtmp.extend(make_Split2b3b_ss(xij0,yij0,zij0,m_x,m_y,m_z, v, param_dict))
                else:
                    print('Wrong V_flag!\n')
                    exit()
            count_vtime = count_vtime + time() - before_time
            Xv.append(Xvtmp)
            Fv.append(Fvtmp)
            atomic_pairs.append(f'{symbols_[i]}')

            # update id of force vector
            ID_vector += 1

    writeout_fp(Xv, Fv, atomic_pairs, nfout=nfout)

    print("prepare matrix cost: {0:.2f} 秒".format(time() - before_time))
    print("read matrix cost: {0:.2f} 秒".format(count_mtime))
    print("make_V cost: {0:.2f} 秒".format(count_vtime))

    #fp_time = time() - main_start
    print("fingerprint cost：{0:.2f} 秒".format(time() - main_start))
    print("Total cost：{0:.2f} 秒".format(time() - main_start))
    return
