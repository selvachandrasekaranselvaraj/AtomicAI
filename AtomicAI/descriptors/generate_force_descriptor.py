# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
from AtomicAI.descriptors.force_descriptor_functions import *
import argparse, math, os, random, copy, sys
from time import time
import numpy as np
from AtomicAI.tools.select_snapshots import select_snapshots

FP_TYPES = ['BP2b', 'Split2b3b_ss']


def _parse_force_args():
    parser = argparse.ArgumentParser(
        description='Generate force descriptors from a trajectory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Available fingerprint types:\n  ' + '\n  '.join(FP_TYPES),
    )
    parser.add_argument('input_file', help='Trajectory file (.xyz)')
    parser.add_argument(
        '--fp-type', '-f',
        choices=FP_TYPES,
        default='Split2b3b_ss',
        dest='fp_type',
        help='Force fingerprint type (default: Split2b3b_ss)',
    )
    parser.add_argument(
        '--rc', '-r',
        type=float,
        default=10.5,
        help='Cutoff radius in Angstrom for 2-body and 3-body terms (default: 10.5)',
    )
    parser.add_argument(
        '--n2b',
        type=int,
        default=20,
        help='Number of 2-body eta functions (default: 20)',
    )
    parser.add_argument(
        '--n3b',
        type=int,
        default=10,
        help='Number of 3-body eta functions (default: 10)',
    )
    return parser.parse_args()


def generate_force_descriptors():
    args = _parse_force_args()
    out_directory = './descriptors/'
    os.makedirs(out_directory, exist_ok=True)
    selected_frames, _ = select_snapshots()
    force_descriptor(selected_frames, fp_type=args.fp_type, rc=args.rc, n2b=args.n2b, n3b=args.n3b)
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


def force_descriptor(selected_frames, fp_type='Split2b3b_ss', rc=10.5, n2b=20, n3b=10):
    """Compute force descriptors for all atoms in selected_frames.

    Parameters
    ----------
    fp_type : str
        Fingerprint type — 'BP2b' (2-body Behler-Parrinello) or 'Split2b3b_ss' (split 2+3-body).
    rc : float
        Cutoff radius in Angstrom applied to both 2-body and 3-body terms.
    n2b : int
        Number of 2-body eta decay functions.
    n3b : int
        Number of 3-body eta decay functions.
    """
    if fp_type not in FP_TYPES:
        print(f'Unknown fingerprint type {fp_type!r}. Choose from: {FP_TYPES}')
        sys.exit(1)

    parameters = {
        'Rc2b': rc,
        'Rc3b': rc,
        'Reta': rc,
        '2b':      [-3.0, 1.0, n2b, 2.5],
        '3b':      [-3.0, 1.0, n3b, rc,  3, 10],
        'split3b': [-3.0, 1.0, n3b],
    }

    seed_ran = 16835
    fp_flag = fp_type

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
