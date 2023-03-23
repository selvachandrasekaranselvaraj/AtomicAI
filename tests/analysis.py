# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")  
import os, sys
import math
import ase.io
import numpy as np
from AtomicAI.io.write_data_in_py import write_data_in_py

Rc=6.5
r_nb = 3.00 
verysmall = 0.00001
###############################################################
def define_mirror_cubic(position, cell, Rc):
    m_min = [0,0,0]
    m_max = [0,0,0]
    for i in range(len(position)):
        if(position[i] < Rc): m_min[i] = -1
        if(cell[i,i]-position[i]) < Rc : m_max[i] = 1
    #print(position, m_min, m_max, Rc)
    m_x = []
    m_y = []
    m_z = []
    for i in range(m_min[0], m_max[0]+1):
        for j in range(m_min[1], m_max[1] + 1):
            for k in range(m_min[2], m_max[2] + 1):
                m_x.append(i* cell[0,0])
                m_y.append(j* cell[1,1])
                m_z.append(k* cell[2,2])
    #print(m_x)
    return m_x, m_y, m_z

###############################################################
def angle_between_three_points(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
# angle = angel_between_three_points(a, b, c) # func_out

###############################################################
def angle_tet(position, nn_atoms):
    angle = []
    for i in range(len(nn_atoms)):
        for j in range(i+1, len(nn_atoms)):
            angle.append(angle_between_three_points(nn_atoms[i], position, nn_atoms[j]))
    return np.array(angle)

###############################################################
def structure_analysis():
    try:
        input_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"structure_analysis traj_file_name with .xyz extension\"")
        print()
        exit()
    bond_lengths, angles, coord_no, edge_dist, std = [], [], [], [], []
    frames = ase.io.read(input_file, index=':')
    for i_frame in range(-len(frames), 0):
        print("Frame No.: ", i_frame)
        atoms_local = frames[i_frame]
        #ase.io.write('1.vasp', atoms_local)
        cell = atoms_local.cell
        numbers = atoms_local.numbers
        positions = atoms_local.positions
        atomcforces = atoms_local.arrays.get('forces')
        natoms = len(numbers)

        #tot_coords = define_extra_atoms(positions, cell, Rc, natoms)
        for atom_i in range(natoms):
            m_x, m_y, m_z = define_mirror_cubic(positions[atom_i], cell, Rc)
            vij = positions - positions[atom_i]
            d_vij_lst=[]
            vij_local = np.empty((0,3))
            for m_i in range(len(m_x)):
                m_shift = np.array([m_x[m_i],m_y[m_i],m_z[m_i]])
                vij_local = np.append(vij_local, vij + m_shift, axis=0)
           
            d_vij = np.linalg.norm(vij_local, axis=1)
           
           
            nn_atoms = vij_local[np.where((d_vij < r_nb) * (d_vij > verysmall))]
            d_vij1 = []
            for nn_atom_i in range(len(nn_atoms)):
                for nn_atom_j in range(len(nn_atoms)):
                    if nn_atom_i > nn_atom_j:
                        d_vij1.append(round(math.dist(nn_atoms[nn_atom_i], nn_atoms[nn_atom_j]), 1))
            edge_dist.append(sorted(d_vij1))
            n_nearest = len(nn_atoms) #np.sum((d_vij < r_nb) * (d_vij > verysmall))
            d_vij0 = np.round(d_vij[(d_vij < r_nb) * (d_vij > verysmall)], 1)
            #ang_mean = np.mean(angle_tet(np.array([0., 0., 0.]), nn_atoms))
            ang = np.round(angle_tet(np.array([0., 0., 0.]), nn_atoms),1)

            bond_lengths.append(sorted(d_vij0))# = np.append(bond_lengths, d_vij0)
            coord_no.append(n_nearest)
            std_edge_dis = np.std(np.array(d_vij1))
            std_ang = np.std(np.array(ang))
            std.append(np.std(np.array([std_edge_dis, std_ang])))
            #print(atom_i, n_nearest, "%7.2f %7.2f %7.2f" %(grand_std, std_edge_dis, std_ang))
            angles.append(sorted(ang))
            #ang_min.append(min(ang))
            #ang_max.append(max(ang))
            #angle_mean = np.append(angle_mean, ang_mean)
    std = np.array(std)
    std[np.where((np.array(coord_no) == 2))] = max(std) + (max(std)-min(std))/40
    std[np.where((np.array(coord_no) == 3))] = max(std) + (max(std)-min(std))/60
    std[np.where((np.array(coord_no) > 4))] = max(std) + (max(std)-min(std))/90
    data_names = ['bond_lengths', 'angles', 'coord_no', 'edge_distance', 'std']
    data_values = [bond_lengths, angles, coord_no, edge_dist, list(std)]
    out_directory = './structure_analysis/'
    py_out_file = 'structure_analysis.py'
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)
    if os.path.isfile(out_directory+py_out_file):
        os.remove(out_directory+py_out_file)

    initialize_variables = {}
    initialize_variables['data_name'] = 'structure_analysis'
    initialize_variables['data'] = '{}'
    write_data_in_py(out_directory+py_out_file, initialize_variables)

    for data_name, data_value in zip(data_names, data_values):
        py_output_data = {}
        py_output_data['data_name'] = f"structure_analysis['{data_name}']"
        py_output_data['data'] = data_value
        write_data_in_py(out_directory+py_out_file, py_output_data)

    return 
structure_analysis()
