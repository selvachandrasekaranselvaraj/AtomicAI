# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")  
import os, sys
import math
import ase.io
import numpy as np
from AtomicAI.io.write_data_in_py import write_data_in_py
from AtomicAI.tools.select_snapshots import select_snapshots
from  AtomicAI.tools.define_mirror_cubic import define_mirror_cubic
from  AtomicAI.tools.angles_in_tetrahedron import angles_in_tetrahedron
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from collections import Counter

Rc=6.5
r_nb = 3.00 
verysmall = 0.00001
###############################################################
def structure_analysis():
    bond_lengths, angles, coord_nos, edge_dists = [], [], [], []
    dist_from_cell_center = []
    frames, symbols = select_snapshots()
    symbols_type = list(set(symbols))

    for i_frame in range(-len(frames), 0):
        print("Frame No.: ", i_frame)
        atoms_local = frames[i_frame]
        #ase.io.write('1.vasp', atoms_local)
        cell = atoms_local.cell
        numbers = atoms_local.numbers
        positions = atoms_local.positions
        natoms = len(numbers)
        defect_center = input("Give defect center (Ex. '0.0, 0.0, 0.0'): ", )

        cij = positions - np.array([cell[0][0]*defect_center[0], cell[1][1]*defect_center[1], cell[2][2]*defect_center[2]])

        d_cij = np.linalg.norm(cij, axis=1)
        sorted_cij = sorted(d_cij)
        print("Frame No.: ", i_frame, sorted_cij[0:4])
        dist_from_cell_center.extend(d_cij)

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
            n_nearest = len(nn_atoms) #np.sum((d_vij < r_nb) * (d_vij > verysmall))
            d_vij0 = np.round(d_vij[(d_vij < r_nb) * (d_vij > verysmall)], 1)
            #ang_mean = np.mean(angle_tet(np.array([0., 0., 0.]), nn_atoms))
            ang = np.round(angles_in_tetrahedron(np.array([0., 0., 0.]), nn_atoms),1)

            bond_lengths.append(sorted(d_vij0))# = np.append(bond_lengths, d_vij0)
            coord_nos.append(n_nearest)
            angles.append(sorted(ang))
            edge_dists.append(sorted(d_vij1))
            #std_edge_dis = np.std(np.array(d_vij1))
            #std_ang = np.std(np.array(ang))
            #std.append(np.std(np.array([std_edge_dis, std_ang])))
            #print(atom_i, n_nearest, "%7.2f %7.2f %7.2f" %(grand_std, std_edge_dis, std_ang))
            #std_angle.append(std_ang)
            #ang_min.append(min(ang))
            #ang_max.append(max(ang))
            #angle_mean = np.append(angle_mean, ang_mean)
    bond_lengths, angles, coord_nos, edge_dists = np.array(bond_lengths), np.array(angles), np.array(coord_nos), np.array(edge_dists)

    data_names = ['bond_lengths', 'angles', 'coord_nos', 'edge_distance', 'Distance_from_vacancy']
    data_values = [list(bond_lengths), list(angles), list(coord_nos), list(edge_dists), list(dist_from_cell_center)]
    out_directory = './sf/'
    py_out_file = 'sf.py'
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)
    if os.path.isfile(out_directory+py_out_file):
        os.remove(out_directory+py_out_file)

    initialize_variables = {}
    initialize_variables['data_name'] = 'sf'
    initialize_variables['data'] = '{}'
    write_data_in_py(out_directory+py_out_file, initialize_variables)

    for data_name, data_value in zip(data_names, data_values):
        py_output_data = {}
        py_output_data['data_name'] = f"sf['{data_name}']"
        py_output_data['data'] = data_value
        write_data_in_py(out_directory+py_out_file, py_output_data)

    plt.figure()
    mu, std_a = norm.fit(std)
    plt.hist(std, bins=25, density=True, alpha=0.6, color='g')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std_a)
    plt.plot(x, p, 'k', linewidth=2)
    #title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    #plt.title(title)
    #plt.figure()
    #plt.scatter(std_angle, std)
    #print(list(np.round(np.array(std_angle),1)))
    #print
    #print(list(np.round(np.array(std),1)))
    #plt.show()

    return