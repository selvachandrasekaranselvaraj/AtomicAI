import warnings
warnings.filterwarnings("ignore")
from numba import jit
import math
import numpy as np
from math import exp, sqrt, cos, pi
from AtomicAI.descriptors.define_mirror_reduce_cube import define_mirror_reduce_cube
from AtomicAI.descriptors.sum_G2b00_all import sum_G2b00_all
from AtomicAI.descriptors.sum_G3b11ss import sum_G3b11ss
from AtomicAI.descriptors.sum_G3b01ss import sum_G3b01ss
from AtomicAI.descriptors.sum_G3b00ss import sum_G3b00ss

# Default variables
small_number = 1.0e-9
verysmall = 1.0e-65
small = 1e-4

sigma = 1.0 / 10
sigma2 = sigma ** 2

@jit
def MultiSplit2b3b_index_ss(variables):
    atomsx, atomID, v, Param_dict = variables
    """

    :param atomsx:
    :param atomID:
    :param v:
    :param Param_dict:
    :return: V
    """
    G2b_eta = Param_dict.get('G2b_eta')
    G2b_Rs = Param_dict.get('G2b_Rs')
    Rc2b = Param_dict.get('Rc2b')
    G3b_eta = Param_dict.get('G3b_eta')
    Rc3b = Param_dict.get('Rc3b')
    #G3b_lst_param_eta = Param_dict.get('G3b_lst_param_eta')

    Rc = max(Rc2b, Rc3b)

    nG2b_eta = len(G2b_eta)
    nG2b_Rs = len(G2b_Rs)
    nG3b_eta = len(G3b_eta)

    #G2b_lst = Param_dict.get('G2b_lst')
    #G3b_lst = Param_dict.get('G3b_lst')

    i = atomID
    numbers = atomsx.numbers

    n = len(numbers)
    atom_name = numbers[atomID]

    X, Y, Z = atomsx.positions[:, 0], atomsx.positions[:, 1], atomsx.positions[:, 2]
    position = np.array([X[atomID], Y[atomID], Z[atomID]])
    xij0, yij0, zij0 = X - position[0], Y - position[1], Z - position[2]

    cell = atomsx.cell
    m_x, m_y, m_z = define_mirror_reduce_cube(position, cell, Rc)
    vx, vy, vz = v[0], v[1], v[2]

    V1_00 = np.zeros((nG2b_eta, nG2b_Rs))  # AA
    V1_01 = np.zeros((nG2b_eta, nG2b_Rs))  # AB
    V2_00 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))
    V2_01 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))
    V2_11 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))
    # print("V1 size len = ", len(R_eta), len(R_Rs))
    # print("V1 size = ", len(V1_00), len(V1_01))

    for j in range(n):

        for mj in range(len(m_x)):

            xij = xij0[j] + m_x[mj]
            yij = yij0[j] + m_y[mj]
            zij = zij0[j] + m_z[mj]
            rij = math.sqrt(xij ** 2 + yij ** 2 + zij ** 2)

            if Rc > rij > small:

                # vec_rij dot F_vector
                coeff_ij = (xij * vx + yij * vy + zij * vz)
                # function of cut off
                frc_ij = 0.5 * (math.cos(pi * rij / Rc) + 1)
                tmp0 = coeff_ij * frc_ij

                # Compare Element name
                if atom_name == numbers[j]:
                    V1_00 = sum_G2b00_all(V1_00, tmp0, rij, G2b_Rs, G2b_eta)

                else:

                    # Bond length Si-Ge of Rc
                    # Eta = [ ], Rc, Rs
                    # Lasso
                    # MD Conquest
                    # MOs
                    V1_01 = sum_G2b00_all(V1_01, tmp0, rij, G2b_Rs, G2b_eta)
                    # V1_01 = sum_G2b00(V1_01, tmp0, rij, R_Rs, R_eta)

                for k in range(j, n):

                    for mk in range(len(m_x)):

                        xik = xij0[k] + m_x[mk]
                        yik = yij0[k] + m_y[mk]
                        zik = zij0[k] + m_z[mk]
                        rik = math.sqrt(xik ** 2 + yik ** 2 + zik ** 2)
                        xjk = xik - xij
                        yjk = yik - yij
                        zjk = zik - zij
                        rjk = math.sqrt(xjk ** 2 + yjk ** 2 + zjk ** 2)

                        #R_3b = math.sqrt(rij ** 2 + rik ** 2 + rjk ** 2)
                        #Rc_3b = math.sqrt(Rc ** 2)
                        # TODO 20230117: Setup cutoffs properly
                        # if (Rc_3b > R_3b > small) and (rik > small) and (rjk > small):
                        if (Rc > rik > small) and (Rc > rjk > small):
                            frc_ik = 0.5 * (math.cos(pi * rik / Rc) + 1)
                            frc_jk = 0.5 * (math.cos(pi * rjk / Rc) + 1)
                            frc_3b = frc_ij * frc_ik * frc_jk

                            proj_ij = (xij + 0) * vx + (yij + 0) * vy + (zij + 0) * vz
                            proj_ik = (0 + xik) * vx + (0 + yik) * vy + (0 + zik) * vz

                            if atom_name == numbers[j] and atom_name == numbers[k]:
                                V2_00 = sum_G3b00ss(V2_00, frc_3b, proj_ij, proj_ik, rij, rik, rjk, G3b_eta)

                            elif atom_name == numbers[j] and atom_name != numbers[k]:
                                # V2_01 = sum_G3b01(V2_01, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta)
                                # V2_01 = sum_G3b01s(V2_01, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta)
                                V2_01 = sum_G3b01ss(V2_01, frc_3b, proj_ij, proj_ik, rij, rik, rjk, G3b_eta)

                            elif atom_name != numbers[j] and atom_name == numbers[k]:
                                # need to change the position of i,j,k --input
                                # j->k
                                # V2_01 = sum_G3b01(V2_01, frc_3b, proj_ik, proj_ij, rik, rij, rjk, Ts_eta)
                                # V2_01 = sum_G3b01s(V2_01, frc_3b, proj_ik, proj_ij, rik, rij, rjk, Ts_eta)
                                V2_01 = sum_G3b01ss(V2_01, frc_3b, proj_ik, proj_ij, rik, rij, rjk, G3b_eta)

                            else:
                                # V2_11 = sum_G3b11(V2_11, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta)
                                # V2_11 = sum_G3b11s(V2_11, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta)
                                V2_11 = sum_G3b11ss(V2_11, frc_3b, proj_ij, proj_ik, rij, rik, rjk, G3b_eta)
    V1_tmp = []
    V2_tmp = []
    for RsID in range(len(G2b_Rs)):
        for etID in range(len(G2b_eta)):
            V1_tmp.append(V1_00[etID][RsID])
    for RsID in range(len(G2b_Rs)):
        for etID in range(len(G2b_eta)):
            V1_tmp.append(V1_01[etID][RsID])

    for eID1 in range(nG3b_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(eID2 + 1):
                V2_tmp.append(V2_00[eID1][eID2][eID3])
    for eID1 in range(nG3b_eta):
        for eID2 in range(nG3b_eta):
            for eID3 in range(eID2 + 1):
                V2_tmp.append(V2_01[eID1][eID2][eID3])
    for eID1 in range(nG3b_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(nG3b_eta):
                V2_tmp.append(V2_11[eID1][eID2][eID3])
    V = V1_tmp
    V.extend(V2_tmp)
    return V
