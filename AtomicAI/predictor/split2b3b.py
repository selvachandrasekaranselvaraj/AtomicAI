"""
    Module for the spli2b3b descriptor
"""

import math
from numba import jit
import numpy as np
from math import exp, sqrt, cos, pi

# Default variables
small_number = 1.0e-9
verysmall = 1.0e-65
small = 1e-4

p = 2

sigma = 1.0 / 10
sigma2 = sigma ** 2

@jit
def fcut(R, Rc) -> float:
    """
        Cosine-based cutoff function
    :param R: distance
    :param Rc: cutoff distance
    :return: a float
    """
    return 0.5 * (math.cos(pi * R / Rc) + 1)

def define_mirror_reduce_cubic(
        position: np.array,
        cell: np.array,
        Rc: float
):
    m_min = [0, 0, 0]
    m_max = [0, 0, 0]

    for i in range(len(position)):

        if position[i] < Rc:
            m_min[i] = -1

        if (cell[i, i] - position[i]) < Rc:
            m_max[i] = 1

    m_x = []
    m_y = []
    m_z = []
    for i in range(m_min[0], m_max[0] + 1):
        for j in range(m_min[1], m_max[1] + 1):
            for k in range(m_min[2], m_max[2] + 1):
                m_x.append(i * cell[0, 0])
                m_y.append(j * cell[1, 1])
                m_z.append(k * cell[2, 2])

    return m_x, m_y, m_z



def set_splitG3b(Param_dict):
    """
        Generate list of parameters.
        # TODO: More details to be added to documentation.
    :param Param_dict:
    :return: G3b_lst_param_eta
    """
    G3b_lst = Param_dict.get('G3b_lst')
    G3b_eta = Param_dict.get('G3b_eta')
    nG3b_eta = len(G3b_eta)

    G3b_lst_param_eta = [[] for _ in range(len(G3b_lst))]

    for etID1 in range(nG3b_eta):
        for etID2 in range(etID1 + 1):
            for etID3 in range(etID2 + 1):
                G3b_lst_param_eta[0].append([etID1, etID2, etID3])

    for etID1 in range(nG3b_eta):
        for etID2 in range(nG3b_eta):
            for etID3 in range(etID2 + 1):
                G3b_lst_param_eta[1].append([etID1, etID2, etID3])

    for etID1 in range(nG3b_eta):
        for etID2 in range(etID1 + 1):
            for etID3 in range(nG3b_eta):
                G3b_lst_param_eta[2].append([etID1, etID2, etID3])

    return G3b_lst_param_eta


@jit
def sum_G3b00ss(V2,
                frc_3b,
                proj_ij, proj_ik,
                rij, rik, rjk,
                Ts_eta, p_exponent=2,
                ):
    """

    :param V2:
    :param frc_3b:
    :param proj_ij:
    :param proj_ik:
    :param rij:
    :param rik:
    :param rjk:
    :param Ts_eta:
    :param p_exponent:
    :return: V2
    """
    nTs_eta = len(Ts_eta)
    tmp_square_ij = rij ** p_exponent
    tmp_square_ik = rik ** p_exponent
    tmp_square_jk = rjk ** p_exponent

    tmp_exp_ij = np.exp(- tmp_square_ij * Ts_eta)
    tmp_exp_ik = np.exp(- tmp_square_ik * Ts_eta)
    tmp_exp_jk = np.exp(- tmp_square_jk * Ts_eta)

    for eID1 in range(nTs_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(eID2 + 1):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]

                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp132 = tmp_exp_ij[eID1] * tmp_exp_ik[eID3] * tmp_exp_jk[eID2]
                tmp_proj132 = -(proj_ij * eta_op1 + proj_ik * eta_op3)
                tmp132 = tmp132 * tmp_proj132

                tmp213 = tmp_exp_ij[eID2] * tmp_exp_ik[eID1] * tmp_exp_jk[eID3]
                tmp_proj213 = -(proj_ij * eta_op2 + proj_ik * eta_op1)
                tmp213 = tmp213 * tmp_proj213

                tmp231 = tmp_exp_ij[eID2] * tmp_exp_ik[eID3] * tmp_exp_jk[eID1]
                tmp_proj231 = -(proj_ij * eta_op2 + proj_ik * eta_op3)
                tmp231 = tmp231 * tmp_proj231

                tmp312 = tmp_exp_ij[eID3] * tmp_exp_ik[eID1] * tmp_exp_jk[eID2]
                tmp_proj312 = -(proj_ij * eta_op3 + proj_ik * eta_op1)
                tmp312 = tmp312 * tmp_proj312

                tmp321 = tmp_exp_ij[eID3] * tmp_exp_ik[eID2] * tmp_exp_jk[eID1]
                tmp_proj321 = -(proj_ij * eta_op3 + proj_ik * eta_op2)
                tmp321 = tmp321 * tmp_proj321

                tmp = (tmp123 + tmp132 + tmp213 + tmp231 + tmp312 + tmp321) * frc_3b

                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


## 1/6 symmtry and ratio_sigma
@jit
def sum_G3b01ss(V2,
                frc_3b,
                proj_ij, proj_ik,
                rij, rik, rjk,
                Ts_eta, p_exponent=2
                ):
    """

    :param V2:
    :param frc_3b:
    :param proj_ij:
    :param proj_ik:
    :param rij:
    :param rik:
    :param rjk:
    :param Ts_eta:
    :param p_exponent:
    :return: V2
    """
    nTs_eta = len(Ts_eta)
    proj_ij = proj_ij
    proj_ik = proj_ik

    tmp_square_ij = rij ** p_exponent
    tmp_square_ik = rik ** p_exponent
    tmp_square_jk = rjk ** p_exponent

    tmp_exp_ij = np.exp(- tmp_square_ij * Ts_eta)
    tmp_exp_ik = np.exp(- tmp_square_ik * Ts_eta)
    tmp_exp_jk = np.exp(- tmp_square_jk * Ts_eta)

    for eID1 in range(nTs_eta):
        for eID2 in range(nTs_eta):
            for eID3 in range(eID2 + 1):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]

                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp132 = tmp_exp_ij[eID1] * tmp_exp_ik[eID3] * tmp_exp_jk[eID2]
                tmp_proj132 = -(proj_ij * eta_op1 + proj_ik * eta_op3)
                tmp132 = tmp132 * tmp_proj132

                tmp = (tmp123 + tmp132) * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


@jit
def sum_G3b11ss(V2,
                frc_3b,
                proj_ij, proj_ik,
                rij, rik, rjk,
                Ts_eta,
                p_exponent=2
                ):
    """

    :param V2:
    :param frc_3b:
    :param proj_ij:
    :param proj_ik:
    :param rij:
    :param rik:
    :param rjk:
    :param Ts_eta:
    :param p_exponent:
    :return: V2
    """
    nTs_eta = len(Ts_eta)
    proj_ij = proj_ij
    proj_ik = proj_ik

    tmp_square_ij = rij ** p_exponent
    tmp_square_ik = rik ** p_exponent
    tmp_square_jk = rjk ** p_exponent

    tmp_exp_ij = np.exp(- tmp_square_ij * Ts_eta)
    tmp_exp_ik = np.exp(- tmp_square_ik * Ts_eta)
    tmp_exp_jk = np.exp(- tmp_square_jk * Ts_eta)

    for eID1 in range(nTs_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(nTs_eta):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]  # TODO 20230117 : why not using eta_op3?

                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp213 = tmp_exp_ij[eID2] * tmp_exp_ik[eID1] * tmp_exp_jk[eID3]
                tmp_proj213 = -(proj_ij * eta_op2 + proj_ik * eta_op1)
                tmp213 = tmp213 * tmp_proj213

                tmp = (tmp123 + tmp213) * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


@jit
def sum_G2b00_all(
        V1,
        factor,
        rij,
        R_Rs, R_eta,
        p_exponent=2
):
    """

    :param V1:
    :param factor:
    :param rij:
    :param R_Rs:
    :param R_eta:
    :param p_exponent:
    :return: V1
    """
    for RsID in range(len(R_Rs)):
        tmp_Rs = rij - R_Rs[RsID]
        for etID in range(len(R_eta)):
            tmp1 = factor * exp(-R_eta[etID] * tmp_Rs ** p_exponent)
            V1[etID][RsID] = V1[etID][RsID] + tmp1

    return V1


@jit
def MultiSplit2b3b_index_ss_nb(
        numbers,
        nb_index,
        xij0, yij0, zij0,
        atomID,
        v,
        Param_dict
):
    """

    :param numbers:
    :param nb_index:
    :param xij0:
    :param yij0:
    :param zij0:
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
    G3b_lst_param_eta = Param_dict.get('G3b_lst_param_eta')

    Rc = max(Rc2b, Rc3b)

    nG2b_eta = len(G2b_eta)
    nG2b_Rs = len(G2b_Rs)
    nG3b_eta = len(G3b_eta)

    G2b_lst = Param_dict.get('G2b_lst')
    G3b_lst = Param_dict.get('G3b_lst')

    n = len(nb_index)
    atom_name = numbers[atomID]

    vx, vy, vz = v[0], v[1], v[2]

    V1_00 = np.zeros((nG2b_eta, nG2b_Rs))  # AA
    V1_01 = np.zeros((nG2b_eta, nG2b_Rs))  # AB
    V2_00 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))
    V2_01 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))
    V2_11 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))
    # print("V1 size len = ", len(R_eta), len(R_Rs))
    # print("V1 size = ", len(V1_00), len(V1_01))

    for j in range(n):

        j_index = nb_index[j]

        xij = xij0[j]
        yij = yij0[j]
        zij = zij0[j]
        rij = sqrt(xij ** 2 + yij ** 2 + zij ** 2)

        # vec_rij dot F_vector
        coeff_ij = (xij * vx + yij * vy + zij * vz)
        # function of cut off
        frc_ij = 0.5 * (cos(pi * rij / Rc) + 1)
        tmp0 = coeff_ij * frc_ij

        # Compare Element name
        if atom_name == numbers[j_index]:
            V1_00 = sum_G2b00_all(V1_00, tmp0, rij, G2b_Rs, G2b_eta)

        else:

            # Bond length Si-Ge of Rc
            # Eta = [ ], Rc, Rs
            # Lasso
            # MD Conquest
            # MOs
            V1_01 = sum_G2b00_all(V1_01, tmp0, rij, G2b_Rs, G2b_eta)
            # V1_01 = sum_G2b00(V1_01, tmp0, rij, R_Rs, R_eta)

        for k in range(j + 1, n):

            k_index = nb_index[k]
            xik = xij0[k]
            yik = yij0[k]
            zik = zij0[k]
            rik = sqrt(xik ** 2 + yik ** 2 + zik ** 2)
            xjk = xik - xij
            yjk = yik - yij
            zjk = zik - zij
            rjk = sqrt(xjk ** 2 + yjk ** 2 + zjk ** 2)

            R_3b = sqrt(rij ** 2 + rik ** 2 + rjk ** 2)
            Rc_3b = sqrt(Rc ** 2)

            if (Rc_3b > R_3b > small_number) and (rik > small_number) and (rjk > small_number):
                # TODO 20230113: cutoff is product of separate cut functions --> check
                # frc_3b = 0.5 * (cos(pi * R_3b / Rc_3b) + 1)
                frc_ij = 0.5 * (cos(pi * rij / Rc) + 1)
                frc_jk = 0.5 * (cos(pi * rjk / Rc) + 1)
                frc_ik = 0.5 * (cos(pi * rik / Rc) + 1)
                frc_3b = frc_ij * frc_jk * frc_ik

                proj_ij = (xij + 0) * vx + (yij + 0) * vy + (zij + 0) * vz
                proj_ik = (0 + xik) * vx + (0 + yik) * vy + (0 + zik) * vz

                if atom_name == numbers[j_index] and atom_name == numbers[k_index]:
                    V2_00 = sum_G3b00ss(V2_00, frc_3b, proj_ij, proj_ik, rij, rik, rjk, G3b_eta)

                elif atom_name == numbers[j_index] and atom_name != numbers[k_index]:
                    # V2_01 = sum_G3b01(V2_01, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta)
                    # V2_01 = sum_G3b01s(V2_01, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta)
                    V2_01 = sum_G3b01ss(V2_01, frc_3b, proj_ij, proj_ik, rij, rik, rjk, G3b_eta)

                elif atom_name != numbers[j_index] and atom_name == numbers[k_index]:
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


@jit
def sum_G2b00_all(
        V1,
        factor,
        rij,
        R_Rs, R_eta
):
    """

    :param V1:
    :param factor:
    :param rij:
    :param R_Rs:
    :param R_eta:
    :return: V1
    """
    for RsID in range(len(R_Rs)):
        tmp_Rs = rij - R_Rs[RsID]
        for etID in range(len(R_eta)):
            tmp1 = factor * math.exp(-R_eta[etID] * tmp_Rs ** p)
            V1[etID][RsID] = V1[etID][RsID] + tmp1
    return V1


# default Split2b3b
@jit
def sum_G3b00(
        V2,
        frc_3b,
        proj_ij, proj_ik,
        rij, rik, rjk,
        Ts_eta
):
    """

    :param V2:
    :param frc_3b:
    :param proj_ij:
    :param proj_ik:
    :param rij:
    :param rik:
    :param rjk:
    :param Ts_eta:
    :return: V2
    """
    nTs_eta = len(Ts_eta)
    tmp_square_ij = rij ** p
    tmp_square_ik = rik ** p
    tmp_square_jk = rjk ** p

    tmp_exp_ij = np.exp(- tmp_square_ij * Ts_eta)
    tmp_exp_ik = np.exp(- tmp_square_ik * Ts_eta)
    tmp_exp_jk = np.exp(- tmp_square_jk * Ts_eta)

    for eID1 in range(nTs_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(eID2 + 1):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]
                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp132 = tmp_exp_ij[eID1] * tmp_exp_ik[eID3] * tmp_exp_jk[eID2]
                tmp_proj132 = -(proj_ij * eta_op1 + proj_ik * eta_op3)
                tmp132 = tmp132 * tmp_proj132

                tmp213 = tmp_exp_ij[eID2] * tmp_exp_ik[eID1] * tmp_exp_jk[eID3]
                tmp_proj213 = -(proj_ij * eta_op2 + proj_ik * eta_op1)
                tmp213 = tmp213 * tmp_proj213

                tmp231 = tmp_exp_ij[eID2] * tmp_exp_ik[eID3] * tmp_exp_jk[eID1]
                tmp_proj231 = -(proj_ij * eta_op2 + proj_ik * eta_op3)
                tmp231 = tmp231 * tmp_proj231

                tmp312 = tmp_exp_ij[eID3] * tmp_exp_ik[eID1] * tmp_exp_jk[eID2]
                tmp_proj312 = -(proj_ij * eta_op3 + proj_ik * eta_op1)
                tmp312 = tmp312 * tmp_proj312

                tmp321 = tmp_exp_ij[eID3] * tmp_exp_ik[eID2] * tmp_exp_jk[eID1]
                tmp_proj321 = -(proj_ij * eta_op3 + proj_ik * eta_op2)
                tmp321 = tmp321 * tmp_proj321

                tmp = (tmp123 + tmp132 + tmp213 + tmp231 + tmp312 + tmp321) * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


# 1/6 symmtry and ratio_sigma
@jit
def sum_G3b01rs(
        V2,
        frc_3b,
        proj_ij, proj_ik,
        rij, rik, rjk,
        Ts_eta
):
    """

    :param V2:
    :param frc_3b:
    :param proj_ij:
    :param proj_ik:
    :param rij:
    :param rik:
    :param rjk:
    :param Ts_eta:
    :return: V2
    """
    nTs_eta = len(Ts_eta)
    proj_ij = proj_ij
    proj_ik = proj_ik

    tmp_square_ij = rij ** p
    tmp_square_ik = rik ** p
    tmp_square_jk = rjk ** p

    tmp_exp_ij = np.exp(- tmp_square_ij * Ts_eta)
    tmp_exp_ik = np.exp(- tmp_square_ik * Ts_eta)
    tmp_exp_jk = np.exp(- tmp_square_jk * Ts_eta)

    for eID1 in range(nTs_eta):
        for eID2 in range(nTs_eta):
            for eID3 in range(eID2 + 1):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]
                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp132 = tmp_exp_ij[eID1] * tmp_exp_ik[eID3] * tmp_exp_jk[eID2]
                tmp_proj132 = -(proj_ij * eta_op1 + proj_ik * eta_op3)
                tmp132 = tmp132 * tmp_proj132

                tmp213 = tmp_exp_ij[eID2] * tmp_exp_ik[eID1] * tmp_exp_jk[eID3]
                tmp_proj213 = -(proj_ij * eta_op2 + proj_ik * eta_op1)
                tmp213 = tmp213 * tmp_proj213

                tmp231 = tmp_exp_ij[eID2] * tmp_exp_ik[eID3] * tmp_exp_jk[eID1]
                tmp_proj231 = -(proj_ij * eta_op2 + proj_ik * eta_op3)
                tmp231 = tmp231 * tmp_proj231

                tmp312 = tmp_exp_ij[eID3] * tmp_exp_ik[eID1] * tmp_exp_jk[eID2]
                tmp_proj312 = -(proj_ij * eta_op3 + proj_ik * eta_op1)
                tmp312 = tmp312 * tmp_proj312

                tmp321 = tmp_exp_ij[eID3] * tmp_exp_ik[eID2] * tmp_exp_jk[eID1]
                tmp_proj321 = -(proj_ij * eta_op3 + proj_ik * eta_op2)
                tmp321 = tmp321 * tmp_proj321

                tmp = (tmp123 + tmp132 + tmp213 + tmp231 + tmp312 + tmp321) * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


@jit
def sum_G3b11rs(V2, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta):
    """

    :param V2:
    :param frc_3b:
    :param proj_ij:
    :param proj_ik:
    :param rij:
    :param rik:
    :param rjk:
    :param Ts_eta:
    :return: V2
    """
    nTs_eta = len(Ts_eta)
    proj_ij = proj_ij
    proj_ik = proj_ik

    tmp_square_ij = rij ** p
    tmp_square_ik = rik ** p
    tmp_square_jk = rjk ** p

    tmp_exp_ij = []
    tmp_exp_ik = []
    tmp_exp_jk = []

    for eID in range(nTs_eta):
        tmp = math.exp(- tmp_square_ij * Ts_eta[eID])
        tmp_exp_ij.append(tmp)

        tmp = math.exp(- tmp_square_ik * Ts_eta[eID])
        tmp_exp_ik.append(tmp)

        tmp = math.exp(- tmp_square_jk * Ts_eta[eID])
        tmp_exp_jk.append(tmp)

    for eID1 in range(nTs_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(nTs_eta):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]
                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp132 = tmp_exp_ij[eID1] * tmp_exp_ik[eID3] * tmp_exp_jk[eID2]
                tmp_proj132 = -(proj_ij * eta_op1 + proj_ik * eta_op3)
                tmp132 = tmp132 * tmp_proj132

                tmp213 = tmp_exp_ij[eID2] * tmp_exp_ik[eID1] * tmp_exp_jk[eID3]
                tmp_proj213 = -(proj_ij * eta_op2 + proj_ik * eta_op1)
                tmp213 = tmp213 * tmp_proj213

                tmp231 = tmp_exp_ij[eID2] * tmp_exp_ik[eID3] * tmp_exp_jk[eID1]
                tmp_proj231 = -(proj_ij * eta_op2 + proj_ik * eta_op3)
                tmp231 = tmp231 * tmp_proj231

                tmp312 = tmp_exp_ij[eID3] * tmp_exp_ik[eID1] * tmp_exp_jk[eID2]
                tmp_proj312 = -(proj_ij * eta_op3 + proj_ik * eta_op1)
                tmp312 = tmp312 * tmp_proj312

                tmp321 = tmp_exp_ij[eID3] * tmp_exp_ik[eID2] * tmp_exp_jk[eID1]
                tmp_proj321 = -(proj_ij * eta_op3 + proj_ik * eta_op2)
                tmp321 = tmp321 * tmp_proj321

                tmp = (tmp123 + tmp132 + tmp213 + tmp231 + tmp312 + tmp321) * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


# default Split2b3b
@jit
def sum_G3b00ss(V2, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta):
    """

    :param V2:
    :param frc_3b:
    :param proj_ij:
    :param proj_ik:
    :param rij:
    :param rik:
    :param rjk:
    :param Ts_eta:
    :return: V2
    """
    nTs_eta = len(Ts_eta)
    tmp_square_ij = rij ** p
    tmp_square_ik = rik ** p
    tmp_square_jk = rjk ** p

    tmp_exp_ij = np.exp(- tmp_square_ij * Ts_eta)
    tmp_exp_ik = np.exp(- tmp_square_ik * Ts_eta)
    tmp_exp_jk = np.exp(- tmp_square_jk * Ts_eta)

    for eID1 in range(nTs_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(eID2 + 1):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]

                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp132 = tmp_exp_ij[eID1] * tmp_exp_ik[eID3] * tmp_exp_jk[eID2]
                tmp_proj132 = -(proj_ij * eta_op1 + proj_ik * eta_op3)
                tmp132 = tmp132 * tmp_proj132

                tmp213 = tmp_exp_ij[eID2] * tmp_exp_ik[eID1] * tmp_exp_jk[eID3]
                tmp_proj213 = -(proj_ij * eta_op2 + proj_ik * eta_op1)
                tmp213 = tmp213 * tmp_proj213

                tmp231 = tmp_exp_ij[eID2] * tmp_exp_ik[eID3] * tmp_exp_jk[eID1]
                tmp_proj231 = -(proj_ij * eta_op2 + proj_ik * eta_op3)
                tmp231 = tmp231 * tmp_proj231

                tmp312 = tmp_exp_ij[eID3] * tmp_exp_ik[eID1] * tmp_exp_jk[eID2]
                tmp_proj312 = -(proj_ij * eta_op3 + proj_ik * eta_op1)
                tmp312 = tmp312 * tmp_proj312

                tmp321 = tmp_exp_ij[eID3] * tmp_exp_ik[eID2] * tmp_exp_jk[eID1]
                tmp_proj321 = -(proj_ij * eta_op3 + proj_ik * eta_op2)
                tmp321 = tmp321 * tmp_proj321

                tmp = (tmp123 + tmp132 + tmp213 + tmp231 + tmp312 + tmp321) * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


## 1/6 symmtry and ratio_sigma
@jit
def sum_G3b01ss(V2, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta):
    """

    :param V2:
    :param frc_3b:
    :param proj_ij:
    :param proj_ik:
    :param rij:
    :param rik:
    :param rjk:
    :param Ts_eta:
    :return: V2
    """
    nTs_eta = len(Ts_eta)
    proj_ij = proj_ij
    proj_ik = proj_ik

    tmp_square_ij = rij ** p
    tmp_square_ik = rik ** p
    tmp_square_jk = rjk ** p

    tmp_exp_ij = np.exp(- tmp_square_ij * Ts_eta)
    tmp_exp_ik = np.exp(- tmp_square_ik * Ts_eta)
    tmp_exp_jk = np.exp(- tmp_square_jk * Ts_eta)

    for eID1 in range(nTs_eta):
        for eID2 in range(nTs_eta):
            for eID3 in range(eID2 + 1):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]

                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp132 = tmp_exp_ij[eID1] * tmp_exp_ik[eID3] * tmp_exp_jk[eID2]
                tmp_proj132 = -(proj_ij * eta_op1 + proj_ik * eta_op3)
                tmp132 = tmp132 * tmp_proj132

                tmp = (tmp123 + tmp132) * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


@jit
def sum_G3b11ss(V2, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta):
    """

    :param V2:
    :param frc_3b:
    :param proj_ij:
    :param proj_ik:
    :param rij:
    :param rik:
    :param rjk:
    :param Ts_eta:
    :return: V2
    """
    nTs_eta = len(Ts_eta)
    proj_ij = proj_ij
    proj_ik = proj_ik

    tmp_square_ij = rij ** p
    tmp_square_ik = rik ** p
    tmp_square_jk = rjk ** p

    tmp_exp_ij = np.exp(- tmp_square_ij * Ts_eta)
    tmp_exp_ik = np.exp(- tmp_square_ik * Ts_eta)
    tmp_exp_jk = np.exp(- tmp_square_jk * Ts_eta)

    for eID1 in range(nTs_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(nTs_eta):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]

                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp213 = tmp_exp_ij[eID2] * tmp_exp_ik[eID1] * tmp_exp_jk[eID3]
                tmp_proj213 = -(proj_ij * eta_op2 + proj_ik * eta_op1)
                tmp213 = tmp213 * tmp_proj213

                tmp = (tmp123 + tmp213) * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


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
    G3b_lst_param_eta = Param_dict.get('G3b_lst_param_eta')

    Rc = max(Rc2b, Rc3b)

    nG2b_eta = len(G2b_eta)
    nG2b_Rs = len(G2b_Rs)
    nG3b_eta = len(G3b_eta)

    G2b_lst = Param_dict.get('G2b_lst')
    G3b_lst = Param_dict.get('G3b_lst')

    i = atomID
    numbers = atomsx.numbers

    n = len(numbers)
    atom_name = numbers[atomID]

    X, Y, Z = atomsx.positions[:, 0], atomsx.positions[:, 1], atomsx.positions[:, 2]
    position = np.array([X[atomID], Y[atomID], Z[atomID]])
    xij0, yij0, zij0 = X - position[0], Y - position[1], Z - position[2]

    cell = atomsx.cell
    m_x, m_y, m_z = define_mirror_reduce_cubic(position, cell, Rc)
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

                        R_3b = math.sqrt(rij ** 2 + rik ** 2 + rjk ** 2)
                        Rc_3b = math.sqrt(Rc ** 2)
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


## 1/6 symmtry and ratio_sigma
@jit
def sum_G3b00_s(V2, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta):
    nTs_eta = len(Ts_eta)
    tmp_square_ij = rij ** p
    tmp_square_ik = rik ** p
    tmp_square_jk = rjk ** p

    tmp_exp_ij = np.zeros(nTs_eta)
    tmp_exp_ik = np.zeros(nTs_eta)
    tmp_exp_jk = np.zeros(nTs_eta)

    for eID in range(nTs_eta):
        tmp_exp_ij[eID] = math.exp(- tmp_square_ij * Ts_eta[eID])
        tmp_exp_ik[eID] = math.exp(- tmp_square_ik * Ts_eta[eID])
        tmp_exp_jk[eID] = math.exp(- tmp_square_jk * Ts_eta[eID])

    for eID1 in range(nTs_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(eID2 + 1):
                eta_op1 = 1.0 / (Ts_eta[eID1] ** p)
                eta_op2 = 1.0 / (Ts_eta[eID2] ** p)
                eta_op3 = 1.0 / (Ts_eta[eID3] ** p)
                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp = tmp123 * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


@jit
def sum_G3b01rs_s(V2, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta):
    nTs_eta = len(Ts_eta)
    proj_ij = proj_ij
    proj_ik = proj_ik

    tmp_square_ij = rij ** p
    tmp_square_ik = rik ** p
    tmp_square_jk = rjk ** p

    tmp_exp_ij = np.zeros(nTs_eta)
    tmp_exp_ik = np.zeros(nTs_eta)
    tmp_exp_jk = np.zeros(nTs_eta)

    for eID in range(nTs_eta):
        tmp_exp_ij[eID] = math.exp(- tmp_square_ij * Ts_eta[eID])
        tmp_exp_ik[eID] = math.exp(- tmp_square_ik * Ts_eta[eID])
        tmp_exp_jk[eID] = math.exp(- tmp_square_jk * Ts_eta[eID])

    for eID1 in range(nTs_eta):
        for eID2 in range(nTs_eta):
            for eID3 in range(eID2 + 1):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]
                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp = tmp123 * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


@jit
def sum_G3b11rs_s(V2, frc_3b, proj_ij, proj_ik, rij, rik, rjk, Ts_eta):
    nTs_eta = len(Ts_eta)
    proj_ij = proj_ij
    proj_ik = proj_ik

    tmp_square_ij = rij ** p
    tmp_square_ik = rik ** p
    tmp_square_jk = rjk ** p

    tmp_exp_ij = np.zeros(nTs_eta)
    tmp_exp_ik = np.zeros(nTs_eta)
    tmp_exp_jk = np.zeros(nTs_eta)

    for eID in range(nTs_eta):
        tmp_exp_ij[eID] = math.exp(- tmp_square_ij * Ts_eta[eID])
        tmp_exp_ik[eID] = math.exp(- tmp_square_ik * Ts_eta[eID])
        tmp_exp_jk[eID] = math.exp(- tmp_square_jk * Ts_eta[eID])

    for eID1 in range(nTs_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(nTs_eta):
                eta_op1 = Ts_eta[eID1]
                eta_op2 = Ts_eta[eID2]
                eta_op3 = Ts_eta[eID3]
                tmp123 = tmp_exp_ij[eID1] * tmp_exp_ik[eID2] * tmp_exp_jk[eID3]
                tmp_proj123 = -(proj_ij * eta_op1 + proj_ik * eta_op2)
                tmp123 = tmp123 * tmp_proj123

                tmp = tmp123 * frc_3b
                V2[eID1][eID2][eID3] = V2[eID1][eID2][eID3] + tmp

    return V2


@jit
def MultiSplit2b3b_khoa(
        atomsx,
        atomID,
        v,
        Param_dict
):
    """
    Modified version for testing purpose (20230117)
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

    Rc = max(Rc2b, Rc3b)

    nG2b_eta = len(G2b_eta)
    nG2b_Rs = len(G2b_Rs)
    nG3b_eta = len(G3b_eta)

    numbers = atomsx.numbers

    number_of_atoms = len(numbers)
    atom_name = numbers[atomID]

    X, Y, Z = atomsx.positions[:, 0], atomsx.positions[:, 1], atomsx.positions[:, 2]
    position = np.array([X[atomID], Y[atomID], Z[atomID]])
    xij0, yij0, zij0 = X - position[0], Y - position[1], Z - position[2]

    cell = atomsx.cell
    m_x, m_y, m_z = define_mirror_reduce_cubic(position, cell, Rc)
    vx, vy, vz = v[0], v[1], v[2]

    V1_00 = np.zeros((nG2b_eta, nG2b_Rs))  # AA
    V1_01 = np.zeros((nG2b_eta, nG2b_Rs))  # AB
    V2_00 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))
    V2_01 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))
    V2_11 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))

    for j in range(number_of_atoms):

        for mj in range(len(m_x)):

            xij = xij0[j] + m_x[mj]
            yij = yij0[j] + m_y[mj]
            zij = zij0[j] + m_z[mj]
            rij = math.sqrt(xij ** 2 + yij ** 2 + zij ** 2)

            if Rc > rij > small:

                # vec_rij dot F_vector
                coeff_ij = (xij * vx + yij * vy + zij * vz)
                # function of cut off
                frc_ij = fcut(rij, Rc)
                # frc_ij = 0.5 * (math.cos(pi * rij / Rc) + 1)
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

                for k in range(j, number_of_atoms):

                    for mk in range(len(m_x)):

                        xik = xij0[k] + m_x[mk]
                        yik = yij0[k] + m_y[mk]
                        zik = zij0[k] + m_z[mk]
                        rik = math.sqrt(xik ** 2 + yik ** 2 + zik ** 2)
                        xjk = xik - xij
                        yjk = yik - yij
                        zjk = zik - zij
                        rjk = math.sqrt(xjk ** 2 + yjk ** 2 + zjk ** 2)

                        R_3b = math.sqrt(rij ** 2 + rik ** 2 + rjk ** 2)
                        Rc_3b = math.sqrt(Rc ** 2)

                        if (Rc_3b > R_3b > small) and (rik > small) and (rjk > small):
                            # frc_ik = 0.5 * (math.cos(pi * rik / Rc) + 1)
                            # frc_jk = 0.5 * (math.cos(pi * rjk / Rc) + 1)
                            frc_ik = fcut(rik, Rc)
                            frc_jk = fcut(rjk, Rc)
                            frc_3b = frc_ij * frc_ik * frc_jk

                            proj_ij = (xij + 0) * vx + (yij + 0) * vy + (zij + 0) * vz
                            proj_ik = (0 + xik) * vx + (0 + yik) * vy + (0 + zik) * vz

                            if atom_name == numbers[j] and atom_name == numbers[k]:
                                V2_00 = sum_G3b00ss(V2_00, frc_3b, proj_ij, proj_ik, rij, rik, rjk, G3b_eta)

                            elif atom_name == numbers[j] and atom_name != numbers[k]:
                                V2_01 = sum_G3b01ss(V2_01, frc_3b, proj_ij, proj_ik, rij, rik, rjk, G3b_eta)

                            elif atom_name != numbers[j] and atom_name == numbers[k]:
                                # need to change the position of i,j,k --input
                                # j->k
                                V2_01 = sum_G3b01ss(V2_01, frc_3b, proj_ik, proj_ij, rik, rij, rjk, G3b_eta)

                            else:
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


@jit
def MultiSplit2b3b_khoa_exec(
        numbers,
        nb_index,
        xij0, yij0, zij0,
        atomID,
        v,
        Param_dict
):
    """
        Calculation of split2b3b during MD
    :param numbers:
    :param nb_index:
    :param xij0:
    :param yij0:
    :param zij0:
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

    Rc = max(Rc2b, Rc3b)

    nG2b_eta = len(G2b_eta)
    nG2b_Rs = len(G2b_Rs)
    nG3b_eta = len(G3b_eta)

    n = len(nb_index)
    atom_name = numbers[atomID]

    vx, vy, vz = v[0], v[1], v[2]

    V1_00 = np.zeros((nG2b_eta, nG2b_Rs))  # AA
    V1_01 = np.zeros((nG2b_eta, nG2b_Rs))  # AB
    V2_00 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))
    V2_01 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))
    V2_11 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))

    for j in range(n):

        j_index = nb_index[j]

        xij = xij0[j]
        yij = yij0[j]
        zij = zij0[j]
        rij = sqrt(xij ** 2 + yij ** 2 + zij ** 2)

        # vec_rij dot F_vector
        coeff_ij = (xij * vx + yij * vy + zij * vz)
        # function of cut off
        frc_ij = 0.5 * (cos(pi * rij / Rc) + 1)
        tmp0 = coeff_ij * frc_ij

        # Compare Element name
        if atom_name == numbers[j_index]:
            V1_00 = sum_G2b00_all(V1_00, tmp0, rij, G2b_Rs, G2b_eta)

        else:

            # Bond length Si-Ge of Rc
            # Eta = [ ], Rc, Rs
            # Lasso
            # MD Conquest
            # MOs
            V1_01 = sum_G2b00_all(V1_01, tmp0, rij, G2b_Rs, G2b_eta)
            # V1_01 = sum_G2b00(V1_01, tmp0, rij, R_Rs, R_eta)

        for k in range(j + 1, n):

            k_index = nb_index[k]
            xik = xij0[k]
            yik = yij0[k]
            zik = zij0[k]
            rik = sqrt(xik ** 2 + yik ** 2 + zik ** 2)
            xjk = xik - xij
            yjk = yik - yij
            zjk = zik - zij
            rjk = sqrt(xjk ** 2 + yjk ** 2 + zjk ** 2)

            R_3b = sqrt(rij ** 2 + rik ** 2 + rjk ** 2)
            Rc_3b = sqrt(Rc ** 2)

            if (Rc_3b > R_3b > small) and (rik > small) and (rjk > small):
                frc_jk = 0.5 * (cos(pi * rjk / Rc) + 1)
                frc_ik = 0.5 * (cos(pi * rik / Rc) + 1)
                frc_3b = frc_ij * frc_jk * frc_ik

                proj_ij = (xij + 0) * vx + (yij + 0) * vy + (zij + 0) * vz
                proj_ik = (0 + xik) * vx + (0 + yik) * vy + (0 + zik) * vz

                if atom_name == numbers[j_index] and atom_name == numbers[k_index]:
                    V2_00 = sum_G3b00ss(V2_00, frc_3b, proj_ij, proj_ik, rij, rik, rjk, G3b_eta)

                elif atom_name == numbers[j_index] and atom_name != numbers[k_index]:
                    V2_01 = sum_G3b01ss(V2_01, frc_3b, proj_ij, proj_ik, rij, rik, rjk, G3b_eta)

                elif atom_name != numbers[j_index] and atom_name == numbers[k_index]:
                    # need to change the position of i,j,k --input
                    # j->k
                    V2_01 = sum_G3b01ss(V2_01, frc_3b, proj_ik, proj_ij, rik, rij, rjk, G3b_eta)

                else:
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
