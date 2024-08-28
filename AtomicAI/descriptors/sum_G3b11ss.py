from numba import jit
import numpy as np

p = 2

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
