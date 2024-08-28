from numba import jit
from math import exp, sqrt, cos, pi
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
    #print(len(rij), tmp1)
    return V1
