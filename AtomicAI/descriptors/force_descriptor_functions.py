import math
import numpy as np
from numba import jit

# 問題に利用されるパラメータの設定を行います。
p = 2
pi = 4.0 * math.atan(1.0)
small = 1.0e-12


parameters = {
    #      cut off for fingerprint
    #        AA
    'Rc2b': 7.0,
    'Rc3b': 7.0,
    'Reta': 7.0,
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

@jit
def define_eta(eta_range,num):
    # eta_select: [eta_min, eta_max, eta_num]
    R0 = eta_range[0]
    R1 = eta_range[1]
    eta_num = int(num)
    eta = np.array([R0])

    for _ in range(eta_num - 1):
        eta = np.append(eta, np.array([(R1 / R0) ** (1 / eta_num) * eta[-1]]))

    return eta

def define_mirror_cubic(position, cell, Rc):
    m_min = [0,0,0]
    m_max = [0,0,0]
    for i in range(len(position)):
        if(position[i] < Rc): m_min[i] = -1
        if(cell[i,i]-position[i]) < Rc : m_max[i] = 1
    m_x = []
    m_y = []
    m_z = []
    for i in range(m_min[0], m_max[0]+1):
        for j in range(m_min[1], m_max[1] + 1):
            for k in range(m_min[2], m_max[2] + 1):
                m_x.append(i* cell[0,0])
                m_y.append(j* cell[1,1])
                m_z.append(k* cell[2,2])

    return m_x, m_y, m_z

# ハイパーパラメータはmake_V typeによって設定します。
def set_param_dict(parameters, fp_flag):
    Rc2b = parameters.get('Rc2b')
    Rc3b = parameters.get('Rc3b')
    #print('Rc2b/Rc3b =', Rc2b, Rc3b)
    param_dict = {"Rc2b": Rc2b}

    G2b_eta_range = list(parameters['2b'][0:2])
    G2b_eta_num = parameters.get('2b')[2]
    G2b_dRs = parameters.get('2b')[3]

    Reta=parameters.get('Reta')
    eta_const = list(np.exp(np.array(G2b_eta_range)) * Reta)
    G2b_eta = define_eta(eta_const, G2b_eta_num)
    #print('G2b_eta length')
    #print(G2b_eta)

    G2b_eta = 0.5 / np.square(np.array(G2b_eta))
    G2b_Rs = list(np.arange(0, Rc2b, G2b_dRs))
    #print('G2b_eta')
    #print(G2b_eta)
    #print('G2b_Rs')
    #print(G2b_Rs)

    param_dict.update(G2b_eta=G2b_eta)
    param_dict.update(G2b_Rs=G2b_Rs)
    nfp = 0
    if fp_flag == 'BP2b':
        #nfp = len(G2b_eta) * len(G2b_Rs)
        nfp = len(G2b_eta)
    elif (fp_flag == 'LA2b3b') or (fp_flag == 'DerMBSF'):
        # parameters for 3body-term
        para_G3b = parameters.get('3b')
        G3b_eta_range = list(para_G3b[0:2])
        G3b_eta_num = para_G3b[2]

        eta_const = list(np.exp(np.array(G3b_eta_range)) * Reta)
        G3b_eta = define_eta(eta_const, G3b_eta_num)
        G3b_eta = 0.5 / np.square(np.array(G3b_eta))

        G3b_dRs = parameters.get('3b')[3]
        G3b_zeta_num = parameters.get('3b')[4]
        G3b_theta_num = parameters.get('3b')[5]
        G3b_Rs = list(np.arange(0, Rc3b, G3b_dRs))

        zeta_lst = [1, 2, 4, 16]
        G3b_zeta = zeta_lst[0:G3b_zeta_num]

        nx = G3b_theta_num
        dx = pi / (nx - 1)
        G3b_theta = [dx * x for x in range(nx)]

        param_dict.update(Rc3b=Rc3b)
        param_dict.update(G3b_eta=G3b_eta)
        param_dict.update(G3b_Rs=G3b_Rs)
        param_dict.update(G3b_zeta=G3b_zeta)
        param_dict.update(G3b_theta=G3b_theta)
        nfp = len(G2b_eta) * len(G2b_Rs) + len(G3b_eta) * len(G3b_zeta) * len(G3b_theta)

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
    else:
        print('Error: No such type of fingerprint(Define Parameters)!')
        exit()

    num_G1_d = len(G2b_eta) * len(G2b_Rs)
    num_G2_d = nfp - num_G1_d
    #print('Number of descriptor = %d(G1=%d,G2=%d)' % (num_G1_d + num_G2_d, num_G1_d, num_G2_d))
    return param_dict, nfp


@jit
def fcut(R,Rc):
    return 0.5 * (math.cos(pi * R / Rc) + 1)

# BP2b
@jit
def make_BP2b(xij0,yij0,zij0,m_x,m_y,m_z, v, Param_dict):
    G2b_eta = Param_dict.get('G2b_eta')
    Rc2b=Param_dict.get('Rc2b')
    nG2b_eta = len(G2b_eta)
    V1 = np.zeros(nG2b_eta)

    n = len(xij0)
    vx, vy, vz = v[0], v[1], v[2]
    for j in range(n):
        for mj in range(len(m_x)):
            xij = xij0[j] + m_x[mj]
            yij = yij0[j] + m_y[mj]
            zij = zij0[j] + m_z[mj]
            rij = math.sqrt(xij ** 2 + yij ** 2 + zij ** 2)

            if Rc2b > rij > small:
                coeff1 = (xij * vx + yij * vy + zij * vz) / rij
                frc_ij = fcut(rij, Rc2b)
                tmp0 = coeff1 * frc_ij
                for eID in range(nG2b_eta):
                    V1[eID] = V1[eID] + tmp0 * math.exp(-G2b_eta[eID] * rij ** p)

    return list(V1.flatten())

@jit
def make_Split2b3b_ss(xij0,yij0,zij0,m_x,m_y,m_z, v, Param_dict):
    G2b_eta = Param_dict.get('G2b_eta')
    G2b_Rs = Param_dict.get('G2b_Rs')
    Rc2b = Param_dict.get('Rc2b')
    G3b_eta = Param_dict.get('G3b_eta')
    Rc3b = Param_dict.get('Rc3b')

    nG2b_eta = len(G2b_eta)
    nG2b_Rs = len(G2b_Rs)
    nG3b_eta = len(G3b_eta)

    V1 = np.zeros((nG2b_eta, nG2b_Rs))
    V2 = np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))

    n=len(xij0)
    Rc = max(Rc2b,Rc3b)
    vx, vy, vz = v[0], v[1], v[2]
    for j in range(n):
        for mj in range(len(m_x)):
            xij = xij0[j] + m_x[mj]
            yij = yij0[j] + m_y[mj]
            zij = zij0[j] + m_z[mj]
            rij = math.sqrt(xij ** 2 + yij ** 2 + zij ** 2)

            if Rc2b > rij > small:
                coeff_ij = (xij * vx + yij * vy + zij * vz)
                frc_ij = fcut(rij, Rc2b)
                tmp0 = coeff_ij * frc_ij
                for RsID in range(nG2b_Rs):
                    tmp_Rs = rij - G2b_Rs[RsID]
                    for etID in range(nG2b_eta):
                        tmp1 = tmp0 * math.exp(-G2b_eta[etID] * tmp_Rs**p)
                        V1[etID][RsID] = V1[etID][RsID] + tmp1

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

                        if (Rc3b > rik > small) and (Rc3b > rjk > small):

                            frc_3b = fcut(rij,Rc3b)*fcut(rik,Rc3b)*fcut(rjk,Rc3b)

                            proj_ij = (xij + 0) * vx + (yij + 0) * vy + (zij + 0) * vz
                            proj_ik = (0 + xik) * vx + (0 + yik) * vy + (0 + zik) * vz
                            tmp_square_ij = rij ** p
                            tmp_square_ik = rik ** p
                            tmp_square_jk = rjk ** p

                            tmp_exp_ij = np.zeros(nG3b_eta)
                            tmp_exp_ik = np.zeros(nG3b_eta)
                            tmp_exp_jk = np.zeros(nG3b_eta)

                            for eID in range(nG3b_eta):
                                tmp_exp_ij[eID] = math.exp(- tmp_square_ij * G3b_eta[eID])
                                tmp_exp_ik[eID] = math.exp(- tmp_square_ik * G3b_eta[eID])
                                tmp_exp_jk[eID] = math.exp(- tmp_square_jk * G3b_eta[eID])

                            for eID1 in range(nG3b_eta):
                                for eID2 in range(eID1 + 1):
                                    for eID3 in range(eID2 + 1):
                                        eta_op1 = G3b_eta[eID1]
                                        eta_op2 = G3b_eta[eID2]
                                        eta_op3 = G3b_eta[eID3]
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


    V2_tmp = []
    for eID1 in range(nG3b_eta):
        for eID2 in range(eID1 + 1):
            for eID3 in range(eID2 + 1):
                V2_tmp.append(V2[eID1][eID2][eID3])
    V2 = np.array(V2_tmp)
    return list(np.r_[V1.flatten(), V2.flatten()])
