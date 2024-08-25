import numpy as np
from math import pi
from AtomicAI.descriptors.define_eta import define_eta
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
