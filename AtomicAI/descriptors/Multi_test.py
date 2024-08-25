#import wrnings
#warnings.filterwarnings("ignore")
from numba import jit
import math
import numpy as np
from math import pi

from AtomicAI.descriptors.sum_G2b00_all import sum_G2b00_all
from AtomicAI.descriptors.sum_G3b00ss import sum_G3b00ss

# Default variables
small_number = 1.0e-9
verysmall = 1.0e-65
small = 1e-4

sigma = 1.0 / 10
sigma2 = sigma ** 2

@jit
def MultiSplit2b3b_index_ss(variables):
    structure, target_position, vforce, Param_dict = variables
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
    Rc = max(Rc2b, Rc3b)
    nG2b_eta = len(G2b_eta)
    nG2b_Rs = len(G2b_Rs)
    nG3b_eta = len(G3b_eta)


    cell = structure.cell
    cell_mag = np.array(list([np.linalg.norm(np.array(cell)[i]) for i in range(3)]))
    if sum(cell_mag < Rc):
        Rc = min(cell_mag)


    inverse_cell = np.linalg.inv(cell)
    positions = structure.positions - target_position + cell_mag * 0.5
    fractional_positions = np.dot(positions, inverse_cell)
    convert_positions = fractional_positions % 1.0
    structure.positions = np.dot(convert_positions, cell)


    species_list = structure.get_chemical_symbols()
    n = len(species_list)
    species = list(set(species_list))
    no_of_2b_loop = len(species)

    V2b, V3b = {}, {}
    for specie in species:
        V2b.update({specie: np.zeros((nG2b_eta, nG2b_Rs))})

    for i in range(len(species)): # len of sorted species_list
        for j in range(len(species)):
            if i>=j:
                V2b.update({f'{species[i]}_{species[i]}': np.zeros((nG3b_eta, nG3b_eta, nG3b_eta))})

    X, Y, Z = structure.positions[:, 0], structure.positions[:, 1], structure.positions[:, 2]
    # = np.array([X[atomID], Y[atomID], Z[atomID]])
    xij, yij, zij = X - c_[0] * 0.5, Y - c_[1] * 0.5, Z - c_[2] * 0.5
    vx, vy, vz = vforce[0], vforce[1], vforce[2]


    for j in range(n):
        rij = math.sqrt(xij ** 2 + yij ** 2 + zij ** 2)

        if Rc > rij > small:
            # vec_rij dot F_vector
            coeff_ij = (xij * vx + yij * vy + zij * vz)
            # function of cut off
            frc_ij = 0.5 * (math.cos(pi * rij / Rc) + 1)
            tmp0 = coeff_ij * frc_ij

            # Compare Element name
            V2b[species_list[j]] = sum_G2b00_all(V2b[species_list[j]], tmp0, rij, G2b_Rs, G2b_eta)

            for k in range(j, n):
                xik, yik, zik = xij[k], yij[k], zij[k]
                rik = math.sqrt(xik ** 2 + yik ** 2 + zik ** 2)
                xjk, yjk, zjk = xik - xij, yik - yij, zik - zij
                rjk = math.sqrt(xjk ** 2 + yjk ** 2 + zjk ** 2)

                # TODO 20230117: Setup cutoffs properly
                # if (Rc_3b > R_3b > small) and (rik > small) and (rjk > small):
                if (Rc > rik > small) and (Rc > rjk > small):
                    frc_ik = 0.5 * (math.cos(pi * rik / Rc) + 1)
                    frc_jk = 0.5 * (math.cos(pi * rjk / Rc) + 1)
                    frc_3b = frc_ij * frc_ik * frc_jk

                    proj_ij = (xij + 0) * vx + (yij + 0) * vy + (zij + 0) * vz
                    proj_ik = (0 + xik) * vx + (0 + yik) * vy + (0 + zik) * vz
                    pair = f'{species_list[j]}_{species_list[k]}'
                    V3b[pair] = sum_G3b00ss(V3b[pair], frc_3b, proj_ij, proj_ik, rij, rik, rjk, G3b_eta)

    V2b_ = list(np.array(list(V2b.values())).flatten())
    return V2b_.extend(list(np.array(list(V3b.values())).flatten()))
