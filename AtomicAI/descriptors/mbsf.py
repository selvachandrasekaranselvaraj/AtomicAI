"""
    Modified Behler-Parrinello symmetry functions (MBSF) [1],
    is a modified version of Behler-Parrinello atom-centered symmetry functions (ACSF)[2].
    Support two-body and three-body fingerprints

    Notice that the order of pair in descriptors follows the order of defined species

    [1] Smith J S, Isayev O and Roitberg A E 2017 Chem. sci. 8 3192
    [2] Jörg Behler, "Atom-centered symmetry functions for constructing high-dimensional
        neural network potentials", The Journal of Chemical
        Physics, 134, 074106 (2011), https://doi.org/10.1063/1.3553717
"""

import numpy as np
from math import cos, sqrt, exp, pi, acos

from numba import jit
from ase.atoms import Atoms
from dscribe.utils.species import symbols_to_numbers


class MBSF():
    def __init__(
            self,
            cutoff_descriptor,
            params_gr=None,
            params_ga=None,
            species=None,
            periodic=True,
    ):
        """
        Args:
            cutoff_descriptor (float): The smooth cutoff value in angstroms. This cutoff
                value is used throughout the calculations for all symmetry
                functions.
            params_gr (n* (2 *np.ndarray) ): A list of :math:`\eta` and :math:'Rs'
                for :math:`G^R` (two-body) functions.
            params_ga (n* (4 *np.ndarray)): A list of :math:`\zeta`, :math:'\theta_s',
                :math:'\eta' and :math:'Rs' for :math:`G^A` (three-body) functions.
            species (iterable): The chemical species as a list of atomic
                numbers or as a list of chemical symbols. Notice that this is not
                the atomic numbers that are present for an individual system, but
                should contain all the elements that are ever going to be
                encountered when creating the descriptors for a set of systems.
                Keeping the number of chemical species as low as possible is
                preferable.
            periodic (bool): Set to true if you want the descriptor output to
                respect the periodicity of the atomic systems (see the
                pbc-parameter in the constructor of ase.Atoms).
        """
        # Setup
        if not species:
            print('need species information of atoms')
            exit()
        self.species = species
        self.params_gr = params_gr
        self.params_ga = params_ga
        self.rcut = cutoff_descriptor
        self.pbc = periodic

        local_descriptor_lst = {'get_gr': get_gr_nblst, 'get_ga': get_ga_nblst}
        self.nfp_gr = 0
        self.nfp_ga = 0
        if params_gr is not None:
            self.nfp_gr = int(len(species) * len(params_gr))
            self.local_descriptor_gr = local_descriptor_lst.get('get_gr')
            if params_ga is not None:
                self.nfp_ga = int(len(params_ga) * len(species) * (len(species) + 1) / 2)
                self.local_descriptor_ga = local_descriptor_lst.get('get_ga')
        else:
            print('Parameters (two-body, three-body) for descriptor are needed!')
            exit()

    def create(self, system, positions=None):
        """Return the ACSF output for the given systems and given positions.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            positions (list): Positions where to calculate ACSF. Can be
                provided as cartesian positions or atomic indices. If no
                positions are defined, the output will be created for all
                atoms in the system. When calculating for multiple
                systems, provide the positions as a list for each system.
        Returns:
            np.ndarray
        """
        # Validate input / combine input arguments
        if isinstance(system, Atoms):
            system = [system]
        if positions is None:
            inp = [(i_sys, np.arange(len(i_sys.numbers))) for i_sys in system]
        else:
            inp = list(zip(system, positions))

        fingerprint_all_ary = []
        for id_system, (target_atoms, target_positions) in enumerate(inp):
            local_positions = target_atoms.positions

            # convert the list of atomic numbers with the order of species
            numbers = target_atoms.numbers
            numbers_of_species = symbols_to_numbers(self.species)
            atomic_type_list = np.zeros(len(numbers))
            for number_id, number in enumerate(numbers_of_species):
                mask = np.argwhere(numbers == int(number)).flatten()
                atomic_type_list[mask] = int(number_id)

            # Get cell parameters
            # Current available cell system : orthorhombic
            cell = target_atoms.cell
            lattice_a, lattice_b, lattice_c = cell[0, 0], cell[1, 1], cell[2, 2]
            lattice_parameters = [lattice_a, lattice_b, lattice_c]

            # Get x, y, z coordination
            x_list = local_positions[:, 0]
            y_list = local_positions[:, 1]
            z_list = local_positions[:, 2]

            fingerprint_vector_ary = np.zeros((len(target_positions), (self.nfp_gr + self.nfp_ga)))

            for i in range(len(target_positions)):
                atom_i = target_positions[i]
                xij_array = x_list - x_list[atom_i]
                yij_array = y_list - y_list[atom_i]
                zij_array = z_list - z_list[atom_i]

                m_x, m_y, m_z = np.zeros(1), np.zeros(1), np.zeros(1)

                if self.pbc:
                    m_x, m_y, m_z = calculate_mirror_cubic(local_positions[atom_i], cell, Rc=self.rcut)

                nb_id, nb_shift, nb_r2 = get_neighbour_lst(xij_array, yij_array, zij_array, m_x, m_y, m_z,
                                                           radius_cutoff=self.rcut)
                if self.nfp_gr:
                    startid, endid = 0, self.nfp_gr
                    fingerprint_vector_ary[i, startid: endid] = self.local_descriptor_gr(
                        xij_array, yij_array, zij_array,
                        atomic_type_list,
                        m_x, m_y, m_z,
                        params_gr_list=self.params_gr,
                        cutoff_descriptor=self.rcut,
                        n_species=len(self.species),
                        nb_index=nb_id,
                        nb_index_shift=nb_shift,
                        nb_r2_lst=nb_r2,
                    )

                if self.nfp_ga:
                    startid, endid = self.nfp_gr, self.nfp_gr + self.nfp_ga
                    fingerprint_vector_ary[i, startid: endid] = self.local_descriptor_ga(
                        xij_array, yij_array, zij_array,
                        atomic_type_list,
                        m_x, m_y, m_z,
                        params_ga_list=self.params_ga,
                        cutoff_descriptor=self.rcut,
                        n_species=len(self.species),
                        nb_index=nb_id,
                        nb_index_shift=nb_shift,
                        nb_r2_lst=nb_r2,
                    )

            fingerprint_all_ary.extend(list(fingerprint_vector_ary))
        fingerprint_all_ary = np.array(fingerprint_all_ary)
        return fingerprint_all_ary


@jit
def get_neighbour_lst(xij: np.float64, yij: np.float64, zij: np.float64,
                      m_x: np.float64, m_y: np.float64, m_z: np.float64,
                      radius_cutoff: float, ):
    index_lst, shift_index_lst, nb_r2_lst = [], [], []
    number_of_atoms = len(xij)
    number_of_mirror = len(m_x)
    radius_cutoff_square = radius_cutoff * radius_cutoff
    for j in range(number_of_atoms):  # Check all potential neighbors
        for mj in range(number_of_mirror):
            dxij = xij[j] + m_x[mj]
            dyij = yij[j] + m_y[mj]
            dzij = zij[j] + m_z[mj]

            if max(abs(dxij), abs(dyij), abs(dzij)) > radius_cutoff:  # Too far --> skip
                continue

            rij2 = dxij * dxij + dyij * dyij + dzij * dzij
            if radius_cutoff_square > rij2 > 1e-15:
                index_lst.append(j)
                shift_index_lst.append(mj)
                nb_r2_lst.append(rij2)

    return np.array(index_lst), np.array(shift_index_lst), np.array(nb_r2_lst)


@jit
def get_gr_nblst(
        xij: np.float64, yij: np.float64, zij: np.float64,
        atomic_type_list: np.int,
        m_x: np.float64, m_y: np.float64, m_z: np.float64,
        params_gr_list: np.float64,
        cutoff_descriptor: float,
        n_species: int,
        nb_index: np.int,
        nb_index_shift: np.int,
        nb_r2_lst: np.float64,
):
    """
        Calculate the fingerprint vector for a single atom i
    :param xij:
    :param yij:
    :param zij:
    :param atomic_type_list:
    :param lattice_a:
    :param lattice_b:
    :param lattice_c:
    :param eta_all_list:
    :param atom_type:
    :param cutoff_descriptor:
    :return:
    """

    """
        Calculate the fingerprint vector
        Only for orthorhombic cell
    """

    number_of_params_gr = len(params_gr_list)
    fingerprint_vector = np.zeros(n_species * number_of_params_gr)

    for index_j, (j, mj, rij2) in enumerate(zip(nb_index, nb_index_shift, nb_r2_lst)):
        # offset of the different type of 2b pair
        params_gr_offset = int(atomic_type_list[j] * number_of_params_gr)

        r_ij = sqrt(rij2)
        for i_2b, (eta, rs) in enumerate(params_gr_list):
            exponent_term = (r_ij - rs)
            exponent_term_square = - eta * exponent_term * exponent_term
            cosine_term = 0.5 * (cos(pi * r_ij / cutoff_descriptor) + 1)
            addition_term = exp(exponent_term_square) * cosine_term
            fingerprint_vector[params_gr_offset + i_2b] += addition_term

    return fingerprint_vector  # Size = number of eta x 2


@jit
def get_gr(
        xij: np.float64, yij: np.float64, zij: np.float64,
        atomic_type_list: np.int,
        lattice_a: np.float64, lattice_b: np.float64, lattice_c: np.float64,
        params_2b_list: np.float64,
        cutoff_descriptor: float,
        n_species: int,
):
    """
        Calculate the fingerprint vector for a single atom i
    :param xij:
    :param yij:
    :param zij:
    :param atomic_type_list:
    :param lattice_a:
    :param lattice_b:
    :param lattice_c:
    :param eta_all_list:
    :param atom_type:
    :param cutoff_descriptor:
    :return:
    """

    """
        Calculate the fingerprint vector
        Only for orthorhombic cell
    """

    number_of_params_2b = len(params_2b_list)
    number_of_atoms = len(xij)
    cutoff_squared = cutoff_descriptor * cutoff_descriptor
    fingerprint_vector = np.zeros(n_species * number_of_params_2b)

    for j in range(number_of_atoms):  # Check all potential neighbors

        dx = 0.5 * lattice_a - ((xij[j] + 1.5 * lattice_a) % lattice_a)
        dy = 0.5 * lattice_b - ((yij[j] + 1.5 * lattice_b) % lattice_b)
        dz = 0.5 * lattice_c - ((zij[j] + 1.5 * lattice_c) % lattice_c)

        # offset of the different type of 2b pair
        params_2b_offset = int(atomic_type_list[j] * number_of_params_2b)

        if max(abs(dx), abs(dy), abs(dz)) > cutoff_descriptor:  # Too far --> skip
            continue

        dx2 = dx * dx
        dy2 = dy * dy
        dz2 = dz * dz
        r2 = dx2 + dy2 + dz2

        if cutoff_squared > r2 > 1e-15:
            r_ij = sqrt(r2)
            for i_2b, (eta, rs) in enumerate(params_2b_list):
                exponent_term = (r_ij - rs)
                exponent_term_square = - eta * exponent_term * exponent_term
                cosine_term = 0.5 * (cos(pi * r_ij / cutoff_descriptor) + 1)
                addition_term = exp(exponent_term_square) * cosine_term
                fingerprint_vector[params_2b_offset + i_2b] = fingerprint_vector[
                                                                  params_2b_offset + i_2b] + addition_term

    return fingerprint_vector  # Size = number of eta x 2


@jit
def get_grga(
        xij: np.float64, yij: np.float64, zij: np.float64,
        atomic_type_list: np.int,
        m_x: np.float64, m_y: np.float64, m_z: np.float64,
        params_g4_list: np.float64,
        cutoff_descriptor: float,
        n_species: int,
):
    """
        Calculate the fingerprint vector for a single atom i
    :param xij:
    :param yij:
    :param zij:
    :param atomic_type_list:
    :param lattice_a:
    :param lattice_b:
    :param lattice_c:
    :param eta_all_list:
    :param atom_type:
    :param cutoff_descriptor:
    :return:
    """

    """
        Calculate the fingerprint vector
        Only for orthorhombic cell
    """
    order_shift_species = np.zeros((n_species, n_species))
    order_shift = 0
    for ni in range(n_species):
        for nj in range(ni, n_species):
            order_shift_species[ni, nj] = order_shift
            order_shift = order_shift + 1

    number_of_params_g4 = len(params_g4_list)
    number_of_atoms = len(xij)
    number_of_mirror = len(m_x)
    cutoff_squared = cutoff_descriptor * cutoff_descriptor
    fingerprint_vector = np.zeros(int(n_species * (n_species + 1) * number_of_params_g4 / 2))

    for j in range(number_of_atoms):  # Check all potential neighbors
        for mj in range(number_of_mirror):
            dxij = xij[j] + m_x[mj]
            dyij = yij[j] + m_y[mj]
            dzij = zij[j] + m_z[mj]

            if max(abs(dxij), abs(dyij), abs(dzij)) > cutoff_descriptor:  # Too far --> skip
                continue

            rij2 = dxij * dxij + dyij * dyij + dzij * dzij
            rij_flag = False
            if cutoff_squared > rij2 > 1e-15:
                rij_flag = True

                for k in range(j, number_of_atoms):
                    for mk in range(number_of_mirror):
                        dxik = xij[k] + m_x[mk]
                        dyik = yij[k] + m_y[mk]
                        dzik = zij[k] + m_z[mk]

                        if max(abs(dxik), abs(dyik), abs(dzik)) > cutoff_descriptor:  # Too far --> skip
                            continue

                        rik_flag = False
                        rik2 = dxik * dxik + dyik * dyik + dzik * dzik
                        if cutoff_squared > rik2 > 1e-15: rik_flag = True

                        dxjk = dxik - dxij
                        dyjk = dyik - dyij
                        dzjk = dzik - dzij

                        if max(abs(dxjk), abs(dyjk), abs(dzjk)) > cutoff_descriptor:  # Too far --> skip
                            continue

                        rjk_flag = False
                        rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk
                        if cutoff_squared > rjk2 > 1e-15: rjk_flag = True

                        if rij_flag * rik_flag * rjk_flag:
                            # check species order of j and k atom to the order from small to large
                            if atomic_type_list[j] > atomic_type_list[k]:
                                species_j = atomic_type_list[k]
                                species_k = atomic_type_list[j]
                            else:
                                species_j = atomic_type_list[j]
                                species_k = atomic_type_list[k]
                            # offset of the different type of g4 triplet
                            params_g4_offset = int(order_shift_species[int(species_j), int(species_k)]
                                                   * number_of_params_g4)

                            r_ij = sqrt(rij2)
                            r_ik = sqrt(rik2)
                            r_jk = sqrt(rjk2)
                            fcut_ij = 0.5 * (cos(pi * r_ij / cutoff_descriptor) + 1)
                            fcut_ik = 0.5 * (cos(pi * r_ik / cutoff_descriptor) + 1)
                            fcut_jk = 0.5 * (cos(pi * r_jk / cutoff_descriptor) + 1)
                            fcut_term = fcut_ij * fcut_ik * fcut_jk

                            cos_theta_jik = (rij2 + rik2 - rjk2) / (2 * r_ij * r_ik)
                            r2_all = rij2 + rik2 + rjk2
                            for i_g4, (eta0, zeta0, lambda0) in enumerate(params_g4_list):
                                cosine_term = pow(2, 1 - zeta0) * (1 + lambda0 * cos_theta_jik) ** zeta0
                                exponent_term = exp(- eta0 * r2_all)
                                addition_term = cosine_term * exponent_term * fcut_term
                                fingerprint_vector[params_g4_offset + i_g4] += addition_term

    return fingerprint_vector  # Size = number_of_params_g4 * n_species * (n_species +1)/2


@jit
def get_ga_nblst(
        xij: np.float64, yij: np.float64, zij: np.float64,
        atomic_type_list: np.int,
        m_x: np.float64, m_y: np.float64, m_z: np.float64,
        params_ga_list: np.float64,
        cutoff_descriptor: float,
        n_species: int,
        nb_index: np.int,
        nb_index_shift: np.int,
        nb_r2_lst: np.float64,
):
    """
        Calculate the fingerprint vector for a single atom i
    :param xij:
    :param yij:
    :param zij:
    :param atomic_type_list:
    :param lattice_a:
    :param lattice_b:
    :param lattice_c:
    :param eta_all_list:
    :param atom_type:
    :param cutoff_descriptor:
    :return:
    """

    """
        Calculate the fingerprint vector
        Only for orthorhombic cell
    """
    order_shift_species = np.zeros((n_species, n_species))
    order_shift = 0
    for ni in range(n_species):
        for nj in range(ni, n_species):
            order_shift_species[ni, nj] = order_shift
            order_shift = order_shift + 1

    number_of_params_ga = len(params_ga_list)
    cutoff_squared = cutoff_descriptor * cutoff_descriptor
    fingerprint_vector = np.zeros(int(n_species * (n_species + 1) * number_of_params_ga / 2))

    number_of_nb = len(nb_index)
    for nb_j in range(number_of_nb):
        j, mj, rij2 = nb_index[nb_j], nb_index_shift[nb_j], nb_r2_lst[nb_j]
        dxij = xij[j] + m_x[mj]
        dyij = yij[j] + m_y[mj]
        dzij = zij[j] + m_z[mj]
        r_ij = sqrt(rij2)

        for nb_k in range(nb_j, number_of_nb):
            k, mk, rik2 = nb_index[nb_k], nb_index_shift[nb_k], nb_r2_lst[nb_k]
            dxik = xij[k] + m_x[mk]
            dyik = yij[k] + m_y[mk]
            dzik = zij[k] + m_z[mk]
            r_ik = sqrt(rik2)

            dxjk = dxik - dxij
            dyjk = dyik - dyij
            dzjk = dzik - dzij

            if max(abs(dxjk), abs(dyjk), abs(dzjk)) > cutoff_descriptor:  # Too far --> skip
                continue

            rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk

            if cutoff_squared > rjk2 > 1e-15:
                # check species order of j and k atom to the order from small to large
                if atomic_type_list[j] > atomic_type_list[k]:
                    species_j = atomic_type_list[k]
                    species_k = atomic_type_list[j]
                else:
                    species_j = atomic_type_list[j]
                    species_k = atomic_type_list[k]
                # offset of the different type of g4 triplet
                params_ga_offset = int(order_shift_species[int(species_j), int(species_k)]
                                       * number_of_params_ga)

                # r_jk = sqrt(rjk2)
                fcut_ij = 0.5 * (cos(pi * r_ij / cutoff_descriptor) + 1)
                fcut_ik = 0.5 * (cos(pi * r_ik / cutoff_descriptor) + 1)
                # fcut_jk = 0.5 * (cos(pi * r_jk / cutoff_descriptor) + 1)
                # fcut_term = fcut_ij * fcut_ik * fcut_jk
                fcut_term = fcut_ij * fcut_ik

                cos_theta_jik = (rij2 + rik2 - rjk2) / (2 * r_ij * r_ik)
                theta_jik = acos(cos_theta_jik)
                # r2_all = rij2 + rik2 + rjk2
                r_ave = 0.5 * (r_ij + r_ik)
                for i_ga, (zeta0, theta_s, eta0, rs) in enumerate(params_ga_list):
                    cosine_term = pow(2, 1 - zeta0) * (1 + cos(theta_jik - theta_s)) ** zeta0
                    exponent_term = exp(- eta0 * (r_ave - rs) ** 2)
                    addition_term = cosine_term * exponent_term * fcut_term
                    fingerprint_vector[params_ga_offset + i_ga] += addition_term

    return fingerprint_vector  # Size = number_of_params_g4 * n_species * (n_species +1)/2


# @jit
def calculate_mirror_cubic(position, cell, Rc):
    m_min = [0, 0, 0]
    m_max = [0, 0, 0]
    for i in range(len(position)):
        if position[i] < Rc: m_min[i] = -1
        if (cell[i, i] - position[i]) < Rc: m_max[i] = 1
    m_x, m_y, m_z = [], [], []
    for i in range(m_min[0], m_max[0] + 1):
        for j in range(m_min[1], m_max[1] + 1):
            for k in range(m_min[2], m_max[2] + 1):
                m_x.append(i * cell[0, 0])
                m_y.append(j * cell[1, 1])
                m_z.append(k * cell[2, 2])
    return np.array(m_x), np.array(m_y), np.array(m_z)
