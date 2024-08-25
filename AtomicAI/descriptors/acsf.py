"""
    Atom-Centered Symmetry Functions only G2 function[1]

    Notice that the order of pair in descriptors follows the order of defined species

    [1] Jörg Behler, "Atom-centered symmetry functions for constructing high-dimensional
        neural network potentials", The Journal of Chemical
        Physics, 134, 074106 (2011), https://doi.org/10.1063/1.3553717
"""
"""
In current version, it is G2 when params_g4 (and params_g5) is None.
inp = list(zip(system, positions)) 
this is a package of system and positions, which is a pair of ase.Atom (snapshot) and selected atom ID.
This kind of package, can be useful in further parallel calculation
"""

import math
import random
from math import cos, sqrt, exp, pi

import numpy as np
from ase.atoms import Atoms
from dscribe.utils.species import symbols_to_numbers
from numba import jit


class ACSF:
    def __init__(
            self,
            cutoff_descriptor: float,
            params_g2=None,
            params_g3=None,
            params_g4=None,
            params_g5=None,
            species=None,
            periodic: bool = True,
    ):
        """
        Args:
            cutoff_descriptor (float): The smooth cutoff value in angstroms. This cutoff
                value is used throughout the calculations for all symmetry
                functions.
            eta (n* np.ndarray): A list of :math:`\eta` for :math:`G^2` functions.
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
        self.params_g2 = params_g2
        self.params_g3 = params_g3
        self.params_g4 = params_g4
        self.params_g5 = params_g5
        self.rcut = cutoff_descriptor
        self.pbc = periodic

        local_descriptor_lst = {'get_g2': get_g2, 'get_g3': get_g3, 'get_g4': get_g4, 'get_g5': get_g5}
        self.nfp_g2 = 0
        self.nfp_g3 = 0
        self.nfp_g4 = 0
        self.nfp_g5 = 0
        flag_params = False
        if params_g2 is not None:
            self.nfp_g2 = int(len(species) * len(params_g2))
            self.local_descriptor_g2 = local_descriptor_lst.get('get_g2')
            flag_params = True
        if params_g3 is not None:
            self.nfp_g3 = int(len(species) * len(params_g3))
            self.local_descriptor_g3 = local_descriptor_lst.get('get_g3')
            flag_params = True
        if params_g4 is not None:
            self.nfp_g4 = int(len(params_g4) * len(species) * (len(species) + 1) / 2)
            self.local_descriptor_g4 = local_descriptor_lst.get('get_g4')
            flag_params = True
        if params_g5 is not None:
            self.nfp_g5 = int(len(params_g5) * len(species) * (len(species) + 1) / 2)
            self.local_descriptor_g5 = local_descriptor_lst.get('get_g5')
            flag_params = True
        if flag_params:
            pass
            #print('Setting Parameters (two-body, three-body) for descriptor')
        else:
            print('Parameters (two-body, three-body) for descriptor are needed!')
            exit()
        self.nfp = self.nfp_g2 + self.nfp_g3 + self.nfp_g4 + self.nfp_g5

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

            fingerprint_vector_ary = np.zeros((len(target_positions), self.nfp))

            for i in range(len(target_positions)):
                atom_i = target_positions[i]
                xij_array = x_list - x_list[atom_i]
                yij_array = y_list - y_list[atom_i]
                zij_array = z_list - z_list[atom_i]
                startid, endid = 0, 0
                if self.nfp_g2:
                    startid = endid
                    endid = endid + self.nfp_g2
                    fingerprint_vector_ary[i, startid: endid] = self.local_descriptor_g2(
                        xij_array, yij_array, zij_array,
                        atomic_type_list,
                        lattice_a, lattice_b, lattice_c,
                        self.params_g2,
                        cutoff_descriptor=self.rcut, n_species=len(self.species)
                    )
                if self.nfp_g3:
                    startid = endid
                    endid = endid + self.nfp_g3
                    fingerprint_vector_ary[i, startid: endid] = self.local_descriptor_g3(
                        xij_array, yij_array, zij_array,
                        atomic_type_list,
                        lattice_a, lattice_b, lattice_c,
                        self.params_g3,
                        cutoff_descriptor=self.rcut, n_species=len(self.species)
                    )
                if self.nfp_g4 or self.nfp_g5:
                    m_x, m_y, m_z = np.zeros(1), np.zeros(1), np.zeros(1)

                    if self.pbc:
                        m_x, m_y, m_z = calculate_mirror_cubic(local_positions[atom_i], cell, Rc=self.rcut)

                    if self.nfp_g4:
                        startid = endid
                        endid = endid + self.nfp_g4
                        fingerprint_vector_ary[i, startid: endid] = self.local_descriptor_g4(
                            xij_array, yij_array, zij_array,
                            atomic_type_list,
                            m_x, m_y, m_z,
                            self.params_g4,
                            cutoff_descriptor=self.rcut, n_species=len(self.species)
                        )
                    if self.nfp_g5:
                        startid = endid
                        endid = endid + self.nfp_g5
                        fingerprint_vector_ary[i, startid: endid] = self.local_descriptor_g5(
                            xij_array, yij_array, zij_array,
                            atomic_type_list,
                            m_x, m_y, m_z,
                            self.params_g5,
                            cutoff_descriptor=self.rcut, n_species=len(self.species)
                        )

            fingerprint_all_ary.extend(list(fingerprint_vector_ary))
        fingerprint_all_ary = np.array(fingerprint_all_ary)
        return fingerprint_all_ary


@jit
def get_g2(
        xij: np.float64, yij: np.float64, zij: np.float64,
        atomic_type_list: np.int,
        lattice_a: np.float64, lattice_b: np.float64, lattice_c: np.float64,
        params_2b_list: np.float64,
        cutoff_descriptor: float,
        n_species: int,
):
    """
        Calculate the fingerprint vector for a single atom i
    :param n_species:
    :param params_2b_list:
    :param xij:
    :param yij:
    :param zij:
    :param atomic_type_list:
    :param lattice_a:
    :param lattice_b:
    :param lattice_c:
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
def get_g3(
        xij: np.float64, yij: np.float64, zij: np.float64,
        atomic_type_list: np.int,
        lattice_a: np.float64, lattice_b: np.float64, lattice_c: np.float64,
        params_g3_list: np.float64,
        cutoff_descriptor: float,
        n_species: int,
):
    """
        Calculate the fingerprint vector for a single atom i
    :param params_g3_list:
    :param n_species:
    :param xij:
    :param yij:
    :param zij:
    :param atomic_type_list:
    :param lattice_a:
    :param lattice_b:
    :param lattice_c:
    :param cutoff_descriptor:
    :return:
    """

    """
        Calculate the fingerprint vector
        Only for orthorhombic cell
    """

    number_of_params_g3 = len(params_g3_list)
    number_of_atoms = len(xij)
    cutoff_squared = cutoff_descriptor * cutoff_descriptor
    fingerprint_vector = np.zeros(n_species * number_of_params_g3)

    for j in range(number_of_atoms):  # Check all potential neighbors

        dx = 0.5 * lattice_a - ((xij[j] + 1.5 * lattice_a) % lattice_a)
        dy = 0.5 * lattice_b - ((yij[j] + 1.5 * lattice_b) % lattice_b)
        dz = 0.5 * lattice_c - ((zij[j] + 1.5 * lattice_c) % lattice_c)

        # offset of the different type of 2b pair
        params_g3_offset = int(atomic_type_list[j] * number_of_params_g3)

        if max(abs(dx), abs(dy), abs(dz)) > cutoff_descriptor:  # Too far --> skip
            continue

        dx2 = dx * dx
        dy2 = dy * dy
        dz2 = dz * dz
        r2 = dx2 + dy2 + dz2

        if cutoff_squared > r2 > 1e-15:
            r_ij = sqrt(r2)
            for i_g3, kappa in enumerate(params_g3_list):
                cosine_term = cos(kappa * r_ij)
                fcut_term = 0.5 * (cos(pi * r_ij / cutoff_descriptor) + 1)
                addition_term = cosine_term * fcut_term
                fingerprint_vector[params_g3_offset + i_g3] += addition_term

    return fingerprint_vector  # Size = number of eta x 2


@jit
def get_g4(
        xij: np.float64, yij: np.float64, zij: np.float64,
        atomic_type_list: np.int,
        m_x: np.float64, m_y: np.float64, m_z: np.float64,
        params_g4_list: np.float64,
        cutoff_descriptor: float,
        n_species: int,
):
    """
        Calculate the fingerprint vector for a single atom i
    :param n_species:
    :param params_g4_list:
    :param m_z:
    :param m_x:
    :param m_y:
    :param xij:
    :param yij:
    :param zij:
    :param atomic_type_list:
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
def get_g5(
        xij: np.float64, yij: np.float64, zij: np.float64,
        atomic_type_list: np.int,
        m_x: np.float64, m_y: np.float64, m_z: np.float64,
        params_g5_list: np.float64,
        cutoff_descriptor: float,
        n_species: int,
):
    """
        Calculate the fingerprint vector of G5 for a single atom i
    :param n_species:
    :param params_g5_list:
    :param m_z:
    :param m_x:
    :param m_y:
    :param xij:
    :param yij:
    :param zij:
    :param atomic_type_list:
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

    number_of_params_g5 = len(params_g5_list)
    number_of_atoms = len(xij)
    number_of_mirror = len(m_x)
    cutoff_squared = cutoff_descriptor * cutoff_descriptor
    fingerprint_vector = np.zeros(int(n_species * (n_species + 1) * number_of_params_g5 / 2))

    for j in range(number_of_atoms):  # Check all potential neighbors
        for mj in range(number_of_mirror):
            dxij = xij[j] + m_x[mj]
            dyij = yij[j] + m_y[mj]
            dzij = zij[j] + m_z[mj]

            if max(abs(dxij), abs(dyij), abs(dzij)) > cutoff_descriptor:  # Too far --> skip
                continue

            rij2 = dxij * dxij + dyij * dyij + dzij * dzij
            if cutoff_squared > rij2 > 1e-15:
                for k in range(j, number_of_atoms):
                    for mk in range(number_of_mirror):
                        dxik = xij[k] + m_x[mk]
                        dyik = yij[k] + m_y[mk]
                        dzik = zij[k] + m_z[mk]

                        if max(abs(dxik), abs(dyik), abs(dzik)) > cutoff_descriptor:  # Too far --> skip
                            continue

                        rik2 = dxik * dxik + dyik * dyik + dzik * dzik
                        if cutoff_squared > rik2 > 1e-15:
                            dxjk = dxik - dxij
                            dyjk = dyik - dyij
                            dzjk = dzik - dzij
                            rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk

                            # check species order of j and k atom to the order from small to large
                            if atomic_type_list[j] > atomic_type_list[k]:
                                species_j = atomic_type_list[k]
                                species_k = atomic_type_list[j]
                            else:
                                species_j = atomic_type_list[j]
                                species_k = atomic_type_list[k]
                            # offset of the different type of g4 triplet
                            params_offset = int(order_shift_species[int(species_j), int(species_k)]
                                                * number_of_params_g5)

                            r_ij = sqrt(rij2)
                            r_ik = sqrt(rik2)
                            fcut_ij = 0.5 * (cos(pi * r_ij / cutoff_descriptor) + 1)
                            fcut_ik = 0.5 * (cos(pi * r_ik / cutoff_descriptor) + 1)
                            fcut_term = fcut_ij * fcut_ik

                            cos_theta_jik = (rij2 + rik2 - rjk2) / (2 * r_ij * r_ik)
                            r2_sum = rij2 + rik2
                            for i_gx, (eta0, zeta0, lambda0) in enumerate(params_g5_list):
                                cosine_term = pow(2, 1 - zeta0) * (1 + lambda0 * cos_theta_jik) ** zeta0
                                exponent_term = exp(- eta0 * r2_sum)
                                addition_term = cosine_term * exponent_term * fcut_term
                                fingerprint_vector[params_offset + i_gx] += addition_term

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


class ACSF_Force:
    def __init__(
            self,
            rcut=6.0,
            params_v2b=None,
            params_v3b=None,
            species=None,
            periodic: bool = True,
    ):
        """
        Temporary class for force fingerprints.
        # TODO: Merge with ACSF (functions create and derivative?)

        Args:
            cutoff_descriptor (float): The smooth cutoff value in angstroms. This cutoff
                value is used throughout the calculations for all symmetry
                functions.
            eta (n* np.ndarray): A list of :math:`\eta` for :math:`G^2` functions.
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
        self.params_v2b = params_v2b
        self.params_v3b = params_v3b
        self.rcut = rcut
        self.pbc = periodic

        local_descriptor_lst = {
            'get_v2b': get_v2b,
            'get_v3b': get_v3b,
        }
        self.nfp_v2b = 0
        flag_params = False
        if params_v2b is not None:
            self.nfp_v2b = int(len(species) * len(params_v2b))
            self.local_descriptor_v2b = local_descriptor_lst.get('get_v2b')
            flag_params = True
        if params_v3b is not None:
            self.nfp_v3b = int(len(species) * len(params_v3b))
            self.local_descriptor_v3b = local_descriptor_lst.get('get_v3b')
            flag_params = True
        if flag_params:
            print('Setting Parameters (two-body, three-body) for descriptor')
        else:
            print('Parameters (two-body, three-body) for descriptor are needed!')
            exit()
        self.nfp = self.nfp_v2b

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

            fingerprint_vector_ary = np.zeros((len(target_positions), 3, self.nfp))

            for i in range(len(target_positions)):
                atom_i = target_positions[i]
                xij_array = x_list - x_list[atom_i]
                yij_array = y_list - y_list[atom_i]
                zij_array = z_list - z_list[atom_i]
                startid, endid = 0, 0

                m_x, m_y, m_z = np.zeros(1), np.zeros(1), np.zeros(1)
                if self.pbc:
                    m_x, m_y, m_z = calculate_mirror_cubic(local_positions[atom_i], cell, Rc=self.rcut)

                if self.nfp_v2b:

                    startid = endid
                    endid = endid + self.nfp_v2b
                    fingerprint_vector_ary[i, :, startid: endid] = self.local_descriptor_v2b(
                        xij_array, yij_array, zij_array,
                        atomic_type_list,
                        m_x, m_y, m_z,
                        self.params_v2b,
                        cutoff_descriptor=self.rcut, n_species=len(self.species)
                    )
                if self.params_v3b is not None:
                    if self.nfp_v3b:

                        startid = endid
                        endid = endid + self.nfp_v3b
                        fingerprint_vector_ary[i, :, startid: endid] = self.local_descriptor_v3b(
                            xij_array, yij_array, zij_array,
                            atomic_type_list,
                            m_x, m_y, m_z,
                            self.params_v3b,
                            cutoff_descriptor=self.rcut, n_species=len(self.species)
                        )

            fingerprint_all_ary.extend(list(fingerprint_vector_ary))
        fingerprint_all_ary = np.array(fingerprint_all_ary)
        return fingerprint_all_ary


@jit
def get_v2b(
        xij: np.float64, yij: np.float64, zij: np.float64,
        atomic_type_list: np.int,
        m_x: np.float64, m_y: np.float64, m_z: np.float64,
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
    :param m_x:
    :param m_y:
    :param m_z:
    :param params_2b_list:
    :param cutoff_descriptor:
    :return:
    """

    """
        Calculate the fingerprint vector
        Only for orthorhombic cell
    """

    number_of_params_2b = len(params_2b_list)
    number_of_atoms = len(xij)
    number_of_mirror = len(m_x)
    cutoff_squared = cutoff_descriptor * cutoff_descriptor
    fingerprint_vector = np.zeros((3, n_species * number_of_params_2b))
    vec = np.ones(3)

    for j in range(number_of_atoms):  # Check all potential neighbors
        for mj in range(number_of_mirror):
            dxij = xij[j] + m_x[mj]
            dyij = yij[j] + m_y[mj]
            dzij = zij[j] + m_z[mj]

            # offset of the different type of 2b pair
            params_2b_offset = int(atomic_type_list[j] * number_of_params_2b)

            if max(abs(dxij), abs(dyij), abs(dzij)) > cutoff_descriptor:  # Too far --> skip
                continue

            r2 = dxij * dxij + dyij * dyij + dzij * dzij
            vec_rij = [dxij, dyij, dzij]

            if cutoff_squared > r2 > 1e-15:
                r_ij = sqrt(r2)
                for i_2b, (eta, rs) in enumerate(params_2b_list):
                    exponent_term = (r_ij - rs)
                    exponent_term_square = - eta * exponent_term * exponent_term
                    cosine_term = 0.5 * (cos(pi * r_ij / cutoff_descriptor) + 1)
                    addition_term = exp(exponent_term_square) * cosine_term
                    for di in range(len(vec)):
                        projection_di = vec_rij[di] / r_ij
                        fingerprint_vector[di, params_2b_offset + i_2b] += projection_di * addition_term

    return fingerprint_vector  # Size = number of eta x 2


@jit
def get_v3b(
        xij: np.float64, yij: np.float64, zij: np.float64,
        atomic_type_list: np.int,
        m_x: np.float64, m_y: np.float64, m_z: np.float64,
        params_v3b_list: np.float64,
        cutoff_descriptor: float,
        n_species: int,
):
    """
        Calculate the fingerprint vector for a single atom i
    :param xij:
    :param yij:
    :param zij:
    :param atomic_type_list:
    :param m_x:
    :param m_y:
    :param m_z:
    :param params_v3b_list:
    :param cutoff_descriptor:
    :return:
    """

    """
        Calculate the fingerprint vector
        Only for orthorhombic cell
    """

    number_of_params_v3b = len(params_v3b_list)
    number_of_atoms = len(xij)
    number_of_mirror = len(m_x)
    cutoff_squared = cutoff_descriptor * cutoff_descriptor
    fingerprint_vector = np.zeros((3, n_species * number_of_params_v3b))
    vec = np.ones(3)

    for j in range(number_of_atoms):  # Check all potential neighbors
        for mj in range(number_of_mirror):
            dxij = xij[j] + m_x[mj]
            dyij = yij[j] + m_y[mj]
            dzij = zij[j] + m_z[mj]

            # offset of the different type of 2b pair
            params_v3b_offset = int(atomic_type_list[j] * number_of_params_v3b)

            if max(abs(dxij), abs(dyij), abs(dzij)) > cutoff_descriptor:  # Too far --> skip
                continue

            r2 = dxij * dxij + dyij * dyij + dzij * dzij
            vec_rij = [dxij, dyij, dzij]

            if cutoff_squared > r2 > 1e-15:
                r_ij = sqrt(r2)
                for i_v3b, kappa in enumerate(params_v3b_list):
                    cosine_term = cos(kappa * r_ij)
                    fcut_term = 0.5 * (cos(pi * r_ij / cutoff_descriptor) + 1)
                    addition_term = cosine_term * fcut_term
                    fingerprint_vector[params_v3b_offset + i_v3b] += addition_term

                    for di in range(len(vec)):
                        projection_di = vec_rij[di] / r_ij
                        fingerprint_vector[di, params_v3b_offset + i_v3b] += projection_di * addition_term

    return fingerprint_vector  # Size = number of kappa x 2


def set_eta(
        r0: float = 0.45,
        cutoff_descriptor: float = 5.0,
        number_of_eta: int = 100,
):
    """
        Calculate the decay function [goes from R0 to cutoff]
    :return: list of eta
    """

    eta_vector = np.array([r0])
    multiplier_value = (cutoff_descriptor / r0) ** (1 / number_of_eta)

    for _ in range(number_of_eta - 1):
        eta_vector = np.append(eta_vector, np.array([multiplier_value * eta_vector[-1]]))

    return eta_vector


# make random vector for force projection
def random_vforce(data_num_rand):
    vforce = []

    for _ in range(data_num_rand):
        zr = random.uniform(0.0, 1.0) * 2 - 1
        pr = random.uniform(0.0, 1.0) * 2 * pi

        vx = math.sqrt(1 - zr ** 2) * math.cos(pr)
        vy = math.sqrt(1 - zr ** 2) * math.sin(pr)
        vz = zr

        vforce.append([vx, vy, vz])

    return vforce
