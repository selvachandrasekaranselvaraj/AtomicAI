"""
    Locally averaged atomic fingerprints (LAAF)
    Only support two-body fingerprints
"""
import os
import random
from math import cos, sqrt

import ase.io
import numpy as np
from numpy import exp
from numba import jit
#from tqdm.auto import tqdm
from AtomicAI.descriptors.acsf import ACSF
from AtomicAI.descriptors.mbsf import MBSF
from dscribe.descriptors import SOAP
import warnings 
warnings.filterwarnings("ignore")

def calculate_eta(r0: float = 0.45, cutoff_descriptor: float = 5.0, number_of_eta: int = 100) -> list:
    """
        Calculate the decay function [goes from R0 to cutoff]
    :return: list of eta
    """

    eta_vector = np.array([r0])
    multiplier_value = (cutoff_descriptor / r0) ** (1 / number_of_eta)

    for _ in range(number_of_eta - 1):
        eta_vector = np.append(eta_vector, np.array([multiplier_value * eta_vector[-1]]))

    return eta_vector


@jit
def calculate_fingerprint_vector(
        x_ij: np.float64, y_ij: np.float64, z_ij: np.float64,
        atomic_type_list: np.int,
        lattice_a: np.float64, lattice_b: np.float64, lattice_c: np.float64,
        # fingerprint_center_atom_i: int,
        eta_all_list: np.float64,
        atom_type: int,
        cutoff_descriptor: float,
):
    # TODO: Should go to analysis.descriptor module
    """
        Calculate the fingerprint vector for a single atom i.
        Original implementation of Botu descriptor.
    :param x_ij:
    :param y_ij:
    :param z_ij:
    :param atomic_type_list:
    :param lattice_a:
    :param lattice_b:
    :param lattice_c:
    :param eta_all_list:
    :param atom_type:
    :param cutoff_descriptor:
    :return:
    """

    number_of_eta = len(eta_all_list)
    number_of_atoms = len(x_ij)

    cutoff_squared = cutoff_descriptor * cutoff_descriptor

    fingerprint_vector = np.zeros(2 * number_of_eta)

    for j in range(number_of_atoms):  # Check all potential neighbors

        dx = 0.5 * lattice_a - ((x_ij[j] + 1.5 * lattice_a) % lattice_a)
        dy = 0.5 * lattice_b - ((y_ij[j] + 1.5 * lattice_b) % lattice_b)
        dz = 0.5 * lattice_c - ((z_ij[j] + 1.5 * lattice_c) % lattice_c)

        if atomic_type_list[j] == atom_type:  # neighbor is same type
            eta_offset = 0
        else:  # neighbor is other type
            eta_offset = number_of_eta

        if max(abs(dx), abs(dy), abs(dz)) > cutoff_descriptor:  # Too far --> skip
            continue

        dx2 = dx * dx
        dy2 = dy * dy
        dz2 = dz * dz

        if cutoff_squared > dx2 + dy2 + dz2 > 1e-15:
            r_ij = sqrt(dx2 + dy2 + dz2)
            exponent_term = -(r_ij / eta_all_list) * (r_ij / eta_all_list)
            cosine_term = 0.5 * (cos(np.pi * r_ij / cutoff_descriptor) + 1)
            addition_term = exp(exponent_term) * cosine_term
            fingerprint_vector[eta_offset:eta_offset + number_of_eta] += addition_term

    return fingerprint_vector  # Size = number of eta x 2


def calculate_average_fingerprint(
        cutoff_descriptor: float = 5,
        cutoff_average: float = 4,
        #input_file: str = "./trajectory.xyz",
        traj_data: str = "traj_data",
        selected_snapshots: str = ':',  # Format of ase for selecting the snapshots in trajectories
        output_file: str = "average_fingerprint.csv",
        append: bool = False,
        target_element: int = 0,
        number_of_data: int = None,  # generated number of data, # TODO: Confirm behaviour if value is None
        number_of_eta: int = 100,
        start_step: int = 0,  # first step to generate data
        final_step: int = None,  # final step to generate data
        element_conversion: dict = None,
        seed: int = None,
        lattice_parameters=None,
        local_descriptor_type=None,
        descriptor=None,
        use_buffer: bool = False,
        descriptor_parameters=None,
        target_neighbor_element: int = None,
):
    # TODO: Replace my AverageFingerprintCalculator for more flexible usage
    """
        Calculate averaged fingerprints for randomly selected center atoms.
        Center atomic sites are selected by target_element, neighbor atomic sites are selected by
        target_neighbor_element.
        REMARK: Descriptor order is different between original and acsf g2:
        original is 0-0/0-1 while acsf g2 is 0-1/0-0
    :param target_neighbor_element:
    :param descriptor_parameters: specific parameters for descriptors. For advanced use!
    :param selected_snapshots: str compliant with ase.io.read function
    :param descriptor: instance of descriptor # TODO: List all available descriptors
    :param local_descriptor_type: name of descriptor # TODO: Can it be inferred from "descriptor"?
    :param append: If True, will append at the end of the csv file. Else overwriting it.
    :param lattice_parameters: For normal xyz files
    :param cutoff_descriptor:
    :param cutoff_average:
    :param input_file:
    :param traj_data:
    :param output_file:
    :param target_element:
    :param number_of_data:
    :param number_of_eta:
    :param start_step: 0 = first selected snapshot
    :param final_step:
    :param element_conversion:
    :param use_buffer: store calculated fingerprint vectors for later random choice
    :param seed:
    :return:
    """

    if descriptor_parameters is None:
        descriptor_parameters = {}
    if target_neighbor_element is None:
        target_neighbor_element = target_element  # Default: Centers and neighbors are of the same type

    fingerprint_buffer = {}

    if element_conversion is None:
        element_conversion = {
            'Si': 0,
            'O': 1,
        }

    # Read input file with ASE and select snapshots
    atoms = traj_data #ase.io.read(input_file, selected_snapshots)

    if final_step is None:  # Default is that final step is last step
        final_step = len(atoms) - 1

    assert final_step < len(atoms), \
        f'Final step exceeds available number of snapshots. Stopping now! ({final_step} >= {len(atoms)})'

    eta_list = calculate_eta(cutoff_descriptor=cutoff_descriptor, number_of_eta=number_of_eta)

    # Initialize the descriptor
    # Available local descriptors
    local_descriptor_dict = {
        'ACSF_G2': ACSF,
        'ACSF_G4': ACSF,
        'ACSF_G2G4': ACSF,
        'SOAP': SOAP,
        'MBSF': MBSF,
    }
    species = list(element_conversion.keys())

    local_descriptor = None
    if descriptor:
        local_descriptor = descriptor
    elif not local_descriptor_type:
        print('Using default local descriptor')
    elif local_descriptor_type == 'ACSF_G2':
        # Set G2 parameters
        # convert current $\eta$ to the $\eta$ in G2 # TODO: What does it mean?
        rs_lst = [0.0]
        params_g2 = []
        for eta in eta_list:
            for rs in rs_lst:
                params_g2.append([1.0 / (eta * eta), rs])

        params_g2 = np.array(params_g2)

        descriptor_parameter_dict = {
            'cutoff_descriptor': cutoff_descriptor,
            'params_g2': params_g2,
            'species': species,
        }
        descriptor_parameter_dict.update(descriptor_parameters)  # Use user input, if any

        local_descriptor = local_descriptor_dict[local_descriptor_type](**descriptor_parameter_dict)

    elif local_descriptor_type == 'ACSF_G2G4':
        # TODO: set parameter defaults vs descriptor is given as input. Think about it.
        # Set parameters
        # G2
        rs_lst = [0.0]
        params_g2 = []
        for eta in eta_list:
            for rs in rs_lst:
                params_g2.append([1.0 / (eta * eta), rs])

        params_g2 = np.array(params_g2)
        # G4
        number_of_eta_g4 = 10
        eta_g4_list = calculate_eta(cutoff_descriptor=cutoff_descriptor, number_of_eta=number_of_eta_g4)
        zeta_lst = [1, 2, 4, 16]
        lambda_list = [-1, 1]
        params_g4 = []
        for eta0 in eta_g4_list:
            for zeta0 in zeta_lst:
                for lambda0 in lambda_list:
                    params_g4.append([1.0 / (eta0 * eta0), zeta0, lambda0])
        params_g4 = np.array(params_g4)

        descriptor_parameter_dict = {
            'cutoff_descriptor': cutoff_descriptor,
            'params_g2': params_g2,
            'params_g4': params_g4,
            'species': species,
        }
        descriptor_parameter_dict.update(descriptor_parameters)  # Use user input, if any

        local_descriptor = local_descriptor_dict[local_descriptor_type](**descriptor_parameter_dict)

    elif local_descriptor_type == 'ACSF_G4':
        # Set parameters
        # G2
        params_g2 = None
        # G4
        number_of_eta_g4 = 10
        eta_g4_list = calculate_eta(cutoff_descriptor=cutoff_descriptor, number_of_eta=number_of_eta_g4)
        zeta_lst = [1, 2, 4, 16]
        lambda_list = [-1, 1]
        params_g4 = []
        for eta0 in eta_g4_list:
            for zeta0 in zeta_lst:
                for lambda0 in lambda_list:
                    params_g4.append([1.0 / (eta0 * eta0), zeta0, lambda0])
        params_g4 = np.array(params_g4)

        descriptor_parameter_dict = {
            'cutoff_descriptor': cutoff_descriptor,
            'params_g2': params_g2,
            'params_g4': params_g4,
            'species': species,
        }
        descriptor_parameter_dict.update(descriptor_parameters)  # Use user input, if any

        local_descriptor = local_descriptor_dict[local_descriptor_type](**descriptor_parameter_dict)

    elif local_descriptor_type == 'MBSF':
        # set parameter for MBSF gr(g2) + ga (\zeta, \theta_s, \eta, Rs)
        rs_lst = [0.0]
        params_gr = []
        for eta in eta_list:
            for rs in rs_lst:
                params_gr.append([1.0 / (eta * eta), rs])

        params_gr = np.array(params_gr)
        # G4
        zeta_lst = [1, 2, 4, 16]
        theta_s_list = np.linspace(0, 1, 5) * np.pi
        number_of_eta_ga = 10
        eta_ga_list = calculate_eta(
            cutoff_descriptor=cutoff_descriptor,
            number_of_eta=number_of_eta_ga,
        )
        rs_lst = [0.0]
        params_ga = []
        for zeta0 in zeta_lst:
            for theta_s0 in theta_s_list:
                for eta0 in eta_ga_list:
                    for rs0 in rs_lst:
                        params_ga.append([zeta0, theta_s0, 1.0 / (eta0 * eta0), rs0])
        params_ga = np.array(params_ga)

        descriptor_parameter_dict = {
            'cutoff_descriptor': cutoff_descriptor,
            'params_gr': params_gr,
            'params_ga': params_ga,
            'periodic': True,
            'species': species,
        }
        descriptor_parameter_dict.update(descriptor_parameters)  # Use user input, if any

        local_descriptor = local_descriptor_dict[local_descriptor_type](**descriptor_parameter_dict)

    elif local_descriptor_type == 'SOAP':

        descriptor_parameter_dict = {
            'species': species,
            'periodic': True,
            'rcut': cutoff_descriptor,
            'nmax': 8,
            'lmax': 8,
        }
        descriptor_parameter_dict.update(descriptor_parameters)  # Use user input, if any

        local_descriptor = local_descriptor_dict[local_descriptor_type](**descriptor_parameter_dict)

    else:
        print('Unknown local_descriptor_type:', local_descriptor_type)
        print('local_descriptor_list:', local_descriptor_type)
        exit()

    # If seed is defined to fix the order, mainly for DEBUG
    if seed:
        print(f'WARNING: As a seed was set ({seed}), this is a deterministic calculation.')
        random.seed(seed)

    result_str_list = []

    # Generate all data points
    for data_index in range(number_of_data):
    #for data_index in tqdm(range(number_of_data), desc=f'LAAF of {os.path.basename(input_file)}'):
        # TODO: Speedup by storing calculated fingerprint vectors as they are created, by snapshot number.
        # TODO: Like a cache, if already calculated --> Reuse. Else calculate the fingerprint vector
        # Select step
        selected_step = random.randint(start_step, final_step)
        selected_snapshot = atoms[selected_step]

        # Get atomic type list
        symbols = selected_snapshot.symbols
        number_of_atoms = len(symbols)
        atomic_type_list = [element_conversion[symbol] for symbol in symbols]
        atomic_type_list = np.array(atomic_type_list)

        # Get cell parameters
        # Current available cell system : orthorhombic
        if lattice_parameters is None:
            cell = selected_snapshot.cell
            lattice_a, lattice_b, lattice_c = cell[0, 0], cell[1, 1], cell[2, 2]
            lattice_parameters = [lattice_a, lattice_b, lattice_c]
        else:
            lattice_a, lattice_b, lattice_c = lattice_parameters

        # Get x, y, z coordination
        positions = selected_snapshot.positions
        x_list = positions[:, 0]
        y_list = positions[:, 1]
        z_list = positions[:, 2]

        if use_buffer:
            if selected_step in fingerprint_buffer.keys():
                fingerprint_vector_list = fingerprint_buffer[selected_step]

            elif local_descriptor:
                fingerprint_vector_list = local_descriptor.create(
                    selected_snapshot)  # TODO: call create without system?
            else:
                # Original Botu atomic fingerprint
                fingerprint_vector_list = np.zeros((number_of_atoms, 2 * number_of_eta))
                eta_list = np.array(eta_list)

                for atom_i in range(number_of_atoms):
                    x_ij_array = x_list - x_list[atom_i]
                    y_ij_array = y_list - y_list[atom_i]
                    z_ij_array = z_list - z_list[atom_i]
                    fingerprint_vector_list[atom_i] = calculate_fingerprint_vector(
                        x_ij_array, y_ij_array, z_ij_array,
                        atomic_type_list,
                        lattice_a, lattice_b, lattice_c,
                        eta_list, atomic_type_list[atom_i],
                        cutoff_descriptor=cutoff_descriptor,
                    )
        else:

            if local_descriptor:
                fingerprint_vector_list = local_descriptor.create(selected_snapshot)
            else:
                # Original Botu atomic fingerprint
                fingerprint_vector_list = np.zeros((number_of_atoms, 2 * number_of_eta))
                eta_list = np.array(eta_list)

                for atom_i in range(number_of_atoms):
                    x_ij_array = x_list - x_list[atom_i]
                    y_ij_array = y_list - y_list[atom_i]
                    z_ij_array = z_list - z_list[atom_i]
                    fingerprint_vector_list[atom_i] = calculate_fingerprint_vector(
                        x_ij_array, y_ij_array, z_ij_array,
                        atomic_type_list,
                        lattice_a, lattice_b, lattice_c,
                        eta_list, atomic_type_list[atom_i],
                        cutoff_descriptor=cutoff_descriptor,
                    )
        if use_buffer:
            fingerprint_buffer[selected_step] = fingerprint_vector_list

        for k in range(number_of_atoms):
            # Select a random atom

            atom_i = random.randint(0, number_of_atoms - 1)

            if atomic_type_list[atom_i] == target_element:
                break

        # single_averaged_fingerprint = np.array([0.0 for k in range(2 * number_of_eta)])
        single_averaged_fingerprint = np.zeros(len(fingerprint_vector_list[atom_i]))
        number_of_fingerprints_in_average = 0

        for atom_j in range(number_of_atoms):

            if atomic_type_list[atom_j] != target_neighbor_element:  # Only keep target elements. Skip if not
                continue

            buffer_dx = 0.5 * lattice_a - ((x_list[atom_j] - x_list[atom_i] + 1.5 * lattice_a) % lattice_a)
            buffer_dy = 0.5 * lattice_b - ((y_list[atom_j] - y_list[atom_i] + 1.5 * lattice_b) % lattice_b)
            buffer_dz = 0.5 * lattice_c - ((z_list[atom_j] - z_list[atom_i] + 1.5 * lattice_c) % lattice_c)

            cutoff_average_squared = cutoff_average * cutoff_average

            dx2 = buffer_dx * buffer_dx
            dy2 = buffer_dy * buffer_dy
            dz2 = buffer_dz * buffer_dz

            r2 = dx2 + dy2 + dz2

            if cutoff_average_squared > r2:
                number_of_fingerprints_in_average += 1
                single_averaged_fingerprint += fingerprint_vector_list[atom_j, :]

        single_averaged_fingerprint = single_averaged_fingerprint / float(number_of_fingerprints_in_average)

        average_fingerprint_list = list(single_averaged_fingerprint)
        fingerprint_str = ["{0: >30.15f}".format(v) for v in average_fingerprint_list]

        result_str_list.append(",".join(fingerprint_str) + '\n')

    if append:
        writing_mode = 'a'
    else:
        writing_mode = 'w'

    with open(output_file, writing_mode) as output_file_pointer:
        output_file_pointer.writelines(result_str_list)
        output_file_pointer.close()


class AverageFingerprintCalculator:
    """
        Main class for calculation of LAAF
    """

    def __init__(self,
                 cutoff_descriptor: float = 5.0,
                 cutoff_average: float = 4.0,
                 traj_data: str = 'traj_data', #"./trajectory.xyz",
                 selected_snapshots: str = ':',  # Format of ase for selecting the snapshots in trajectories
                 number_of_eta: int = 100,
                 element_conversion: dict = None,
                 descriptor_type='custom',
                 descriptor=None,
                 descriptor_parameters: dict = None,
                 ):
        """

        :param cutoff_descriptor:
        :param cutoff_average:
        :param traj_data:  #input_file:
        :param selected_snapshots:
        :param number_of_eta:
        :param element_conversion:
        :param descriptor_type:
        :param descriptor: Descriptor instance (Only tested for SOAP)
        :param descriptor_parameters:
        """
        # TODO: Choose types of atoms to be included in averaging
        if descriptor_parameters is None:
            descriptor_parameters = {}
        self.fingerprint_buffer = {}  # Reduce redundant operations

        self.cutoff_descriptor = cutoff_descriptor
        self.cutoff_average = cutoff_average
        #self.input_file = input_file
        self.traj_data = traj_data
        self.selected_snapshots = selected_snapshots
        self.number_of_eta = number_of_eta

        if element_conversion is None:
            element_conversion = {
                'Si': 0,
                'O': 1,
            }
        self.element_conversion = element_conversion

        # Read input file with ASE and select snapshots
        self.atoms = traj_data #ase.io.read(input_file, selected_snapshots)

        self.species = list(element_conversion.keys())

        # Allowed descriptor types: 'custom', 'ACSF_G2', 'ACSF_G2G4'
        allowed_descriptor_types = ['custom', 'ACSF_G2', 'ACSF_G4', 'ACSF_G2G4', 'SOAP', 'MBSF']
        self.descriptor_type = descriptor_type
        assert descriptor_type in allowed_descriptor_types, f'Invalid descriptor type {descriptor_type}. Stopping now'

        local_descriptor_dict = {
            'ACSF_G2': ACSF,
            'ACSF_G4': ACSF,
            'ACSF_G2G4': ACSF,
            'SOAP': SOAP,
            'MBSF': MBSF,
        }

        if descriptor_type == 'custom':
            assert descriptor is not None, 'For a custom descriptor type, you must provide a descriptor instance'
            self.descriptor = descriptor
        else:
            #print('Using local descriptor. Any other descriptor in input will be discarded.')
            # Initialize the list of available local descriptors

            eta_list_g2 = calculate_eta(cutoff_descriptor=cutoff_descriptor, number_of_eta=number_of_eta)

            if descriptor_type == 'ACSF_G2':
                # Set G2 parameters
                rs_lst = [0.0]
                params_g2 = []
                for eta in eta_list_g2:
                    for rs in rs_lst:
                        params_g2.append([1.0 / (eta * eta), rs])

                params_g2 = np.array(params_g2)

                descriptor_parameter_dict = {
                    'cutoff_descriptor': cutoff_descriptor,
                    'params_g2': params_g2,
                    'species': self.species,
                }
                descriptor_parameter_dict.update(descriptor_parameters)  # Use user input, if any

                self.descriptor = local_descriptor_dict[descriptor_type](**descriptor_parameter_dict)

            elif descriptor_type == 'ACSF_G4':
                # Set G2 parameters
                params_g2 = None

                # Set G4 parameters
                number_of_eta_g4 = 10
                eta_list_g4 = calculate_eta(cutoff_descriptor=cutoff_descriptor, number_of_eta=number_of_eta_g4)
                zeta_lst = [1, 2, 4, 16]
                lambda_list = [-1, 1]
                params_g4 = []
                for eta0 in eta_list_g4:
                    for zeta0 in zeta_lst:
                        for lambda0 in lambda_list:
                            params_g4.append([1.0 / (eta0 * eta0), zeta0, lambda0])
                params_g4 = np.array(params_g4)

                descriptor_parameter_dict = {
                    'cutoff_descriptor': cutoff_descriptor,
                    'params_g2': params_g2,
                    'params_g4': params_g4,
                    'species': self.species,
                }
                descriptor_parameter_dict.update(descriptor_parameters)  # Use user input, if any

                self.descriptor = local_descriptor_dict[descriptor_type](**descriptor_parameter_dict)

            elif descriptor_type == 'ACSF_G2G4':
                # Set G2 parameters
                rs_lst = [0.0]
                params_g2 = []
                for eta in eta_list_g2:
                    for rs in rs_lst:
                        params_g2.append([1.0 / (eta * eta), rs])

                params_g2 = np.array(params_g2)
                # Set G4 parameters
                number_of_eta_g4 = 10
                eta_list_g4 = calculate_eta(cutoff_descriptor=cutoff_descriptor, number_of_eta=number_of_eta_g4)
                zeta_lst = [1, 2, 4, 16]
                lambda_list = [-1, 1]
                params_g4 = []
                for eta0 in eta_list_g4:
                    for zeta0 in zeta_lst:
                        for lambda0 in lambda_list:
                            params_g4.append([1.0 / (eta0 * eta0), zeta0, lambda0])
                params_g4 = np.array(params_g4)

                descriptor_parameter_dict = {
                    'cutoff_descriptor': cutoff_descriptor,
                    'params_g2': params_g2,
                    'params_g4': params_g4,
                    'species': self.species,
                }
                descriptor_parameter_dict.update(descriptor_parameters)  # Use user input, if any

                self.descriptor = local_descriptor_dict[descriptor_type](**descriptor_parameter_dict)

            elif descriptor_type == 'MBSF':
                # set parameter for MBSF gr(g2) + ga (\zeta, \theta_s, \eta, Rs)
                rs_lst = [0.0]
                params_gr = []
                for eta in eta_list_g2:
                    for rs in rs_lst:
                        params_gr.append([1.0 / (eta * eta), rs])

                params_gr = np.array(params_gr)
                # G4
                zeta_lst = [1, 2, 4, 16]
                theta_s_list = np.linspace(0, 1, 5) * np.pi
                number_of_eta_ga = 10
                eta_ga_list = calculate_eta(
                    cutoff_descriptor=cutoff_descriptor,
                    number_of_eta=number_of_eta_ga,
                )
                rs_lst = [0.0]
                params_ga = []
                for zeta0 in zeta_lst:
                    for theta_s0 in theta_s_list:
                        for eta0 in eta_ga_list:
                            for rs0 in rs_lst:
                                params_ga.append([zeta0, theta_s0, 1.0 / (eta0 * eta0), rs0])
                params_ga = np.array(params_ga)

                descriptor_parameter_dict = {
                    'cutoff_descriptor': cutoff_descriptor,
                    'params_gr': params_gr,
                    'params_ga': params_ga,
                    'periodic': True,
                    'species': self.species,
                }
                descriptor_parameter_dict.update(descriptor_parameters)  # Use user input, if any

                self.descriptor = local_descriptor_dict[descriptor_type](**descriptor_parameter_dict)

            elif descriptor_type == 'SOAP':

                descriptor_parameter_dict = {
                    'species': self.species,
                    'periodic': True,
                    'rcut': cutoff_descriptor,
                    'nmax': 8,
                    'lmax': 8,
                }
                descriptor_parameter_dict.update(descriptor_parameters)  # Use user input, if any

                self.descriptor = local_descriptor_dict[descriptor_type](**descriptor_parameter_dict)

    def compute_averaged_fingerprints_random(
            self,
            output_file: str = "average_fingerprint.csv",
            append: bool = False,
            number_of_data: int = None,  # generated number of data,
            start_step: int = 0,  # first step to generate data
            final_step: int = None,  # final step to generate data
            target_element: int = 0,
            target_neighbor_element: int = None,
            seed: int = None,
    ):
        if target_neighbor_element is None:
            target_neighbor_element = target_element

        # If seed is defined to fix the order, mainly for DEBUG
        if seed:
            print(f'WARNING: A seed was set ({seed}). Debugging mode...')
            random.seed(seed)

        atoms = self.atoms
        if final_step is None:  # Default is that final step is last step
            final_step = len(atoms) - 1
        assert final_step < len(atoms), \
            f'Final step exceeds available number of snapshots. Stopping now! ({final_step} >= {len(atoms)})'

        # Generate all data points
        result_str_list = []

        #for data_index in tqdm(range(number_of_data), desc=f'LAAF of {os.path.basename(self.input_file)}'):
        for data_index in range(number_of_data):
            # TODO: Select tuples of (selected_step,data_index) to avoid duplicate data

            # Select step
            selected_step = random.randint(start_step, final_step)
            selected_snapshot = atoms[selected_step]

            # Get atomic type list
            symbols = selected_snapshot.symbols
            number_of_atoms = len(symbols)
            atomic_type_list = [self.element_conversion[symbol] for symbol in symbols]
            atomic_type_list = np.array(atomic_type_list)

            # Get cell parameters
            # Current available cell system : orthorhombic

            cell = selected_snapshot.cell
            lattice_a, lattice_b, lattice_c = cell[0, 0], cell[1, 1], cell[2, 2]
            # lattice_parameters = [lattice_a, lattice_b, lattice_c] # not used

            # Get x, y, z coordination
            positions = selected_snapshot.positions
            x_list = positions[:, 0]
            y_list = positions[:, 1]
            z_list = positions[:, 2]

            if selected_step in self.fingerprint_buffer.keys():
                fingerprint_vector_list = self.fingerprint_buffer[selected_step]

            else:
                fingerprint_vector_list = self.descriptor.create(selected_snapshot)  # TODO: call create without system?

            self.fingerprint_buffer[selected_step] = fingerprint_vector_list

            """
            this_species_atom_indices = [i for i in range(number_of_atoms) if
                                         atomic_type_list[i] == target_element]
            atom_i = random.choice(this_species_atom_indices)
            """
            for k in range(number_of_atoms):
                # Select a random atom
                atom_i = random.randint(0, number_of_atoms - 1)
                if atomic_type_list[atom_i] == target_element:
                    break

            single_averaged_fingerprint = np.zeros(len(fingerprint_vector_list[atom_i]))
            number_of_fingerprints_in_average = 0

            for atom_j in range(number_of_atoms):

                if atomic_type_list[atom_j] != target_neighbor_element:  # Only keep target elements. Skip if not
                    continue

                buffer_dx = 0.5 * lattice_a - ((x_list[atom_j] - x_list[atom_i] + 1.5 * lattice_a) % lattice_a)
                buffer_dy = 0.5 * lattice_b - ((y_list[atom_j] - y_list[atom_i] + 1.5 * lattice_b) % lattice_b)
                buffer_dz = 0.5 * lattice_c - ((z_list[atom_j] - z_list[atom_i] + 1.5 * lattice_c) % lattice_c)

                cutoff_average_squared = self.cutoff_average * self.cutoff_average

                dx2 = buffer_dx * buffer_dx
                dy2 = buffer_dy * buffer_dy
                dz2 = buffer_dz * buffer_dz

                r2 = dx2 + dy2 + dz2

                if cutoff_average_squared > r2:
                    number_of_fingerprints_in_average += 1
                    single_averaged_fingerprint += fingerprint_vector_list[atom_j, :]

            single_averaged_fingerprint = single_averaged_fingerprint / float(number_of_fingerprints_in_average)

            average_fingerprint_list = list(single_averaged_fingerprint)
            fingerprint_str = ["{0: >30.15f}".format(v) for v in average_fingerprint_list]

            result_str_list.append(",".join(fingerprint_str) + '\n')

        if append:
            writing_mode = 'a'
        else:
            writing_mode = 'w'

        with open(output_file, writing_mode) as output_file_pointer:
            output_file_pointer.writelines(result_str_list)
            output_file_pointer.close()

    def compute_averaged_fingeprints_selection(
            self,
            output_file: str = "average_fingerprint.csv",
            selected_atoms: list = None,
            append: bool = False,
            selected_steps: list = None,
            target_element: int = 0,
            target_neighbor_element: int = None,
    ):
        if target_neighbor_element is None:
            target_neighbor_element = target_element

        atoms = self.atoms

        # Generate all data points
        result_str_list = []

        if selected_atoms is None:
            selected_atoms = range(len(atoms[0].symbols))

        if selected_steps is None:
            selected_steps = range(len(atoms))

        #for this_step in tqdm(selected_steps):
        for this_step in selected_steps:

            selected_snapshot = atoms[this_step]
            # Get atomic type list
            symbols = selected_snapshot.symbols
            number_of_atoms = len(symbols)
            atomic_type_list = [self.element_conversion[symbol] for symbol in symbols]
            atomic_type_list = np.array(atomic_type_list)
            selected_atoms = np.where(atomic_type_list == target_element)[0]

            cell = selected_snapshot.cell
            lattice_a, lattice_b, lattice_c = cell[0, 0], cell[1, 1], cell[2, 2]

            # Get x, y, z coordination
            positions = selected_snapshot.positions
            x_list = positions[:, 0]
            y_list = positions[:, 1]
            z_list = positions[:, 2]

            if this_step in self.fingerprint_buffer.keys():
                fingerprint_vector_list = self.fingerprint_buffer[this_step]
            else:
                fingerprint_vector_list = self.descriptor.create(selected_snapshot)  # TODO: call create without system?
                #fingerprint_vector_list = self.descriptor.create(selected_atoms)  # TODO: call create without system?

            self.fingerprint_buffer[this_step] = fingerprint_vector_list

            for atom_i in selected_atoms:
                # single_averaged_fingerprint = np.array([0.0 for k in range(2 * number_of_eta)])
                single_averaged_fingerprint = np.zeros(len(fingerprint_vector_list[atom_i]))
                number_of_fingerprints_in_average = 0

                for atom_j in range(number_of_atoms):

                    if atomic_type_list[atom_j] != target_neighbor_element:  # Only keep target elements. Skip if not
                        continue

                    buffer_dx = 0.5 * lattice_a - ((x_list[atom_j] - x_list[atom_i] + 1.5 * lattice_a) % lattice_a)
                    buffer_dy = 0.5 * lattice_b - ((y_list[atom_j] - y_list[atom_i] + 1.5 * lattice_b) % lattice_b)
                    buffer_dz = 0.5 * lattice_c - ((z_list[atom_j] - z_list[atom_i] + 1.5 * lattice_c) % lattice_c)

                    cutoff_average_squared = self.cutoff_average * self.cutoff_average

                    dx2 = buffer_dx * buffer_dx
                    dy2 = buffer_dy * buffer_dy
                    dz2 = buffer_dz * buffer_dz

                    r2 = dx2 + dy2 + dz2

                    if cutoff_average_squared > r2:
                        number_of_fingerprints_in_average += 1
                        single_averaged_fingerprint += fingerprint_vector_list[atom_j, :]

                single_averaged_fingerprint = single_averaged_fingerprint / float(number_of_fingerprints_in_average)

                average_fingerprint_list = list(single_averaged_fingerprint)
                fingerprint_str = ["{0: >30.16e}".format(v) for v in average_fingerprint_list]

                result_str_list.append("".join(fingerprint_str) + '\n')

        if append:
            writing_mode = 'a'
        else:
            writing_mode = 'w'

        with open(output_file, writing_mode) as output_file_pointer:
            output_file_pointer.writelines(result_str_list)
            output_file_pointer.close()
