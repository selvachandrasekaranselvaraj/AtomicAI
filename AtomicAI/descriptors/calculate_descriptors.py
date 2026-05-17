import sys, os, argparse
import time, multiprocessing
import numpy as np

from AtomicAI.descriptors.laaf import AverageFingerprintCalculator
from AtomicAI.data.data_lib import descriptor_cutoff, no_mpi_processors
from AtomicAI.tools.select_snapshots import select_snapshots

DESCRIPTOR_TYPES = [
    'ACSF_G2',
    'ACSF_G3',
    'ACSF_G4',
    'ACSF_G2G4',
    'ACSF_G2G4G5',
    'SOAP',
    'MBSF',
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate averaged atomic descriptors from a trajectory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Available descriptor types:\n  ' + '\n  '.join(DESCRIPTOR_TYPES),
    )
    parser.add_argument('input_file', help='Trajectory file (.xyz)')
    parser.add_argument(
        '--descriptor', '-d',
        choices=DESCRIPTOR_TYPES,
        nargs='+',
        default=['ACSF_G2', 'ACSF_G2G4', 'SOAP'],
        metavar='TYPE',
        help='Descriptor type(s) to compute (default: ACSF_G2 ACSF_G2G4 SOAP)',
    )
    parser.add_argument(
        '--n-eta', '-n',
        type=int,
        default=50,
        dest='n_eta',
        help='Number of eta decay functions (default: 50)',
    )
    return parser.parse_args()


def _build_jobs(descriptor_types, number_of_eta):
    job_variables = []
    out_directory = './descriptors/'
    os.makedirs(out_directory, exist_ok=True)

    frames, symbols = select_snapshots()
    print('No. of frames:', len(frames))
    symbols_type = sorted(set(symbols))
    target_elements = {sym: i for i, sym in enumerate(symbols_type)}

    for des_type in descriptor_types:
        for i, t_specie in enumerate(symbols_type):
            for j, tne in enumerate(symbols_type):
                if i >= j:
                    key = f'{t_specie}_{tne}'
                    if key not in descriptor_cutoff:
                        print(f'Descriptor cutoff not available for {t_specie}-{tne}.'
                              f' Add it to AtomicAI/data/data_lib.py.')
                        sys.exit(1)
                    d_cutoff, a_cutoff = descriptor_cutoff[key]
                    for d in d_cutoff:
                        for a in a_cutoff:
                            job_variables.append([
                                out_directory,
                                round(float(d), 1),
                                round(float(a), 1),
                                des_type,
                                frames,
                                number_of_eta,
                                target_elements,
                                t_specie,
                                tne,
                            ])
    return job_variables, frames


def _calc_descriptor(variables):
    out_directory, r_d, r_a, des_type, frames, number_of_eta, target_elements, t_specie, tne = variables
    # Use the part after the first underscore as the file-name prefix (e.g. G2, G2G4, SOAP)
    des_name = des_type.split('_', 1)[1] if '_' in des_type else des_type

    calculator = AverageFingerprintCalculator(
        cutoff_descriptor=r_d,
        cutoff_average=r_a,
        traj_data=frames,
        selected_snapshots=':',
        number_of_eta=number_of_eta,
        element_conversion=target_elements,
        descriptor_type=des_type,
    )
    out_file = f'{out_directory}{des_name}_{r_d}_{r_a}_{t_specie}_{tne}.dat'
    print(out_file)
    calculator.compute_averaged_fingeprints_selection(
        output_file=out_file,
        target_element=target_elements[t_specie],
        target_neighbor_element=target_elements[tne],
        selected_atoms=None,
    )


def calculate_descriptors():
    args = _parse_args()
    t0 = time.perf_counter()
    job_variables, _ = _build_jobs(args.descriptor, args.n_eta)
    with multiprocessing.Pool(no_mpi_processors) as pool:
        jobs = [pool.apply_async(_calc_descriptor, args=(v,)) for v in job_variables]
        [j.get() for j in jobs]
    print(f'Finished in {time.perf_counter() - t0:.1f} s')
