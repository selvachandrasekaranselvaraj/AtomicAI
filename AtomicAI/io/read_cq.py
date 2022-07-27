from ase import Atoms
from numpy import *
from matplotlib.pyplot import *
from collections import *
from operator import itemgetter, attrgetter
import sys
from collections import Counter

def read_cq(file_name):
    cq_input_lines = []
    with open('Conquest_input', 'r') as cq_input_file_data:
        for cq_input_line in cq_input_file_data:
            cq_input_lines.append(cq_input_line)

    for j in range(0, len(cq_input_lines)):
        if "ChemicalSpeciesLabel" in cq_input_lines[j]:
            index = j

    species_type = {}
    for i in range(1,10):
        try:
            if '%endblock' in cq_input_lines[i+index]:
                continue
            else:
                dummy = cq_input_lines[index+i].split()
                species_type.update({dummy[0]:dummy[2]})
        except:
            continue

    coords_data = []
    with open(file_name, 'r') as coords_lines:
        for line in coords_lines:
           coords_data.append(line.split())

    # Extract data from coords_data
    data = {'file_format': ['dat', 'CQ', 'cq']}
    data['file_name'] = file_name[:-4]
    data['title'] = 'CQ to vasp by Selva'
    data['lat_scale'] = 1.0
    data['lattice_a']   = 0.529177*float(coords_data[0][0])
    data['lattice_b']   = 0.529177*float(coords_data[1][1])
    data['lattice_c']   = 0.529177*float(coords_data[2][2])
    tot_no_of_atoms = int(coords_data[3][0])
    atom_sy = []
    for i in range(tot_no_of_atoms):
        atom_sy.append(species_type[coords_data[4+i][3]])
    species = Counter(atom_sy)
    data['species'] = dict(species) 
    line = 4
    positions = []
    for specie, specie_count in species.items():
        xyz_data = []
        for i in range(0, specie_count):
            item = coords_data[line]
            xyz_data.append([float(item[0])*data['lattice_a'], 
                                      float(item[1])*data['lattice_b'], 
                                      float(item[2])*data['lattice_c'],
                                      item[4], item[5], item[6]])  # including atomic constains
            positions.append([float(item[0])*data['lattice_a'], 
                                      float(item[1])*data['lattice_b'], 
                                      float(item[2])*data['lattice_c']
                                      ])  # including atomic constains
            line += 1
        data[specie] = xyz_data
    data['coordinate_scale'] = 'Cartesian'
    symbols = str()
    for symbol, n_atoms in (Counter(data['species']).items()):
        symbols += symbol+str(n_atoms)
    ase_data = Atoms(
            symbols,
            cell=[data['lattice_b'],data['lattice_b'],data['lattice_c']],
            positions = positions,
            pbc = ['True', 'True', 'True'],

            )
    return ase_data
