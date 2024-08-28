import warnings, os
warnings.filterwarnings("ignore")
#from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation
from pymatgen.io.vasp.sets import batch_write_input, MPRelaxSet
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core.surface import SlabGenerator
from ase.io import read, write
from pymatgen.core.structure import Structure
import numpy as np

def updatepositions(atoms):
    cell = atoms.cell
    # print('Updating positions. Cell is:', cell)
    inverse_cell = np.linalg.inv(cell)
    positions = atoms.positions
    species = np.array(list(atoms.symbols))
    #print(species.shape)
    species_modified = replace_cations(species).reshape(-1, 1)
    
    sy_pos = np.concatenate((species_modified, positions), axis=1)
    
    # Define a function to get the atomic symbol from a data row
    def get_atomic_symbol(row):
     return row[0]
    sorted_data = np.array(sorted(sy_pos, key=get_atomic_symbol))
    
    atoms.symbols = (sorted_data[:, 0:1]).flatten()
    atoms.positions = sorted_data[:, 1:]

    return atoms


def replace_cations(species):
    # Count the number of 'Co' occurrences
    num_o = np.sum(species == 'Co')

    # Probabilities for replacement
    probabilities = [0.6, 0.2, 0.2]

    # Replacement choices
    replacement_choices = ['Ni', 'Mn', 'Co']

    # Replace 'O' atoms with 'S', 'P', or 'Cl' based on probabilities
    replacement = np.random.choice(replacement_choices, size=num_o, p=probabilities)
    

    # Get the indices where 'O' atoms are present
    indices = np.where(species == 'Co')[0]


    # Update the species array with replacements
    species[indices] = replacement
    return species    

def build_interface():
    replace_cations=True
    log = open('out.log', 'w')

    # Define the directory to check/create
    directory = "interfaces_structures"
    
    # Check if the directory exists
    if not os.path.exists(directory):
        # If not, create the directory
        os.makedirs(directory)

    # Check all input files
    pwd = "./"

    # List of files to check
    files_to_check = ['film.vasp', 'sub.vasp']

    # Check for the existence of each file
    for filename in files_to_check:
        file_path = os.path.join(pwd, filename)
        if os.path.isfile(file_path):
            print(f"{filename} exists in the directory.")
        else:
            print()
            print("#######ERROR##########")
            print(f"{filename} does not exist in the directory.")
            print("#######ERROR##########")
            exit()

    film_structure = Structure.from_file("film.vasp")
    substrate_structure = Structure.from_file("sub.vasp")
    sub_indices = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 4)]
    film_index = (1, 0, 0)
    for sub_index in sub_indices:
        zsl = ZSLGenerator(max_area=700)
        cib = CoherentInterfaceBuilder(film_structure=film_structure,
                                       substrate_structure=substrate_structure,
                                       film_miller=film_index,
                                       substrate_miller=sub_index,
                                       zslgen=zsl)
        
        film_terminations = []
        substrate_terminations = []
        for ter in cib.terminations:
            film_terminations.append(ter[1])
            substrate_terminations.append(ter[0])
            
        log.write(f"Film terminations: {set(film_terminations)} \n")
        log.write(f"substrate terminations: {set(substrate_terminations)} \n") 
       
        for i, termination in enumerate(cib.terminations):
            log.write(f"Let's use {termination} to build interface with substrate index {sub_index} and film index {film_index} \n")
            interfaces = list(
                cib.get_interfaces(termination=termination,
                                   gap=1.5,
                                   vacuum_over_film=1.5,
                                   film_thickness=40,
                                   substrate_thickness=40,
                                   in_layers=False))
        
            log.write(f"No. of interfaces are: {len(interfaces)} \n")
        
            for j, inter in enumerate(interfaces[:2]): 
                filename = f"./{directory}/interface_{i}{j}_{''.join(map(str, sub_index))}_{''.join(map(str, film_index))}.vasp"
                log.write(f"{filename} \n")
                log.write(f"No. of atoms: {len(interfaces[j])} \n")
                log.write(f"Cell size: {interfaces[j].lattice.parameters} \n")   
                log.write(f"Volume: {interfaces[j].volume} \n")   
                log.write("########## \n")
                log.write("\n")
                
                interfaces[0].to_file('POSCAR')
                

                if replace_cations:
                    write(filename, updatepositions(read('POSCAR')))
                else:
                    os.rename('POSCAR', filename)
                print(f"{filename} is done")
    log.close()          
    return

