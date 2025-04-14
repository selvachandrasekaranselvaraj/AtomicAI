import sys
import numpy as np
from ase import Atoms
from ase.io import read, write
#from ase.atoms import symbols
from ase.data import chemical_symbols

def find_positionsB(atomsA, atomsB):
    positionsA = atomsA.positions
    positionsB = atomsB.positions
    heightA = max(positionsA[:, 2])
    heightB = max(positionsB[:, 2])
    gap = (atomsA.cell[2][2] + atomsB.cell[2][2]) - (heightA + heightB)

    z_offset = gap * 0.5
    positionsB = atomsB.positions
    #if z_offset < 1.0:
    #    z_offset = 1.0
    positionsB[:, 2] +=  (heightA + z_offset)
    return positionsB


def updatepositions(input_data):
    out_file = "combined_structure.vasp"
    atoms_all = []
    mean_cell = np.mean([c.cell for c in input_data], axis=0) 

    # Update positions and stack structures
    z_offset = 0.0
    z_max  = 0.0
    for i, atoms in enumerate(input_data):
        cell = atoms.cell
        positions = atoms.positions
        if i == 0:
            atoms.positions[:, 2] += z_max 
        else:
            #atoms.positions[:, 2] += z_max #find_positionsB(atoms_ini, atoms, z_max) 
            atoms.positions = find_positionsB(combined_atoms, atoms) 

        
        # Combine all atoms into a single Atoms object
        atoms_all.extend(atoms)
        z_max += cell[2][2] 
        combined_atoms = Atoms(atoms_all)
        
        # Update the cell of the combined structure
        mean_cell[2][2] = z_max
        combined_atoms.cell = mean_cell #[max_x, max_y, total_z]
    
    # Sort atoms by chemical symbol
    sorted_indices = sorted(range(len(combined_atoms)), key=lambda k: chemical_symbols.index(combined_atoms[k].symbol))
    combined_atoms = combined_atoms[sorted_indices]
    
    # Write the combined structure to a file
    write(out_file, combined_atoms, vasp5=True, sort=True)
        
    return combined_atoms

def vasp2vasp():
    input_data = []
    try:
        input_file = sys.argv[1:]
    except IndexError:
        print("Input error!!!!")
        print("Usage: \"vasp2xyz vasp_file_name1 vasp_file_name2 vasp_file_name3 vasp_file_name4\"")
        print()
        sys.exit(1)

    for input_f in input_file:
        if input_f in ['POSCAR', 'CONTCAR'] or input_f.endswith('.vasp') or input_f.endswith('.xml'):
            input_data.append(read(input_f))
        else:
            print('No file format matches!!!')
            sys.exit(1)
    
    updatepositions(input_data)

if __name__ == "__main__":
    vasp2vasp()
