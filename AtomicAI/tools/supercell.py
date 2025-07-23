import sys
import numpy as np
import ase.io
from ase.build import make_supercell
from ase.io import write

def generate_supercell_vasp(input_file, scaling_matrix, output_vasp, cartesian=False):
    atoms = ase.io.read(input_file)
    
    # Original cell vectors
    original_cell = atoms.get_cell()

    # Build transformation matrix (scaling in the original basis)
    transformation = np.diag(scaling_matrix)
    
    # Make supercell (respects original cell shape)
    supercell_atoms = make_supercell(atoms, transformation)

    sorted_indices = np.argsort(supercell_atoms.get_chemical_symbols())
    structure_sorted = supercell_atoms[sorted_indices]
    
    # New cell vectors (should scale properly)
    print("Supercell vectors:")
    print(structure_sorted.get_cell())
    
    # Write in Cartesian or Direct (fractional) coordinates
    write(output_vasp, structure_sorted, format='vasp', direct=not cartesian, vasp5=True)
    
    print(f"✅ Written supercell POSCAR to: {output_vasp}")
    print(f"🔹 Original atoms: {len(atoms)}, Supercell atoms: {len(supercell_atoms)}")
    print(f"🔹 Output format: {'Cartesian' if cartesian else 'Direct (fractional)'}")

def supercell():
    if len(sys.argv) != 2:
        print("❌ Usage: python supercell_vasp.py input.vasp")
        sys.exit(1)
    
    input_file = sys.argv[1]

    atoms = ase.io.read(input_file)
    # Original cell vectors
    original_cell = atoms.get_cell()
    print("Original cell vectors:")
    print(original_cell)
    
    
    try:
        x = int(input("Enter repetitions in x: "))
        y = int(input("Enter repetitions in y: "))
        z = int(input("Enter repetitions in z: "))
    except ValueError:
        print("❌ Please enter integers for scaling factors.")
        sys.exit(1)
    
    cartesian = input("Output in Cartesian coordinates? (y/n): ").strip().lower() == 'y'
    
    scaling = [x, y, z]
    base_name = input_file.split('.')[0]
    output_filename = f"{x}x{y}x{z}_{base_name}.vasp"
    
    generate_supercell_vasp(input_file, scaling, output_filename, cartesian)
