import os
import sys
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp import Poscar

def surfaces(filename):
    # Read the input structure
    structure = Structure.from_file(filename) #* (2,2,2)
   
    # Define the Miller indices for the surfaces we want to generate
    miller_indices = [(0,0,1), (0,1,1), (1,1,1)]
     
    # Create the "surfaces" folder if it doesn't exist
    folder = "bulk_surfaces"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Get the base name of the input file (without extension)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Generate and save surfaces
    for index in miller_indices:
        slab_gen = SlabGenerator(structure, 
                   miller_index = index, 
                   min_vacuum_size = 1,
                   min_slab_size =30,
                   in_unit_planes = True,
                   #lll_reduce = True,
                   primitive = True, 
                   reorient_lattice = True,
                   bulk_like = True, 
                   center_slab = False) ### If you want the slab to be center set center_slab=True
        slabs = slab_gen.get_slabs() #symmetrize=True, tol= 0.1, ftol=0.1, max_broken_bonds=0)
        for si, slab in enumerate(slabs):
            outfile = f"{folder}/{base_name}_{''.join(map(str, index))}_{si+1}.vasp"
            slab = slab.get_orthogonal_c_slab().get_sorted_structure()
            Poscar(slab).write_file(outfile)
            
    
    print(f"Surface structures for {filename} have been generated and saved in the {folder} folder.")

def structures():
    input_files = sys.argv[1:]  # Get all command-line arguments as input files
    
    if not input_files:
        print("Input error!!!!")
        print("Usage: python script.py vasp_file_name1 [vasp_file_name2 ...]")
        sys.exit(1)

    for input_file in input_files:
        if input_file in ['POSCAR', 'CONTCAR'] or input_file.endswith('.vasp') or input_file.endswith('.xml'):
            surfaces(input_file)
        else:
            print(f'File format not supported for {input_file}')

structures()            
