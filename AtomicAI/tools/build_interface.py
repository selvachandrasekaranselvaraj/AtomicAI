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
import re, os, shutil
from ase.io import lammpsdata

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
    cal_dir = "./calculations"
    
    # Check if the directory exists
    if not os.path.exists(cal_dir):
        # If not, create the directory
        os.makedirs(cal_dir)

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
    sub_indices = [(1, 0, 0)] #[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 4)]
    film_index = (1, 0, 0)
    for sub_index in sub_indices:
        
        #Calcualtion directory
        sur_dir = f"{''.join(map(str, sub_index))}_{''.join(map(str, film_index))}"

        # Check if the directory exists
        dir_1 = f"{cal_dir}/{sur_dir}"
        if not os.path.exists(dir_1):
            # If not, create the directory
            os.makedirs(dir_1)
        
        zsl = ZSLGenerator(max_area=2000)
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

            # different termination directory
            ter_dir = f"{termination[1].split('_')[0]}_{termination[0].split('_')[0]}"

            # Check if the directory exists
            dir_2 = f"{cal_dir}/{sur_dir}/{ter_dir}"
            if not os.path.exists(dir_2):
                # If not, create the directory
                os.makedirs(dir_2)

            # Check if the directory exists
            dir_npt = f"{cal_dir}/{sur_dir}/{ter_dir}/npt"
            if not os.path.exists(dir_npt):
                # If not, create the directory
                os.makedirs(dir_npt)

            # Check if the directory exists
            dir_nvt = f"{cal_dir}/{sur_dir}/{ter_dir}/nvt"
            if not os.path.exists(dir_nvt):
                # If not, create the directory
                os.makedirs(dir_nvt)
        
            log.write(f"Let's use {termination} to build interface with substrate index {sub_index} and film index {film_index} \n")
            interfaces = list(
                cib.get_interfaces(termination=termination,
                                   gap=1.0,
                                   vacuum_over_film=1.0,
                                   film_thickness=40,
                                   substrate_thickness=40,
                                   in_layers=False))
        
            log.write(f"No. of interfaces are: {len(interfaces)} \n")
        
            for j, inter in enumerate(interfaces[:1]): 
                #  NPT
                filename = f"./{dir_npt}/interface_{i}{j}_{''.join(map(str, sub_index))}_{''.join(map(str, film_index))}.vasp"
                log.write(f"{filename} \n")
                log.write(f"No. of atoms: {len(interfaces[j])} \n")
                log.write(f"Cell size: {interfaces[j].lattice.parameters} \n")   
                log.write(f"Volume: {interfaces[j].volume} \n")   
                log.write("########## \n")
                log.write("\n")
                
                poscar_file = f"{dir_npt}/POSCAR"
                lmp_data_file = f"{dir_npt}/data.lmp_data"
                vasp_file = f"{dir_npt}/data.vasp"
                interfaces[0].to_file(poscar_file)
                

                if replace_cations:
                    write(vasp_file, updatepositions(read(poscar_file)))
                    atoms_data = read(vasp_file)
                    lammpsdata.write_lammps_data(lmp_data_file, atoms_data, masses=True)
                    #os.rename(filename, vasp_file)
                    os.remove(poscar_file)
                else:
                    os.rename(poscar_file, vasp_file)
                    #os.copy(filename, 'data.vasp')
                    atoms_data = read(vasp_file)
                    lammpsdata.write_lammps_data(lmp_data_file, atoms_data, masses=True)
                generate_lammps_npt_inputs(dir_npt, lmp_data_file)

                #  NVT
                shutil.copy(lmp_data_file, f"{dir_nvt}/old_data.lmp_data")
                generate_lammps_npt_inputs(dir_npt, lmp_data_file)
                generate_lammps_nvt_inputs(dir_nvt, f"{dir_nvt}/old_data.lmp_data")
                print(f"{dir_nvt} is done")

    
    log.close()          
    return
    
def get_elements(data_file):
    data = lammpsdata.read_lammps_data(data_file)
    return list(data.symbols.formula.count().keys())

def generate_lammps_npt_inputs(directory, data_file):
    if os.path.isfile(data_file):
        pass
    else:
        print(f"{data_file} does not exist in the directory.")
        exit()


    elements = ' '.join(get_elements(data_file))
    lammps_input_content = f"""# Structure
units           metal
boundary        p p p
atom_style      atomic

# Variables
variable read_data_file string "data.lmp_data"
variable pair_style_type string "deepmd mlpot.dp"
variable pair_coeff_type string "* *"
variable dump_file1 string "dump_unwrapped.lmp"
variable dump_file2 string "dump.lmp"

# Numeric Variables
variable nx equal 1
variable ny equal 1
variable nz equal 1
variable run_1 equal 20000 #NPT run
variable temp_init equal 0.01
variable temp_final equal 0.01
variable timestep equal 0.001
variable thermo_freq equal 1
variable dump_freq_1 equal 100
variable temp_damp equal 50.0*${{timestep}}
variable velocity_seed equal 87287
variable neighbor_distance equal 0.5
variable neigh_modify_every equal 20
variable neigh_modify_delay equal 0
variable neigh_modify_check equal "no"

read_data       ${{read_data_file}} 
replicate       ${{nx}} ${{ny}} ${{nz}}

region interface1 block      INF INF INF INF 0.0 2.0
group  interface1  region interface1
region interface2 block      INF INF INF INF 2.0 INF
group  interface2  region interface2


pair_style	    ${{pair_style_type}}
pair_coeff	    ${{pair_coeff_type}}

neighbor        ${{neighbor_distance}} bin
neigh_modify    every ${{neigh_modify_every}} delay ${{neigh_modify_delay}} check ${{neigh_modify_check}}

# Timestep
timestep        ${{timestep}}

# Output settings
thermo ${{thermo_freq}}
thermo_style custom step time temp press pe vol 

# Dump settings
#dump 1 all custom ${{dump_freq_1}} ${{dump_file1}} id type xu yu zu vx vy vz element
dump 1 interface2 custom ${{dump_freq_1}} ${{dump_file1}} id type xu yu zu element
dump_modify 1 every ${{dump_freq_1}} element {elements} 
dump 2 interface2 custom ${{dump_freq_1}} ${{dump_file2}} id type x y z element
dump_modify 2 every ${{dump_freq_1}} element {elements} 

# Energy Minimization (Geometry Optimization)
#fix 10 all box/relax aniso 0.0 vmax 0.001
#min_style cg
#minimize   1.0e-25 1.0e-25 100000 100000
#unfix 10

# Initial velocity
velocity  interface2 create ${{temp_init}} ${{velocity_seed}} loop geom

# NPT
fix    1 interface2 npt temp ${{temp_init}} ${{temp_final}} ${{temp_damp}} iso 1.0 1.0 100
run ${{run_1}}
unfix 1

# Write the minimized structure to a file
write_data minimized_structure.dat
"""

    lammps_file = f"{directory}/in.lammps"
    # Save the LAMMPS input to a file
    with open(lammps_file, 'w') as file:
        file.write(lammps_input_content)
    
    #print("LAMMPS input file generated successfully.")
    
    job_name = 'npt'+''.join([s[0] for s in get_elements(data_file)])
    
    sub_file_content = f"""#!/bin/bash
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -A LTC
#PBS -l walltime=72:00:00
#PBS -N {job_name}
##PBS -o vasp.out
#PBS -j n
#PBS -m e

cd $PBS_O_WORKDIR
NNODES=`wc -l < $PBS_NODEFILE`
echo "NNODES=" $NNODES

module add gcc/13.2.0 openmpi/4.1.6-gcc-13.2.0 aocl/4.1.0-gcc-13.1.0
export PATH=/soft/software/custom-built/vasp/5.4.4/bin:$PATH
export UCX_NET_DEVICES=mlx5_0:1

#mpirun -np $NNODES vasp_std
#autopsy dump.lmp 128
mpirun -np $NNODES lmp_mpi -in in.lammps
"""

    sub_file = f"{directory}/improv.sh"
    
    # Save the sub_file_content to a file
    with open(sub_file, 'w') as file:
        file.write(sub_file_content)
    
    #print("improv.sh file generated successfully.")

    # Check all input files
    # Define the directory to check

    # List of files to check
    files_to_check = ['data.lmp_data', 'mlpot.dp', 'in.lammps', 'improv.sh']

    # Check for the existence of each file
    for filename in files_to_check:
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            pass
            
        else:
            if os.path.isfile('mlpot.dp'):
                shutil.copy('mlpot.dp', file_path)
            else:
                print()
                print("#######ERROR##########")
                print(f"{filename} does not exist in the directory.")
                print("#######ERROR##########")
                os.exit()
    return

def generate_lammps_nvt_inputs(directory, data_file):
    if os.path.isfile(data_file):
        pass
    else:
        print(f"{data_file} does not exist in the directory.")
        exit()


    elements = ' '.join(get_elements(data_file))
    lammps_input_content = f"""# Structure
units           metal
boundary        p p p
atom_style      atomic

# Variables
variable read_data_file string "minimized_structure.dat"
variable pair_style_type string "deepmd mlpot.dp"
variable pair_coeff_type string "* *"
variable dump_file1 string "dump_unwrapped.lmp"
variable dump_file2 string "dump.lmp"

# Numeric Variables
variable nx equal 1
variable ny equal 1
variable nz equal 1
variable run_1 equal 2000000 #NVT run
variable temp_init equal 300.0
variable temp_final equal 300.0
variable timestep equal 0.001
variable thermo_freq equal 1
variable dump_freq_1 equal 100
variable temp_damp equal 50.0*${{timestep}}
variable velocity_seed equal 87287
variable neighbor_distance equal 0.5
variable neigh_modify_every equal 20
variable neigh_modify_delay equal 0
variable neigh_modify_check equal "no"

read_data       ${{read_data_file}} 
replicate       ${{nx}} ${{ny}} ${{nz}}

region interface1 block      INF INF INF INF 0.0 2.0
group  interface1  region interface1
region interface2 block      INF INF INF INF 2.0 INF
group  interface2  region interface2


pair_style	    ${{pair_style_type}}
pair_coeff	    ${{pair_coeff_type}}

neighbor        ${{neighbor_distance}} bin
neigh_modify    every ${{neigh_modify_every}} delay ${{neigh_modify_delay}} check ${{neigh_modify_check}}

# Timestep
timestep        ${{timestep}}

# Output settings
thermo ${{thermo_freq}}
thermo_style custom step time temp press pe vol 

# Dump settings
#dump 1 all custom ${{dump_freq_1}} ${{dump_file1}} id type xu yu zu vx vy vz element
dump 1 interface2 custom ${{dump_freq_1}} ${{dump_file1}} id type xu yu zu element
dump_modify 1 every ${{dump_freq_1}} element {elements} 
dump 2 interface2 custom ${{dump_freq_1}} ${{dump_file2}} id type x y z element
dump_modify 2 every ${{dump_freq_1}} element {elements} 

# Energy Minimization (Geometry Optimization)
#fix 10 all box/relax aniso 0.0 vmax 0.001
#min_style cg
#minimize   1.0e-25 1.0e-25 100000 100000
#unfix 10

# Initial velocity
velocity  interface2 create ${{temp_init}} ${{velocity_seed}} loop geom


# NPT
fix    1 interface2 nvt temp ${{temp_init}} ${{temp_final}} ${{temp_damp}} 
run ${{run_1}}
unfix 1

# Write the minimized structure to a file
write_data after_nvt.dat
"""

    lammps_file = f"{directory}/in.lammps"
    # Save the LAMMPS input to a file
    with open(lammps_file, 'w') as file:
        file.write(lammps_input_content)
    
    #print("LAMMPS input file generated successfully.")
    
    job_name = 'nvt'+''.join([s[0] for s in get_elements(data_file)])
    
    sub_file_content = f"""#!/bin/bash
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -A LTC
#PBS -l walltime=72:00:00
#PBS -N {job_name}
##PBS -o vasp.out
#PBS -j n
#PBS -m e

cd $PBS_O_WORKDIR
NNODES=`wc -l < $PBS_NODEFILE`
echo "NNODES=" $NNODES

module add gcc/13.2.0 openmpi/4.1.6-gcc-13.2.0 aocl/4.1.0-gcc-13.1.0
export PATH=/soft/software/custom-built/vasp/5.4.4/bin:$PATH
export UCX_NET_DEVICES=mlx5_0:1

#mpirun -np $NNODES vasp_std
#autopsy dump.lmp 128
mpirun -np $NNODES lmp_mpi -in in.lammps
"""

    sub_file = f"{directory}/improv.sh"
    
    # Save the sub_file_content to a file
    with open(sub_file, 'w') as file:
        file.write(sub_file_content)
    
    #print("improv.sh file generated successfully.")

    # Check all input files
    # Define the directory to check

    # List of files to check
    files_to_check = ['mlpot.dp', 'in.lammps', 'improv.sh']

    # Check for the existence of each file
    for filename in files_to_check:
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            pass
            
        else:
            if os.path.isfile('mlpot.dp'):
                shutil.copy('mlpot.dp', file_path)
            else:
                print()
                print("#######ERROR##########")
                print(f"{filename} does not exist in the directory.")
                print("#######ERROR##########")
                os.exit()


    return