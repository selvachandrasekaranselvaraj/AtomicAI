import re, os
from ase.io import lammpsdata
import numpy as np
from AtomicAI.tools.submission_file import job_submit

def get_elements(data_file):
    data = lammpsdata.read_lammps_data(data_file)
    return list(sorted(data.symbols.formula.count().keys()))


def get_supercell_size(data):
    positions = data.positions
    n_atoms = positions.shape[0]
    a, b, c, _, _, _ = data.cell.cellpar()
    
    # Initial supercell multipliers
    na = nb = nc = 1
    
    # Array of cell dimensions and corresponding multipliers
    abc = np.array([a, b, c])
    nabc = np.array([na, nb, nc])
    
    # Keep increasing the smallest dimension until n_atoms in the supercell is less than or equal to 2000
    while n_atoms < 2000:
        # Find the index of the smallest dimension
        min_index = np.argmin(abc * nabc)
        
        # Increase the multiplier for the smallest dimension
        nabc[min_index] += 1
        
        # Calculate the new total number of atoms in the supercell
        n_atoms = positions.shape[0] * np.prod(nabc)
    
    return nabc

def generate_lammps_npt_inputs():
    data_file = 'data.lmp_data'
    if os.path.isfile(data_file):
        print(f"{data_file} exists in the directory.")
    else:
        print(f"{data_file} does not exist in the directory.")
        exit()

    elements = ' '.join(get_elements(data_file))
    data = lammpsdata.read_lammps_data(data_file)
    nx, ny, nz = get_supercell_size(data)

    lammps_input_content = f"""# Structure
units           metal
boundary        p p p
atom_style      atomic

# Variables
variable read_data_file string "{data_file}"
variable pair_style_type string "mlpot.dp"
variable dump_file1 string "dump_unwrapped.lmp"
variable dump_file2 string "dump.lmp"

# Numeric Variables
variable nx equal {nx}
variable ny equal {ny}
variable nz equal {nz}
variable run_1 equal 10000 #NPT run
variable temp_init equal 300
variable temp_final equal 300
variable timestep equal 0.001
variable thermo_freq equal 1
variable dump_freq_1 equal 10
variable temp_damp equal 100.0*${{timestep}}
variable velocity_seed equal 87287
variable neighbor_distance equal 1.0
variable neigh_modify_every equal 20
variable neigh_modify_delay equal 0
variable neigh_modify_check equal "no"

read_data       ${{read_data_file}} 
replicate       ${{nx}} ${{ny}} ${{nz}}

pair_style	deepmd ${{pair_style_type}}
pair_coeff	* * 

neighbor        ${{neighbor_distance}} bin
neigh_modify    every ${{neigh_modify_every}} delay ${{neigh_modify_delay}} check ${{neigh_modify_check}}

# Timestep
timestep        ${{timestep}}

# Output settings
thermo ${{thermo_freq}}
thermo_style custom step time temp press pe cella cellb cellc cellalpha cellbeta cellgamma

# Dump settings
#dump 1 all custom ${{dump_freq_1}} ${{dump_file1}} id type xu yu zu vx vy vz element
dump 1 all custom ${{dump_freq_1}} ${{dump_file1}} id type xu yu zu element
dump_modify 1 every ${{dump_freq_1}} element {elements} 
dump 2 all custom ${{dump_freq_1}} ${{dump_file2}} id type x y z element
dump_modify 2 every ${{dump_freq_1}} element {elements} 

# Energy Minimization (Geometry Optimization)
#fix 10 all box/relax aniso 0.0 vmax 0.001
#min_style cg
#minimize   1.0e-25 1.0e-25 100000 100000
#unfix 10

# Initial velocity
velocity    all create ${{temp_init}} ${{velocity_seed}} loop geom


# Fix for NPT ensemble
# Equilibrate at high temperature

#fix             1 all npt temp 2000 2000 0.1 iso 1.0 1.0 1.0
#run             10000  # 10 ps
#unfix           1

# Rapid quenching
#fix             2 all temp/berendsen 2000 300 1.0
#run             20000  # 20 ps
#unfix           2

# Switch to NPT ensemble at low temperature
# NPT
fix    3 all npt temp ${{temp_init}} ${{temp_final}} ${{temp_damp}} iso 1.0 1.0 1.0
run ${{run_1}}
unfix 3

# Write the minimized structure to a file
write_data minimized_structure.dat
"""

    # Save the LAMMPS input to a file
    with open('in.lammps', 'w') as file:
        file.write(lammps_input_content)
    
    print("LAMMPS input file generated successfully.")
    
    job_name = 'npt'+''.join([s[0] for s in get_elements(data_file)])
    
    job_submit(job_name)

    # Check all input files
    # Define the directory to check
    directory = "./"

    # List of files to check
    files_to_check = ['data.lmp_data', 'mlpot.dp', 'in.lammps', 'sub.sh']

    # Check for the existence of each file
    for filename in files_to_check:
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            print(f"{filename} exists in the directory.")
        else:
            print()
            print("#######ERROR##########")
            print(f"{filename} does not exist in the directory.")
            print("#######ERROR##########")
            exit()


    return