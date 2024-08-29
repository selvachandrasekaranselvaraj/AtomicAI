import re, os, sys
import numpy as np
from ase.io import lammpsdata
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
    try:
        temp_init = sys.argv[1]
        temp_final = sys.argv[2]
    except:
        print("Input error.")
        print("lammps_nvt_input init_temp final_temp")
        exit()



    data_file = 'data.lmp_data'
    if os.path.isfile(data_file):
        print(f"{data_file} exists in the directory.")
    else:
        print(f"{data_file} does not exist in the directory.")
        exit()

    elements = ' '.join(get_elements(data_file))
    data = lammpsdata.read_lammps_data(data_file)
    nx, ny, nz = get_supercell_size(data)

    lammps_npt_input = f"""# Structure
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
variable run_1 equal 40000 #NPT run
variable temp_init equal {temp_init}
variable temp_final equal {temp_final}
variable timestep equal 0.001
variable thermo_freq equal 1
variable dump_freq_1 equal 10
variable temp_damp equal 50.0*${{timestep}}
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

# Initial velocity
velocity    all create ${{temp_init}} ${{velocity_seed}} loop geom

# Energy Minimization (Geometry Optimization)
fix 10 all box/relax aniso 0.0 vmax 0.001
min_style cg
minimize   1.0e-25 1.0e-25 100000 100000
unfix 10
"""
    lammps_npt_input_high_temp = f"""

# Fix for NPT ensemble

# NPT
fix    1 all npt temp ${{temp_init}} ${{temp_final}} ${{temp_damp}} aniso 1.0 1.0 1.0
run    10000
unfix  1

# Equilibrate at high temperature
fix    2 all nvt temp ${{temp_final}} ${{temp_final}} ${{temp_damp}}
run    ${{run_1}} 
unfix  2

fix    3 all nvt temp ${{temp_final}} ${{temp_init}} ${{temp_damp}}
run    5000  # 10 ps
unfix  3

fix    4 all npt temp ${{temp_final}} ${{temp_final}} ${{temp_damp}} aniso 1.0 1.0 1.0
run    ${{run_1}}  
unfix  4

# Write the minimized structure to a file
write_data minimized_structure.dat
"""
    
    lammps_npt_input_room_temp = f"""
# NPT
fix    1 all npt temp ${{temp_init}} ${{temp_final}} ${{temp_damp}} iso 1.0 1.0 1.0
run ${{run_1}}
unfix 1

# Write the minimized structure to a file
write_data minimized_structure.dat
"""

    # Save the LAMMPS input to a file
    with open('in.lammps', 'w') as file:
        file.write(lammps_npt_input)
    if temp_final > temp_init and temp_final > 800:
        with open('in.lammps', 'a') as file:
            file.write(lammps_npt_input_high_temp)
    elif temp_final == temp_init and temp_final < 400:
        with open('in.lammps', 'a') as file:
            file.write(lammps_npt_input_room_temp)
    else:
        print(f"Input temperatures are not correct: {temp_final, temp_init}")
        exit

    
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
