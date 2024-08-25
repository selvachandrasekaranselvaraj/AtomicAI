import re, os, sys
from ase.io import lammpsdata

def get_atom_style(data_file):
    with open(data_file, 'r') as dfile:
        for line in dfile:
            if 'charge' in line:
                atom_s = False
                kspace = "kspace_style    pppm 1.0e-5 \nkspace_modify   gewald 0.45"
                return 'charge', kspace
         
            elif 'atomic' in line:
                atom_s = False
                kspace = " "
                return 'atomic', kspace
        if not atom_s:
            print("#######ERROR#########")
            print(f"No atom_style found in the {data_file}...")
            exit()

def get_elements(data_file):
    data = lammpsdata.read_lammps_data(data_file)
    return list(data.symbols.formula.count().keys())

def generate_lammps_nvt_inputs():
    try:
        temp = sys.argv[1]
    except:
        print("Input error.")
        print("lammps_nvt_input temprature")
        exit()

    data_file = 'minimized_structure.dat'
    if os.path.isfile(data_file):
        print(f"{data_file} exists in the directory.")
    else:
        print(f"{data_file} does not exist in the directory.")
        exit()

    elements = ' '.join(get_elements(data_file))
    atom_style, kspace = get_atom_style(data_file)
    lammps_input_content = f"""# Structure
units           metal
boundary        p p p
atom_style      {atom_style}

# Variables
variable read_data_file string "{data_file}"
variable pair_style_type string "mlpot.dp"
variable dump_file1 string "dump_unwrapped.lmp"
variable dump_file2 string "dump.lmp"

# Numeric Variables
variable nx equal 1
variable ny equal 1
variable nz equal 1
variable run_1 equal 3000000 #NVT run
variable temp_init equal {temp}
variable temp_final equal {temp}
variable timestep equal 0.001
variable thermo_freq equal 1
variable dump_freq_1 equal 200
variable temp_damp equal 50.0*${{timestep}}
variable velocity_seed equal 87287
variable neighbor_distance equal 1.0
variable neigh_modify_every equal 20
variable neigh_modify_delay equal 0
variable neigh_modify_check equal "no"

read_data       ${{read_data_file}} 
replicate       ${{nx}} ${{ny}} ${{nz}}

pair_style	deepmd    ${{pair_style_type}}
pair_coeff	 * * 

{kspace}
#dielectric      4.0

neighbor        ${{neighbor_distance}} bin
neigh_modify    every ${{neigh_modify_every}} delay ${{neigh_modify_delay}} check ${{neigh_modify_check}}

# Timestep
timestep        ${{timestep}}

# Output settings
thermo ${{thermo_freq}}
thermo_style custom step time temp press pe vol 

# Dump settings
dump 1 all custom ${{dump_freq_1}} ${{dump_file1}} id type xu yu zu element
dump_modify 1 every ${{dump_freq_1}} element {elements} 
dump 2 all custom ${{dump_freq_1}} ${{dump_file2}} id type x y z element
dump_modify 2 every ${{dump_freq_1}} element {elements} 

# Initial velocity
velocity    all create ${{temp_init}} ${{velocity_seed}} loop geom

# NVT Runs
fix 1 all nvt temp ${{temp_init}} ${{temp_final}} ${{temp_damp}}
run ${{run_1}}
unfix 1

# Write the nvt structure to a file
write_data after_nvt_.dat
"""

    # Save the LAMMPS input to a file
    with open('in.lammps', 'w') as file:
        file.write(lammps_input_content)
    
    print("LAMMPS input file generated successfully.")
    
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
    
    # Save the sub_file_content to a file
    with open('improv.sh', 'w') as file:
        file.write(sub_file_content)
    
    print("improv.sh file generated successfully.")

    # Check all input files
    # Define the directory to check
    directory = "./"
    
    # List of files to check
    files_to_check = ['minimized_structure.dat', 'mlpot.dp', 'in.lammps', 'improv.sh']
    
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
