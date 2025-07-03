import socket

def job_submit(job_name):
    hostname = socket.gethostname()

    improv_file_content = f"""#!/bin/bash
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -A LTC
#PBS -l walltime=72:00:00
#PBS -N {job_name}
#PBS -j n
#PBS -m e

cd $PBS_O_WORKDIR
NNODES=`wc -l < $PBS_NODEFILE`
echo "NNODES=" $NNODES

module add gcc/13.2.0 openmpi/4.1.6-gcc-13.2.0 aocl/4.1.0-gcc-13.1.0
export PATH=/soft/software/custom-built/vasp/5.4.4/bin:$PATH
export UCX_NET_DEVICES=mlx5_0:1

#mpirun -np $NNODES vasp_std
mpirun -np $NNODES /home/schandrasekaran/myopt/improv/lammps/bin/lmp_mpi -in in.lammps
#autopsy dump_unwrapped.lmp #--atoms Li
"""

    bebop_file_content = f"""#!/bin/bash -l
#PBS -A LIO2SS
#PBS -l select=1:mpiprocs=36
#PBS -l walltime=4:00:00
#PBS -N {job_name}
#PBS -j n
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
NNODES=`wc -l < $PBS_NODEFILE`
echo "NNODES=" $NNODES

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
#echo Running on nodes `cat $PBS_NODEFILE`

ulimit -s unlimited

module load  binutils/2.42   gcc/11.4.0   openmpi/5.0.3-gcc-11.4.0

mpirun -np $NNODES /home/schandrasekaran/myopt/bebop/test/lammps/src/lmp_mpi <in.lammps

#autopsy dump_unwrapped.lmp #--atoms Li
"""

    nrel_file_content = f"""#!/bin/sh
#SBATCH -J {job_name}
#SBATCH --ntasks-per-node=104
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -A ltctest
#SBATCH --error=%J.stderr
#SBATCH --output=%J.stdout
ulimit -s unlimited
#module add gcc/9.2.0-pkmzczt
#module add intel-parallel-studio/cluster.2020.2-y7ijupg
mpirun -np $SLURM_NTASKS ~/myopt/lammps/src/lmp_mpi < in.lammps
#autopsy dump_unwrapped.lmp #--atoms Li
"""
    if "bebop" in hostname:
        # Save the sub_file_content to a file
        with open('sub.sh', 'w') as file:
            file.write(bebop_file_content)
        print("sub.sh file generated successfully.")

    elif 'kl' in hostname and len(hostname) == 3:
        # Save the sub_file_content to a file
        with open('sub.sh', 'w') as file:
            file.write(nrel_file_content)
        print("sub.sh file generated successfully.")

    elif 'ilogin' in hostname:
        # Save the sub_file_content to a file
        with open('sub.sh', 'w') as file:
            file.write(improv_file_content)
        print("sub.sh file generated successfully.")
    else:
        print(f"{hostname} is the hostname. So, no sub.sh file created")
        exit
    
    return

