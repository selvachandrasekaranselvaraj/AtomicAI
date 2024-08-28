import socket

def job_submit(job_name):
    hostname = socket.gethostname()

    improv_file_content = f"""#!/bin/bash
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
mpirun -np $NNODES /home/schandrasekaran/myopt/improv/lammps/bin/lmp_mpi -in in.lammps
"""

    bebop_file_content = f"""#!/bin/bash -l
#PBS -A LIO2SS
#PBS -l select=2:mpiprocs=36
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
"""

    if "bebop" in hostname:
        # Save the sub_file_content to a file
        with open('sub.sh', 'w') as file:
             file.write(bebop_file_content)
    else:
        # Save the sub_file_content to a file
        with open('sub.sh', 'w') as file:
             file.write(improv_file_content)

    print("sub.sh file generated successfully.")
    return

