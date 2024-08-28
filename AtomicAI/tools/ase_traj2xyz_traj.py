from ase.io.trajectory import Trajectory
from ase.io import write
import sys

def ase_traj2xyz_traj():
    try:
        traj_file = sys.argv[1]
    except:
        print("Input error!!!!")
        print("Usage: \"ase_traj2xyz_traj ase trajectory file with .traj extension\"")
        print()
        exit()

    frames = Trajectory(traj_file)
    write(traj_file[:-4]+'xyz', frames, format='extxyz')
