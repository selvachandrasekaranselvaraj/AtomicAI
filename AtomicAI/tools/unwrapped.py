import sys
from ovito.io import import_file
from ovito.io import export_file
from ovito.modifiers import UnwrapTrajectoriesModifier

def unwrapped():

    filenames = []
    for i in range(1, 10):
        try:
            filenames.append(sys.argv[i])
        except IndexError:
            if i == 1:
                print("No dump.lmp file is available HERE!!!")
                print("Usage: python wrapped1.lmp wrapped2.lmp wrpped3.lmp ...")
                exit()
    for traj_file in filenames:
        # Load a simulation trajectory:
        pipeline = import_file(traj_file,  columns = ["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])
        
        # Insert the unwrap modifier into the pipeline. 
        # Note that the modifier should typically precede any other modifiers in the pipeline. 
        pipeline.modifiers.append(UnwrapTrajectoriesModifier())
        
        # For demonstration purposes, request last frame of the trajectory. 
        # The returned DataCollection will contain modified particle coordinates, which are computed 
        # by the unwrap modifier by tracing the trajectories of the particles and unfolding
        # them whenever crossings of the periodic cell boundaries are detected.
        data = pipeline.compute(pipeline.source.num_frames - 1)
        #data = pipeline.compute()
        # Define the old file path
        old_file = traj_file.split("/")[-1]
        new_file = f"unwrapped_{old_file}"


        # Define the new file path by replacing 'dump.lmp' with 'unwrapped.lmp'
        new_file_path = traj_file.replace(old_file, new_file)
        
        export_file(pipeline, new_file_path, "lammps/dump", columns = ["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"], multiple_frames=True)
    return 
