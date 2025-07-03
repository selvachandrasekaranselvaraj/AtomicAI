#!/usr/bin/env python3
import sys, os, re, shutil, socket
import numpy as np
from glob import glob
from ase.io import read, write
from ase.build import make_supercell
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances


hostname = socket.gethostname()
if hostname.startswith("bebop"):
    base_dir = "/lcrc/project/LiO2SS/selva/dft_for_mlff/database"
else:
    base_dir = "/Users/selva/workspace/mlff/database" 
lda_u_elements = {"V": 3.1, "Cr": 3.5, "Mn": 4.0, "Fe": 4.0, "Co": 3.3, "Ni": 6.2, "Ce": 5.0, "U": 4.5}
incar_base = """SYSTEM = {system}
ENCUT = {encut}
EDIFF = 1E-5
ISMEAR = 0
SIGMA = 0.05
IBRION = 2
ISIF = 3
NSW = 2000
PREC = Accurate
LREAL = Auto
ALGO = Normal
LWAVE = .FALSE.
LCHARG = .FALSE.

{lda_u_section}
"""
lda_u_template = """LDAU = .TRUE.
LDAUTYPE = 2
LDAUL = {ldau_l}
LDAUU = {ldau_u}
LDAUJ = {ldau_j}
LDAUPRINT = 1
"""

def expand_supercell_if_needed(poscar_file, min_natoms=90, max_natoms=160):
    structure = read(poscar_file, format="vasp")
    natoms = len(structure)
    if natoms >= min_natoms:
        return poscar_file, natoms
    scaling_matrix = [1, 1, 1]
    cell_lengths = np.linalg.norm(structure.get_cell(), axis=1)
    while True:
        est_natoms = natoms * np.prod(scaling_matrix)
        if min_natoms <= est_natoms <= max_natoms:
            break
        if est_natoms > max_natoms:
            break
        min_axis = np.argmin(cell_lengths * scaling_matrix)
        scaling_matrix[min_axis] += 1
    structure = make_supercell(structure, np.diag(scaling_matrix))
    sorted_indices = np.argsort(structure.get_chemical_symbols())
    structure_sorted = structure[sorted_indices]
    expanded_file = poscar_file + "_expanded"
    write(expanded_file, structure_sorted, format="vasp")
    print(f"[INFO] POSCAR expanded with scaling matrix {scaling_matrix} -> {len(structure_sorted)} atoms.")
    return expanded_file, len(structure_sorted)

def make_output_folder(elements, quantities):
    name = "".join(f"{el}{qty}" for el, qty in sorted(zip(elements, quantities)))
    path = os.path.join(base_dir, name)
    if os.path.exists(path):
        print(f"\033[91m[WARNING] Folder {path} already exists.\033[0m")
        print("Exits...")
        #print(f"Check {path} before creating one.")
        print(f"\033[91m***************************************\033[0m")
        return None # path
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Created folder: {path}")
    return path

def read_poscar_elements(poscar_file):
    with open(poscar_file) as f: lines = f.readlines()
    return lines[5].split()

def read_poscar_quantities(poscar_file):
    with open(poscar_file) as f: lines = f.readlines()
    return list(map(int, lines[6].split()))

def generate_ldau_section(elements):
    l, u, j = [], [], []
    for el in elements:
        if el in lda_u_elements:
            l.append("2" if el not in ["Ce", "U"] else "3")
            u.append(str(lda_u_elements[el]))
            j.append("0.0")
        else:
            l.append("-1")
            u.append("0.0")
            j.append("0.0")
    if any(el in lda_u_elements for el in elements):
        return lda_u_template.format(ldau_l=" ".join(l), ldau_u=" ".join(u), ldau_j=" ".join(j))
    return ""

def generate_potcar(elements, folder):
    if hostname.startswith("bebop"):
        potcar_path = "/lcrc/soft/vasp/potpaw_PBE.64"
    else:
        potcar_path = "/Users/selva/myopt/vasp/potential/potpaw_PBE" #"/lcrc/soft/vasp/potpaw_PBE.64"

    with open(os.path.join(folder, "POTCAR"), "wb") as fout:
        for el in elements:
            fpath = os.path.join(potcar_path, el, "POTCAR")
            if not os.path.exists(fpath):
                fpath = os.path.join(potcar_path, f"{el}_sv", "POTCAR")
            if not os.path.exists(fpath):
                print(f"[ERROR] POTCAR for {el} not found!")
                sys.exit(1)
            with open(fpath, "rb") as f: fout.write(f.read())

def get_max_enmax_from_potcar(potcar_path):
    vals = []
    with open(potcar_path) as f:
        for line in f:
            if "ENMAX" in line:
                nums = re.findall(r"[\d.]+", line)
                if nums: vals.append(float(nums[0]))
    if not vals:
        print("[ERROR] No ENMAX found!")
        sys.exit(1)
    max_enmax = max(vals)
    if max_enmax < 200: return max_enmax * 1.5
    if max_enmax < 350: return max_enmax * 1.3
    if max_enmax < 400: return max_enmax * 1.2
    return max_enmax * 1.08

def vector_magnitude(x, y, z): return np.sqrt(x**2 + y**2 + z**2)
def get_kpoints(mag):
    if mag <= 2: return 12
    elif mag <= 3: return 10
    elif mag <= 4: return 8
    elif mag <= 5: return 6
    elif mag <= 6: return 5
    elif mag <= 7: return 4
    elif mag <= 10: return 3
    return 1

def generate_kpoints(poscar_file, folder):
    with open(poscar_file) as f: lines = f.readlines()
    a = list(map(float, lines[2].split()))
    b = list(map(float, lines[3].split()))
    c = list(map(float, lines[4].split()))
    kx, ky, kz = get_kpoints(vector_magnitude(*a)), get_kpoints(vector_magnitude(*b)), get_kpoints(vector_magnitude(*c))
    with open(os.path.join(folder, "KPOINTS"), "w") as f:
        f.write(f"Auto KPOINTS\n0\nMP\n{kx} {ky} {kz}\n0 0 0\n")

def write_submission_script(folder):
    hostname = socket.gethostname()
    if hostname.startswith("bebop"):
        sub_script = """#!/bin/bash -l
#PBS -A LiTFSi-LiPF6  
#PBS -l select=1:mpiprocs=36
#PBS -l walltime=72:00:00
#PBS -N test1
#PBS -j n

#  Availabel projects for comupting hours: LiTFSi-LiPF6 hexagon  SSE-LiO2   hydration_surface
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
NNODES=wc -l < $PBS_NODEFILE
echo "NNODES=" $NNODES

echo Jobid: $PBS_JOBID
echo Running on host hostname
#echo Running on nodes cat $PBS_NODEFILE

ulimit -s unlimited

module load  binutils/2.42   gcc/11.4.0   openmpi/5.0.3-gcc-11.4.0 vasp/6.4.3
mpirun -np $NNODES vasp_std
"""
    else:
        sub_script = """#!/bin/bash
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -A AlN
#PBS -l walltime=72:00:00
#PBS -N lgpslco
##PBS -o vasp.out
#PBS -j n
#PBS -m e

cd $PBS_O_WORKDIR
NNODES=wc -l < $PBS_NODEFILE
echo "NNODES=" $NNODES

module add gcc/13.2.0 openmpi/4.1.6-gcc-13.2.0 aocl/4.1.0-gcc-13.1.0
export PATH=/soft/software/custom-built/vasp/5.4.4/bin:$PATH
export UCX_NET_DEVICES=mlx5_0:1

mpirun -np $NNODES vasp_std
"""
    script_path = os.path.join(folder, "sub.sh")
    with open(script_path, "w") as f:
        f.write(sub_script)
    os.chmod(script_path, 0o755)  # Make it executable



def find_safe_midpoint(point, vertices, cell, pbc):
    """
    Move the given point away from its nearest three vertices if it's too close.
    The movement is directed along the normal to the plane defined by the three atoms.
    """
    def nearest_three(p):
        dists = get_distances(p, vertices, cell, pbc=pbc)[1][0]
        nearest_indices = np.argsort(dists)[:3]
        return dists[nearest_indices], vertices[nearest_indices]

    point = point.copy()
    best_point = point.copy()
    best_min_dist = 0.0

    for _ in range(2000):  # Limit iterations
        dists, nearest_points = nearest_three(point)
        min_dist = dists[0]
        a, b, c = nearest_points

        # Define two vectors in the plane of the three nearest atoms
        ab = b - a
        ac = c - a
        ap = point - a
        bp = point - b
        # Normal vector to the plane
        normal = np.cross(ab, ap)
        norm = np.linalg.norm(normal)
        if norm == 0:
            normal = np.random.rand(3)
            norm = np.linalg.norm(normal)
        unit_vector = normal / norm

        delta = 0.10  # step size
        point = (point + unit_vector * delta) % cell.diagonal()
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_point = point.copy()
        else:
            break

    return best_point, best_min_dist, _

def setup_aimd_folders(base_folder, vc_only=False):
    vc_dir = os.path.join(base_folder, "vc")
    os.makedirs(vc_dir, exist_ok=True)
    shutil.move(os.path.join(base_folder, "POSCAR"), os.path.join(vc_dir, "POSCAR"))
    elements = read_poscar_elements(os.path.join(vc_dir, "POSCAR"))
    quantities = read_poscar_quantities(os.path.join(vc_dir, "POSCAR"))
    generate_potcar(elements, vc_dir)
    encut = get_max_enmax_from_potcar(os.path.join(vc_dir, "POTCAR"))
    incar = incar_base.format(system="-".join(elements), encut=f"{encut:.1f}", lda_u_section=generate_ldau_section(elements))
    with open(os.path.join(vc_dir, "INCAR"), "w") as f: f.write(incar)
    generate_kpoints(os.path.join(vc_dir, "POSCAR"), vc_dir)
    write_submission_script(vc_dir)

    contcar = os.path.join(vc_dir, "POSCAR")
    if not os.path.isfile(contcar):
        print(f"[WARNING] CONTCAR not found in {vc_dir}. Run VC calculation first.")
        return

    if vc_only:
        return  # Stop after VC setup

    # === AIMD folders ===
    for name, scale in [("aimd_300", 1.00), ("c5_aimd_300", 0.95), ("e5_aimd_300", 1.05)]:
        run_dir = os.path.join(base_folder, name)
        shutil.copytree(vc_dir, run_dir, dirs_exist_ok=True)
        poscar = os.path.join(run_dir, "POSCAR")
        shutil.copy(contcar, poscar)
        if scale != 1.00:
            with open(poscar) as f: lines = f.readlines()
            lines[1] = f"{scale:.4f}\n"
            with open(poscar, "w") as f: f.writelines(lines)
        generate_potcar(read_poscar_elements(poscar), run_dir)
        encut = get_max_enmax_from_potcar(os.path.join(run_dir, "POTCAR"))
        incar = incar_base.format(system="-".join(elements), encut=f"{encut:.1f}", lda_u_section=generate_ldau_section(elements))
        incar = incar.replace("ISIF = 3", "ISIF = 2")
        incar = incar.replace("IBRION = 2", "IBRION = 0") + "\nTEBEG = 200\nTEEND  = 500\nSMASS = 0\nPOTIM = 2.0\nISYM = 0\n"
        with open(os.path.join(run_dir, "INCAR"), "w") as f: f.write(incar)
        generate_kpoints(poscar, run_dir)
        write_submission_script(run_dir)

    # === Surface folder ===
    run_dir = os.path.join(base_folder, "surface")
    os.makedirs(os.path.join(run_dir), exist_ok=True)
    structure = read(contcar, format="vasp")
    cell = structure.get_cell()
    cell[2, 2] += 12.0  # add vacuum along c-axis
    structure.set_cell(cell, scale_atoms=False)
    write(os.path.join(run_dir, "POSCAR"), structure, format="vasp")
    generate_potcar(read_poscar_elements(poscar), run_dir)
    with open(os.path.join(run_dir, "INCAR"), "w") as f: f.write(incar)
    generate_kpoints(poscar, run_dir)
    write_submission_script(run_dir)

    # === Defect folder ===
    run_dir = os.path.join(base_folder, "defect")
    os.makedirs(os.path.join(run_dir), exist_ok=True)
    structure = read(contcar, format="vasp")

    symbols = structure.get_chemical_symbols()
    unique_elements = sorted(set(symbols))
    cell = structure.get_cell()
    pbc = structure.get_pbc()
    cell_lengths = np.linalg.norm(cell, axis=1)
    positions = structure.positions.copy()
    moved = []
   
    # Move 1 atom of each element
    for el in unique_elements:
        moved_count = 0
        for i, sym in enumerate(symbols):
            if sym == el and moved_count < 1:
                pos = positions[i].copy()
                offset = np.random.uniform(4.0, 5.0, size=3)
                positions[i] = (pos + offset) % cell.diagonal()
                print(f"[INFO] {el}: {pos} → {positions[i]} (initial defect move)")
                moved.append(i)
                moved_count += 1
   
    # Check and adjust short distances
    for idx in moved:
        pos_i = positions[idx]
        dists = get_distances(pos_i, positions, cell, pbc=pbc)[1][0]
        nearest_indices = np.argsort(dists)[1:11]
        nearest_positions = positions[nearest_indices]
        best_point, best_min_dist, _ = find_safe_midpoint(pos_i, nearest_positions, cell, pbc)
        positions[idx] = best_point.copy()
        print(f"[FIXED] Atom {idx} moved to avoid short distance: min_dist = {best_min_dist:.3f} with {_}th step")



    structure.set_positions(positions)
    write(os.path.join(run_dir, "POSCAR"), structure, format="vasp")
    generate_potcar(read_poscar_elements(poscar), run_dir)
    with open(os.path.join(run_dir, "INCAR"), "w") as f: f.write(incar)
    generate_kpoints(poscar, run_dir)
    write_submission_script(run_dir)
"""
if __name__=="__main__":
    input_files = sys.argv[1:] if len(sys.argv) > 1 else glob("*.vasp")
    if not input_files:
        print("[ERROR] No input files given or found (*.vasp).")
        sys.exit(1)

    for input_file in input_files:
        if input_file in ["POSCAR","CONTCAR"] or input_file.endswith(".vasp"):
            if not os.path.isfile(input_file):
                print(f"[ERROR] {input_file} not found, skipping.")
                continue

            print(f"\n=== Processing {input_file} ===")

            # Step 1: Expand supercell if needed (returns expanded POSCAR file path)
            expanded_file, natoms = expand_supercell_if_needed(input_file)
            print(f"[INFO] Total atoms after possible expansion: {natoms}")

            # Step 2: Read elements and quantities from the expanded POSCAR
            elements = read_poscar_elements(expanded_file)  # you must have this function defined
            quantities = read_poscar_quantities(expanded_file)  # you must have this function defined
            print(f"[INFO] Elements: {elements}, Quantities: {quantities}")

            # Step 3: Create unique output folder (skip if exists)
            target_folder = make_output_folder(elements, quantities)
            if target_folder is None:
                if expanded_file:
                    os.remove(expanded_file)
                continue  # skip if folder already exists

            # Step 4: Copy expanded POSCAR into target folder
            shutil.move(expanded_file, os.path.join(target_folder, "POSCAR"))

            # Step 5: Generate inputs directly into target folder
            setup_aimd_folders(target_folder)

        else:
            print(f"[WARNING] {input_file} is not POSCAR/CONTCAR/.vasp, skipping.") 
"""
