#!/usr/bin/env python3.10
from AtomicAI.io.read import read
from AtomicAI.io.write_cq import write_cq_file
import os, sys
import ase.io
import numpy as np
import pandas as pd
def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0])
    comment = lines[1].strip()
    atoms = []
    coords = []
    for line in lines[2:2 + natoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords), comment

def write_poscar(atoms, coords, comment='Generated from XYZ', padding=2.0, output='input.vasp'):
    unique_elements = sorted(set(atoms), key=atoms.index)
    counts = [atoms.count(el) for el in unique_elements]

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    lengths = max_coords - min_coords + padding

    shifted_coords = coords - min_coords

    a = [lengths[0], 0.0, 0.0]
    b = [0.0, lengths[1], 0.0]
    c = [0.0, 0.0, lengths[2]]

    with open(output, 'w') as f:
        f.write(f"{comment}\n")
        f.write("1.0\n")
        f.write(f"{a[0]:>12.6f} {a[1]:>12.6f} {a[2]:>12.6f}\n")
        f.write(f"{b[0]:>12.6f} {b[1]:>12.6f} {b[2]:>12.6f}\n")
        f.write(f"{c[0]:>12.6f} {c[1]:>12.6f} {c[2]:>12.6f}\n")
        f.write("  " + "  ".join(unique_elements) + "\n")
        f.write("  " + "  ".join(str(x) for x in counts) + "\n")
        f.write("Cartesian\n")
        for coord in shifted_coords:
            f.write(f"{coord[0]:>12.6f} {coord[1]:>12.6f} {coord[2]:>12.6f}\n")

    print(f"Written: {output} with {len(atoms)} atoms [{', '.join(unique_elements)}]")

def xyz2vasp():
    if len(sys.argv) != 2:
        print("Usage: python script.py input.xyz")
        sys.exit(1)

    xyz_file = sys.argv[1]
    if not os.path.isfile(xyz_file):
        print(f"Error: File '{xyz_file}' not found.")
        sys.exit(1)

    try:
        data = ase.io.read(xyz_file)
        print(np.array(data.cell)) 
        elements, positions = list(data.symbols), data.positions.T
        array = np.array([elements, positions[0], positions[1], positions[2]])
        df = pd.DataFrame(array.T, columns=['Sy', 'x', 'y', 'z'])
        df.index = elements
        df = df.sort_index()
        data.symbols = np.array(df['Sy'])
        array = np.array([list(df['x']), list(df['y']), list(df['z'])]).T
        data.set_positions(array)
        out_file = xyz_file[:-3]+'vasp'
        ase.io.write(out_file, data, format='vasp')
        return
    except:
        atoms, coords, comment = read_xyz(xyz_file)
        base_name = os.path.splitext(xyz_file)[0]
        output_file = base_name + '.vasp'
        write_poscar(atoms, coords, comment=comment, output=output_file)
