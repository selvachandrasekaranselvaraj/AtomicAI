import numpy as np
from matplotlib.pyplot import *
from collections import *
from operator import itemgetter
import sys
from AtomicAI.data.atomic_weight import atomic_weight 
import ase.io

def write_cq_file(ase_data, file_name):
    symbols = list(ase_data.symbols)
    ang2bohr = 1/0.52917
    positions = ase_data.positions*ang2bohr
    data = {}
    cell = ase_data.cell*ang2bohr
    ax, ay, az = cell[0]
    bx, by, bz = cell[1]
    cx, cy, cz = cell[2]
    lat_a = np.linalg.norm(cell[0])
    lat_b = np.linalg.norm(cell[1])
    lat_c = np.linalg.norm(cell[2])
    with open(file_name+'.dat', 'w') as f:
        f.write("  "+(" %12.8f    %12.8f    %12.8f" %(ax, ay, az))+'\n')
        f.write("  "+(" %12.8f    %12.8f    %12.8f" %(bx, by, bz))+'\n')
        f.write("  "+(" %12.8f    %12.8f    %12.8f" %(cx, cy, cz))+'\n')
        f.write("   "+str(len(symbols))+'\n')
        for symbol_ID, symbol_type in enumerate(set(symbols)):
            for symbol, position in zip(symbols, positions):
                if symbol == symbol_type:
                    x, y, z = position[0]/lat_a, position[1]/lat_b, position[2]/lat_c
                    f.write(("%11.6f    %11.6f    %11.6f" %(x, y, z)) + "  " + str(symbol_ID+1) + '  T  T  T \n')

    def define_kpoint(lattice_constant):
        if lattice_constant < 3:
            return 12
        elif 3 < lattice_constant < 4:
            return 8
        elif 4 < lattice_constant < 6:
            return 6
        elif 6 < lattice_constant < 8:
            return 4
        elif 8 < lattice_constant < 14:
            return 2
        elif lattice_constant > 14:
            return 1


    with open('Conquest_input', 'w') as f1:
        f1.write("AtomMove.TypeOfRun                  sqnm  #cg #static  "+'\n')
        f1.write("IO.Title                            " + str(ase_data.symbols) +'\n')
        f1.write("SC.MaxIters                         200"+'\n')
        f1.write("IO.Coordinates                      " + file_name+'.dat'+'\n')
        f1.write("IO.FractionalAtomicCoords           T"+'\n')
        f1.write("General.NumberOfSpecies             " + str(len(set(symbols)))+'\n')
        f1.write("Basis.BasisSet                      PAOs"+'\n')
        f1.write("Grid.GridCutoff                     120"+'\n')
        f1.write("Diag.MPMesh                         T"+'\n')
        f1.write("Diag.MPMeshX                        "+ str(define_kpoint(lat_a*0.53)) + '\n')
        f1.write("Diag.MPMeshY                        "+ str(define_kpoint(lat_b*0.53)) + '\n')
        f1.write("Diag.MPMeshZ                        "+ str(define_kpoint(lat_c*0.53)) + '\n')
        #f1.write("Diag.GammaCentred                  T"+'\n')
        f1.write("%block ChemicalSpeciesLabel"+'\n')
        for symbol_ID, symbol_type in enumerate(set(symbols)):
            f1.write(" " + str(symbol_ID+1)+"  "+atomic_weight[symbol_type]+"  "+symbol_type+'\n')
        f1.write("%endblock"+'\n')
    return
