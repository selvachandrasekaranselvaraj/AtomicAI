# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")  
import os, sys
import ase.io
import numpy as np
from  AtomicAI.tools.angle_between_three_points import angle_between_three_points

###############################################################
def angles_in_tetrahedron(position, nn_atoms):
    angle = []
    for i in range(len(nn_atoms)):
        for j in range(i+1, len(nn_atoms)):
            angle.append(angle_between_three_points(nn_atoms[i], position, nn_atoms[j]))
    return np.array(angle)
###############################################################
