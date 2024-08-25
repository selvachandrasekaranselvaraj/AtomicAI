import math
import numpy as np
from numba import jit
def define_mirror_cubic(position, cell, Rc):
    m_min = [0,0,0]
    m_max = [0,0,0]
    for i in range(len(position)):
        if(position[i] < Rc): m_min[i] = -1
        if(cell[i,i]-position[i]) < Rc : m_max[i] = 1
    m_x = []
    m_y = []
    m_z = []
    for i in range(m_min[0], m_max[0]+1):
        for j in range(m_min[1], m_max[1] + 1):
            for k in range(m_min[2], m_max[2] + 1):
                m_x.append(i* cell[0,0])
                m_y.append(j* cell[1,1])
                m_z.append(k* cell[2,2])

    return m_x, m_y, m_z
