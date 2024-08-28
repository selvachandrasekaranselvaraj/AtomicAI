import math
import numpy as np
from numba import jit

@jit
def define_eta(eta_range,num):
    # eta_select: [eta_min, eta_max, eta_num]
    R0 = eta_range[0]
    R1 = eta_range[1]
    eta_num = int(num)
    eta = np.array([R0])

    for _ in range(eta_num - 1):
        eta = np.append(eta, np.array([(R1 / R0) ** (1 / eta_num) * eta[-1]]))
    return eta
