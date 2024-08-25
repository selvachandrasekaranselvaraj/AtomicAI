import random, math
import numpy as np
# make random vector for force projection
def prepare_vforce(no_of_data):
    vforce = []
    pi = np.pi
    for _ in range(no_of_data):
        zr = random.uniform(0.0, 1.0) * 2 - 1
        pr = random.uniform(0.0, 1.0) * 2 * pi

        vx = math.sqrt(1 - zr**2) * math.cos(pr)
        vy = math.sqrt(1 - zr**2) * math.sin(pr)
        vz = zr

        vforce.append([vx, vy, vz])

    return vforce
