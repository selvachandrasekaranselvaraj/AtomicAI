import warnings

warnings.filterwarnings("ignore")

import numpy as np
import copy, os, pickle
from numba import jit

from ase.calculators.calculator import Calculator, all_changes
from AtomicAI.descriptors.MultiSplit2b3b_index_ss import MultiSplit2b3b_index_ss

from AtomicAI.mlff.mlff import get_mlff
from AtomicAI.descriptors.get_parameter import get_parameters
import time, multiprocessing

class ML(Calculator):
    implemented_properties = ['energy', 'energies', 'forces', 'free_energy']
    implemented_properties += ['stress', 'stresses']  # bulk properties

    nolabel = True

    def __init__(self, timestep = None, forces0 = None, atoms0 = None, **kwargs):
        """
        Parameters
        ----------
        Rc2b:
        Rc3b:
        Reta:
        2b:
        3b:
        split3b:

        """

        self.forces0 = forces0
        self.atoms0 = atoms0
        try:
            self.updatepositions0()
        except:
            pass

        if timestep is not None:
            self.timestep = timestep
        Calculator.__init__(self, **kwargs)

        self.param_dict = get_parameters() #, nfp = set_param_dict(parameters, fp_flag)
        self.nl = None
        if not os.path.isfile('MLFF.obj'):
            self.mlff = get_mlff()
            with open('MLFF.obj', 'wb') as handle:
                pickle.dump(self.mlff, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            with open('MLFF.obj', 'rb') as f:
                self.mlff = pickle.load(f)
                
    # --added 20210903 update positions to on general cell
    def updatepositions0(self):
        cell = self.atoms0.cell
        # print('Updating positions. Cell is:', cell)
        inverse_cell = np.linalg.inv(cell)

        positions = self.atoms0.positions
        fractional_positions = np.dot(positions, inverse_cell)
        convert_positions = fractional_positions % 1.0
        self.atoms0.positions = np.dot(convert_positions, cell)

    def updatepositions(self):
        cell = self.atoms.cell
        # print('Updating positions. Cell is:', cell)
        inverse_cell = np.linalg.inv(cell)

        positions = self.atoms.positions
        fractional_positions = np.dot(positions, inverse_cell)
        convert_positions = fractional_positions % 1.0
        self.atoms.positions = np.dot(convert_positions, cell)

    def storage(self):
        # storage information for calculating potential energy
        self.atoms0 = copy.deepcopy(self.atoms)
        self.forces0 = copy.deepcopy(self.results['forces'])

##################################################################################
    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Update the positions
        self.updatepositions()  # TODO: 20220520 : UNCOMMENT?
        # --added 20210901
        
        natoms = len(self.atoms)
        positions = self.atoms.positions
        cell = self.atoms.cell
        elements = list(self.atoms.symbols)
        mlff = self.mlff

        # define finger print vector variables
        fp_vectors = [[], [], []]
        for i in range(3):
            fp_vectors[i] = [[] for _ in range(natoms)]


        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        #stresses = np.zeros((natoms, 3, 3))

        vd = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        pool = multiprocessing.Pool(4)
        inputs = [(self.atoms, i, vd[vID], self.param_dict) for vID in range(3) for i in range(natoms)]
        fp_vectors_ = np.array(pool.map(MultiSplit2b3b_index_ss, inputs))
        fp_vectors = fp_vectors_.reshape((3, natoms, fp_vectors_.shape[1]))

        #print(fp_vectors.shape)
        #with open('MLFF_17_2:28.pickle', 'rb') as f:
        #    mlff = pickle.load(f)
        ml = [mlff[element]['Lasso_model'] for element in elements]
        vts = [mlff[element]['VT_model'] for element in elements]
        sds = [mlff[element]['SDS_model'] for element in elements]

        for xyz_i in range(3):
            forces[:, xyz_i] = [
                ml[i].predict(sds[i].transform(vts[i].transform(
                    fp_vectors[xyz_i, i].reshape(1, 1440))))
                for i in range(natoms)
            ]

        f = [forces[:, 0], forces[:, 1], forces[:, 2]]
        f = np.array(f).T


        self.results['forces'] = forces
        #print(forces.shape)

        velocities = self.atoms.get_velocities()
        xvec = velocities * self.timestep
        # xvec = self.atoms.positions - self.atoms0.positions
        self.storage() # renew the information of n-1th step
        # take average force (F_{n-1} + F_{n})/2
        fave = 0.5 * (f + self.forces0)
        energies[:] = np.sum(fave * xvec, axis=1)
        self.results['forces'] = forces
        self.results['energies'] = energies
        self.results['energy'] = np.sum(energies)
        #self.energy += np.sum(self.energies)


        velocities = self.atoms.get_velocities()
        xvec = velocities * self.timestep
        fave = 0.5 * (f + self.forces0)
        energies[:] = np.sum(fave * xvec, axis=1)
