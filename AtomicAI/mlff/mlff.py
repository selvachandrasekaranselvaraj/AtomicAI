import random
import numpy as np
import pandas as pd
from AtomicAI.descriptors.MultiSplit2b3b_index_ss import MultiSplit2b3b_index_ss
from AtomicAI.mlff.select_data_from_trajectory import select_data_from_trajectory
from AtomicAI.descriptors.prepare_vforce import prepare_vforce
from AtomicAI.descriptors.get_parameter import get_parameters
from AtomicAI.mlff.LassoLarCV import LassoLarCV_VT_SDS
from AtomicAI.mlff.plot_vt_r2 import plot_line

from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def get_mlff():  # traj_file name
    selected_frames, MLFF = select_data_from_trajectory()
    param_dict = get_parameters()
    for element in MLFF.keys():
        no_of_data = MLFF[element]['Number of data']

        # initialize random seed
        seed_ran = 16835
        seed_rand = seed_ran
        random.seed(seed_rand)
        num_frames = len(selected_frames)
        vforce = prepare_vforce(no_of_data)
        #random_choice_time = time() - before_time

        diter = 1000
        # collect results of fingerprint
        Xv, Fv = [], []
        ID_vector = 0
        diter_local = 0
        vforce_no = 0
        for selected_frame_index, selected_atoms_indices in zip(
                MLFF[element]['Selected_frames_indices'],
                MLFF[element]['Selected_atoms_indices']):
            atoms_local = selected_frames[selected_frame_index]
            cell = atoms_local.cell
            numbers = atoms_local.numbers
            symbols = list(atoms_local.symbols)
            positions = atoms_local.positions
            atomicforces = atoms_local.arrays.get('momenta')
            if atomicforces is None:
               atomicforces = atoms_local.arrays.get('forces')
            natoms = len(numbers)

            """
            if ID_vector >= diter_local:
                print(element, 'Process : %d' % ID_vector)
                diter_local += diter


            for selected_atom_index in selected_atoms_indices:
                Force_xyz = atomicforces[selected_atom_index]
                Fvtmp = np.dot(
                    Force_xyz, vforce[vforce_no]
                )  # Fv : atomic force projection along the random vector

                # prepare feature: fingerprint value(descriptor value)
                Xvtmp = []

                inputs = (atoms_local, selected_atom_index, vforce[vforce_no], param_dict)
                Xvtmp = MultiSplit2b3b_index_ss(inputs)
                vforce_no += 1
                Xv.append(Xvtmp)
            natoms = len(numbers)
            """

            if ID_vector >= diter_local:
                print(element, 'Process : %d' % ID_vector)
                diter_local += diter


            for selected_atom_index in selected_atoms_indices:
                Force_xyz = atomicforces[selected_atom_index]
                Fvtmp = np.dot(
                    Force_xyz, vforce[vforce_no]
                )  # Fv : atomic force projection along the random vector

                # prepare feature: fingerprint value(descriptor value)
                Xvtmp = []

                inputs = (atoms_local, selected_atom_index, vforce[vforce_no], param_dict)
                Xvtmp = MultiSplit2b3b_index_ss(inputs)
                vforce_no += 1
                Xv.append(Xvtmp)
                Fv.append(Fvtmp)
                # update id of force vector
                ID_vector += 1
        print(np.array(Xvtmp).shape)
        descriptors = {'Descriptors': Xv, 'Forces': Fv}
        MLFF[element].update(descriptors)
        MLFF[element].update(param_dict)

    ###############################################################
    for element in MLFF.keys():
        descriptor = np.array(MLFF[element]['Descriptors'])
        labels = np.array(MLFF[element]['Forces'])
        no_of_data = descriptor.shape[0]
        print(f'No. of features in {element} are: {descriptor.shape[1]} and no of data are {no_of_data}')
        data_split_at = int(no_of_data * 0.8)
        train_features = descriptor[0:data_split_at]
        test_features = descriptor[data_split_at:no_of_data]
        train_labels = labels[0:data_split_at]
        test_labels = labels[data_split_at:no_of_data]
        df = pd.DataFrame(train_features)
        variances = list(df.var().sort_values())
        #print(min(variances), max(variances))
        r2, no_of_dimensions_vt = [], []
        vts = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-20, 1e-40, 0.0] # variances[:-2]
        for variance in vts:
            predicted_test_lables, no_of_dimensions_vt_ = LassoLarCV_VT_SDS(
                train_features, test_features, train_labels, variance)
            r2.append(round(r2_score(test_labels, predicted_test_lables), 3))
            no_of_dimensions_vt.append(no_of_dimensions_vt_)
        data_ = {'No of dimensions after VT': no_of_dimensions_vt,
                 'Predictd accuracy for VTs': r2,
                 'Variances' : vts,
                }
        MLFF[element].update(data_)
        no_of_dimensions_vt = MLFF[element]['No of dimensions after VT']
        r2 = MLFF[element]['Predictd accuracy for VTs']
        vts = MLFF[element]['Variances']
        plot_line(vts, no_of_dimensions_vt, r2, element)
    ###############################################################
    for element in MLFF.keys():
        r2 = np.array(MLFF[element]['Predictd accuracy for VTs'])
        vts = np.array(MLFF[element]['Variances'])
        optimized_vt = {'optimized_vt': vts[r2==max(r2)]}
        MLFF[element].update(optimized_vt)
        descriptors = np.array(MLFF[element]['Descriptors'])

        # VT
        vt = VarianceThreshold(threshold=MLFF[element]['optimized_vt'][0])
        descriptos_vt = vt.fit_transform(descriptors)

        # SDS
        sds = StandardScaler()   # with_mean=False)
        descriptos_vt_sds = sds.fit_transform(descriptos_vt)

        # Create regression object
        lasso = LassoLarsCV(fit_intercept=False, max_iter=5000)
        lasso.fit(descriptos_vt_sds, np.array(MLFF[element]['Forces']))

        # Models
        models = {
            'VT_model' : vt,
            'SDS_model': sds,
            'Lasso_model': lasso

        }

        MLFF[element].update(models)

        forces = np.array(MLFF[element]['Forces'])
        lasso_model = MLFF[element]['Lasso_model']
        sds_model =  MLFF[element]['SDS_model']
        vt_model =  MLFF[element]['VT_model']
        descriptors = np.array(MLFF[element]['Descriptors'])
        predicted_forces = lasso_model.predict(sds_model.transform(vt_model.transform(descriptors)))
        print(round(r2_score(forces, predicted_forces), 3))

    return MLFF
