import warnings
warnings.filterwarnings("ignore")
import sys, os
import numpy as np
from AtomicAI.data.descriptor_cutoff import descriptor_cutoff
from AtomicAI.dim_reduction.select_descriptors_and_manual_labels import select_descriptors_and_manual_labels
from AtomicAI.dim_reduction.lpp import lpp
from AtomicAI.dim_reduction.tslpp import tslpp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

def dim_reduction():
    '''

    '''
    in_dir = './descriptors/'
    des_files = sorted([f for f in os.listdir(in_dir) if '.dat' in f])
    if len(des_files) > 0:
        print(f"Availabel files are \n {', '.join(des_files)}")
    else:
        print("No des_file.dat file is availabel HERE!!!")
        exit()
    out_directory = './dim_reduction/'
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)
    rd = descriptor_cutoff['Si_Si']
    ra = [1.0] + rd
    descriptors = ['G2', 'G2G4', 'SOAP']
    dim_reductions = [2]
    dim_reduc_models = ['UMAP']#['PCA', 'TsNE', 'UMAP', 'LPP', 'TsLPP']
    count = 1 
    for reduced_dim in dim_reductions:
        for dim_reduc_model in dim_reduc_models:
            for descriptor in descriptors:
                variables = (reduced_dim, dim_reduc_model, descriptor, ra, rd)
                outfile_labels = []
                outfile = f'{out_directory}{str(reduced_dim)}D_{dim_reduc_model}_{descriptor}.dat'
                print(f'Running {outfile}')
                for a in ra:
                    for d in rd:
                       #try:
                            des_file = f'{descriptor}_{d}_{a}_Si_Si.dat'
                            if_label, selected_descriptors = select_descriptors_and_manual_labels(in_dir+des_file, count)
                            count += 1
                            train_features = selected_descriptors[0]
                            print(des_file, np.shape(train_features))
                            train_features_vt = VarianceThreshold(threshold=1e-8).fit_transform(train_features)
                            train_features_vt_sds = StandardScaler().fit_transform(train_features_vt)


                            if dim_reduc_model == 'PCA':
                                model = PCA(n_components = reduced_dim)
                                dim_reduc = model.fit_transform(train_features_vt_sds).transpose()
                                outfile_labels.extend((f'{a}_{d}_D1', f'{a}_{d}_D2'))

                            elif dim_reduc_model == 'LPP':
                                dim_reduc, sigma_labels = lpp(train_features_vt_sds, reduced_dim) 
                                for s_i in range(len(sigma_labels)):
                                    sigma_labels[s_i] = f'{a}_{d}_' + sigma_labels[s_i]
                                outfile_labels.extend(sigma_labels)
                         
                            elif dim_reduc_model == 'TsLPP':
                                dim_reduc, sigma_labels = tslpp(train_features_vt_sds, reduced_dim) 
                                for s_i in range(len(sigma_labels)):
                                    sigma_labels[s_i] = f'{a}_{d}_' + sigma_labels[s_i]
                                outfile_labels.extend(sigma_labels)

                            elif dim_reduc_model == 'TsNE':
                                model = TSNE(n_components = reduced_dim, init = 'random', random_state = 0, 
                                        learning_rate = 'auto', n_iter = 1000, perplexity = 50)
                                dim_reduc = model.fit_transform(train_features_vt_sds).transpose()
                                outfile_labels.extend((f'{a}_{d}_D1', f'{a}_{d}_D2'))

                            elif dim_reduc_model == 'UMAP':
                                model = umap.UMAP()
                                dim_reduc = model.fit_transform(train_features_vt_sds).transpose()
                                outfile_labels.extend((f'{a}_{d}_D1', f'{a}_{d}_D2'))

                            if a == ra[0] and d == rd[0]:
                                dim_reduc_data = dim_reduc
                            else:
                                dim_reduc_data = np.vstack((dim_reduc_data, dim_reduc))
                       #except:
                       #    print(f'{descriptor}_{d}_{a}_Si_Si.dat is NOT availabel')
                print(f'Writing {outfile}')
                with open(outfile, mode='w') as nf:
                    lines = []
                    #lines, data_format = [], ["{0: >30.16e}" for dummy in range(0, len(outfile_labels))]
                    #labels_data_format = ["{0: >30.16s}" for dummy in range(0, len(outfile_labels))]
                    labels_list = ["{0: >30.16s}".format(label) for label in outfile_labels]
                    lines.append("".join(labels_list)+'\n')
                    for i in range(len(dim_reduc[0])):
                        data_list = ["{0: >30.16e}".format(dim_reduc_data[j][i]) for j in range(0, len(dim_reduc_data))]
                        lines.append("".join(data_list) + '\n')
                    nf.writelines(lines)
                    nf.close()

    return
