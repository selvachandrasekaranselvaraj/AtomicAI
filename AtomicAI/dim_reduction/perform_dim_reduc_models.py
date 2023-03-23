import warnings
warnings.filterwarnings("ignore")
import sys, os
import numpy as np
import pandas as pd

from AtomicAI.dim_reduction.lpp import lpp
from AtomicAI.dim_reduction.ts_lpp import TsLpp
#from AtomicAI.data.data_lib import sigmas
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pickle 
from AtomicAI.dim_reduction.select_descriptors import select_descriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def perform_reduce_dimensions(variables):
    descriptor_file, reduced_dim, descriptor, a, d, out_directory, reduced_dim_model, sigma, intermediate_dimension = variables
    descriptor_file = variables[0]
    selected_descriptors = select_descriptors(descriptor_file, descriptor)
    train_features = selected_descriptors[0]
    train_labels = selected_descriptors[3]
    #train_features_vt = VarianceThreshold(threshold=1e-32).fit_transform(train_features)
    #train_features_vt_sds = StandardScaler().fit_transform(train_features_vt)
    train_features_vt_sds = train_features
    tot_dim = np.shape(train_features)[1] 
    label = f'{reduced_dim}D_{reduced_dim_model}_{descriptor}_{a}_{d}_{sigma}_{intermediate_dimension}_{tot_dim}'
    outfile = f'{out_directory}{str(reduced_dim)}D_{reduced_dim_model}_{descriptor}_{a}_{d}_{sigma}_{intermediate_dimension}_{tot_dim}'

    reduced_dimensions_data = None
    outfile_labels = []

    if reduced_dim_model == 'PCA':
        model = PCA(n_components = reduced_dim)
        reduced_dimensions_data = model.fit_transform(train_features_vt_sds).transpose()
        #pickle.dump(model, open(outfile+'.pkl', 'wb'))

    elif reduced_dim_model == 'PCA+LPP':
        pca_components = int(np.shape(train_features_vt_sds)[1] * 0.5)
        model = PCA(n_components = pca_components)
        reduced_dimensions_data_pca = model.fit_transform(train_features_vt_sds)
        reduced_dimensions_data = lpp(dim_reduc_pca, reduced_dim, sigma)
        #pickle.dump(model, open(outfile+'.pkl', 'wb'))


    elif reduced_dim_model == 'PCA+TsNE':
        pca_components = int(np.shape(train_features_vt_sds)[1] * 0.5)
        model = PCA(n_components = pca_components)
        reduced_dimensions_data_pca = model.fit_transform(train_features_vt_sds)
        model = TSNE(n_components = reduced_dim, init = 'random', random_state = 0,
                learning_rate = 'auto', n_iter = 1000, perplexity = 50)
        reduced_dimensions_data = model.fit_transform(dim_reduc_pca).transpose()
        #pickle.dump(model, open(outfile+'.pkl', 'wb'))

    elif reduced_dim_model == 'PCA+UMAP':
        pca_components = int(np.shape(train_features_vt_sds)[1] * 0.5)
        model = PCA(n_components = pca_components)
        reduced_dimensions_data_pca = model.fit_transform(train_features_vt_sds)
        model = umap.UMAP()
        reduced_dimensions_data = model.fit_transform(dim_reduc_pca).transpose()
        #pickle.dump(model, open(outfile+'.pkl', 'wb'))

    elif reduced_dim_model == 'PCA+PCA':
        pca_components = int(np.shape(train_features_vt_sds)[1] * 0.5)
        model = PCA(n_components = pca_components)
        reduced_dimensions_data_pca = model.fit_transform(train_features_vt_sds)
        model = PCA(n_components = reduced_dim)
        reduced_dimensions_data = model.fit_transform(dim_reduc_pca).transpose()
        #pickle.dump(model, open(outfile+'.pkl', 'wb'))


    elif reduced_dim_model == 'LPP':
        reduced_dimensions_data = lpp(train_features_vt_sds, reduced_dim, sigma)


    elif reduced_dim_model == 'TsLPP':
        model = TsLpp()
        try:
            reduced_dimensions_data = model.fit(np.array(train_features_vt_sds),
                                                inter_dimension = intermediate_dimension, 
                                                final_dimension = reduced_dim,
                                                sigma = sigma,
                                                ).transpose()
            #pickle.dump(model, open(outfile+'.pkl', 'wb'))
        except np.linalg.LinAlgError:
            print(outfile, "EXITS due to error.")
            exit()

    elif reduced_dim_model == 'TsNE':
        model = TSNE(n_components = reduced_dim, init = 'random', random_state = 0,
                learning_rate = 'auto', n_iter = 1000, perplexity = 50)
        reduced_dimensions_data = model.fit_transform(train_features_vt_sds).transpose()
        #pickle.dump(model, open(outfile+'.pkl', 'wb'))

    elif reduced_dim_model == 'UMAP':
        model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=reduced_dim, metric='euclidean')
        reduced_dimensions_data = model.fit_transform(train_features_vt_sds).transpose()
        #pickle.dump(model, open(outfile+'.pkl', 'wb'))
    
    if reduced_dimensions_data is not None: 
        df = pd.DataFrame()
        for i in range(len(reduced_dimensions_data)):
            writing_dim = i + 1
            df[label + f'_D{writing_dim}'] = reduced_dimensions_data[i]
        df['m_labels'] = train_labels
        df.to_csv(outfile+'.csv', sep='\t', encoding='utf-8')
        print(outfile, "is DONE.")
    else:
        pass
    return

    #with open(outfile, mode='w') as nf:
        #lines = []
        #lines, data_format = [], ["{0: >30.16e}" for dummy in range(0, len(outfile_labels))]
        #labels_data_format = ["{0: >30.16s}" for dummy in range(0, len(outfile_labels))]
        #labels_list = ["{0: >30.29s}".format(f'{reduced_dim_model}_{descriptor}_{label}') for label in outfile_labels]
        #labels_list = [f'{reduced_dim_model}_{descriptor}_{label}' for label in outfile_labels]
        #lines.append("".join(labels_list)+'\n')
        #df = pd.DataFrame()
        #for i in range(len(reduced_dimensions_data[0])):
            #data_list = ["{0: >30.16e}".format(reduced_dimensions_data_data[j][i]) for j in range(0, len(dim_reduc_data))]
            #data_list = [reduced_dimensions_data_data[j][i] for j in range(0, len(dim_reduc_data))]
            #df[label_list[i]] = [reduced_dimensions_data_data[j][i] for j in range(0, len(dim_reduc_data))]
            #lines.append("".join(data_list) + '\n')
        #nf.writelines(lines)
        #nf.close()
