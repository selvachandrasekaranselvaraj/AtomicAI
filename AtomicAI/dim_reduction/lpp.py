"""
    Module for analysis with LPP
    - Used by TS-LPP as well
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm.auto import tqdm

import sys, os, random, scipy, scipy.stats, sklearn
from scipy.spatial import distance
from sklearn.cluster import KMeans

def lpp(features, reduced_dim, sigma):
    #sigma_list=np.arange(5, 100, 10)
    #sigma_list = np.append(sigma_list, np.arange(100, 600, 100))
    knn = 7
    data = perform_lpp(
        features,
        graph_nearest_neighbors = knn,
        number_of_dimensions=reduced_dim,
        sigma=sigma,
        #k_nearest_neighbors=knn,
        #silent=True,
        #dimension_list=[5, 10, 20],
        #sigma_list=[20],
        )
    return data


def new_centering(reference_data, my_data):
    """
        Keep only columns that have some variance.
    :param reference_data:
    :param my_data:
    :return:
    """
    reference_std = np.std(reference_data, 0)
    reference_index = np.where(reference_std != 0)
    return (my_data[:, reference_index[0]] -
            np.mean(reference_data[:, reference_index[0]], 0)) / reference_std[reference_index[0]]


def compute_lpp(
        descriptor_data: np.float64,
        graph_nearest_neighbors: int,
        sigma: float,
        n_components_target: int,
):
    """
        Locally preserving projection.
    :param n_components_target:
    :param sigma:
    :param descriptor_data: 2D array (typically, numpy.array)
    :param graph_nearest_neighbors: number (knn)
    :return:
    """

    # Create distance matrix
    distance_matrix = distance.cdist(descriptor_data, descriptor_data, metric='euclidean')

    weighted_adjacency_matrix = np.exp(-np.power(distance_matrix, 2) / 2.0 / sigma / sigma)

    for data_number in range(len(descriptor_data)):
        weighted_adjacency_matrix[data_number, data_number] = 0.0

    # Create neighbor graph
    for data_number in range(len(descriptor_data)):
        del_list = np.argsort(weighted_adjacency_matrix[data_number])[::-1][graph_nearest_neighbors:]
        weighted_adjacency_matrix[data_number, del_list] = 0.0

    # Symmetrical of W
    weighted_adjacency_matrix = np.maximum(weighted_adjacency_matrix.T, weighted_adjacency_matrix)

    # Create D
    degree_matrix = np.diag(np.sum(weighted_adjacency_matrix, axis=1))

    # Create L
    graph_laplacian = degree_matrix - weighted_adjacency_matrix

    # SVD of X1
    delta = 1e-7
    U, Sig, VT = np.linalg.svd(descriptor_data, full_matrices=False)
    rk = np.sum(Sig / Sig[0] > delta)
    Sig = np.diag(Sig)
    U1 = U[:, 0:rk]
    VT1 = VT[0:rk, :]
    Sig1 = Sig[0:rk, 0:rk]

    # Positive definite for L
    Lp = np.dot(U1.T, np.dot(graph_laplacian, U1))
    Lp = (Lp + Lp.T) / 2

    # Positive definite for D
    Dp = np.dot(U1.T, np.dot(degree_matrix, U1))
    Dp = (Dp + Dp.T) / 2

    # Generalized eigenvalue problem
    eig_val, eig_vec = scipy.linalg.eigh(Lp, Dp)

    # Projection for low dimension
    tmp1 = np.dot(VT1.T, scipy.linalg.solve(Sig1, eig_vec))
    Trans_eig_vec = tmp1.T

    # Mapping matrix (Y)
    mapping_matrix_1 = Trans_eig_vec[0:n_components_target]

    x_transformed = np.dot(mapping_matrix_1, descriptor_data.T).T

    return x_transformed, mapping_matrix_1


def perform_lpp(
        descriptor_data,
        graph_nearest_neighbors=None,
        sigma=None, 
        number_of_dimensions=None,
):
    X = new_centering(descriptor_data, descriptor_data)
    number_of_lines, number_of_features = np.shape(descriptor_data)
    # sigma_num = sigma_list[-1]

   #PseudoF_max = 0.0
    lpp_projection_for_all_sigmas = []
    outfile_labels = []
    #for sigma in sigma_list:
        ############
        # LPP x 1
        ############
    X_final_lpp, Y_final_lpp = compute_lpp(X, graph_nearest_neighbors, sigma, number_of_dimensions)
       #xx, xy = np.transpose(X_final_lpp)
       #lpp_projection_for_all_sigmas.append(xx)
       #lpp_projection_for_all_sigmas.append(xy)
       #outfile_labels.append(str(sigma)+'_D1')
       #outfile_labels.append(str(sigma)+'_D2')
 
        ## Clustering by K means
        #for number_of_cluster in range(2, 7): # if need increase range
        #    k_means_model = KMeans(n_clusters=number_of_cluster, random_state=10).fit(X_final_lpp)
        #    labels = k_means_model.labels_
        # 
        #    PseudoF = sklearn.metrics.calinski_harabasz_score(X_final_lpp, labels)
        #    if PseudoF > PseudoF_max:
        #        # Update best hyperparameters
        #        PseudoF_opt = PseudoF
        #        sigma_opt = sigma
        #        no_of_cluster_opt = number_of_cluster

      
   ##Output the best result
   #X_final_lpp, Y_final_lpp = compute_lpp(X, graph_nearest_neighbors, sigma_opt, number_of_dimensions)
   #k_means_model = KMeans(n_clusters=no_of_cluster_opt, random_state=10).fit(X_final_lpp)
   #labels = k_means_model.labels_
   #PseudoF = sklearn.metrics.calinski_harabasz_score(X_final_lpp, labels)
   #print(f'Best sigma: {sigma_opt}  pseudoF score: {PseudoF}  best k-cluster: {no_of_cluster_opt}')
   #lpp_projection = np.transpose(X_final_lpp)

    return np.transpose(X_final_lpp) #np.array(lpp_projection_for_all_sigmas), outfile_labels
