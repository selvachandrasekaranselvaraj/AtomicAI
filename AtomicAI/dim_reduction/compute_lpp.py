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
    #print(n_components_target, descriptor_data.shape, mapping_matrix_1.shape, x_transformed.shape)

    return x_transformed, mapping_matrix_1
