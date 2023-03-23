import numpy as np

# Variables
no_mpi_processors = 6
#no_of_train_indices  = 5000
descriptors = ['G2', 'G2G4', 'SOAP']
#descriptors = ['SOAP']
#dim_reduction_models = ['PCA+LPP', 'PCA+TsNE', 'PCA+PCA', 'PCA+UMAP', 'PCA', 'TsNE', 'UMAP', 'LPP', 'TsLPP'] 
dim_reduction_models = ['TsLPP'] #['PCA', 'TsNE', 'UMAP', 'TsLPP'] 
#sigmas = list(range(1, 16, 5)) + list(range(20, 100, 10)) + list(range(200, 540, 200))
#sigmas = [20]
#intermediate_reduced_dimensions = [40]
#intermediate_reduced_dimensions = list(range(1, 11, 2)) + list(range(12, 51, 5)) + list(range(55, 500, 15)) 
final_reduced_dimensions = [2]
cluster_numbers = list(range(6, 7, 1))

# Dictionaries
descriptor_cutoff = {}


descriptor_cutoff['Si_Si'] = [7.0], [3.0]
#descriptor_cutoff['Si_Si'] = [5.1, 7.0], [3.0, 4.3]
#descriptor_cutoff['Si_Si'] = [3.0, 4.3, 5.1, 6.0, 7.0], [1.0, 3.0, 4.3, 5.1]
descriptor_cutoff['Ni_Ni'] = [3.0, 4.3, 5.1], [1.0, 3.0, 4.3, 5.1]
descriptor_cutoff['Si_Ni'] = [3.0, 4.3, 5.1], [1.0, 3.0, 4.3, 5.1]
descriptor_cutoff['Ni_Si'] = [3.0, 4.3, 5.1], [1.0, 3.0, 4.3, 5.1]
