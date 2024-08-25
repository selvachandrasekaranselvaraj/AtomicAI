import pandas as pd
import numpy as np
import os
import pickle

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from AtomicAI.dim_reduction.compute_lpp import compute_lpp, new_centering
#import plotly.express as px

class TsLpp:
    """
        Class for TS-LPP model.
        Needed information to process new data:
        - Original data set (normalization - new_centering function <- From .csv file)
        - Y_middle and Y_final matrices (for dimensionality reduction <- stored?)
    """

    def __init__(self):
        self.x_data = None
        self.data_training = None
        self.labels_training = None
        self.projections_training = None
        self.y_middle = None
        self.y_final = None
        self.data_test = None
        self.labels_test = None
        self.projections_test = None

    def fit(self,
            X_train,
            inter_dimension: int = None,
            final_dimension: int = 2,
            k_nearest_neighbors: int = 7,
            sigma: int = 1,
            silent: bool = True,
            ):
        """
            Train TS-LPP model using training data
        :param X_train:
        :param number_of_intermediate_dimensions:
        :param numbet of final reduced dimensions:
        :param k_nearest_neighbors:
        :param sigma value:
        :param silent:
        :return:
        """

        if not silent:
            print('Shape of data set: ', np.shape(X_train))
        self.data_training = X_train

        #X = new_centering(X_train, X_train)

        # Number of lines and features
        number_of_lines, number_of_features = np.shape(X_train)

        if not silent:
            print('*************')
            print('Training data')
            print('Number of data =', len(X))
            print('Number of features =', number_of_features)
            print('Number of clusters =', number_of_clusters)
            print('Reduced dimension =', number_of_dimensions)
            print('*************')

        """
            Perform TS-LPP
        """
        X_middle, y_middle = compute_lpp(X_train, k_nearest_neighbors, sigma, inter_dimension)
        X_final, y_final = compute_lpp(X_middle, k_nearest_neighbors, sigma, final_dimension)

        self.projections_training = X_final
        self.y_middle = y_middle
        self.y_final = y_final
        return X_final

    def transform(self, data_test):
        """
            Use trained model to transform data into low-dimensional space
        :param data_test:
        :return:
        """
        self.data_test = data_test
        #normalized_test_data = new_centering(self.data_training, data_test)
        projections_test = np.dot(np.dot(self.data_test, self.y_middle.T), self.y_final.T)
        self.projections_test = projections_test
        return projections_test

    def predict(self, data_test):
        """
           Placeholder: predict class of test data
        :param data_test:
        :return:
        """
        transformed_data = self.transform(data_test=data_test)
        print(transformed_data.shape)
        k_mean_model = self.clustering_model
        prediction = k_mean_model.predict(transformed_data)
        return prediction


    def show_projections(self):
        pass
