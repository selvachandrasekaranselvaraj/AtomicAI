import numpy as np
import pandas as pd
import os, sys, imp, random
from AtomicAI.data import data_lib


def select_descriptors(descriptor_filename, descriptor):
    if descriptor_filename[-3:] == 'dat':
        data = np.asarray(np.loadtxt(descriptor_filename, skiprows=0))
        features = np.asarray(np.loadtxt(descriptor_filename, skiprows=0))
        row, column = np.shape(data)
        features = data[:, 0:column-1]   # Descreptors
        labels = data[:, column-1].reshape(row, 1) # Force
        
        # train, test, validation sets are about20%, 60% and 20%.
        train_indices = np.arange(0, int(20/100 * row))
        val_indices = np.arange(int(20/100 * row), int(40/100*row))
        test_indices = np.arange(int(40/100 * row), row)
        train_indices_file = './descriptors/train_indices.txt'
        test_indices_file = './descriptors/test_indices.txt'
        val_indices_file = './descriptors/val_indices.txt'
        train_indices = np.asarray(np.loadtxt(train_indices_file, skiprows=0), dtype ='int32')
        train_indices = np.arange(0, row)
        f_train, f_test, f_val = features[train_indices], features[test_indices], features[val_indices]
        l_train, l_test, l_val = labels[train_indices], labels[test_indices], labels[val_indices]
    elif descriptor_filename[-3:] == 'csv':
        df = pd.read_csv(descriptor_filename, header=0, sep='\t', encoding='utf-8', index_col=0)
        l_train = np.array(df['m_labels'])

        df = df.drop(columns= ['m_labels', 'm_sublabels'])
       #if descriptor == 'SOAP':
       #    no_of_features = 324
       #elif descriptor == 'G2G4':
       #    no_of_features = 130
       #elif descriptor == 'G2':
       #    no_of_features = 50
        f_train = np.array(df)
        f_test, f_val, l_test, l_val = None, None, None, None
    return (f_train, f_test, f_val, l_train, l_test, l_val)

