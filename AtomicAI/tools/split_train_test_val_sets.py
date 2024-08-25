import numpy as np
import os, sys
def split_train_test_val_sets(descriptor_filename):
    if descriptor_filename == 'force_descriptors.dat':
        is_labels = True
        data = np.asarray(np.loadtxt(descriptor_filename, skiprows=0))
        row, column = np.shape(data)
        features = data[:, 0:column-1]   # Descreptors
        labels = data[:, column-1].reshape(row, 1) # Force
        
        # train, test, validation sets are about20%, 60% and 20%.
        train_indices = np.arange(0, int(20/100 * row))
        val_indices = np.arange(int(20/100 * row), int(40/100*row))
        test_indices = np.arange(int(40/100 * row), row)
        train_data = [0, int(80/100 * row)]
        test_data = [int(80/100 * row), row]
        f_train, f_test, f_val = features[train_indices], features[test_indices], features[val_indices]
        l_train, l_test, l_val = labels[train_indices], labels[test_indices], labels[val_indices]
        return is_labels, (f_train, f_test, f_val, l_train, l_test, l_val)

    else:
        is_labels = False
        features = np.asarray(np.loadtxt(descriptor_filename, skiprows=0))
        row, column = np.shape(data)
    
        # train, test, validation sets are about20%, 60% and 20%.
        train_indices = np.arange(0, int(20/100 * row))
        val_indices = np.arange(int(20/100 * row), int(40/100*row))
        test_indices = np.arange(int(40/100 * row), row)
        train_data = [0, int(80/100 * row)]
        test_data = [int(80/100 * row), row]
        f_train, f_test, f_val = features[train_indices], features[test_indices], features[val_indices]
        return is_labels, (f_train, f_test, f_val)

