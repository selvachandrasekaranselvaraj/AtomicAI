import warnings
warnings.filterwarnings("ignore")
import sys, os
import numpy as np
import pandas as pd

import time, multiprocessing
from AtomicAI.dim_reduction.perform_dim_reduc_models import perform_reduce_dimensions
from AtomicAI.dim_reduction.inputs_for_dim_reduction import inputs_for_dim_reduction
from AtomicAI.dim_reduction.outputs_for_dim_reduction import outputs_for_dim_reduction
from AtomicAI.data.data_lib import no_mpi_processors


def dim_reduction_mpi():
    pool = multiprocessing.Pool(no_mpi_processors)
    jobs = []
    input_variables = inputs_for_dim_reduction()
    for variables in input_variables: 
        #print(variables)
        # variables = des_file, reduced_dim, descriptor, a, d, out_directory, dim_reduc_model, sigma, inter_dim
        jobs.append(pool.apply_async(perform_reduce_dimensions, args=(variables,)))
    results = [job.get() for job in jobs]
    print()
    print('All Jobs done')
    print('*************')
   #print("Concatenating output files")
   #outputs_for_dim_reduction()
    return
