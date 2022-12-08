#!/usr/bin/env python
# coding: utf-8
import sys, os, importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from split_GstLAL_data import split_GstLAL_data
repo_paths = ['/home/simonealbanesi/repos/IPAM2021_ML/', '/home/simone/repos/IPAM2021_ML/']
for rp in repo_paths:
    if os.path.isdir(rp):
        repo_path = rp
        break
sys.path.insert(0, repo_path+'utils/')
import classyNN as cnn

dict_name      = 'crossval_dicts/GstLAL.dict'
neurons_max    = 750
neurons_step   = 50
epochs         = 100
batch_size     = 128 

# default without constraints in the output layer
learning_rate  = 0.001;
linear_output  = True
compact_scaler = True
std_scaler     = True
sigma0         = 1

compact_bounds      = {}
compact_bounds['A'] = [0.75, -1, -1, 0.75]
compact_bounds['B'] = [400,   1,  1,  120]


m1_cutoff = None
Mc_min    = None

data_path        = repo_path+'datasets/GstLAL/'
fname_train_data = data_path+'train_NS.csv'
fname_test_data  = data_path+'test_NS.csv'
train_datasets   = cnn.extract_data(fname_train_data, skip_header=True)
test_datasets    = cnn.extract_data(fname_test_data, skip_header=True)

train_data_split = split_GstLAL_data(train_datasets, features='mass&spin', m1_cutoff=m1_cutoff, Mc_min=Mc_min)
test_data_split  = split_GstLAL_data(test_datasets,  features='mass&spin', m1_cutoff=m1_cutoff, Mc_min=Mc_min)
train_inj = train_data_split['inj']
train_rec = train_data_split['rec']
test_inj  =  test_data_split['inj']
test_rec  =  test_data_split['rec']

nfeatures = len(train_inj[0,:])

## Cross-validation
CV = cnn.CrossValidator(nfeatures=nfeatures, neurons_max=neurons_max, neurons_step=neurons_step, dict_name=dict_name,
                        xtrain=train_rec, ytrain=train_inj, xtest=test_rec, ytest=test_inj,
                        epochs=epochs, batch_size=batch_size, out_intervals=None, seed=None,
                        compact_bounds=compact_bounds, linear_output=linear_output,
                        compact_scaler=compact_scaler, standard_scaler=std_scaler, sigma0=sigma0)
CV.crossval(verbose=True)

CV.plot(feature_idx=-1, threshold=None)






