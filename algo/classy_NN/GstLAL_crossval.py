#!/usr/bin/env python
# coding: utf-8

import sys, os, importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
repo_paths = ['/home/simonealbanesi/repos/IPAM2021_ML/', '/home/simone/repos/IPAM2021_ML/', 
             '/Users/simonealbanesi/repos/IPAM2021_ML/']
for rp in repo_paths:
    if os.path.isdir(rp):
        repo_path = rp
        break
sys.path.insert(0, repo_path+'utils/')
import sklassyNN  as sknn


# ## Input
seed            = 1
verbose_train   = True
epochs          = 50
batch_size      = 128
learning_rate   = 0.001;

compact_bounds  = {}
#features2use    = 'm1Mcchi1chi2'
#compact_bounds['A'] = [0.75, 0.75, -1, -1]
#compact_bounds['B'] = [500,  130,  1,  1] # before MCD6 was [400,1,1,120]
features2use = 'm1m2chi1chi2'
compact_bounds['A'] = [0.75, 0.75, -1, -1]
compact_bounds['B'] = [500,   250,  1,  1]

dict_name = 'crossval_dicts/'+features2use+'_GstLAL.dict'

neurons_max  = 800
neurons_step = 200

data_path = repo_path+'datasets/GstLAL/'

train_inj_all = sknn.extract_data(data_path+'complete_ytrain.csv') # m1, m2, Mc, chi1, chi2
train_rec_all = sknn.extract_data(data_path+'complete_xtrain.csv')
test_inj_all  = sknn.extract_data(data_path+'complete_ytest.csv')
test_rec_all  = sknn.extract_data(data_path+'complete_xtest.csv')

if features2use=='m1Mcchi1chi2':
    train_inj = np.delete(train_inj_all, 1, axis=1)
    train_rec = np.delete(train_rec_all, 1, axis=1)
    test_inj  = np.delete( test_inj_all, 1, axis=1)
    test_rec  = np.delete( test_rec_all, 1, axis=1)
    names     = ['m1', 'Mc', 'chi1', 'chi2']
elif features2use=='m1m2chi1chi2':
    train_inj = np.delete(train_inj_all, 2, axis=1)
    train_rec = np.delete(train_rec_all, 2, axis=1)
    test_inj  = np.delete( test_inj_all, 2, axis=1)
    test_rec  = np.delete( test_rec_all, 2, axis=1)
    names     = ['m1', 'm2', 'chi1', 'chi2']

nfeatures = len(train_inj[0,:])

CV = sknn.CrossValidator(neurons_max=neurons_max, neurons_step=neurons_step, dict_name=dict_name,
                    xtrain=train_rec, ytrain=train_inj, xtest=test_rec, ytest=test_inj,
                    epochs=epochs, batch_size=batch_size, seed=seed, learning_rate=learning_rate,
                    compact_bounds=compact_bounds)
CV.crossval(verbose=False)

for i in range(-1,nfeatures):
    CV.plot(feature_idx=i, threshold=None)




