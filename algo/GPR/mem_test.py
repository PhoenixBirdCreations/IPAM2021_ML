import csv
import numpy as np

import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import DotProduct, RationalQuadratic, RBF, ConstantKernel, WhiteKernel

from read_data import *

# Read data from files
xtrain = extractData('v0c0train_x.csv')
ytrain = extractData('v0c0train_y.csv')
xtest = extractData('v0c0test_x.csv')
ytest = extractData('v0c0test_y.csv')

# Remove the last column (M_chirp) of the dataset
xtrain = xtrain[:,:-1]
ytrain = ytrain[:,:-1]
xtest = xtest[:,:-1]
ytest = ytest[:,:-1]

# Standardize the data
scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain_scaled = scaler.transform(xtrain)
scaler = preprocessing.StandardScaler().fit(xtest)
xtest_scaled = scaler.transform(xtest)
scaler = preprocessing.StandardScaler().fit(ytrain)
ytrain_scaled = scaler.transform(ytrain)
scaler = preprocessing.StandardScaler().fit(ytest)
ytest_scaled = scaler.transform(ytest)

##Only take part of the data to train and test
#xtrain_scaled = xtrain_scaled[:10000,:]
#ytrain_scaled = ytrain_scaled[:10000,:]
#xtest_scaled = xtest_scaled[:2000,:]
#ytest_scaled = ytest_scaled[:2000,:]

# Train GPR and predict data
kernel = DotProduct() + RationalQuadratic(length_scale=1.0, alpha=2, length_scale_bounds=(.1, 10), alpha_bounds=(1e-4, 4))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-7, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=3, copy_X_train=False)
gpr.fit(xtrain_scaled, ytrain_scaled)
predicted_data, std = gpr.predict(xtest_scaled, return_std=True)
print('The score is: ', gpr.score(xtest_scaled, ytest_scaled))

# Write results in csv file
writeResult('GPR_DP+RQ_nochirp_v0c0.csv',predicted_data)
