import sys
import csv
import numpy as np

import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing

from plotting_tools import *
sys.path.insert(0, '/home/IPAMNET/lmzertuche/Documents/LIGO_ML/IPAM2021_ML')
from read_data import *


from sklearn.gaussian_process.kernels import DotProduct, RationalQuadratic, RBF, ConstantKernel, WhiteKernel

xtrain = extractData('../NewRealistic/v0c0train_x.csv')
ytrain = extractData('../NewRealistic/v0c0train_y.csv')
xtest = extractData('../NewRealistic/v0c0test_x.csv')
ytest = extractData('../NewRealistic/v0c0test_y.csv')

xtrain = xtrain[:,:-1]
ytrain = ytrain[:,:-1]
xtest = xtest[:,:-1]
ytest = ytest[:,:-1]

scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain_scaled = scaler.transform(xtrain)
scaler = preprocessing.StandardScaler().fit(xtest)
xtest_scaled = scaler.transform(xtest)
scaler = preprocessing.StandardScaler().fit(ytrain)
ytrain_scaled = scaler.transform(ytrain)
scaler = preprocessing.StandardScaler().fit(ytest)
ytest_scaled = scaler.transform(ytest)

kernel = DotProduct() + RationalQuadratic(length_scale=1.0, alpha=2, length_scale_bounds=(.1, 10), alpha_bounds=(1e-4, 4))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-7, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, copy_X_train=False)
gpr.fit(xtrain_scaled, ytrain_scaled)
predicted_data, std = gpr.predict(xtest_scaled, return_std=True)
gpr.score(xtest_scaled, ytest_scaled)
