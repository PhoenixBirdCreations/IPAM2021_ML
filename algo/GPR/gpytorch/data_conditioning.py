import sys

import numpy as np
import torch
import gpytorch
from sklearn import preprocessing
from sklearn.utils import shuffle
import scipy
from scipy.special import logit, expit

sys.path.insert(0, '/Users/Lorena/ML_IPAM/IPAM2021_ML/utils')
from utils import *

def map_to_inf(xtrain, ytrain, xtest, ytest, shuffle_data=True):
    if shuffle_data==True:
        xtrain = shuffle(xtrain, random_state=5)
        ytrain = shuffle(ytrain, random_state=5)
        xtest = shuffle(xtest, random_state=42)
        ytest = shuffle(ytest, random_state=42)
    
    min_mass = 0.95 # bounds from template bank
    max_mass = 2.4

    # mapping into (0,1) range
    A = np.array([[min_mass, 1], [max_mass, 1]])
    B = np.array([0, 1])
    X = np.linalg.solve(A,B)
    a = X[0]
    b = X[1]
    y = lambda x : (a * x) + b
    ytrain_01 = y(ytrain)
    xtrain_01 = y(xtrain)
    ytest_01 = y(ytest)
    xtest_01 = y(xtest)

    # mapping into (-inf,inf) range
    ytrain_inf = scipy.special.logit(ytrain_01)
    xtrain_inf = scipy.special.logit(xtrain_01)
    ytest_inf = scipy.special.logit(ytest_01)
    xtest_inf = scipy.special.logit(xtest_01)
    return xtrain_inf, ytrain_inf, xtest_inf, ytest_inf

def map_from_inf(predicted_data_inf):
    predicted_data_01 = expit(predicted_data_inf)

    min_mass = 0.95 # bounds from template bank
    max_mass = 2.4

    A = np.array([[min_mass, 1], [max_mass, 1]])
    B = np.array([0, 1])
    X = np.linalg.solve(A,B)
    a = X[0]
    b = X[1]
    x = lambda y : (-b + y) * (1/a)
    return x(predicted_data_01)

def standardize(xtrain_inf, ytrain_inf, xtest_inf, ytest_inf):
    xtrain_scaler = preprocessing.StandardScaler().fit(xtrain_inf)
    xtrain_scaled = xtrain_scaler.transform(xtrain_inf)
    ytrain_scaler = preprocessing.StandardScaler().fit(ytrain_inf)
    ytrain_scaled = ytrain_scaler.transform(ytrain_inf)

    xtest_scaler = preprocessing.StandardScaler().fit(xtest_inf)
    xtest_scaled = xtest_scaler.transform(xtest_inf)
    ytest_scaler = preprocessing.StandardScaler().fit(ytest_inf)
    ytest_scaled = ytest_scaler.transform(ytest_inf)

    return xtrain_scaled, ytrain_scaled, xtest_scaled, ytest_scaled, ytest_scaler

def unstandardize(predictions, ytest_scaler):
    preds = torch.transpose(predictions.mean,0,1).numpy()
    return ytest_scaler.inverse_transform(preds)

def torchify(xtrain_scaled, ytrain_scaled, xtest_scaled, ytest_scaled):
    train_x = torch.from_numpy(xtrain_scaled).float()
    train_y = torch.from_numpy(ytrain_scaled).float()
    test_x = torch.from_numpy(xtest_scaled).float()
    test_y = torch.from_numpy(ytest_scaled).float()

    train_x = train_x.unsqueeze(0).repeat(2, 1, 1)
    train_y = train_y.transpose(-2, -1)
    return train_x, train_y, test_x, test_y
