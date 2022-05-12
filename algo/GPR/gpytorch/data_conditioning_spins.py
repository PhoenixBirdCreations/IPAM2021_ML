import sys

import csv
import numpy as np
import torch
import gpytorch
from sklearn import preprocessing
from sklearn.utils import shuffle
import scipy
from scipy.special import logit, expit

#sys.path.insert(0, '/Users/Lorena/ML_IPAM/IPAM2021_ML/utils')
#from utils import *

def extractData(filename, verbose=False):
    lst=[]
    header = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header.append(next(csv_reader))
        for row in csv_reader:
            lst.append(row)
    data=np.array(lst, dtype=float)
    if verbose:
        print(filename, 'loaded')
    return header, data

def writeResult(filename, data, verbose=False):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in data:
            spamwriter.writerow(row)
    if verbose:
        print(filename,'saved')

def map_to_inf(xtrain, ytrain, xtest, ytest, shuffle_data=True):
    if shuffle_data==True:
        xtrain = shuffle(xtrain, random_state=5)
        ytrain = shuffle(ytrain, random_state=5)
        xtest = shuffle(xtest, random_state=42)
        ytest = shuffle(ytest, random_state=42)
    
    #Define mass mapping
    min_mass = 0.75 # bounds from template bank
    max_mass = 400.
    A = np.array([[min_mass, 1], [max_mass, 1]])
    B = np.array([0, 1])
    X = np.linalg.solve(A,B)
    a = X[0]
    b = X[1]
    y = lambda x : (a * x) + b
    #Define spin mapping
    min_spin = -1. # bounds from template bank
    max_spin = 1.
    As = np.array([[min_spin, 1], [max_spin, 1]])
    Bs = np.array([0, 1])
    Xs = np.linalg.solve(As,Bs)
    sa = Xs[0]
    sb = Xs[1]
    v = lambda u : (sa * u) + sb
    
    #Map
    ytrain_01 = ytrain
    xtrain_01 = xtrain
    ytest_01 = ytest
    xtest_01 = xtest
    ytrain_01[:,0:2] = y(ytrain[:,0:2])
    xtrain_01[:,0:2] = y(xtrain[:,0:2])
    ytest_01[:,0:2] = y(ytest[:,0:2])
    xtest_01[:,0:2] = y(xtest[:,0:2])
    ytrain_01[:,2:] = v(ytrain[:,2:])
    xtrain_01[:,2:] = v(xtrain[:,2:])
    ytest_01[:,2:] = v(ytest[:,2:])
    xtest_01[:,2:] = v(xtest[:,2:]) 

    # mapping into (-inf,inf) range
    ytrain_inf = ytrain_01
    xtrain_inf = xtrain_01
    ytest_inf = ytest_01
    xtest_inf = xtest_01
    ytrain_inf[:,0:2] = scipy.special.logit(ytrain_01[:,0:2])
    xtrain_inf[:,0:2] = scipy.special.logit(xtrain_01[:,0:2])
    ytest_inf[:,0:2] = scipy.special.logit(ytest_01[:,0:2])
    xtest_inf[:,0:2] = scipy.special.logit(xtest_01[:,0:2])
    ytrain_inf[:,2:] = scipy.special.logit(ytrain_01[:,2:])
    xtrain_inf[:,2:] = scipy.special.logit(xtrain_01[:,2:])
    ytest_inf[:,2:] = scipy.special.logit(ytest_01[:,2:])
    xtest_inf[:,2:] = scipy.special.logit(xtest_01[:,2:])
    return xtrain_inf, ytrain_inf, xtest_inf, ytest_inf

def map_from_inf(predicted_data_inf):
    predicted_data_01 = predicted_data_inf
    predicted_data_01[:,:2] = expit(predicted_data_inf[:,:2])
    predicted_data_01[:,2:] = expit(predicted_data_inf[:,2:])
    
    #Define mass mapping
    min_mass = 0.75 # bounds from template bank
    max_mass = 400.
    A = np.array([[min_mass, 1], [max_mass, 1]])
    B = np.array([0, 1])
    X = np.linalg.solve(A,B)
    a = X[0]
    b = X[1]
    x = lambda y : (-b + y) * (1/a)
    #Define spin mapping
    min_spin = -1. # bounds from template bank
    max_spin = 1.
    As = np.array([[min_spin, 1], [max_spin, 1]])
    Bs = np.array([0, 1])
    Xs = np.linalg.solve(As,Bs)
    sa = Xs[0]
    sb = Xs[1]
    u = lambda v : (-sb + v) * (1/sa)
    
    predicted_data = predicted_data_01
    predicted_data[:,:2] = x(predicted_data_01[:,:2])
    predicted_data[:,2:] = u(predicted_data_01[:,2:])

    return predicted_data

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

    train_x = train_x.unsqueeze(0).repeat(4, 1, 1)
    train_y = train_y.transpose(-2, -1)
    return train_x, train_y, test_x, test_y
