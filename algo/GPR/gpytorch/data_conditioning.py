import sys
import csv
import numpy as np
import torch
import gpytorch
from sklearn import preprocessing
import scipy
from scipy.special import logit, expit

def extract_data(filename, header=False, verbose=False):
    lst=[]
    header_list = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        if header==True:
            header_list.append(next(csv_reader))
            for row in csv_reader:
                lst.append(row)
        else:
            for row in csv_reader:
                lst.append(row)
    data=np.array(lst, dtype=float)
    if verbose:
        print(filename, 'loaded')
    return header_list, data

def writeResult(filename, data, verbose=False):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in data:
            spamwriter.writerow(row)
    if verbose:
        print(filename,'saved')

def flip_mass_values(data):
    sorted_data = data.copy()
    for ind,row in enumerate(data):
        m1, m2, chi1, chi2 = row
        if m2>m1:
            sorted_data[ind,0] = m2
            sorted_data[ind,1] = m1
            sorted_data[ind,2] = chi2
            sorted_data[ind,3] = chi1
        else:
            continue
    return sorted_data

def map_to_inf(xtrain, ytrain, xtest, ytest):
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
    sA = np.array([[min_spin, 1], [max_spin, 1]])
    sB = np.array([0, 1])
    sX = np.linalg.solve(sA,sB)
    sa = sX[0]
    sb = sX[1]
    v = lambda u : (sa * u) + sb

    #Map
    ytrain_01 = np.zeros(np.shape(xtrain)) 
    xtrain_01 = np.zeros(np.shape(xtrain))
    ytest_01 = np.zeros(np.shape(xtest))
    xtest_01 = np.zeros(np.shape(xtest))
    ytrain_01[:,0:2] = y(ytrain[:,0:2])
    xtrain_01[:,0:2] = y(xtrain[:,0:2])
    ytest_01[:,0:2] = y(ytest[:,0:2])
    xtest_01[:,0:2] = y(xtest[:,0:2])
    ytrain_01[:,2:] = v(ytrain[:,2:])
    xtrain_01[:,2:] = v(xtrain[:,2:])
    ytest_01[:,2:] = v(ytest[:,2:])
    xtest_01[:,2:] = v(xtest[:,2:])
    
    # mapping into (-inf,inf) range
    ytrain_inf = np.zeros(np.shape(xtrain))
    xtrain_inf = np.zeros(np.shape(xtrain))
    ytest_inf = np.zeros(np.shape(xtest))
    xtest_inf = np.zeros(np.shape(xtest))
    ytrain_inf[:,:] = scipy.special.logit(ytrain_01[:,:])
    xtrain_inf[:,:] = scipy.special.logit(xtrain_01[:,:])
    ytest_inf[:,:] = scipy.special.logit(ytest_01[:,:])
    xtest_inf[:,:] = scipy.special.logit(xtest_01[:,:])
    return xtrain_inf, ytrain_inf, xtest_inf, ytest_inf

def map_from_inf(predicted_data_inf):
    predicted_data_01 = np.zeros(np.shape(predicted_data_inf))
    predicted_data_01[:,:] = expit(predicted_data_inf[:,:])
    
    #Define mass and spin mappings
    min_mass = 0.75 # bounds from template bank
    max_mass = 400.
    A = np.array([[min_mass, 1], [max_mass, 1]])
    B = np.array([0, 1])
    X = np.linalg.solve(A,B)
    a = X[0]
    b = X[1]
    x = lambda y : (-b + y) * (1/a)

    min_spin = -1. # bounds from template bank
    max_spin = 1.
    sA = np.array([[min_spin, 1], [max_spin, 1]])
    sB = np.array([0, 1])
    sX = np.linalg.solve(sA,sB)
    sa = sX[0]
    sb = sX[1]
    u = lambda v : (-sb + v) * (1/sa)
                                                
    predicted_data = np.zeros(np.shape(predicted_data_01))
    predicted_data[:,0:2] = x(predicted_data_01[:,0:2])
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
    preds = torch.transpose(predictions.mean.cpu(),0,1).numpy()
    return ytest_scaler.inverse_transform(preds)

def torchify(xtrain_scaled, ytrain_scaled, xtest_scaled, ytest_scaled):
    train_x = torch.from_numpy(xtrain_scaled).float()
    train_y = torch.from_numpy(ytrain_scaled).float()
    test_x = torch.from_numpy(xtest_scaled).float()
    test_y = torch.from_numpy(ytest_scaled).float()

    train_x = train_x.unsqueeze(0).repeat(4, 1, 1)
    train_y = train_y.transpose(-2, -1)

    #send tensors to be on GPUs
    return train_x, train_y, test_x, test_y
