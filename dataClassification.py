#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
Practically Marina's code, just with small changes
"""

from sklearn import datasets
import numpy as np
import csv


#--------------------------------------------------------------------------------------------------------------
#Generate tags: somewhat conditional exclusive tags
# hasNS   hasRMN    category    criteria
#   N        N        0          m2>3     
#   Y        N        1          m2<3 && f<0
#   Y        Y        2          m2<3 && f>0
def f_conditional(x):
    m1=x[0]; m2=x[1]; s1x=x[2]; s1y=x[3]; s1z=x[4]; s2x=x[5]; s2y=x[6]; s2z=x[7];
    s1=s1x+s1y+s1z; s2=s2x+s2y+s2z;
    value=(-m2/2+m1*s1-m2*s2)/np.sqrt(m1*m2)
    
    if (m2<3 and value>=0):
        return int(2)
    elif (m2<3 and value<0):
        return int(1)
    elif (m2>=3):
        return int(0)

def generateEvents(Nsample, dsample=9):
    y=np.zeros((Nsample,dsample))
    for i in range (0,Nsample):
        s=np.random.randint(0,2)
        m1=3+np.random.rand()*(s*2.0+(s-1)*2.0)
        m2=3+np.random.rand()*(s*2.0+(s-1)*2.0)
        aux=np.maximum(m1,m2)
        m2=np.minimum(m1,m2)
        y[i][0]=aux
        y[i][1]=m2
        y[i][2:8]=-1+2*np.random.random_sample((1,6))   #components of spins
        norm=np.sqrt(y[i][2]**2+y[i][3]**2+y[i][4]**2)
        y[i][2]=y[i][2]/norm; y[i][3]=y[i][3]/norm; y[i][4]=y[i][4]/norm;
        norm=np.sqrt(y[i][5]**2+y[i][6]**2+y[i][7]**2)
        y[i][5]=y[i][5]/norm; y[i][6]=y[i][6]/norm; y[i][7]=y[i][7]/norm;
        y[i][8]=np.dot([s1x,s1y,s1z],[s2x,s2y,s2z])/(np.dot([s1x,s1y,s1z],[s1x,s1y,s1z])*np.dot([s2x,s2y,s2z],[s2x,s2y,s2z]))  #cos angle between spins
    #    y[i][9]=(y[i][0]*y[i][1])**(3.0/5)/(y[i][0]+y[i][1])**(1.0/5)   #chirp mass. Maybe we add it 
    return y

def categorize(x,talk=False):
    N=len(x)
    tags=np.zeros(N,dtype=int)
    for i in range (0,N):
        tags[i]=f_conditional(x[i])
    if(talk):
        count_arr=np.bincount(tags)
        print('No NS ', count_arr[0]/N*100,'%')
        print('Has NS, no remnant: ', count_arr[1]/N*100,'%')
        print('Has NS, yes remnant: ', count_arr[2]/N*100,'%')
    return tags

#--------------------------------------------------------------------------------------------------------------
def SetSizes(Nfeatures, Nsample, test_ratio, max_Nfeatures, max_Nsample, default_Nsample=-1):
        if default_Nsample<0:
            default_Nsample = max_Nsample

        if Nfeatures<0:
            Nfeatures = max_Nfeatures
        if Nsample<0:
            Nsample  = default_Nsample
        if Nfeatures>max_Nfeatures:
            Nfeatures = max_Nfeatures
            print('Too many features! Setting Nfeatures=', max_Nfeatures, sep='')
        if Nsample>max_Nsample:
            Nsample = max_Nsample
            print('Too many samples! Setting Nsample=', max_Nsample, sep='')
        
        Nsample = int(Nsample)
        Ntrain  = int((1-test_ratio)*Nsample) 
        Ntest   = Nsample-Ntrain
        return Nfeatures, Nsample, Ntrain, Ntest

def LoadData(dataset_name, Nsample=-1, test_ratio=0.3, Nfeatures=-1, seed=-1):
    if seed>=0:
        np.random.seed(seed)

    if dataset_name=='iris':
        max_Nfeatures   = 4
        max_Nsample     = 150
        Nfeatures, Nsample, Ntrain, Ntest = SetSizes(Nfeatures, Nsample, test_ratio, max_Nfeatures, max_Nsample)
        iris = datasets.load_iris()
        X = iris.data[:, :Nfeatures]  # Here is taking just the first two features. You can take up until the 4th one
        y = iris.target #Here are the correspondent labels
        Nsample = len(X);
        irisall=np.zeros((Nsample,Nfeatures+1))
        for i in range(0,Nsample):
            for j in range(0,Nfeatures):
                irisall[i][j]=X[i][j]
                irisall[i][Nfeatures]=y[i]
        np.random.shuffle(irisall)
        xtrain = irisall[:Ntrain,   :Nfeatures]
        ytrain = irisall[:Ntrain,    Nfeatures]
        xtest  = irisall[Ntrain:-1, :Nfeatures]
        ytest  = irisall[Ntrain:-1,  Nfeatures]

    elif dataset_name=='glass':
        max_Nfeatures   = 9
        max_Nsample     = 214
        Nfeatures, Nsample, Ntrain, Ntest = SetSizes(Nfeatures, Nsample, test_ratio, max_Nfeatures, max_Nsample)
        file = open('../glass.csv')
        csvreader = csv.reader(file)
        next(csvreader)
        rows = []
        for row in csvreader:
                rows.append(row)
        file.close()
        nrows = len(row)
        np.random.shuffle(rows) 
        glassall = np.zeros((Nsample, Nfeatures+1))
        for i in range(0, nrows):
            row = rows[i]
            glassall[i][0:Nfeatures] = row[:Nfeatures]
            glassall[i][-1] = row[9]
        np.random.shuffle(glassall)
        xtrain = glassall[:Ntrain  , :Nfeatures]
        ytrain = glassall[:Ntrain  ,  Nfeatures]
        xtest  = glassall[Ntrain:-1, :Nfeatures]
        ytest  = glassall[Ntrain:-1,  Nfeatures]

    elif dataset_name=='realistic_fake':
        max_Nfeatures   = 15
        max_Nsample     = 10000000
        default_Nsample = 1000
        Nfeatures, Nsample, Ntrain, Ntest = SetSizes(Nfeatures, Nsample, test_ratio, \
                max_Nfeatures, max_Nsample, default_Nsample)
        
        x = generateEvents(Nsample)
        y = categorize(x,talk=False) 
        # randomness in generateEvents, no need to shuffle
        xtrain = x[:Ntrain , :Nfeatures]
        ytrain = y[:Ntrain]
        xtest  = x[Ntrain:-1, :Nfeatures]
        ytest  = y[Ntrain:-1]

    else:
        print("'",dataset_name,"' not found! Returning empty arrays",sep='')
        xtrain   = []
        ytrain   = []
        xtest    = []
        ytest    = []
        Ntrain   = 0
        Ntest    = 0

    # wrap data into a dictionary
    dataset  = {}
    dataset['name']     = dataset_name
    dataset['xtrain']   = xtrain
    dataset['ytrain']   = ytrain
    dataset['xtest']    = xtest
    dataset['ytest']    = ytest
    dataset['Ntrain']   = Ntrain
    dataset['Ntest']    = Ntest
    dataset['Nsample']  = Ntrain+Ntest 
    dataset['Nclasses'] = len(ytrain)
    
    return dataset
