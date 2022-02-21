import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
sys.path.insert(0, "../../../../utils/")
import utils as ut

X = np.loadtxt('X.dat')
X = shuffle(X)
N = len(X[:,0])

fig, axs = plt.subplots(1,2, figsize=(11, 4))
axs[0].scatter(X[:,2], X[:,3])
axs[0].plot(X[:,2], X[:,2], 'r')
axs[1].scatter(X[:,11], X[:,12])
axs[1].plot(X[:,11], X[:,11], 'r')
plt.show()

for i in range(0,N):
    m1 = X[i,2]
    m2 = X[i,3]
    
    if m2>m1:
        tmp     = X[i,2]
        X[i,2]  = X[i,3]
        X[i,3]  = tmp
        # recovered masses are already ordered

fig, axs = plt.subplots(1,2, figsize=(11, 4))
axs[0].scatter(X[:,2], X[:,3])
axs[0].plot(X[:,2], X[:,2], 'r')
axs[1].scatter(X[:,11], X[:,12])
axs[1].plot(X[:,11], X[:,11], 'r')
plt.show()

inj = np.copy(X[:,2:5])
rec = np.copy(X[:,11:14])
SNR    = np.copy(np.reshape(X[:,20], (N,1)))
labels = np.copy(np.reshape(X[:,24], (N,1)))
split = 0.33

Ntest  = round(N*split)
Ntrain = N-Ntest

inj_train = inj[:Ntrain,:]
inj_test  = inj[Ntrain:,:]
rec_train = rec[:Ntrain,:]
rec_test  = rec[Ntrain:,:]
SNR_train = SNR[:Ntrain]
#SNR_test  = SNR[Ntrain:]
labels_train = labels[:Ntrain]
labels_test  = labels[Ntrain:] 

ut.writeResult('xtrain.csv', rec_train)
ut.writeResult('ytrain.csv', inj_train)
ut.writeResult('xtest.csv',  rec_test)
ut.writeResult('ytest.csv',  inj_test)
ut.writeResult('SNRtrain.csv', SNR_train)
#ut.writeResult('SNRtest.csv', SNR_test)
ut.writeResult('labels_train.csv', labels_train)
ut.writeResult('labels_test.csv' , labels_test)

