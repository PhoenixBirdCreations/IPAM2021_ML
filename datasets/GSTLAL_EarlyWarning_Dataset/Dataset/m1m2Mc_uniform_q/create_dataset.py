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

#fig, axs = plt.subplots(1,2, figsize=(11, 4))
#axs[0].scatter(X[:,2], X[:,3])
#axs[0].plot(X[:,2], X[:,2], 'r')
#axs[1].scatter(X[:,11], X[:,12])
#axs[1].plot(X[:,11], X[:,11], 'r')
#plt.show()

for i in range(0,N):
    m1 = X[i,2]
    m2 = X[i,3]
    
    if m2>m1:
        tmp     = X[i,2]
        X[i,2]  = X[i,3]
        X[i,3]  = tmp
        # recovered masses are already ordered

#fig, axs = plt.subplots(1,2, figsize=(11, 4))
#axs[0].scatter(X[:,2], X[:,3])
#axs[0].plot(X[:,2], X[:,2], 'r')
#axs[1].scatter(X[:,11], X[:,12])
#axs[1].plot(X[:,11], X[:,11], 'r')
#plt.show()

inj = np.copy(X[:,2:5])
rec = np.copy(X[:,11:14])

q_thresholds = np.array([0.8, 0.85, 0.9, 0.95, 1])
nbins = 4; 
N = len(inj[:,0])

# divide in q-bins
# idx0: inj/rec, idx1: bin, idx2: n, idx3: features
bins = np.zeros((2, nbins, N, len(inj[0,:])))
k_indeces = np.zeros((nbins,))
for i in range(0,nbins):
    for j in range(0,N):
        m1 = inj[j,0]
        m2 = inj[j,1]
        q  = m2/m1
        if q>q_thresholds[i] and q<=q_thresholds[i+1]:
            k = round(k_indeces[i])           
            bins[0, i, k, :] = inj[j,:]
            bins[1, i, k, :] = rec[j,:]
            k_indeces[i] += 1

print(np.shape(bins))
#uniform 
kmin = round(min(k_indeces))
inj_uniform = np.zeros((kmin*nbins, len(inj[0,:])))
rec_uniform = np.zeros((kmin*nbins, len(inj[0,:])))
for i in range(0,nbins):
    inj_uniform[i*kmin:(i+1)*kmin,:] = bins[0, i, 0:kmin, :]
    rec_uniform[i*kmin:(i+1)*kmin,:] = bins[1, i, 0:kmin, :]
 
inj_uniform, rec_uniform = shuffle(inj_uniform, rec_uniform)

N         = len(inj_uniform[:,0])
split     = 0.33
Ntest     = round(N*split)
Ntrain    = N-Ntest
inj_train = inj_uniform[:Ntrain,:]
inj_test  = inj_uniform[Ntrain:,:]
rec_train = rec_uniform[:Ntrain,:]
rec_test  = rec_uniform[Ntrain:,:]

print(Ntrain)
print(Ntest)


fig, axs = plt.subplots(1,3, figsize=(11, 3))
axs[0].hist(inj_train[:,0],bins=np.arange(1.2,1.9,0.1), alpha=1, label='train')
axs[0].hist(inj_test[:,0],bins=np.arange(1.2,1.9,0.1), alpha=1, label='test')
axs[0].set_xlabel('m1')
axs[0].legend()
axs[1].hist(inj_train[:,1],bins=np.arange(1.2,1.9,0.1), alpha=1, label='train')
axs[1].hist(inj_test[:,1],bins=np.arange(1.2,1.9,0.1), alpha=1, label='test')
axs[1].set_xlabel('m2')
axs[1].legend()
qtrain = inj_train[:,1]/inj_train[:,0]
qtest  = inj_test[:,1]/inj_test[:,0]
axs[2].hist(qtrain,bins=np.arange(0.8,1.01,0.05), alpha=1, label='train')
axs[2].hist(qtest, bins=np.arange(0.8,1.01,0.05), alpha=1, label='test')
axs[2].set_xlabel('q')
axs[2].legend()
plt.show()

ut.writeResult('xtrain.csv', rec_train)
ut.writeResult('ytrain.csv', inj_train)
ut.writeResult('xtest.csv',  rec_test)
ut.writeResult('ytest.csv',  inj_test)

