import numpy as np
import sys,os 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 

repo_paths = ['/Users/Lorena/ML_IPAM/IPAM2021_ML/', '/Users/simonealbanesi/repos/IPAM2021_ML/']
for rp in repo_paths:
    if os.path.isdir(rp):
        repo_path = rp
        break
sys.path.insert(0, repo_path+'utils/')
from utils import chirpMass, extract_data, write_result

O2_data = extract_data('complete_dataset.csv', skip_header=True)

#After reading it as 'O2_data'
ydata = np.array(O2_data[:,3:7])  #injected m1,m2,chi1,chi2
xdata = np.array(O2_data[:,8:12]) #recovered
cfar  = np.array(O2_data[:,13])

#Injections have sometimes m2>m1, so running code below flips them
sorted_ydata = ydata.copy()
for ind,row in enumerate(ydata):
    m1, m2, chi1, chi2 = row
    if m2>m1:
        #print(m1,m2)
        sorted_ydata[ind,0] = m2
        sorted_ydata[ind,1] = m1
        sorted_ydata[ind,2] = chi2
        sorted_ydata[ind,3] = chi1
    else:
        continue

m1_rec   = xdata[:,0]
m2_rec   = xdata[:,1]
chi1_rec = xdata[:,2]
chi2_rec = xdata[:,3]
Mc_rec   = chirpMass(m1_rec, m2_rec)

m1_inj   = sorted_ydata[:,0]
m2_inj   = sorted_ydata[:,1]
chi1_inj = sorted_ydata[:,2]
chi2_inj = sorted_ydata[:,3]
Mc_inj   = chirpMass(m1_inj, m2_inj)

X = np.column_stack((m1_rec, m2_rec, Mc_rec, chi1_rec, chi2_rec, cfar))
Y = np.column_stack((m1_inj, m2_inj, Mc_inj, chi1_inj, chi2_inj, cfar))

# Shuffle them the same way.
X, Y, cfar = shuffle(X, Y, cfar, random_state=42)

# Separate data 70/30 for training/testing.
sep    = int(len(xdata)*.70)
xtrain = X[:sep,:]
ytrain = Y[:sep,:]
xtest  = X[sep:,:]
ytest  = Y[sep:,:]

#plt.figure
#plt.subplot(1,2,1)
#plt.scatter(ytrain[:,0], ytrain[:,1], s=1)
#plt.subplot(1,2,2)
#plt.scatter(ytrain[:,2], ytrain[:,3], s=1)
#plt.show()
write_result('complete_xtrain.csv', xtrain)
write_result('complete_ytrain.csv', ytrain)
write_result('complete_xtest.csv',  xtest)
write_result('complete_ytest.csv',  ytest)



