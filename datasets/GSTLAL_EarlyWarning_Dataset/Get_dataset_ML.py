import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import matplotlib
from matplotlib import pyplot as plt

script_start = time.time()

## set the parameters ##
#M_rem_cut = 0.025
split_ratio = 0.7
para_inj = ['mass1', 'mass2']
para_rec = ['mass1_recover', 'mass2_recover']
output = "./Dataset/"


## read the raw dataset ###
df_raw = pd.read_csv("./Features_rawdata_56Hz.csv")

# define m1 as the heavier mass 
df_raw.mass1, df_raw.mass2 = np.where(df_raw.mass1 < df_raw.mass2, [df_raw.mass2, df_raw.mass1], [df_raw.mass1, df_raw.mass2])

# drop the M_rem = 0
#df_raw = df_raw.replace(0, pd.np.nan).dropna(axis=0, how='any', subset=["M_rem"]).fillna(0).astype(int)
#df_raw.drop(df_raw.index[df_raw['M_rem'] == np.nan], inplace = True)

df_raw = df_raw[df_raw.M_rem.notnull()]
df_raw.to_csv("Dataset/Features_rawdata_56Hz_v1.csv")

df_raw.drop("Unnamed: 0",1,inplace=True)
nevents = df_raw.shape[0]
print ('There are ',nevents,' events totally')

n_train = int(nevents*split_ratio)
print ("There are ",n_train," for training")
print ("There are ",nevents - n_train," for testing")

## get the injected parameters and the recoverd parameters
df_inject = df_raw[para_inj]
df_recover = df_raw[para_rec]


## Split the dataset into training set and testing set for regression
df_recover_train = df_recover[:n_train]
df_recover_test = df_recover[n_train:]

df_inject_train = df_inject[:n_train]
df_inject_test = df_inject[n_train:]


df_recover_train.to_csv(output + "train_recover.csv",header=False,index=False)
df_inject_train.to_csv(output + "train_inject.csv",header=False,index=False)

df_recover_test.to_csv(output + "test_recover.csv",header=False,index=False)
df_inject_test.to_csv(output + "test_inject.csv",header=False,index=False)

plt.hist(df_raw["M_rem"])
plt.savefig("test.png")

## generate labels for the testing set above. It can be used in classfication
df_label = df_raw['label']
df_label_test = df_label[n_train:]
df_label_test.to_csv(output + "test_label.csv",header=False,index=False)

##------------------------------------
#get some plot##
##------------------------------------
plt.close()

m1 =df_raw["mass1"]
m1r = df_raw["mass1_recover"]
m2 =df_raw["mass2"]
m2r = df_raw["mass2_recover"]


plt.figure(figsize=(12,8))
plt.subplot(2, 2, 1)
plt.scatter(m1,m2,c=np.log(df_raw['SNR']))
plt.plot(m1,m1,color='red')
plt.xlabel("m1_inject")
plt.ylabel("m2_inject")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.scatter(m1r,m2r,c=np.log(df_raw['SNR']))
plt.plot(m1r,m1r,color='red')
plt.xlabel("m1_recover")
plt.ylabel("m2_recover")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.scatter(m1r,m1,c=np.log(df_raw['SNR']))
plt.plot(m1,m1,color='red')
plt.xlabel("m1_recover")
plt.ylabel("m1_inject")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.scatter(m2r,m2,c=np.log(df_raw['SNR']))
plt.plot(m2,m2,color='red')
plt.xlabel("m2_recover")
plt.ylabel("m2_inject")
plt.colorbar()
plt.savefig(output+"masses.png")
plt.close()

plt.hist(df_raw["M_rem"])
plt.title("M_rem")
plt.savefig(output+"M_rem.png")
plt.close()

plt.hist(df_raw["label"])
plt.title("label")
plt.savefig(output+"label.png")
plt.close()
#-----------------------------------------------------------------------
print('')
print('Total runtime: {:.1f} seconds!'.format(time.time() - script_start))
print('')
