import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import time
script_start = time.time()

path = '/home/yanyan.zheng/projects/EMbright/IPAM2021_ML/datasets/GSTLAL_EarlyWarning_Dataset/Dataset/m1m2Mc/'
output = "./"

xtrain = np.loadtxt(path + 'xtrain.csv', delimiter=",")
ytrain = np.loadtxt(path + 'ytrain.csv', delimiter=",")
xtest = np.loadtxt(path + 'xtest.csv', delimiter=",")
ytest = np.loadtxt(path + 'ytest.csv', delimiter=",")
   
    
scaler = StandardScaler()
scaler.fit(xtrain)  
xtrain_s = scaler.transform(xtrain)  
# apply same transformation to test data
xtest_s = scaler.transform(xtest)

# train the model    
regr = MLPRegressor(random_state=1, max_iter=2000,activation = 'tanh',solver = 'lbfgs',hidden_layer_sizes = 100,learning_rate='invscaling').fit(xtrain_s, ytrain)
y_predict = regr.predict(xtest_s)
np.savetxt("NN_prediction.csv", y_predict, delimiter=",")


score = regr.score(xtest_s, ytest)
print ("-------------------------")
print ("The score is {:0.3f}".format(score))
print ("-------------------------")


def regrPredictionPlots(ytest, ypredicted, labels, scaler=None, show=False, save=True, figname='injVSpred.png'):
    """
    the usual injected vs predicted plots
    """
    if scaler is not None:
        ytest      = scaler.inverse_transform(ytest)
        ypredicted = scaler.inverse_transform(ypredicted)
    
    Nfeatures = len(ytest[0,:])
    if Nfeatures!=len(labels) or Nfeatures!=len(ypredicted[0,:]):
        print('Wrong input! Check shapes')
        sys.exit()

    if Nfeatures<3:
        plot_cols = Nfeatures
    else:
        plot_cols = 3
    
    rows = max(round(Nfeatures/plot_cols),1)
    if rows>1:
        fig, axs  = plt.subplots(rows, plot_cols, figsize = (25,17))
    else: 
        fig, axs  = plt.subplots(rows, plot_cols, figsize = (22,9))
    feature = 0
    for i in range(0,rows):
        for j in range(0,plot_cols):
            if feature>=Nfeatures:
                break
            if rows>1:
                ax = axs[i,j]
            else: 
                ax = axs[j]
            ytest_1d      = ytest[:,feature]
            ypredicted_1d = ypredicted[:,feature]
            diff = np.abs(ytest_1d-ypredicted_1d)
            ax.scatter(ytest_1d, ypredicted_1d, s=15, c=diff, cmap="gist_rainbow")
            ax.plot(ytest_1d, ytest_1d, 'k')
            ymax = max(ytest_1d)
            xmin = min(ytest_1d)
            if xmin<0:
                xpos = xmin*0.7
            else:
                xpos = xmin*1.3

            if ymax<0:
                ypos = ymax*0.7
            else:
                ypos = ymax*1.3
            label = labels[feature]
            ax.set_ylabel('predicted - '+label, fontsize=25)
            ax.set_xlabel('injected - '+label, fontsize=25)
            feature+=1;
            
    if save:
        plt.savefig(figname,dpi=200,bbox_inches='tight')
    if show:
        plt.show()
    
    return
columns =["mass1","mass2","q"]
regrPredictionPlots(ytest,y_predict,columns)