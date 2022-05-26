"""
Implementation of KNN algo for classification.

"""

import csv, sys, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import cross_val_score
import sklearn.utils as utils
from sklearn.utils     import shuffle
from sklearn.metrics import confusion_matrix, roc_curve


sys.path.insert(0, "/Users/miquelmiravet/Projects/IPAM_LA/ML_group/IPAM2021_ML/utils")
import utils as ut
import fancyplots as fp
import realistic

import seaborn as sns
from matplotlib.ticker import PercentFormatter
from matplotlib import rc
rc('text', usetex=True)
rc('font',family='serif')

#######################################################################
# Read data, write output
#######################################################################

def extractData(filename):
    lst=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            lst.append(row)
    extracted_data = np.array(lst[1:], dtype=float)
    print('File shape : ', extracted_data.shape)
    data = extracted_data[:,[9,10,11,12,18]]
    label = extracted_data[:,19]
    return data, label

def writeResult(filename, data):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in data:
            spamwriter.writerow(row)
    return()

#######################################################################
# Class for the Classification KNN
#######################################################################

class ClassificationKNN:

    def __init__(self,haswhat = None):

        self.metric = 'mahalanobis'
        self.algo = 'auto'
        self.weights = 'distance'

        if haswhat == 'REM':
            self.label = 'REM'
        elif haswhat == 'NS':
            self.label = 'NS'

        else:
            print('Chosee HasNS (NS) or HasREM (REM)!')
            exit

    def load_datasets(self, path):
        if self.label == 'REM':
            train = path+'ipam_REM_set/train_REM.csv'
            test = path+'ipam_REM_set/test_REM.csv'
        elif self.label == 'NS':
            train = path+'ipam_NS_set/train_NS.csv'
            test = path+'ipam_NS_set/test_NS.csv'

        xtrain, ytrain = extractData(train)
        xtest, ytest = extractData(test)

        print('xtrain shape : ', xtrain.shape)
        print('xtest shape : ', xtest.shape)

        xtrain, ytrain = shuffle(xtrain, ytrain)
        xtest, ytest = shuffle(xtest, ytest)

        self.xtrain = xtrain[:]
        self.xtest = xtest[:]
        self.ytrain = ytrain[:]
        self.ytest = ytest[:]
        self.x = xtrain
        self.y = ytrain
        return

    def CrossVal(self):
        K             = 11
        x             = self.x
        y             = self.y
        metric        = self.metric
        algo          = self.algo
        weights       = self.weights

        neigh = KNeighborsClassifier(n_neighbors= K, metric = metric, metric_params = {'V': np.cov(np.swapaxes(self.x,0,1))} , algorithm = algo, weights = weights)

        print('*'*60)
        print('CROSS VALIDATION')
        print('-'*60)

        score_mean = cross_val_score(neigh, x, y.ravel(), cv=10, scoring='accuracy')

        print('Mean score                : ', score_mean.mean())

        self.k_opt = 11
        self.model = self.build_model()

        return

    def build_model(self):
        self.k_opt = 11
        return KNeighborsClassifier(n_neighbors=self.k_opt, metric = self.metric, metric_params = {'V': np.cov(np.swapaxes(self.xtrain,0,1))}, algorithm = self.algo, weights = self.weights)

    def train_test(self):
        self.model = self.build_model()
        xtrain = self.xtrain
        ytrain = self.ytrain
        xtest = self.xtest
        ytest = self.ytest

        print('*'*60)
        print('Training model...')
        t0 = time.perf_counter()
        model = self.model
        model.fit(xtrain,ytrain.ravel())
        self.train_time = time.perf_counter() - t0
        self.score = model.score(xtest,ytest)
        print('Model score : ', self.score)
        print('Training time (s) : ', self.train_time)
        print('*'*60)
        print('Predicting probabilities...')
        t0 = time.perf_counter()
        self.probab = model.predict_proba(xtest)
        self.predict = model.predict(xtest)
        self.test_time = time.perf_counter()-t0
        print('Testing time (s) : ', self.test_time)
        return

    def plot_confmatrix(self):
        cm = confusion_matrix(self.ytest, self.predict)
        plt.figure(figsize = (10,7))
        sns.heatmap(cm, annot=True)
        #plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_REM_set/plots_chatt/conf_matrix_REM.pdf')
        plt.show()
        return

    def write_probabilities(self):
        print('*'*60)
        print('Some predictions...')
        print('-'*60)
        for i in range(0,40):
            prediction = round(self.predict[i]);
            probability = self.probab
            print("Predicted: {:d}, {:8.2f} % ([{:.2f}, {:.2f}])".format(\
            prediction, probability[i, prediction]*100,
            probability[i, 0], probability[i, 1]))
        print('*'*60)
        return

    def scatter_plot(self):
        xtest = self.xtest
        proba = self.probab

        m1     = xtest[:,0]
        m2     = xtest[:,1]
        prob1d = proba[:,1]

        plt.figure(figsize=(7,6))
        sc=plt.scatter(m1, m2, c=prob1d, vmin=0, vmax=1, s=15, cmap='viridis')
        plt.colorbar(sc)
        plt.xlabel('m1', fontsize=18)
        plt.ylabel('m2', fontsize=18)
        #plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_REM_set/plots_chatt/m1m2prob_REM.pdf')
        plt.show()

        plt.figure(figsize=(7,6))
        q = m2/m1
        sc=plt.scatter(m1, q, c=prob1d, vmin=0, vmax=1, s=15, cmap='viridis')
        plt.colorbar(sc)
        plt.xlabel('m1', fontsize=18)
        plt.ylabel('q',  fontsize=18)
        #plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_REM_set/plots_chatt/m1qprob_REM.pdf')
        plt.show()

        return

    def ROC_plot(self):
        ytest = self.ytest
        proba = self.probab
        proba1d = proba[:,1]

        fpr, tpr, thresholds = roc_curve(ytest, proba1d)
        self.FPR = fpr
        self.TPR = tpr
        self.THR = thresholds

        #fp.plotROC(ytest, proba1d,'/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_REM_set/plots_chatt/ROCplot_REM.pdf')

        return

#######################################################################

if __name__ == '__main__':

    KNN = ClassificationKNN()

    path = "/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/"

    KNN.load_datasets(path)

    print('#'*60)
    print('Chatterjee KNN label Has',KNN.label)
    print('-'*60)
    print('Nº neighbors = 11')
    print('Metric : ', KNN.metric)
    print('Algorithm : ', KNN.algo)
    print('Weights : ', KNN.weights)
    print('#'*60)

    #KNN.CrossVal()
    KNN.train_test()

    #plots and stuff

    KNN.write_probabilities()
    KNN.plot_confmatrix()
    KNN.scatter_plot()
    KNN.ROC_plot()

    print('*'*60)
    print('PAPER KNN')
    print('THRESHOLD    TPR(HasREM)    FPR(HasREM)')
    print('-'*60)
    thrvec = [0.07,0.27,0.51,0.80,0.94]

    for x in range(0,len(thrvec)):
        i = np.where( KNN.THR < thrvec[x])[0][0]
        print('%.3f \t\t %.3f \t\t %.3f'%(thrvec[x],KNN.TPR[i],KNN.FPR[i]))
    print('*'*60)
