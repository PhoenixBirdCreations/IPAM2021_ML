"""
Implementation of KNN algo for classification.

"""

import pandas as pd
import csv, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import cross_val_score
import sklearn.utils as utils
from sklearn.utils     import shuffle
from sklearn.metrics import confusion_matrix


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
    ID = np.array(extracted_data[:,0], dtype = int)
    return data, label, ID

def writeResult(filename, data):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for x in range(0,np.size(data, axis = 1)):
            spamwriter.writerow(data[:,x])
    return()

#######################################################################
# Class for the Classification KNN
#######################################################################

class ClassificationKNN:

    def __init__(self, metric = 'manhattan', algorithm = 'auto',weights = 'uniform'):

        self.metric = metric
        self.algo = algorithm
        self.weights = weights


    def load_datasets(self, path):
        train = path+'train_NS.csv'
        test = path+'test_NS.csv'

        xtrain, ytrain,IDtrain = extractData(train)
        xtest, ytest, IDtest = extractData(test)

        print('xtrain shape : ', xtrain.shape)
        print('xtest shape : ', xtest.shape)

        #xtrain, ytrain = shuffle(xtrain, ytrain)
        #xtest, ytest = shuffle(xtest, ytest)

        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.x = xtrain
        self.y = ytrain
        self.idtest = IDtest

        return

    def CrossVal(self):
        K_max         = 30
        K_step        = 1
        K_vec         = list(i for i in range(1,K_max,K_step))
        score_vec     = np.zeros(len(K_vec))
        x             = self.x
        y             = self.y
        metric        = self.metric
        algo          = self.algo
        weights       = self.weights

        for k in range(0,len(K_vec)):
            neigh = KNeighborsClassifier(n_neighbors=K_vec[k], metric = metric, algorithm = algo, weights = weights)

            scores = cross_val_score(neigh, x, y.ravel(), cv=10, scoring='accuracy')
            score_vec[k] = scores.mean()

        score_max = np.amax(score_vec)
        index_max = np.argmax(score_vec)

        K_opt = K_vec[index_max]

        print('*'*60)
        print('CROSS VALIDATION')
        print('-'*60)

        plt.plot(K_vec, score_vec, linestyle = 'solid',color = 'magenta')
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Cross-Validated Score')
        plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_NS_set/plots_miq/CrossValK.pdf')
        plt.show()

        print('Optimal number of kneighbors : ', K_opt)
        print('Maximum score                : ', score_max)
        print('*'*60)

        self.k_opt = K_opt
        self.model = self.build_model()

        return

    def build_model(self):
        return KNeighborsClassifier(n_neighbors=self.k_opt, metric = self.metric, algorithm = self.algo, weights = self.weights)

    def train_test(self):
        xtrain = self.xtrain
        ytrain = self.ytrain
        xtest = self.xtest
        ytest = self.ytest

        print('*'*60)
        print('Training model...')
        model = self.model
        model.fit(xtrain,ytrain.ravel())
        self.score = model.score(xtest,ytest)
        print('Model score : ', self.score)
        print('*'*60)

        self.probab = model.predict_proba(xtest)
        self.predict = model.predict(xtest)

        return

    def write_probab_pred(self,filename):

        data = np.array([self.idtest,self.predict,self.probab[:,1]])
        writeResult(filename,data)
        return

    #def read_probab(filename):


    def plot_confmatrix(self):
        cm = confusion_matrix(self.ytest, self.predict)
        plt.figure(figsize = (10,7))
        sns.heatmap(cm, annot=True)
        plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_NS_set/plots_miq/conf_matrix.pdf')
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
        plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_NS_set/plots_miq/m1m2prob.pdf')
        plt.show()

        plt.figure(figsize=(7,6))
        q = m2/m1
        sc=plt.scatter(m1, q, c=prob1d, vmin=0, vmax=1, s=15, cmap='viridis')
        plt.colorbar(sc)
        plt.xlabel('m1', fontsize=18)
        plt.ylabel('q',  fontsize=18)
        plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_NS_set/plots_miq/m1qprob.pdf')
        plt.show()

        return

    def ROC_plot(self):
        ytest = self.ytest
        proba = self.probab
        proba1d = proba[:,1]

        fp.plotROC(ytest, proba1d,'/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_NS_set/plots_miq/ROCplot.pdf')

        return

#######################################################################

if __name__ == '__main__':

    KNN = ClassificationKNN(metric = 'manhattan', algorithm = 'auto',weights = 'uniform')

    path = "/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_NS_set/"

    KNN.load_datasets(path)

    KNN.CrossVal()
    KNN.train_test()

    #plots and stuff

    KNN.write_probabilities()
    KNN.write_probab_pred(path+'predictions_prob_NS.csv')

    KNN.plot_confmatrix()
    KNN.scatter_plot()
    KNN.ROC_plot()
