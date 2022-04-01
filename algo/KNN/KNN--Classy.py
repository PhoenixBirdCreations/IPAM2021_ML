"""
Implementation of KNN algo for classification.

"""

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
    data=np.array(lst, dtype=float)
    return data

def writeResult(filename, data):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in data:
            spamwriter.writerow(row)
    return()

def split(x,y,split_ratio=0.7):
    nevents = len(x)
    print('-'*60)
    print ('There are ',nevents,' events totally')
    n_train = int(nevents*split_ratio)
    print('-'*60)
    print ("There are ",n_train," for training")
    print ("There are ",nevents - n_train," for testing")
    print('-'*60)

    if n_train>nevents:
        print('Reduce Ntrain! Nsample=', Nsample, sep='')
        sys.exit()

    xtrain =  x[:n_train]
    ytrain =  y[:n_train]

    xtest =  x[n_train:]
    ytest =  y[n_train:]

    return (x,y,xtrain,ytrain,xtest,ytest)

#######################################################################
# Class for the Classification KNN
#######################################################################

class ClassificationKNN:

    def __init__(self, metric = 'manhattan', algorithm = 'auto',weights = 'uniform'):

        self.metric = metric
        self.algo = algorithm
        self.weights = weights


    def load_datasets(self, pathx,pathy):
        pred = pathx
        label = pathy

        X = ut.extractData(pred)
        Y = ut.extractData(label)

        X, Y = shuffle(X, Y)

        x,y,xtrain, ytrain, xtest, ytest = split(X,Y,split_ratio=0.7)

        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.x = x
        self.y = y
        return

    def CrossVal(self):
        K_max         = 300
        K_step        = 3
        K_vec         = list(i for i in range(1,K_max,K_step))
        score_vec     = np.zeros(len(K_vec))
        x             = self.x
        y             = self.y
        metric        = self.metric
        algo          = self.algo
        weights       = self.weights

        for k in range(0,len(K_vec)):
            neigh = KNeighborsClassifier(n_neighbors=K_vec[k], metric = metric, algorithm = algo, weights = weights)

            scores = cross_val_score(neigh, x, y.ravel(), cv=5, scoring='accuracy')
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

        model = self.model
        model.fit(xtrain,ytrain.ravel())

        self.score = model.score(xtest,ytest)

        self.probab = model.predict_proba(xtest)
        self.predict = model.predict(xtest)
        return

    def plot_confmatrix(self):
        cm = confusion_matrix(self.ytest, self.predict)
        plt.figure(figsize = (10,7))
        sns.heatmap(cm, annot=True)
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
        plt.show()

        plt.figure(figsize=(7,6))
        q = m2/m1
        sc=plt.scatter(m1, q, c=prob1d, vmin=0, vmax=1, s=15, cmap='viridis')
        plt.colorbar(sc)
        plt.xlabel('m1', fontsize=18)
        plt.ylabel('q',  fontsize=18)
        plt.show()

        return

    def ROC_plot(self):
        ytest = self.ytest
        proba = self.probab
        proba1d = proba[:,1]

        fp.plotROC(ytest, proba1d)

        return

#######################################################################

if __name__ == '__main__':

    KNN = ClassificationKNN(metric = 'manhattan', algorithm = 'auto',weights = 'uniform')

    path = "/Users/miquelmiravet/Projects/IPAM_LA/ML_group/IPAM2021_ML/datasets/GSTLAL_EarlyWarning_Dataset/Dataset/m1m2Mc/"

    KNN.load_datasets(path+'xtrain.csv',path+'labels_train.csv')

    KNN.CrossVal()
    KNN.train_test()

    #plots and stuff

    KNN.write_probabilities()
    KNN.plot_confmatrix()
    KNN.scatter_plot()
    KNN.ROC_plot()
