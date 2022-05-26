"""
Implementation of KNN algo for classification.

"""

import time
import csv, sys
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

    def __init__(self, haswhat=None):
        if haswhat == 'REM':
            self.metric = 'manhattan'
            self.algo = 'auto'
            self.weights = 'distance'
            self.label = 'REM'
        elif haswhat == 'NS':
            self.metric = 'manhattan'
            self.algo = 'auto'
            self.weights = 'uniform'
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


        print('*'*60)
        print('CROSS VALIDATION')
        print('-'*60)

        for k in range(0,len(K_vec)):
            neigh = KNeighborsClassifier(n_neighbors=K_vec[k], metric = metric, algorithm = algo, weights = weights)

            scores = cross_val_score(neigh, x, y.ravel(), cv=10, scoring='accuracy')
            score_vec[k] = scores.mean()

        score_max = np.amax(score_vec)
        index_max = np.argmax(score_vec)

        K_opt = K_vec[index_max]

        plt.plot(K_vec, score_vec, linestyle = 'solid',color = 'magenta')
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Cross-Validated Score')
        #plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_REM_set/plots_miq/CrossValK_REM.pdf')
        plt.show()

        print('Optimal number of kneighbors : ', K_opt)
        print('Maximum score                : ', score_max)

        self.k_opt = K_opt
        self.model = self.build_model()

        return

    def build_model(self):
        if self.label == 'REM':
            self.k_opt = 6
        elif self.label == 'NS':
            self.k_opt = 10
        return KNeighborsClassifier(n_neighbors=self.k_opt, metric = self.metric,algorithm = self.algo, weights = self.weights)

    def train_test(self):
        self.model = self.build_model()
        xtrain = self.xtrain
        ytrain = self.ytrain
        xtest = self.xtest
        ytest = self.ytest

        print('*'*60)
        t0 = time.perf_counter()
        print('Training model...')
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

    def write_probab_pred(self,filename):

        data = np.array([self.idtest,self.predict,self.probab[:,1]])
        writeResult(filename,data)
        return

    #def read_probab(filename):


    def plot_confmatrix(self):
        cm = confusion_matrix(self.ytest, self.predict)
        plt.figure(figsize = (10,7))
        sns.heatmap(cm, annot=True)
        #plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_REM_set/plots_miq/conf_matrix_REM.pdf')
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
        #plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_REM_set/plots_miq/m1m2prob_REM.pdf')
        plt.show()

        plt.figure(figsize=(7,6))
        q = m2/m1
        sc=plt.scatter(m1, q, c=prob1d, vmin=0, vmax=1, s=15, cmap='viridis')
        plt.colorbar(sc)
        plt.xlabel('m1', fontsize=18)
        plt.ylabel('q',  fontsize=18)
        #plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_REM_set/plots_miq/m1qprob_REM.pdf')
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
        fp.plotROC(ytest, proba1d)
        #plt.savefig('/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/ipam_REM_set/plots_miq/ROCplot_REM.pdf')

        return

#######################################################################

if __name__ == '__main__':

    KNN = ClassificationKNN()

    path = "/Users/miquelmiravet/Projects/IPAM_LA/ML_group/KNN_miq/"

    KNN.load_datasets(path)

    print('#'*60)
    print('Our KNN, prob Has',KNN.label)
    print('-'*60)
    print('Metric : ', KNN.metric)
    print('Algorithm : ', KNN.algo)
    print('Weights : ', KNN.weights)
    print('#'*60)

    #KNN.CrossVal()
    KNN.train_test()


    #plots and stuff

    KNN.write_probabilities()
    #KNN.write_probab_pred(path+'predictions_prob_REM.csv')

    KNN.plot_confmatrix()
    KNN.scatter_plot()
    KNN.ROC_plot()

    print('*'*60)
    print('OUR KNN')
    print('THRESHOLD    TPR(HasNS)    FPR(HasNS)')
    print('-'*60)
    thrvec = [0.07,0.27,0.51,0.80,0.94]

    for x in range(0,len(thrvec)):
        i = np.where( KNN.THR < thrvec[x])[0][0]
        print('%.3f \t\t %.3f \t\t %.3f'%(thrvec[x],KNN.TPR[i],KNN.FPR[i]))
    print('*'*60)
