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
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import joblib


sys.path.insert(0, "/home/miquelmiravet/University/Doctorat/Projects/IPAM_ML/IPAM2021_ML/utils")
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
    data = extracted_data[:,[1,2,3,4,8]]
    label = extracted_data[:,[9,10]]
#    data = extracted_data[:,[9,10,11,12,18]]
#    label = extracted_data[:,[19,20]]
#    ID = np.array(extracted_data[:,0], dtype = int)
    return data, label

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

    def __init__(self, save=False, show=True):

        self.save_plots=save
        self.show_plots=show

        return

    def load_datasets(self, path_train, path_test,path_orig):

        xtrain, ytrain = extractData(path_train)
        xtest, ytest = extractData(path_test)
        xcross, ycross = extractData(path_orig)

        self.Nfeatures  = len(xtrain[0,:])

        print('*'*60)
        print('Loading data...')

        print('xtrain shape : ', xtrain.shape)
        print('xtest shape : ', xtest.shape)

        print('Datasets loaded!')
        print('*'*60)

        self.xtrain = xtrain
        self.xtest = xtest
        self.label_train = ytrain
        self.label_test = ytest
        self.x_cross = xcross
        self.label_cross = ycross

        return

    def load_original_dataset(self,path_orig,name_file):

        x_orig, y_orig = extractData(path_orig+name_file)
        self.Nfeatures  = len(x_orig[0,:])
        N = len(x_orig[:,0])
        test_N=int(np.floor(0.3*N))
        indexes_test=list(np.arange(0,test_N));
        all_indexes=(np.arange(0,N)).tolist()
        indexes_train=list(set(all_indexes) - set(indexes_test))

        self.xtrain = x_orig[indexes_train,:]
        labelREM = y_orig[indexes_train,1]
        labelBNS = y_orig[indexes_train,0]
        self.label_train = labelREM+labelBNS

        self.xtest = x_orig[indexes_test,:]
        labelREM = y_orig[indexes_test,1]
        labelBNS = y_orig[indexes_test,0]
        self.label_test = labelREM+labelBNS

        print('*'*60)
        print('Loading data...')
        print('Nº of features: ', self.Nfeatures)
        print('Nº of events: ', N)
        print('Nº of events for training: ', N-test_N)
        print('Nº of events for testing: ', test_N)
        print('Datasets loaded!')
        print('*'*60)

        self.x_cross = x_orig
        self.label_cross = y_orig[:,0]+y_orig[:,1]

        return

    def saveModel(self, path, filename='KNN'):
        joblib.dump(self.model, path+filename+".joblib")

        return

    def loadModel(self, path, filename="KNN"):
        print("loading ",path+filename+".joblib")
        self.model = joblib.load(path+filename+".joblib")

        return


    def CrossVal(self):
        K_max         = 20
        K_step        = 1
        K_min         = 3
        K_vec         = list(i for i in range(K_min,K_max,K_step))
        score_vec     = []
        best_score    = -1
        x             = self.x_cross
        y             = self.label_cross
        metrics       = ['euclidean','manhattan','cityblock']
        #metrics       = ['euclidean','haversine','manhattan']
        algo          = ['ball_tree','kd_tree','brute','auto']
        weights       = ['uniform','distance']


        print('*'*60)
        print('CROSS VALIDATION')
        print('-'*60)

        for k in K_vec:
            for m in metrics:
                for alg in algo:
                    for w in weights:
                        print("Doing ",k, " neighbors, metric ",m," algo ",alg," and ", w," weights")
                        t0=time.perf_counter()
                        neigh = KNeighborsClassifier(n_neighbors= k, metric = m, algorithm = alg, weights = w)
                        scores = cross_val_score(neigh, x, y.ravel(), cv=10, scoring='accuracy')
                        total_time=time.perf_counter()-t0
                        print("Time elapsed: ",total_time, "s")
                        print('-'*60)
                        score_vec.append(scores.mean())
                        if scores.mean()>best_score:
                            best_score = scores.mean()
                            config=[k,m,alg,w]

        print("Standard deviation of score during crossvalidation: ",np.std(score_vec),". Mean: ",np.mean(score_vec))
        print('Optimal number of kneighbors : ', config[0])
        print('Optimal metric : ', config[1])
        print('Optimal algorithm : ', config[2])
        print('Optimal weights: ', config[3])
        print('Maximum score                : ', best_score)

        self.optimal = {}
        self.optimal["k"] = config[0]
        self.optimal["metric"] = config[1]
        self.optimal["algo"] = config[2]
        self.optimal["weight"] = config[3]

        return

    def GridSearchCrossVal(self):

        K_max         = 20
        K_step        = 1
        K_min         = 3
        K_vec         = list(i for i in range(K_min,K_max,K_step))
        score_vec     = []
        best_score    = -1
        x             = self.x_cross
        y             = self.label_cross

        parameters = {'n_neighbors':K_vec,'weights':('uniform','distance'),'algorithm':('ball_tree','kd_tree','brute','auto'),'metric':('euclidean','manhattan','cityblock')}


        knn = KNeighborsClassifier()
        clf = GridSearchCV(knn, parameters,refit = True, cv = 10)
        clf.fit(x,y)

        best_score = clf.best_score_
        best_params = clf.best_params_
        print('Best score: ',best_score)
        print('Optimal parameters: ')
        sorted(best_params.keys())

        self.model = clf.best_estimator_

        print('*'*60)
        t0 = time.perf_counter()
        print('Training best model...')
        self.model.fit(self.xtrain,self.label_train.ravel())
        self.train_time = time.perf_counter() - t0
        print('Training time (s) : ', self.train_time)
        print('*'*60)

        return


    def build_train_model(self, n_neigh , metric, algo, weights):

        self.model = KNeighborsClassifier(n_neighbors=n_neigh, metric = metric,algorithm = algo, weights = weights)

        print('*'*60)
        t0 = time.perf_counter()
        print('Training model...')
        self.model.fit(self.xtrain,self.label_train.ravel())
        self.train_time = time.perf_counter() - t0
        print('Training time (s) : ', self.train_time)
        print('*'*60)

        return

    def predict_model(self,x):

        x = np.array(x)
        # if the input is given as a 1d-array...
        if len(x.shape)==1:
            if len(x)==self.Nfeatures:
                x = x.reshape((1,self.Nfeatures)) # ...transform as row-vec
            else:
                raise ValueError('Wrong input-dimension')

        self.predict = self.model.predict(x)

        return


    def compute_metrics(self):

        self.metrics = {}
        self.predict_model(self.xtest)
        self.metrics["score"] = self.model.score(self.xtest,self.label_test)
        print('Model score : ', self.metrics["score"])
        print('Predicting probabilities...')
        t0 = time.perf_counter()
        self.metrics["probab"] = self.model.predict_proba(self.xtest)
        self.test_time = time.perf_counter()-t0
        print('Testing time (s) : ', self.test_time)

        self.metrics["cm"] = confusion_matrix(self.label_test, self.predict,normalize='true')
        cm = self.metrics["cm"]
        fnr = cm.sum(axis=1) - np.diag(cm)
        fprr = cm.sum(axis=0) - np.diag(cm)
        tp = np.diag(cm)
        sens = tp/(tp+fnr)
        self.SENS = sens[1]
        precision = tp/(tp+fprr)
        self.PREC = precision[1]
        f1 = 2*precision*sens/(precision+sens)
        self.F1 = f1[1]

        return

    def write_probab_pred(self,filename):

        data = np.array([self.idtest,self.predict,self.metrics["probab"][:,1]])
        writeResult(filename,data)
        return

    def plot_confmatrix(self):
        cm = confusion_matrix(self.label_test, self.predict,normalize='true')
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
            probability = self.metrics["probab"]
            print("Predicted: {:d}, {:8.2f} % ([{:.2f}, {:.2f}])".format(\
            prediction, probability[i, prediction]*100,
            probability[i, 0], probability[i, 1]))
        print('*'*60)
        return


    def hist_NS(self, figname="histNS"):
        plt.rcParams["font.size"]=14

        probs=self.model.predict_proba(self.xtest)
        pred=1-probs[:,0]
        truelabel=self.label_test
        index_events_has=(np.where((truelabel == 1) | (truelabel == 2)))
        p_events_has=pred[index_events_has]
        index_events_nohas=np.where(truelabel==0)
        p_events_nohas=pred[index_events_nohas]

        plt.hist(p_events_nohas,bins=np.linspace(0,1,20),color='green',alpha=0.5, label='No NS')
        plt.hist(p_events_has,bins=np.linspace(0,1,20),color=(0.1, 0.2, 0.5, 0.),edgecolor='black', hatch="/",label='Has NS')
        plt.yscale('log')
        plt.yticks([1e2,1e3,1e4,1e5])
        plt.xticks([0,0.2,0.4,0.6,0.8,1])
        plt.ylim([4.5e1,1.9e5])
        plt.xlabel('P(HasNS)')
        plt.axvline(x=0.5,color='black',ls='--')
        plt.grid(ls='--')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1,0]
        #add legend to plot
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=12,loc=1)

        if self.save_plots:
            plt.savefig(figname+".png",dpi=200,bbox_inches='tight')
        if self.show_plots:
            plt.show()

        return

    def hist_REM(self, figname="histREM"):
        probs=self.model.predict_proba(self.xtest)
        pred=probs[:,2]
        truelabel=self.label_test
        index_events_has=np.where(truelabel == 2)
        p_events_has=pred[index_events_has]
        index_events_nohas=np.where((truelabel==0) | (truelabel==1))
        p_events_nohas=pred[index_events_nohas]

        plt.hist(p_events_nohas,bins=np.linspace(0,1,20),color='green',alpha=0.5, label='No Remnant')
        plt.hist(p_events_has,bins=np.linspace(0,1,20),color=(0.1, 0.2, 0.5, 0.),edgecolor='black', hatch="/",label='Has Remnant')
        plt.yscale('log')
        plt.yticks([1e2,1e3,1e4,1e5])
        plt.xticks([0,0.2,0.4,0.6,0.8,1])
        plt.ylim([4.5e1,1.9e5])
        plt.xlabel('P(HasRemnant)')
        plt.axvline(x=0.5,color='black',ls='--')
        plt.grid(ls='--')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1,0]
        #add legend to plot
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=12,loc=1)

        if self.save_plots:
            plt.savefig(figname+".png",dpi=200,bbox_inches='tight')
        if self.show_plots:
            plt.show()
        return

    def ROC_NS(self, thr_wanted = [], figname="ROC_NS"):
        allprob = self.model.predict_proba(self.xtest)
        v_prob_NS = 1-allprob[:,0]
        events_have_NS = np.where((self.label_test==1) | (self.label_test==2))[0]
        N = len(events_have_NS)
        M = len(self.label_test) - N
        threshold = np.linspace(0,1,101)[1:-1]
        TP = np.zeros(99)
        FP = np.zeros(99)
        i=0
        for thr in threshold:
            index_say_yes = np.where(v_prob_NS>=thr)[0]
            count_yes = 0.0; count_no = 0.0
            for index in index_say_yes:
                if index in events_have_NS:
                    count_yes=count_yes + 1.0
                else:
                    count_no=count_no + 1.0
            TP[i] = count_yes/N
            FP[i] = count_no/M
            if (len(thr_wanted)>0 and (thr in thr_wanted)):
                print("Threshold ",thr, "TP: {:.3f}, FP {:.3f}".format(TP[i], FP[i]))
            i = i + 1

        plt.figure()
        sc=plt.scatter(FP, TP, c=threshold, cmap='viridis')
        plt.colorbar(sc, label="Threshold")
        plt.xlabel("False Positive")
        plt.ylabel("True Positive")
        plt.grid(ls='--')
        plt.ylim([0.82,1.02])
        plt.xlim([0,0.2])
        plt.yticks(np.linspace(0.825,1,8))
        if self.save_plots:
            plt.savefig(figname+'.png',dpi=200,bbox_inches='tight')
        if self.show_plots:
            plt.show()
        return np.asarray(FP), np.asarray(TP), threshold

    def ROC_REM(self, thr_wanted = [], figname="ROC_REM"):
        allprob = self.model.predict_proba(self.xtest)
        v_prob_REM = allprob[:,2]
        events_have_REM = np.where(self.label_test==2)[0]
        N = len(events_have_REM)
        M = len(self.label_test) - N
        threshold = np.linspace(0,1,101)[1:-1]
        TP = np.zeros(99)
        FP = np.zeros(99)
        i=0
        for thr in threshold:
            index_say_yes = np.where(v_prob_REM>=thr)[0]
            count_yes = 0.0; count_no = 0.0
            for index in index_say_yes:
                if index in events_have_REM:
                    count_yes=count_yes + 1.0
                else:
                    count_no=count_no + 1.0
            TP[i] = count_yes/N
            FP[i] = count_no/M
            if (len(thr_wanted)>0 and (thr in thr_wanted)):
                print("Threshold ",thr, "TP: {:.3f}, FP {:.3f}".format(TP[i], FP[i]))
            i = i + 1

        plt.figure()
        sc=plt.scatter(FP, TP, c=threshold, cmap='viridis')
        plt.colorbar(sc, label="Threshold")
        plt.xlabel("False Positive")
        plt.ylabel("True Positive")
        plt.grid(ls='--')
        plt.ylim([0.82,1.02])
        plt.xlim([0,0.2])
        plt.yticks(np.linspace(0.825,1,8))
        if self.save_plots:
            plt.savefig(figname+'.png',dpi=200,bbox_inches='tight')
        if self.show_plots:
            plt.show()
        return np.asarray(FP), np.asarray(TP), threshold


#######################################################################
