"""
Final stand-alone implementation of the classfication RF using scikitlearn

"""

# TODO: - save/load model
#       - add mean errors and maybe a simple histo-plot

import os, sys, time, csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_curve
import seaborn 
from sklearn.ensemble import RandomForestClassifier
import joblib

#######################################################################
# Usual I/O functiony by Marina
#######################################################################
def extractData(filename, header=False, verbose=False):
    """ Reads data from csv file and returns it in array form.
    """
    lst=[]
    
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        if(header):
            next(csv_reader, None)
        for row in csv_reader:
            lst.append(row)
    data=np.array(lst, dtype=float)
    if verbose:
        print(filename, 'loaded')
    return data



#######################################################################
# Class for the Regression Neural Newtork
#######################################################################
class ClassificationRF:
    """ Class to do classification using a RF from scikitlearn
    """
    def __init__(self,verbose=False, save=False, show=True):
        self.verbose = verbose
        self.save_plots=save
        self.show_plots=show
        self.headers=["ID", "m1_inj" , "m2_inj", "chi1_inj", "chi2_inj", "mc_inj", "q_inj", "R_isco_inj", "Compactness_inj", "m1_rec", "m2_rec", "chi1_rec", "chi2_rec", "mc_rec", "frac_mc_err", "q_rec", "R_isco_rec", "Compactness_rec", "snr", "label"]
        return
   
    def saveModel(self, path, filename='random_forest'):
        joblib.dump(self.model, path+filename+".joblib")
        return
    
    def loadModel(self, path, filename="random_forest"):
        print("loading ",path+filename+".joblib")
        self.model = joblib.load(path+filename+".joblib")
        return

    def __check_attributes(self, attr_list):
        for i in range(0, len(attr_list)):
            attr = attr_list[i]
            if not hasattr(self, attr):
                raise ValueError ('Error: '+attr+' is not defined')
        return

    def load_dataset(self, path, fname_x='xtrain.csv', fname_y='ytrain.csv'):
        """ Load datasets in CSV format 
        """
        xtrain = extractData(path+fname_x, verbose=False)
        ytrain = extractData(path+fname_y, verbose=False)
        
        if (len(xtrain)!=len(ytrain)):
            raise ValueError('data and labels must have the same amount of rows')
        
        self.data       = xtrain
        self.labels     = ytrain
        self.Nfeatures  = len(xtrain[0,:])
        return
    
    def load_train_dataset(self, path, fname_x='xtrain.csv'):
        """ Load datasets in CSV format 
        """
        xtrain = extractData(path+fname_x, verbose=False)
        self.data_train_all       = xtrain
        self.labels_train     = xtrain[:,-1]
        return
    
    def load_original_dataset(self, path, fname_x='xtrain.csv'):
        """ Load datasets in CSV format 
        """
        xtrain = extractData(path+fname_x, verbose=False)
        self.data_train_all       = xtrain
        print("loaded")
        return
    
    def load_test_dataset(self, path, fname_x='xtrain.csv'):
        """ Load datasets in CSV format 
        """
        xtrain = extractData(path+fname_x, verbose=False)
        self.data_test_all       = xtrain
        self.labels_test     = xtrain[:,-1]
        return
        
    def subset_features(self, indexes):
        print("Training and testing using:")
        for i in indexes:
            print(self.headers[i])
        self.Nfeatures = len(indexes)
        self.data_train=self.data_train_all[:,indexes]
        self.data_test=self.data_test_all[:,indexes]
        return
    
    def split_train_test(self, pct=0.7):
        Ntrain=int(pct*len(self.data))
        
        self.data_train = self.data[:Ntrain]
        self.labels_train = self.labels[:Ntrain]
        
        self.data_test = self.data[Ntrain:]
        self.labels_test = self.labels[Ntrain:]

        if self.verbose:
            print("Using ",Ntrain, "for training, ",len(self.data) ," for testing") 
            
        return
    
    
    def crossvalidation(self, trees=np.arange(100, 1100, 100, dtype=int), info=['gini','entropy'], max_depth=np.arange(5, 40, 5, dtype=int)):
        self.__check_attributes(['data_train', 'labels_train', 'data_test', 'labels_test'])
        best_score = -1
        scores=[]
        print("trees, depth, criteria)-> score. Time")
        for tree in trees:
            for criteria in info:
                for depth in max_depth:
                    clf = RandomForestClassifier(n_estimators=tree, criterion=criteria, 
                                                 max_features='sqrt', max_depth=depth, random_state=42) 
                    t0=time.perf_counter()
                    clf.fit(self.data_train, np.ravel(self.labels_train))
                    total_time=time.perf_counter()-t0
                    score = clf.score(self.data_test,self.labels_test)
                    print("(",tree, depth, criteria,") -> {:.6f}  Time:{:.3f}s".format(score, total_time))
                    scores.append(score)
                    if score>best_score:
                        best_score = score
                        config=[tree,criteria,depth]
        
        print("Standard deviation of score during crossvalidation: ",np.std(scores),". Mean: ",np.mean(scores))
        
        print("Score ",best_score,". Optimum forest found: ",config[0]," trees, ",config[1], " criteria and ",config[2]," max depth")
        return config
        
    
    def train(self, trees=100, criterion='gini', max_depth=None): 
        """ Train the model
        """
        self.__check_attributes(['data_train', 'labels_train'])
        self.Nfeatures=len(self.data_train[0])

        self.model=RandomForestClassifier(n_estimators=trees, criterion=criterion, 
                                          max_features='sqrt', max_depth=max_depth,random_state=42) 
        self.model.fit(self.data_train, np.ravel(self.labels_train))
        
        return
    
    def compute_prediction(self, x):
        """ Classify an event, or an array of them
        """
        x = np.array(x)
        # if the input is given as a 1d-array...
        if len(x.shape)==1:
            if len(x)==self.Nfeatures:
                x = x.reshape((1,self.Nfeatures)) # ...transform as row-vec
            else:
                raise ValueError('Wrong input-dimension')
        
        prediction = self.model.predict(x) 
        
        return prediction

    def compute_metrics(self):
        """ Compute evaluation metrics: score and confusion matrix 
        """
        
        self.__check_attributes(['model','data_train', 'labels_train', 'data_test', 'labels_test'])
        
        score = self.model.score(self.data_test, self.labels_test)
        self.test_prediction = self.model.predict(self.data_test)
        test_prob = self.model.predict_proba(self.data_test)
        confusion_matrix=cm(self.labels_test, self.test_prediction, normalize='true')  #PONER LA VERSIÓN NORMALIZADA
        
        self.metric_dic={}
        self.metric_dic["score"] = score
        self.metric_dic["prob"] = test_prob
        self.metric_dic["conf_matrix"] = confusion_matrix
        
        return

    def print_metrics(self,filename='cm.png'):
        """ Print (and eventually compute) evaluation metrics 
        """
        self.compute_metrics()

        print("Score on testing: ", self.metric_dic["score"])
        print("******Confusion matrix******")
        seaborn.heatmap(self.metric_dic["conf_matrix"], annot=True)
        if self.save_plots:
            plt.savefig(filename,dpi=200,bbox_inches='tight')
        if self.show_plots:
            plt.show()
        plt.clf()
        return

    def hist_NS(self, figname="histNS"):
        plt.rcParams["font.size"]=14
        
        probs=self.model.predict_proba(self.data_test)
        pred=1-probs[:,0]
        truelabel=self.labels_test
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
        probs=self.model.predict_proba(self.data_test)
        pred=probs[:,2]
        truelabel=self.labels_test
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
        allprob = self.model.predict_proba(self.data_test)
        v_prob_NS = 1-allprob[:,0] 
        events_have_NS = np.where((self.labels_test==1) | (self.labels_test==2))[0]
        N = len(events_have_NS)
        M = len(self.labels_test) - N
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
        allprob = self.model.predict_proba(self.data_test)
        v_prob_REM = allprob[:,2]
        events_have_REM = np.where(self.labels_test==2)[0]
        N = len(events_have_REM)
        M = len(self.labels_test) - N
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
    
    

if __name__ == '__main__':

    print("wrong loading of classyRF")


