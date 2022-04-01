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

#######################################################################
# Usual I/O functiony by Marina
#######################################################################
def extractData(filename, verbose=False):
    """ Reads data from csv file and returns it in array form.
    """
    lst=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            lst.append(row)
    data=np.array(lst, dtype=float)
    if verbose:
        print(filename, 'loaded')
    return data

#def writeResult(filename, data, verbose=False):
#    """ Writes data predicted by trained algorithm into a csv file.
#    """
#    with open(filename, 'w') as csvfile:
#        spamwriter = csv.writer(csvfile, delimiter=',')
#        for row in data:
#            spamwriter.writerow(row)
#    if verbose:
#        print(filename, 'saved')





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
        
    
    def split_train_test(self, pct=0.7):
        Ntrain=int(pct*len(self.data))
        
        self.data_train = self.data[:Ntrain]
        self.labels_train = self.labels[:Ntrain]
        
        self.data_test = self.data[Ntrain:]
        self.labels_test = self.labels[Ntrain:]

        if self.verbose:
            print("Using ",Ntrain, "for training, ",len(self.data) ," for testing") 
            
        return
    
    
    def crossvalidation(self, trees=np.arange(100, 1100, 100, dtype=int), info=['gini','entropy'], features=[None,'sqrt']):
        self.__check_attributes(['data_train', 'labels_train', 'data_test', 'labels_test'])
        best_score = -1
        scores=[]
        
        for tree in trees:
            for criteria in info:
                for feature in features:
                    if self.verbose:
                        print("Doing ",tree, " trees, criterion ",criteria," and ",feature," features") 
                    clf = RandomForestClassifier(n_estimators=tree, criterion=criteria, max_features=feature) 
                    clf.fit(self.data_train, np.ravel(self.labels_train))
                    score = clf.score(self.data_test,self.labels_test)
                    scores.append(score)
                    if score>best_score:
                        best_score = score
                        config=[tree,criteria,feature]
        
        print("Standard deviation of score during crossvalidation: ",np.std(scores),". Mean: ",np.mean(scores))
        
        print("Score ",best_score,". Optimum forest found: ",config[0]," trees, ",config[1], " criteria and ",config[2]," max features")
        return config
        
    
    def train(self, trees=100, criterion='gini', max_features='sqrt'): #we could add more attributes to tune if we want
        """ Train the model
        """
        self.__check_attributes(['data_train', 'labels_train'])

        self.model=RandomForestClassifier(n_estimators=trees, criterion=criterion, max_features=max_features) 
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

    def __compute_metrics(self):
        """ Compute evaluation metrics: score and confusion matrix 
        """
        
        self.__check_attributes(['model','data_train', 'labels_train', 'data_test', 'labels_test'])
        
        score = self.model.score(self.data_test, self.labels_test)
        self.test_prediction = self.model.predict(self.data_test)
        test_prob = self.model.predict_proba(self.data_test)
        confusion_matrix=cm(self.labels_test, self.test_prediction)
        
        self.metric_dic={}
        self.metric_dic["score"] = score
        self.metric_dic["prob"] = test_prob
        self.metric_dic["conf_matrix"] = confusion_matrix
        
        return

    def print_metrics(self):
        """ Print (and eventually compute) evaluation metrics 
        """
        if not hasattr(self, 'metric_dic'):
            self.__compute_metrics()

        print("Score on testing: ", self.metric_dic["score"])
        print("******Confusion matrix******")
        seaborn.heatmap(self.metric_dic["conf_matrix"], annot=True)
        return

    def ROC_plot(self, name='roc_curve.png'):
        prob_being_1 = self.metric_dic["prob"][:,1]
        fpr, tpr, thresholds = roc_curve(self.labels_test, prob_being_1)
        plt.figure()
        plt.title("Category 1")
        sc=plt.scatter(fpr[1:-1], tpr[1:-1], c=thresholds[1:-1], cmap='viridis')
        plt.colorbar(sc)
        plt.xlabel("false positive rate", fontsize=14)
        plt.ylabel("true positive rate",  fontsize=14)

        if self.save_plots:
            plt.savefig(name,dpi=200,bbox_inches='tight')
        if self.show_plots:
            plt.show()
        plt.clf()
  

        return
    
    def scatterplot_prob(self, index_m1=0, index_m2=1,name='scatterplot.png'):
        if not hasattr(self, 'metric_dic'):
            self.__compute_metrics()
        m1 = self.data_test[:,index_m1]
        m2 = self.data_test[:,index_m2]
        prob_being_1 = self.metric_dic["prob"][:,1]
        
        sc=plt.scatter(m1, m2, c= prob_being_1, vmin=0, vmax=1, s=15, cmap='viridis')
        plt.colorbar(sc)
        plt.xlabel('m1', fontsize=18)
        plt.ylabel('m2', fontsize=18)
        if self.save_plots:
            plt.savefig(name,dpi=200,bbox_inches='tight')
        if self.show_plots:
            plt.show()
        plt.clf()
        
        return
    
    def analysis_plot(self):
        self.scatterplot_prob()
        self.ROC_plot()

if __name__ == '__main__':

    print("wrong loading of classyRF")


