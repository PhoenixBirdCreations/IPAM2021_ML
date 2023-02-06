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

def extractData(filename, header=False, verbose=False):
    """ Reads data from csv file and returns it in array form.
    """
    lst=[]
    
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        if(header):
            next(csv_reader, None)
        for row in csv_reader:
            lst.append(row[:])
    data=lst
    if verbose:
        print(filename, 'loaded')
    return data

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
        
    def load_real_dataset(self, path, fname):
        """ Load datasets in CSV format 
        """
        data = extractData(path+fname, header = True, verbose = False)
        indices_vars = [2,3,4,5,6]
        xtest = []
        label_test = []
        eventid = []
        graceid = []
        prob_ns = []
        prob_rem = []

        for dat in data:
            dat = np.array(dat)
            xtest.append(dat[indices_vars])
            eventid.append(dat[0])
            graceid.append(dat[1])
            prob_ns.append(dat[-2])
            prob_rem.append(dat[-1])
        self.data = np.array(xtest,dtype = float)
        self.Nfeatures  = len(self.data[0][:]) 
        self.eventid = np.array(eventid)
        self.graceid = np.array(graceid)
        self.prob_ns = np.array(prob_ns,dtype = float)
        self.prob_rem = np.array(prob_rem,dtype= float)
        return
        
    def loadModel(self, path, filename="KNN"):
        print("loading ",path+filename+".joblib")
        self.model = joblib.load(path+filename+".joblib")

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
        self.probab = self.model.predict_proba(x)

        return



    def write_probab_pred(self,filename):

        data = np.array([self.eventid,self.graceid,self.prob_ns,self.prob_rem,self.predict,1-self.probab[:,0],self.probab[:,2]])
        writeResult(filename,data)
        return
