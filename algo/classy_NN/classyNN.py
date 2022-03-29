"""
Final stand-alone implementation of the regression NN using TensorFlow

The idea here is to have a short and clea{r,n} code.
Here we only implement the state-of-the-art NN.
For testing, plots and stuff see the notebooks
in the folder algo/NN_tf/
"""

import os, sys, time, csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

# usual I/O functiony by Marina
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

def writeResult(filename, data, verbose=False):
    """ Writes data predicted by trained algorithm into a csv file.
    """
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in data:
            spamwriter.writerow(row)
    if verbose:
        print(filename, 'saved')

# R2 metric
def R2metric(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred ))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1-SS_res/SS_tot
    #if tf.math.is_nan(r2):
    #    r2 = 0.
    return r2

def R2_numpy(y_true, y_pred):
    SS_res = np.sum((y_true - y_pred )**2)
    SS_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1-SS_res/SS_tot


# Linear Scaler, similar to MinMaxScaler
class LinearScaler:
    """ Linear map between [A,B] <--> [C,D] 
    """
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B 
        self.C = C
        self.D = D
        
    def transform(self,x):
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        return np.transpose((D-C)*(np.transpose(x)-A)/(B-A)+C)

    def inverse_transform(self,x):
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        return np.transpose((B-A)*(np.transpose(x)-C)/(D-C)+A)

# Neural Newtork
class NeuralNetwork:
    """ Class to do regression using a NN from Tensorflow 
    and Keras backend
    """
    def __init__(self,
                 epochs            = 100,
                 batch_size        = 64,
                 learning_rate     = 0.001, 
                 Nfeatures         = 6,
                 validation_split  = 0.1,
                 hlayers_sizes     = (100,),
                 out_intervals     = None,
                 verbose           = False
        ):

        # input
        self.epochs           = epochs
        self.batch_size       = batch_size 
        self.learning_rate    = learning_rate
        self.Nfeatures        = Nfeatures
        self.validation_split = validation_split
        self.hlayers_sizes    = hlayers_sizes
        self.verbose          = verbose

        if out_intervals is not None:
            out_intervals = np.array(out_intervals)
        self.out_intervals = out_intervals
        
        # build architecture
        self.hidden_activation = 'relu'
        self.model              = self.build_architecture()
        if verbose:
            self.print_info()

    def build_architecture(self):
        #def output_activation_linear(x):
        #    return x
        def output_activation_linear_cut(x):
            signs = K.switch(x>0, 1+x*0, -1+x*0) # x*0 in order to broadcast to correct dimension
            return K.switch(abs(x)<1, x, signs)
        def output_activation_linear_cut_lb(x):
            return K.switch(x>-1, x, -1+x*0)
        hlayers_sizes     = self.hlayers_sizes
        Nfeatures         = self.Nfeatures
        hidden_activation = self.hidden_activation
        model_input = tf.keras.Input(shape=(Nfeatures))
        # hidden layers
        x = Dense(hlayers_sizes[0], kernel_initializer='normal', activation=hidden_activation)(model_input)
        for i in range(1, len(hlayers_sizes)):
            x = Dense(hlayers_sizes[i], kernel_initializer='normal', activation=hidden_activation)(x)
        # output layer: use only lower-boundary-cut for masses 
        if Nfeatures>2:
            branchA = Dense(2, kernel_initializer='normal', activation=output_activation_linear_cut_lb)(x)
            branchB = Dense(Nfeatures-2, kernel_initializer='normal', activation=output_activation_linear_cut)(x)
            out = tf.keras.layers.concatenate([branchA, branchB])
        else:
            out = Dense(Nfeatures, kernel_initializer='normal',activation=output_activation_linear_cut)(x)
        return tf.keras.Model(model_input, out)
    
    def print_info(self):
        self.model.summary()
        return

    def load_train_dataset(self, path, xtrain_fname='xtrain.csv', ytrain_fname='ytrain.csv'):
        xtrain_notnormalized = extractData(path+xtrain_fname, verbose=False)
        ytrain_notnormalized = extractData(path+ytrain_fname, verbose=False)
        Nfeatures = self.Nfeatures
        if Nfeatures!=len(xtrain_notnormalized[0,:]):
            raise ValueError('Incompatible data size')
        # create scalers
        if self.out_intervals is None:
            print('No output-intervals specified, using MinMaxScaler')
            Ax = np.reshape(xtrain_notnormalized.min(axis=0), (Nfeatures,1))
            Bx = np.reshape(xtrain_notnormalized.max(axis=0), (Nfeatures,1))
            Ay = np.reshape(ytrain_notnormalized.min(axis=0), (Nfeatures,1))
            By = np.reshape(ytrain_notnormalized.max(axis=0), (Nfeatures,1))
        else:
            Ax = np.reshape(self.out_intervals[:,0], (Nfeatures,1)) 
            Bx = np.reshape(self.out_intervals[:,1], (Nfeatures,1)) 
            Ay = Ax
            By = Bx
        self.scaler_x = LinearScaler(Ax,Bx,-1,1)
        self.scaler_y = LinearScaler(Ay,By,-1,1)
        xtrain              = self.scaler_x.transform(xtrain_notnormalized)
        ytrain              = self.scaler_y.transform(ytrain_notnormalized)
        self.xtrain         = xtrain
        self.ytrain         = ytrain
        self.xtrain_notnorm = xtrain_notnormalized
        self.ytrain_notnorm = ytrain_notnormalized
        return
        
    def load_test_dataset(self, path, xtest_fname='xtest.csv', ytest_fname='ytest.csv'):
        xtest_notnormalized = extractData(path+xtest_fname, verbose=False)
        ytest_notnormalized = extractData(path+ytest_fname, verbose=False)
        xtest               = self.scaler_x.transform(xtest_notnormalized)
        ytest               = self.scaler_y.transform(ytest_notnormalized)
        self.xtest          = xtest
        self.ytest          = ytest
        self.xtest_notnorm  = xtest_notnormalized
        self.ytest_notnorm  = ytest_notnormalized
        return
    
    def training(self):
        loss    = MeanSquaredError()
        metrics = [loss, R2metric]
        model   = self.model
        model.compile(loss=loss, metrics=metrics, optimizer=Adam(learning_rate=self.learning_rate))
        history = model.fit(self.xtrain, self.ytrain, 
            epochs           = self.epochs, 
            batch_size       = self.batch_size,
            validation_split = self.validation_split,
            verbose          = self.verbose) 
        self.history = history
        return
    
    def prediction(self, x, store_prediction=False):
        model                  = self.model
        scaled_prediction = model.predict(x) 
        prediction        = self.scaler_y.inverse_transform(scaled_prediction)
        if store_prediction:
            self.scaled_predictiion = scaled_prediction
            self.prediction         = prediction
        return prediction, scaled_prediction

    def compute_metrics_dict(self):
        Nfeatures  = self.Nfeatures
        model      = self.model
        xtest      = self.xtest
        ytest      = self.ytest
        _, ypredicted  = self.prediction(xtest, store_prediction=False) 
        R2_vec     = np.zeros((Nfeatures,))
        for i in range(0,Nfeatures):
             R2_vec[i]  = R2_numpy(ytest[:,i], ypredicted[:,i])
        metrics         = model.metrics
        metrics_results = model.evaluate(xtest, ytest, verbose=0)
        metrics_dict    = {};
        for i in range(0, len(metrics)):
            metrics_dict[metrics[i].name] = metrics_results[i]
        metrics_dict["R2"]     = R2_vec
        metrics_dict["R2mean"] = np.mean(R2_vec)
        self.metrics_dict = metrics_dict
        return

    def print_metrics(self):
        metrics_dict = self.metrics_dict
        print('\nFinal loss     : {:.5f}'.format(metrics_dict["loss"]))
        print('Final R2 mean  : {:.5f}'.format(metrics_dict["R2mean"]))
        i = 0
        R2_vec = metrics_dict["R2"]
        for R2 in metrics_dict["R2"]:
            print('R2[{:2d}]         : {:.5f}'.format(i,R2))
            i+=1
        return

if __name__ == '__main__':

    myNN = NeuralNetwork(Nfeatures=2)








