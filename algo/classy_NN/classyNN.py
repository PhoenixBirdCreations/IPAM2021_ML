"""
Final stand-alone implementation of the regression NN using TensorFlow

The idea here is to have a short and clea{r,n} code.
Here we only implement the state-of-the-art NN.
For testing, plots and stuff see the notebooks
in the folder algo/NN_tf/
"""

# TODO: - add mean errors and maybe a simple histo-plot
#       - maybe change logic for test-data 

import os, sys, csv, pickle, types
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

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

def writeResult(filename, data, verbose=False):
    """ Writes data predicted by trained algorithm into a csv file.
    """
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in data:
            spamwriter.writerow(row)
    if verbose:
        print(filename, 'saved')

#######################################################################
# Linear Scaler, slightly more general than MinMaxScaler
#######################################################################
class LinearScaler:
    """ Linear (vectorized) map between [A,B] <--> [C,D] 
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
    
    def print_info(self):
        for i in range(0,len(self.A)): 
            print('----------------------')
            print('Feature n.', i, sep='')
            print('A: ', self.A[i])
            print('B: ', self.B[i])
            print('C: ', self.C[i])
            print('D: ', self.D[i])
        return

    def return_dict(self):
        scaler_dict = {}
        scaler_dict['A'] = self.A
        scaler_dict['B'] = self.B
        scaler_dict['C'] = self.C
        scaler_dict['D'] = self.D
        return scaler_dict 

#######################################################################
# Class for the Regression Neural Newtork
#######################################################################
class RegressionNN:
    """ Class to do regression using a NN from Tensorflow 
    and Keras backend.
    If load_model='cool_model', load the model and the scalers saved in cool_model/ by save_model(),
    otherwise build a new model according to Nfeatures and hlayers_sizes.
    The scalers will be defined when loading the train dataset.
    """
    def __init__(self, Nfeatures=3, hlayers_sizes=(100,), out_intervals=None, load_model=None, verbose=False):
        # input
        self.Nfeatures        = Nfeatures
        self.hlayers_sizes    = hlayers_sizes
        if out_intervals is not None:
            out_intervals = np.array(out_intervals)
        self.out_intervals = out_intervals
        self.hidden_activation = 'relu'
        
        if load_model is not None:
            self.__load_model(load_model, verbose=verbose)
            if Nfeatures!=self.Nfeatures or hlayers_sizes!=self.hlayers_sizes:
                error_message  = 'Trying to load model that is incosistent with input!\n'
                error_message += 'instance input: Nfeatures={:}, hlayers_sizes={:}\n'.format(Nfeatures, hlayers_sizes)
                error_message += 'loaded model  : Nfeatures={:}, hlayers_sizes={:}\n'.format(self.Nfeatures, self.hlayers_sizes)
                raise ValueError(error_message)
        else:
            self.__build_model()

    def __check_attributes(self, attr_list):
        """ Check that all the attributes in the list
        are defined
        """
        for i in range(0, len(attr_list)):
            attr = attr_list[i]
            if not hasattr(self, attr):
                if attr=='fit_output':
                    raise ValueError ('Error: '+attr+' is not defined, no model fitting has been performed in this istance')
                else:
                    raise ValueError ('Error: '+attr+' is not defined')
        return

    def __build_model(self):
        """ Build the architecture of the NeuralNewtork
        """
        def output_activation_lin_constraint(x):
            signs = K.switch(x>0, 1+x*0, -1+x*0) # x*0 in order to broadcast to correct dimension
            return K.switch(abs(x)<1, x, signs)
        hlayers_sizes     = self.hlayers_sizes
        Nfeatures         = self.Nfeatures
        hidden_activation = self.hidden_activation
        model_input       = tf.keras.Input(shape=(Nfeatures))
        # hidden layers
        x = Dense(hlayers_sizes[0], kernel_initializer='normal', activation=hidden_activation)(model_input)
        for i in range(1, len(hlayers_sizes)):
            x = Dense(hlayers_sizes[i], kernel_initializer='normal', activation=hidden_activation)(x)
        out = Dense(Nfeatures, kernel_initializer='normal',activation=output_activation_lin_constraint)(x)
        self.model = tf.keras.Model(model_input, out)
        return
    
    def __compile_model(self):
        """ Standard compilation of the model
        """
        def R2metric(y_true, y_pred):
            SS_res = K.sum(K.square(y_true - y_pred ))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            r2 = 1-SS_res/SS_tot
            #if tf.math.is_nan(r2):
            #    r2 = 0.
            return r2
        loss    = MeanSquaredError()
        metrics = [loss, R2metric]
        self.model.compile(loss=loss, metrics=metrics, optimizer=Adam(learning_rate=self.learning_rate))
        return
    
    def __save_pickle(self, fname, mydict):
        with open(fname, 'wb') as f:
            pickle.dump(mydict, f)
        return
    
    def __load_pickle(self, fname):
        with open(fname, 'rb') as f:
            mydict = pickle.load(f)
        return mydict

    def save_model(self, model_name=None, verbose=False, overwrite=True):
        """ Save weights of the model, scalers and fit options
        """
        attr2save = ['Nfeatures', 'hlayers_sizes', 'batch_size', 'epochs', 'validation_split', \
                     'learning_rate', 'Ntrain', 'out_intervals']
        self.__check_attributes(['model', 'scaler_x', 'scaler_y']+attr2save)
        
        if model_name is None:
            model_name = 'model_Nfeatures'+str(self.Nfeatures)+'_'+datetime.today().strftime('%Y-%m-%d')
            if not overwrite:
                i = 1
                model_name_v0 = model_name
                while os.path.isdir(model_name):
                    model_name = model_name_v0 + '_v'+str(i)
                    i += 1
                if i>1:
                    print('+++ warning +++: ', model_name_v0, ' already exists and overwrite is False.\n',
                          'Renaming the new model as ', model_name, sep='')

        self.model.save_weights(model_name+'/checkpoint')
        train_info = {}
        for a in attr2save:
            train_info[a] = getattr(self, a)
        scaler_x_dict = self.scaler_x.return_dict() 
        scaler_y_dict = self.scaler_y.return_dict() 
        self.__save_pickle(model_name+'/scaler_x.pkl'  , scaler_x_dict)
        self.__save_pickle(model_name+'/scaler_y.pkl'  , scaler_y_dict)
        self.__save_pickle(model_name+'/train_info.pkl', train_info)
        if verbose:
            print(model_name, 'saved')
        return
    
    def __load_model(self, model_name, verbose=False):
        """ Load things saved by self.save_model() and compile the model
        """
        if not os.path.isdir(model_name):
            raise ValueError(model_name+' not found!')
        scaler_x_dict = self.__load_pickle(model_name+'/scaler_x.pkl')
        scaler_y_dict = self.__load_pickle(model_name+'/scaler_y.pkl')
        train_info    = self.__load_pickle(model_name+'/train_info.pkl')
        Ax = scaler_x_dict['A']
        Bx = scaler_x_dict['B']
        Cx = scaler_x_dict['C']
        Dx = scaler_x_dict['D']
        self.scaler_x = LinearScaler(Ax, Bx, Cx, Dx)
        Ay = scaler_y_dict['A']
        By = scaler_y_dict['B']
        Cy = scaler_y_dict['C']
        Dy = scaler_y_dict['D']
        self.scaler_y = LinearScaler(Ay, By, Cy, Dy)
        train_info_keys = list(train_info.keys())
        for key in train_info_keys:
            setattr(self, key, train_info[key])
        self.__build_model() 
        self.model.load_weights(model_name+'/checkpoint')
        self.__compile_model()
        if verbose:
            print(model_name, 'loaded')
        return

    def load_train_dataset(self, fname_x='xtrain.csv', fname_y='ytrain.csv', verbose=False):
        """ Load datasets in CSV format 
        """
        self.__check_attributes(['Nfeatures'])
        if hasattr(self, 'scaler_x'):
            raise RuntimeError('scaler_x is already defined, i.e. the train dataset has been already loaded.')
        xtrain_notnormalized = extractData(fname_x, verbose=verbose)
        ytrain_notnormalized = extractData(fname_y, verbose=verbose)
        Nfeatures = self.Nfeatures
        if Nfeatures!=len(xtrain_notnormalized[0,:]):
            raise ValueError('Incompatible data size')
        # create scalers
        if self.out_intervals is None:
            print('No output-intervals specified, using MinMaxScaler')
            self.out_intervals      = np.zeros((Nfeatures,2))
            self.out_intervals[:,0] = xtrain_notnormalized.min(axis=0)
            self.out_intervals[:,1] = xtrain_notnormalized.max(axis=0)
        Ax = np.reshape(xtrain_notnormalized.min(axis=0), (Nfeatures,1))
        Bx = np.reshape(xtrain_notnormalized.max(axis=0), (Nfeatures,1))
        Ay = np.reshape(self.out_intervals[:,0], (Nfeatures,1)) 
        By = np.reshape(self.out_intervals[:,1], (Nfeatures,1)) 
        ones = np.ones(np.shape(Ax))
        self.scaler_x = LinearScaler(Ax,Bx,-1*ones, ones)
        self.scaler_y = LinearScaler(Ay,By,-1*ones, ones)
        xtrain              = self.scaler_x.transform(xtrain_notnormalized)
        ytrain              = self.scaler_y.transform(ytrain_notnormalized)
        self.xtrain         = xtrain
        self.ytrain         = ytrain
        self.Ntrain         = len(xtrain[:,0])
        self.xtrain_notnorm = xtrain_notnormalized
        self.ytrain_notnorm = ytrain_notnormalized
        return
        
    def load_test_dataset(self, fname_x='xtest.csv', fname_y='ytest.csv', verbose=False):
        self.__check_attributes(['scaler_x', 'scaler_y'])
        xtest_notnormalized = extractData(fname_x, verbose=verbose)
        ytest_notnormalized = extractData(fname_y, verbose=verbose)
        xtest               = self.scaler_x.transform(xtest_notnormalized)
        ytest               = self.scaler_y.transform(ytest_notnormalized)
        self.xtest          = xtest
        self.ytest          = ytest
        self.xtest_notnorm  = xtest_notnormalized
        self.ytest_notnorm  = ytest_notnormalized
        return
    
    def training(self, verbose=False, epochs=100, batch_size=64, learning_rate=0.001, validation_split=0.):
        """ Train the model with the options given in input
        """
        self.__check_attributes(['xtrain', 'ytrain', 'model'])
        self.epochs           = epochs
        self.batch_size       = batch_size 
        self.learning_rate    = learning_rate
        self.validation_split = validation_split
        self.__compile_model()
        fit_output = self.model.fit(self.xtrain, self.ytrain, 
            epochs           = epochs, 
            batch_size       = batch_size,
            validation_split = validation_split,
            verbose          = verbose) 
        self.fit_output = fit_output
        return
    
    def compute_prediction(self, x, transform_output=False, transform_input=False):
        """ Prediction, can be used only after training
        If you want to remove the normalization, i.e. 
        to have the prediction in physical units, then use 
        transform_output=True (default is False, so that 
        NN.compute_prediction() is equivalent to model.prediction())
        If the input (i.e. x) is not already normalized, use
        transform_input = True
        """
        self.__check_attributes(['Nfeatures', 'model'])
        x = np.array(x)
        # if the input is given as a 1d-array...
        if len(x.shape)==1:
            if len(x)==self.Nfeatures:
                x = x.reshape((1,self.Nfeatures)) # ...transform as row-vec
            else:
                raise ValueError('Wrong input-dimension')

        if transform_input:
            self.__check_attributes(['scaler_x'])
            x = self.scaler_x.transform(x)
        
        prediction = self.model.predict(x) 
        
        if transform_output:
            self.__check_attributes(['scaler_y'])
            out = self.scaler_y.inverse_transform(prediction)
        else:
            out = prediction
        return out
    
    def print_summary(self):
        self.model.summary()
        return

    def print_info(self):
        attrs = dir(self)
        attr2skip = ['xtrain', 'ytrain', 'xtrain_notnorm', 'ytrain_notnorm', 
                     'xtest' , 'ytest' , 'xtest_notnorm' , 'ytest_notnorm'  ]
        for attr in attrs:
            value = getattr(self,attr)
            if (not '__' in attr) and (not type(value)==types.MethodType) and (not attr in attr2skip):
                if attr=='out_intervals':
                    out_intervals = self.out_intervals
                    print('{:20s}: ['.format(attr),end='')
                    for i in range(0,self.Nfeatures):
                        print('[{:},{:}]'.format(out_intervals[i][0],out_intervals[i][1]),end='')
                        if i<self.Nfeatures-1:
                            print(',',end='')
                    print(']')
                else:
                    print('{:20s}: {:}'.format(attr, value))
        return

    def __compute_metrics_dict(self):
        """ Compute evaluation metrics 
        """
        def R2_numpy(y_true, y_pred):
            SS_res = np.sum((y_true - y_pred )**2)
            SS_tot = np.sum((y_true - np.mean(y_true))**2)
            return 1-SS_res/SS_tot
        self.__check_attributes(['Nfeatures', 'model', 'xtest', 'ytest'])
        Nfeatures  = self.Nfeatures
        model      = self.model
        xtest      = self.xtest
        ytest      = self.ytest
        prediction = self.compute_prediction(xtest)
        R2_vec     = np.zeros((Nfeatures,))
        for i in range(0,Nfeatures):
             R2_vec[i]  = R2_numpy(ytest[:,i], prediction[:,i])
        metrics         = model.metrics # this is empty for loaded models, not a big issue
        metrics_results = model.evaluate(xtest, ytest, verbose=0)
        metrics_dict    = {};
        for i in range(0, len(metrics)):
            metrics_dict[metrics[i].name] = metrics_results[i]
        metrics_dict["R2"]     = R2_vec
        metrics_dict["R2mean"] = np.mean(R2_vec)
        self.metrics_dict = metrics_dict
        return

    def print_metrics(self):
        """ Print (and eventually compute) evaluation metrics 
        """
        if not hasattr(self, 'metrics_dict'):
            self.__compute_metrics_dict()
        metrics_dict = self.metrics_dict
        #print('\nFinal loss     : {:.5f}'.format(metrics_dict["loss"])) # problems with loaded model: loss is not defined
        print('Final R2 mean  : {:.5f}'.format(metrics_dict["R2mean"]))
        i = 0
        R2_vec = metrics_dict["R2"]
        for R2 in metrics_dict["R2"]:
            print('R2[{:2d}]         : {:.5f}'.format(i,R2))
            i+=1
        return

    def plot_predictions(self, x):
        """ Simple plot. For more 'elaborate' plots we rely
        on other modules (i.e. do not overcomplicate 
        this code with useless graphical functions)
        """
        self.__check_attributes(['Nfeatures', 'ytest_notnorm'])
        Nfeatures     = self.Nfeatures
        ytest_notnorm = self.ytest_notnorm
        prediction    = self.compute_prediction(x, transform_output=True)
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
                ytest_notnorm_1d = ytest_notnorm[:,feature]
                prediction_1d    = prediction[:,feature]
                diff = np.abs(ytest_notnorm_1d-prediction_1d)
                ax.scatter(ytest_notnorm_1d, prediction_1d, s=15, c=diff, cmap="gist_rainbow")
                ax.plot(ytest_notnorm_1d, ytest_notnorm_1d, 'k')
                ymax = max(ytest_notnorm_1d)
                xmin = min(ytest_notnorm_1d)
                if xmin<0:
                    xpos = xmin*0.7
                else:
                    xpos = xmin*1.3

                if ymax<0:
                    ypos = ymax*0.7
                else:
                    ypos = ymax*1.3
                ax.set_ylabel('predicted - '+str(feature), fontsize=25)
                ax.set_xlabel('injected - '+str(feature), fontsize=25)
                feature+=1;
            plt.show()
        return 
    
    def plot_history(self): 
        """ History plot
        history is one attribute of the ouput of model.compile() in TensorFlow
        """
        self.__check_attributes(['fit_output'])
        history_dict = self.fit_output.history
        acc      = history_dict['R2metric']
        loss     = history_dict['loss']
        epochs_plot=range(1,len(acc)+1)   
        if self.validation_split>0:
            plt.figure(figsize=(10,5))
            ax1=plt.subplot(121)
        else:
            plt.figure(figsize=(5,5))
            ax1=plt.subplot(111)
        ax1.plot(epochs_plot,acc,'b',label='Training R2')
        ax1.plot(epochs_plot,loss,'r',label='Training loss')
        ax1.set_title('loss and R2 of Training')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        if self.validation_split>0:
            val_acc  = history_dict['val_R2metric']
            val_loss = history_dict['val_loss']
            ax2=plt.subplot(122)
            ax2.plot(epochs_plot,val_acc,'b',label='Validation R2')
            ax2.plot(epochs_plot,val_loss,'r',label='Validation loss')
            ax2.set_title('loss and R2 of Validation')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('R2')
            ax2.legend()
        plt.show()
        return 
            

if __name__ == '__main__':

    out_intervals = [[1,2.2],[1,1.8],[0.9,1.6]]
    NN = RegressionNN(Nfeatures=3, hlayers_sizes=(100,), out_intervals=out_intervals)

    path = "/home/simone/repos/IPAM2021_ML/datasets/GSTLAL_EarlyWarning_Dataset/Dataset/m1m2Mc/"
    xtrain = path+'xtrain.csv'
    ytrain = path+'ytrain.csv'
    xtest  = path+'xtest.csv'
    ytest  = path+'ytest.csv'
    
    NN.load_train_dataset(fname_x=xtrain, fname_y=ytrain)

    NN.print_summary()

    NN.training(verbose=True, epochs=10, validation_split=0.)

    NN.load_test_dataset(fname_x=xtest, fname_y=ytest) 
    NN.print_metrics()

    #NN.plot_predictions(NN.xtest)
    #NN.plot_history()
    dashes = '-'*60
    print(dashes, 'Save and load test:', dashes, sep='\n')
    NN.save_model(verbose=True, overwrite=True)
    NN2 = RegressionNN(load_model='model_Nfeatures3_'+datetime.today().strftime('%Y-%m-%d'), verbose=True)
    NN2.load_test_dataset(fname_x=xtest, fname_y=ytest) 
    NN2.print_metrics() 
    
    print(dashes)
    NN.print_info()
    print(dashes)
    NN2.print_info()
