"""
Final stand-alone implementation of the regression NN using TensorFlow

The idea here is to have a short and clea{r,n} code.
Here we only implement the state-of-the-art NN.
For testing, plots and stuff see the notebooks
in the folder algo/NN_tf/

Update: The plan failed. The code is not short and clear.
"""

import os, sys, csv, types, dill, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from keras.utils.layer_utils import count_params
from keras.initializers import RandomNormal
from scipy import stats
from sklearn.preprocessing import QuantileTransformer

#######################################################################
# Default values used in the classes RegressionNN and CrossValidator
#######################################################################
NFEATURES        = 3
EPOCHS           = 25
BATCH_SIZE       = 64
HLAYERS_SIZES    = (100,)
LEARNING_RATE    = 0.001
VALIDATION_SPLIT = 0.
SEED             = None

#######################################################################
# Usual I/O functions by Marina
#######################################################################
def extract_data(filename, verbose=False, skip_header=False):
    """ Reads data from csv file and returns it in array form.
    """
    lst=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            lst.append(row)
    if skip_header:
        lst = lst[1:]
    data=np.array(lst, dtype=float)
    if verbose:
        print(filename, 'loaded')
    return data

def write_result(filename, data, verbose=False):
    """ Writes data predicted by trained algorithm into a csv file.
    """
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in data:
            spamwriter.writerow(row)
    if verbose:
        print(filename, 'saved')
    
#######################################################################
# Save/load dictionary that eventually contains lambda-objects
#######################################################################
def save_dill(fname, mydict, verbose=False):
    out_dict_dir = './'
    dict_name = out_dict_dir+fname
    dill.dump(mydict, open(dict_name, 'wb'))
    if verbose:
        print(dict_name, 'saved') 
    return 

def load_dill(fname, verbose=False):
    out_dict_dir = './'
    dict_name = out_dict_dir+fname
    if os.path.exists(dict_name):
        loaded_dict = dill.load(open(dict_name, 'rb'))
        if verbose:
            print(dict_name, 'loaded')
    else:
        loaded_dict = {}
        if verbose:
            print(dict_name, 'not found, returning empty dictionary')
    return loaded_dict    

#######################################################################
# Scaler
#######################################################################
class CustomScaler:
    """ Linear (vectorized) map between [A,B] <--> [C,D]
    You can also use a quantile-scaler befor linear mapping
    """
    def __init__(self, A, B, C, D, quantile=False, qscaler=None):
        self.A = A
        self.B = B 
        self.C = C
        self.D = D
        self.quantile = quantile # flag
        self.qscaler  = qscaler
        if qscaler is not None and quantile is False:
            raise ValueError('Quantile scaler given in input but quantile flag is False!')

    def transform(self,x):
        if self.quantile:
            if self.qscaler is None:
                qscaler = QuantileTransformer(output_distribution='normal')
                self.qscaler = qscaler.fit(x)
            x = self.qscaler.transform(x)
            
            # update A and B!
            A = np.array(self.A)
            B = np.array(self.B)
            new_A = self.qscaler.transform(A.reshape(1, -1))
            new_B = self.qscaler.transform(B.reshape(1, -1))
            nfeatures = len(self.A)
            self.A = new_A.reshape(nfeatures,1)
            self.B = new_B.reshape(nfeatures,1)

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
        y = np.transpose((B-A)*(np.transpose(x)-C)/(D-C)+A)
        if self.quantile:
            y = self.qscaler.inverse_transform(y)
        return y

    def print_info(self):
        print('quantile : ', self.quantile)
        for i in range(0,len(self.A)): 
            print('----------------------')
            print('Feature n.', i, sep='')
            print('A        : ', self.A[i])
            print('B        : ', self.B[i])
            print('C        : ', self.C[i])
            print('D        : ', self.D[i])
        return

    def return_dict(self):
        scaler_dict = {}
        scaler_dict['A']        = self.A
        scaler_dict['B']        = self.B
        scaler_dict['C']        = self.C
        scaler_dict['D']        = self.D
        scaler_dict['quantile'] = self.quantile
        scaler_dict['qscaler']  = self.qscaler
        return scaler_dict 

#######################################################################
# Class for the Regression Neural Newtork
#######################################################################
class RegressionNN:
    """ Class to do regression using a NN from Tensorflow 
    and Keras backend.
    If load_model='cool_model', load the model and the scalers saved in cool_model/ by save_model(),
    otherwise build a new model according to nfeatures and hlayers_sizes.
    The scalers will be defined when loading the train dataset.
    """
    def __init__(self, nfeatures=NFEATURES, hlayers_sizes=HLAYERS_SIZES, out_intervals=None, load_model=None, verbose=False,
                 seed=SEED):
        # input
        self.nfeatures        = nfeatures
        self.hlayers_sizes    = hlayers_sizes
        if out_intervals is not None:
            out_intervals = np.array(out_intervals)
        self.out_intervals = out_intervals
        self.hidden_activation = 'relu'
        
        if seed is None:
            seed = np.random.randint(1,10000)
        self.seed = seed
        if load_model is not None:
            self.__load_model(load_model, verbose=verbose)
            if nfeatures!=self.nfeatures or hlayers_sizes!=self.hlayers_sizes:
                error_message  = 'Trying to load model that is incosistent with input!\n'
                error_message += 'instance input: nfeatures={:}, hlayers_sizes={:}\n'.format(nfeatures, hlayers_sizes)
                error_message += 'loaded model  : nfeatures={:}, hlayers_sizes={:}\n'.format(self.nfeatures, self.hlayers_sizes)
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
        nfeatures         = self.nfeatures
        hidden_activation = self.hidden_activation
        seed              = self.seed
        model_input       = tf.keras.Input(shape=(nfeatures))
        # hidden layers
        x = Dense(hlayers_sizes[0], kernel_initializer=RandomNormal(seed=seed), activation=hidden_activation)(model_input)
        nlayers = len(hlayers_sizes)
        for i in range(1, nlayers):
            x = Dense(hlayers_sizes[i], kernel_initializer=RandomNormal(seed=seed+i), activation=hidden_activation)(x)
        out = Dense(nfeatures, kernel_initializer=RandomNormal(seed=seed+nlayers),activation=output_activation_lin_constraint)(x)
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
    
    def save_model(self, model_name=None, verbose=False, overwrite=True):
        """ Save weights of the model, scalers and fit options
        """
        attr2save = ['nfeatures', 'hlayers_sizes', 'batch_size', 'epochs', 'validation_split', \
                     'learning_rate', 'ntrain', 'out_intervals', 'seed', 'training_time']
        self.__check_attributes(['model', 'scaler_x', 'scaler_y']+attr2save)
        if model_name is None:
            model_name = 'model_nfeatures'+str(self.nfeatures)+'_'+datetime.today().strftime('%Y-%m-%d')
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
        save_dill(model_name+'/scaler_x.dill'  , scaler_x_dict)
        save_dill(model_name+'/scaler_y.dill'  , scaler_y_dict)
        save_dill(model_name+'/train_info.dill', train_info)
        if verbose:
            print(model_name, 'saved')
        return
    
    def __load_model(self, model_name, verbose=False):
        """ Load things saved by self.save_model() and compile the model
        """
        if not os.path.isdir(model_name):
            raise ValueError(model_name+' not found!')
        scaler_x_dict = load_dill(model_name+'/scaler_x.dill')
        scaler_y_dict = load_dill(model_name+'/scaler_y.dill')
        train_info    = load_dill(model_name+'/train_info.dill')
        Ax = scaler_x_dict['A']
        Bx = scaler_x_dict['B']
        Cx = scaler_x_dict['C']
        Dx = scaler_x_dict['D']
        quantile = scaler_x_dict['quantile']
        qscaler  = scaler_x_dict['qscaler']
        self.scaler_x = CustomScaler(Ax, Bx, Cx, Dx, quantile=quantile, qscaler=qscaler)
        Ay = scaler_y_dict['A']
        By = scaler_y_dict['B']
        Cy = scaler_y_dict['C']
        Dy = scaler_y_dict['D']
        quantile = scaler_y_dict['quantile']
        qscaler  = scaler_y_dict['qscaler']
        self.scaler_y = CustomScaler(Ay, By, Cy, Dy, quantile=quantile, qscaler=qscaler)
        train_info_keys = list(train_info.keys())
        for key in train_info_keys:
            setattr(self, key, train_info[key])
        self.__build_model() 
        self.model.load_weights(model_name+'/checkpoint')
        self.__compile_model()
        if verbose:
            print(model_name, 'loaded')
        return

    def load_train_dataset(self, fname_xtrain='xtrain.csv', fname_ytrain='ytrain.csv', xtrain_data=None, ytrain_data=None, 
                           verbose=False, quantile=False):
        """ Load datasets in CSV format 
        """
        self.__check_attributes(['nfeatures'])
        if hasattr(self, 'scaler_x'):
            raise RuntimeError('scaler_x is already defined, i.e. the train dataset has been already loaded.')

        if xtrain_data is None:
            xtrain_notnormalized = extract_data(fname_xtrain, verbose=verbose)
        else:
            xtrain_notnormalized = xtrain_data
        if ytrain_data is None:
            ytrain_notnormalized = extract_data(fname_ytrain, verbose=verbose)
        else:
            ytrain_notnormalized = ytrain_data

        nfeatures = self.nfeatures
        if nfeatures!=len(xtrain_notnormalized[0,:]):
            raise ValueError('Incompatible data size')
        # create scalers
        if self.out_intervals is None:
            if verbose:
                print('No output-intervals specified, using MinMaxScaler')
            self.out_intervals      = np.zeros((nfeatures,2))
            self.out_intervals[:,0] = xtrain_notnormalized.min(axis=0)
            self.out_intervals[:,1] = xtrain_notnormalized.max(axis=0)
        Ax = np.reshape(xtrain_notnormalized.min(axis=0), (nfeatures,1))
        Bx = np.reshape(xtrain_notnormalized.max(axis=0), (nfeatures,1))
        Ay = np.reshape(self.out_intervals[:,0], (nfeatures,1)) 
        By = np.reshape(self.out_intervals[:,1], (nfeatures,1)) 
        ones = np.ones(np.shape(Ax))
        self.scaler_x       = CustomScaler(Ax,Bx,-1*ones, ones, quantile=quantile)
        self.scaler_y       = CustomScaler(Ay,By,-1*ones, ones, quantile=quantile)
        xtrain              = self.scaler_x.transform(xtrain_notnormalized)
        ytrain              = self.scaler_y.transform(ytrain_notnormalized)
        self.xtrain         = xtrain
        self.ytrain         = ytrain
        self.ntrain         = len(xtrain[:,0])
        self.xtrain_notnorm = xtrain_notnormalized
        self.ytrain_notnorm = ytrain_notnormalized
        return
        
    def load_test_dataset(self, fname_xtest='xtest.csv', fname_ytest='ytest.csv', xtest_data=None, ytest_data=None, verbose=False):
        self.__check_attributes(['scaler_x', 'scaler_y'])
        if xtest_data is None:
            xtest_notnormalized = extract_data(fname_xtest, verbose=verbose)
        else:
            xtest_notnormalized = xtest_data
        if ytest_data is None:
            ytest_notnormalized = extract_data(fname_ytest, verbose=verbose)
        else:
            ytest_notnormalized = ytest_data
        xtest               = self.scaler_x.transform(xtest_notnormalized)
        ytest               = self.scaler_y.transform(ytest_notnormalized)
        self.xtest          = xtest
        self.ytest          = ytest
        self.xtest_notnorm  = xtest_notnormalized
        self.ytest_notnorm  = ytest_notnormalized
        return
    
    def training(self, verbose=False, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                 learning_rate=LEARNING_RATE, validation_split=VALIDATION_SPLIT):
        """ Train the model with the options given in input
        """
        self.__check_attributes(['xtrain', 'ytrain', 'model', 'seed'])
        self.epochs           = epochs
        self.batch_size       = batch_size 
        self.learning_rate    = learning_rate
        self.validation_split = validation_split
        tf.random.set_seed(self.seed)
        self.__compile_model()
        t0 = time.perf_counter()
        fit_output = self.model.fit(self.xtrain, self.ytrain, 
            epochs           = epochs, 
            batch_size       = batch_size,
            validation_split = validation_split,
            verbose          = verbose) 
        self.training_time = time.perf_counter()-t0
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
        self.__check_attributes(['nfeatures', 'model'])
        x = np.array(x)
        if len(x.shape)==1:
            # if the input is given as a 1d-array...
            if len(x)==self.nfeatures:
                x = x.reshape((1,self.nfeatures)) # ...transform as row-vec
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
                    for i in range(0,self.nfeatures):
                        print('[{:},{:}]'.format(out_intervals[i][0],out_intervals[i][1]),end='')
                        if i<self.nfeatures-1:
                            print(',',end='')
                    print(']')
                else:
                    print('{:20s}: {:}'.format(attr, value))
        return

    def compute_metrics_dict(self,x,y):
        """ Compute evaluation metrics 
        """
        def R2_numpy(y_true, y_pred):
            SS_res = np.sum((y_true - y_pred )**2)
            SS_tot = np.sum((y_true - np.mean(y_true))**2)
            return 1-SS_res/SS_tot
        self.__check_attributes(['nfeatures', 'model'])
        nfeatures   = self.nfeatures
        model       = self.model
        prediction  = self.compute_prediction(x)
        R2_vec      = np.zeros((nfeatures,))
        for i in range(0,nfeatures):
             R2_vec[i]        = R2_numpy(y[:,i], prediction[:,i])
        metrics         = model.metrics # this is empty for loaded models, not a big issue
        metrics_results = model.evaluate(x, y, verbose=0)
        metrics_dict    = {};
        for i in range(0, len(metrics)):
            metrics_dict[metrics[i].name] = metrics_results[i]
        metrics_dict["R2"]          = R2_vec
        metrics_dict["R2mean"]      = np.mean(R2_vec)
        return metrics_dict

    def print_metrics(self):
        """ Print (and eventually compute) evaluation metrics 
        """
        self.__check_attributes(['xtest', 'ytest'])
        metrics_dict = self.compute_metrics_dict(self.xtest,self.ytest)
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
        self.__check_attributes(['nfeatures', 'ytest_notnorm'])
        nfeatures     = self.nfeatures
        ytest_notnorm = self.ytest_notnorm
        prediction    = self.compute_prediction(x, transform_output=True)
        if nfeatures<3:
            plot_cols = nfeatures
        else:
            plot_cols = 3
        rows = int(np.ceil(nfeatures/plot_cols))
        if rows>1:
            fig, axs = plt.subplots(rows, plot_cols, figsize = (25,17))
        else: 
            fig, axs = plt.subplots(rows, plot_cols, figsize = (22,9))
        feature = 0
        for i in range(0,rows):
            for j in range(0,plot_cols):
                if feature>=nfeatures:
                    break
                if rows>1:
                    ax = axs[i,j]
                else: 
                    ax = axs[j]
                ytest_notnorm_1d = ytest_notnorm[:,feature]
                prediction_1d    = prediction[:,feature]
                diff = np.abs(ytest_notnorm_1d-prediction_1d)
                ax.scatter(ytest_notnorm_1d, prediction_1d, s=2, c=diff, cmap="gist_rainbow")
                ax.plot(ytest_notnorm_1d, ytest_notnorm_1d, 'k')
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
    
    def plot_err_histogram(self, feature_idx=0, color_rec=[0.7,0.7,0.7], color_pred=[0,1,0], nbins=31, 
                           logscale=False, name=None, abs_diff=False, fmin=None, fmax=None, verbose=False,
                           alpha_rec=1, alpha_pred=0.5):
        """ Plot error-histogram for one feature. 
        The feature is chosen by feature_idx
        """
        self.__check_attributes(['fit_output', 'ytest_notnorm', 'xtest_notnorm'])
        xtest_notnorm = self.xtest_notnorm
        ytest_notnorm = self.ytest_notnorm
        prediction = self.compute_prediction(xtest_notnorm, transform_output=True, transform_input=True)  
        inj  = ytest_notnorm[:,feature_idx]
        rec  = xtest_notnorm[:,feature_idx]
        pred =    prediction[:,feature_idx]
        if abs_diff:
            errors_rec  = (inj- rec)/inj
            errors_pred = (inj-pred)/inj
            xlab        = r'$\Delta y$'
        else:
            errors_rec  = (inj- rec)
            errors_pred = (inj-pred)
            xlab        = r'$\Delta y/y$'
        
        if fmin is None:
            min_rec  = min(errors_rec)
            min_pred = min(errors_pred)
            fmin     = min(min_rec, min_pred)
        if fmax is None:
            max_rec  = max(errors_rec)
            max_pred = max(errors_pred)
            fmax     = max(max_rec, max_pred)
        
        pred_min_outliers = 0
        pred_max_outliers = 0
        for i in range(len(errors_pred)):
            if errors_pred[i]<fmin:
                pred_min_outliers += 1
        for i in range(len(errors_pred)):
            if errors_pred[i]>fmax:
                pred_max_outliers += 1 
        rec_min_outliers = 0
        rec_max_outliers = 0
        for i in range(len(errors_rec)):
            if errors_rec[i]<fmin:
                rec_min_outliers += 1
        for i in range(len(errors_rec)):
            if errors_rec[i]>fmax:
                rec_max_outliers += 1 
        
        if verbose:
            print('prediction below fmin={:6.2f}: {:d}'.format(fmin, pred_min_outliers))
            print('recovery   below fmin={:6.2f}: {:d}'.format(fmin,  rec_min_outliers))
            print('prediction above fmax={:6.2f}: {:d}'.format(fmax, pred_max_outliers))
            print('recovery   above fmax={:6.2f}: {:d}'.format(fmax,  rec_max_outliers))

        fstep = (fmax-fmin)/nbins
        plt.figure
        plt.hist(errors_rec , bins=np.arange(fmin, fmax, fstep), alpha=alpha_rec,  color=color_rec, label='rec')
        plt.hist(errors_pred, bins=np.arange(fmin, fmax, fstep), alpha=alpha_pred, color=color_pred, label='pred')
        plt.legend(fontsize=20)
        plt.xlabel(xlab, fontsize=15)
        if logscale:
            plt.yscale('log', nonposy='clip')
        if name is not None:
            plt.title(name, fontsize=20)
        plt.show() 
        return

#######################################################################
# Cross-validator (on layers/architecture)
#######################################################################
class CrossValidator:
    """ Cross validation on architecture.
    Consider 1 and 2 layer(s) architectures and do a cross-val on the number of neurons
    for each layer. 
    Options for Box-Cox not implemented here (also because for some reason Box-Cox gives NaN
    during training)
    """
    def __init__(self, nfeatures=NFEATURES, dict_name=None, neurons_max=300, neurons_step=50, out_intervals=None,
                 epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, verbose=False,
                 fname_xtrain=None, fname_ytrain=None, fname_xtest=None, fname_ytest=None, seed=SEED):
        nlayers_max        = 2 # hard-coded for now, but should be ok (i.e. no NN with >2 layers needed)
        self.nfeatures     = nfeatures
        self.nlayers_max   = nlayers_max
        self.neurons_max  = neurons_max
        self.neurons_step  = neurons_step
        if out_intervals is not None:
            out_intervals = np.array(out_intervals)
        self.out_intervals = out_intervals
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.seed          = seed
        
        if fname_xtrain is None or fname_ytrain is None or fname_xtest is None or fname_ytest is None:
            raise ValueError('Incomplete data-input! Specifiy fname_xtrain, fname_ytrain, fname_xtest, fname_ytest')
        self.fname_xtrain = fname_xtrain
        self.fname_ytrain = fname_ytrain
        self.fname_xtest  = fname_xtest
        self.fname_ytest  = fname_ytest
        hlayers_sizes_list = []
        for i in range(neurons_step, neurons_max, neurons_step):
            for j in range(0, neurons_max, neurons_step):
                if j>0:
                    hlayers_size = (i,j)
                else:
                    hlayers_size = (i,)
                hlayers_sizes_list.append(hlayers_size)
        self.hlayers_sizes_list = hlayers_sizes_list
        if dict_name is None:
            dict_name = 'dict_nfeatures'+str(nfeatures)+'.dict'
        self.dict_name = dict_name
        cv_dict = load_dill(dict_name, verbose=verbose)
        self.cv_dict = cv_dict 
        return 

    def __param_to_key(self, hlayers_sizes):
        seed          = self.seed
        epochs        = self.epochs
        batch_size    = self.batch_size
        learning_rate = self.learning_rate
        out_intervals = self.out_intervals
        nfeatures     = self.nfeatures
        nlayers = len(hlayers_sizes)
        key  = 'e:'+str(epochs)+'-bs:'+str(batch_size)+'-alpha:'+str(learning_rate)+'-'
        key += str(nlayers) + 'layers:'
        for i in range(0, nlayers):
            key += str(hlayers_sizes[i])
            if i<nlayers-1:
                key += '+'
        key += '-'
        if out_intervals is not None:
            key += 'oc:['
            for i in range(0, nfeatures):
                key += '['+str(out_intervals[i][0])+','+str(out_intervals[i][1])+']'
                if i<nfeatures-1:
                    key += ','
            key += ']'
        else:
            key += 'no_oc'
        key += '-seed:'+str(self.seed)
        return key

    def crossval(self, verbose=False):
        """ Do cross-validation
        """
        nfeatures          = self.nfeatures
        seed               = self.seed
        epochs             = self.epochs
        batch_size         = self.batch_size
        learning_rate      = self.learning_rate
        out_intervals      = self.out_intervals
        hlayers_sizes_list = self.hlayers_sizes_list 
        xtrain_data = extract_data(self.fname_xtrain,verbose=verbose)
        ytrain_data = extract_data(self.fname_ytrain,verbose=verbose)
        xtest_data  = extract_data(self.fname_xtest,verbose=verbose)
        ytest_data  = extract_data(self.fname_ytest,verbose=verbose)
        for hlayers_sizes in hlayers_sizes_list:
            key = self.__param_to_key(hlayers_sizes)
            if key in self.cv_dict:
                if verbose:
                    #print('{:90s} already saved in {:}'.format(key,self.dict_name))
                    print('key already present:',key)
            else:
                NN = RegressionNN(nfeatures=nfeatures, hlayers_sizes=hlayers_sizes, seed=seed)
                NN.load_train_dataset(xtrain_data=xtrain_data, ytrain_data=ytrain_data)
                NN.training(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
                NN.load_test_dataset(xtest_data=xtest_data, ytest_data=ytest_data)
                metrics_dict = NN.compute_metrics_dict(NN.xtest, NN.ytest)
                prediction   = NN.compute_prediction(NN.xtest_notnorm, transform_output=True, transform_input=True) 
                npars        = count_params(NN.model.trainable_weights) 
                del NN
                struct               = lambda:0
                struct.metrics       = metrics_dict
                struct.hlayers_sizes = hlayers_sizes
                struct.prediction    = prediction
                struct.npars         = npars
                struct.nlayers       = len(hlayers_sizes)
                struct.epochs        = epochs
                struct.batch_size    = batch_size 
                struct.out_intervals = self.out_intervals
                struct.learning_rate = self.learning_rate
                struct.seed          = seed
                self.cv_dict[key] = struct 
                cv_dict           = self.cv_dict
                save_dill(self.dict_name, cv_dict)
                if verbose:
                    #print('{:90s} saved in {:}'.format(key,self.dict_name))
                    print('saving key:',key)
        return

    def plot(self, threshold=0.6, npars_lim=1e+6, feature_idx=-1):
        """ Plots to check which NN-architecture produces the best results
        The metric used is R2. Use feature_idx=-1 to plot the mean of R2
        """
        if not hasattr(self, 'cv_dict'):
            raise ValueError('cross-val dict not defined! Call self.crossval() befor self.plot()')
        cv_dict   = self.cv_dict
        dict_keys = cv_dict.keys()
        i = 0
        max_neurons_l1 = 0
        max_neurons_l2 = 0
        max_score_l1   = 0
        max_score_l2   = 0
        max_score      = 0
        scores         = []
        npars          = []
        hlayers        = []
        layer1_size    = []
        layer2_size    = []
        tot_neurons    = []
        for key in dict_keys:
            s = cv_dict[key]
            if feature_idx<0:
                score = s.metrics["R2mean"]
                mytitle = "mean of R2"
            else:
                score = s.metrics["R2"][feature_idx]
                mytitle = "R2 of feature n."+str(feature_idx)
            mytitle += ", threshold: "+str(threshold)
            
            if self.__param_to_key(s.hlayers_sizes)==key:
                scores.append(score)
                npars.append(s.npars)
                hlayers.append(s.hlayers_sizes)
                neurons_l1 = s.hlayers_sizes[0]
                layer1_size.append(neurons_l1)
                tot_neurons_tmp = neurons_l1
                if s.nlayers>1:
                    neurons_l2 = s.hlayers_sizes[1]
                else:
                    neurons_l2 = 0
                layer2_size.append(neurons_l2)
                tot_neurons_tmp += neurons_l2
                tot_neurons.append(tot_neurons_tmp)
                if neurons_l1>max_neurons_l1:
                    max_neurons_l1 = neurons_l1
                if neurons_l2>max_neurons_l2:
                    max_neurons_l2 = neurons_l2
                if score>max_score:
                    max_score = score
                    max_score_l1 = neurons_l1
                    max_score_l2 = neurons_l2
                i += 1
        if i==0:
            print('no models found (or threshold too big)!')
            sys.exit()
        fig, axs = plt.subplots(1,2, figsize=(12, 4))
        sc=axs[0].scatter(layer1_size, layer2_size, c=scores, cmap='gist_rainbow')
        cbar = plt.colorbar(sc,ax=axs[0])
        cbar.set_label('score')
        axs[0].scatter(max_score_l1, max_score_l2, linewidth=2, s=150, facecolor='none', edgecolor=(0, 1, 0))
        axs[0].title.set_text(mytitle)
        axs[0].set_xlabel('n. neurons - layer 1')
        axs[0].set_ylabel('n. neurons - layer 2')
        axs[0].set_xlim(-5,max_neurons_l1+5)
        axs[0].set_ylim(-5,max_neurons_l2+5)
        sc=axs[1].scatter(npars, scores, c=tot_neurons, cmap='viridis')
        cbar = plt.colorbar(sc,ax=axs[1])
        cbar.set_label('total n. neurons')
        axs[1].title.set_text(mytitle)
        axs[1].set_xlabel('n. parameters')
        axs[1].set_ylabel('score')
        axs[1].set_ylim(threshold, min(np.max(scores)*1.005, 1)) 
        plt.subplots_adjust(wspace=0.4)
        plt.show()
        return

#######################################################################
# Example
#######################################################################
if __name__ == '__main__':

    out_intervals = [[1,2.2],[1,1.8],[0.9,1.6]]

    path = "/home/simone/repos/IPAM2021_ML/datasets/GSTLAL_EarlyWarning_Dataset/Dataset/m1m2Mc/"
    xtrain = path+'xtrain.csv'
    ytrain = path+'ytrain.csv'
    xtest  = path+'xtest.csv'
    ytest  = path+'ytest.csv'
   
    NN = RegressionNN(nfeatures=3, hlayers_sizes=(100,), out_intervals=out_intervals, seed=None)
    NN.load_train_dataset(fname_xtrain=xtrain, fname_ytrain=ytrain, quantile=True)
    NN.print_summary()
    NN.training(verbose=True, epochs=10, validation_split=0.)
    NN.load_test_dataset(fname_xtest=xtest, fname_ytest=ytest) 
    NN.print_metrics()
    NN.plot_predictions(NN.xtest)

    dashes = '-'*80
    print(dashes, 'Save and load test:', dashes, sep='\n')
    NN.save_model(verbose=True, overwrite=True)
    NN2 = RegressionNN(load_model='model_nfeatures3_'+datetime.today().strftime('%Y-%m-%d'), verbose=True)
    NN2.load_test_dataset(fname_xtest=xtest, fname_ytest=ytest) 
    NN2.print_metrics() 
     
    print(dashes)
    NN.print_info()
    print(dashes)
    NN2.print_info()
    print(dashes)

    out_intervals = None 
    CV = CrossValidator(neurons_step=100, fname_xtrain=xtrain, fname_ytrain=ytrain, fname_xtest=xtest, \
                        fname_ytest=ytest, epochs=10, batch_size=128, out_intervals=out_intervals, seed=None)
    CV.crossval(verbose=True)
    CV.plot(feature_idx=-1, threshold=0.82)
