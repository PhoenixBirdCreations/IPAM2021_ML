import sys
import numpy as np
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils.layer_utils import count_params

#########################################################################
# Neural Newtork with TensorFlow
#########################################################################

# define the ouput activation functions
def output_activation_sigmoid(x):
    return K.sigmoid(x)*2-1

def output_activation_linear(x):
    return x

def output_activation_linear_cut(x):
    signs = K.switch(x>0, 1+x*0, -1+x*0) # x*0 in order to broadcast to correct dimension
    return K.switch(abs(x)<1, x, signs)

def output_activation_linear_cut_T3(x):
    return 2/(K.exp(-(2*x+2/3*x*x*x))+1)-1

def output_activation_linear_cut_lb(x):
    return K.switch(x>-1, x, -1+x*0)

# define dense NN using Functional API
def ArchitectureDenseNN(hlayers_sizes, Nfeatures, hidden_activation='relu', out_activation='linear'):
    model_input = tf.keras.Input(shape=(Nfeatures))
    # hidden layers
    x = Dense(hlayers_sizes[0], kernel_initializer='normal', activation=hidden_activation)(model_input)
    for i in range(1, len(hlayers_sizes)):
        x = Dense(hlayers_sizes[i], kernel_initializer='normal', activation=hidden_activation)(x)
    # output layer 
    if out_activation=="sigmoid":
        out = Dense(Nfeatures, kernel_initializer='normal',\
                    activation=output_activation_sigmoid)(x)
    elif out_activation=="linear":
        out = Dense(Nfeatures, kernel_initializer='normal',\
                    activation=output_activation_linear)(x)
    elif out_activation=="linear_cut":
        out = Dense(Nfeatures, kernel_initializer='normal',\
                    activation=output_activation_linear_cut)(x)
    elif out_activation=="linear_cut_T3":
        out = Dense(Nfeatures, kernel_initializer='normal',\
                    activation=output_activation_linear_cut_T3)(x)
    elif out_activation=="linear_cut_lb":
        out = Dense(Nfeatures, kernel_initializer='normal',\
                    activation=output_activation_linear_cut_lb)(x)
    elif out_activation=="linear_cut_mod":
        if Nfeatures==11:
            neuronsA = 2 # m1, m2
            neuronsB = 6 # s1,s2,
            neuronsC = 1 # theta
            neuronsD = 2 # q, Mc
        elif Nfeatures==9:
            neuronsA = 1 # m1
            neuronsB = 6 # s1,s2
            neuronsC = 1 # theta
            neuronsD = 1 # Mc
        else:
            print("'linear_cut_mod' is hardcoded for Nfeatures=9,11!")
            sys.exit()
        branchA = Dense(neuronsA, kernel_initializer='normal', activation=output_activation_linear_cut_lb)(x)
        branchB = Dense(neuronsB, kernel_initializer='normal', activation=output_activation_linear_cut)(x)
        branchC = Dense(neuronsC, kernel_initializer='normal', activation=output_activation_linear)(x)
        branchD = Dense(neuronsD, kernel_initializer='normal', activation=output_activation_linear_cut_lb)(x)
        out = tf.keras.layers.concatenate([branchA, branchB, branchC, branchD])
    else:
        print("'", out_activation,"' is not valid activation function for the output layer!", sep='')
        sys.exit()

    return tf.keras.Model(model_input, out)

#########################################################################
# Regression pipeline
#########################################################################
"""
Build, train and return the regression pipeline.
xtrain and ytrain must be NOT normalized, they will be normalized
by the scaler; scalers implemented: 'standard', 'minmax', 'mixed'
"""
def neuralNewtorkRegression(xtrain_notnormalized, ytrain_notnormalized, scaler_type='standard',\
                            epochs=10, batch_size=32, learning_rate=0.001,  \
                            validation_split=0.1, verbose=False, \
                            hlayers_sizes=(100,), out_activation='linear', hidden_activation='relu'):
    # Fit the scaler and normalize data
    if scaler_type=="standard":
        scaler_x = StandardScaler().fit(xtrain_notnormalized)
        scaler_y = StandardScaler().fit(ytrain_notnormalized)
    elif scaler_type=="minmax":
        scaler_x = MinMaxScaler(feature_range=(-1, 1)).fit(xtrain_notnormalized)
        scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(ytrain_notnormalized)
    elif scaler_type=="mixed":
        scaler_x = StandardScaler().fit(xtrain_notnormalized)
        scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(ytrain_notnormalized)
    else:
        print("scaler '",scaler_type,"' not recognized! Use 'standard', 'minmax' or 'mixed'.",sep="")
    xtrain = scaler_x.transform(xtrain_notnormalized)
    ytrain = scaler_y.transform(ytrain_notnormalized)
    Nfeatures = len(xtrain[0,:])
    # build and compile the model
    model = ArchitectureDenseNN(hlayers_sizes, Nfeatures,\
        out_activation=out_activation,\
        hidden_activation=hidden_activation)
    mse = MeanSquaredError()
    model.compile(loss=mse, metrics=[mse, 'accuracy'], optimizer=Adam(learning_rate=learning_rate))
    if verbose:
        model.summary()
    # train the model and save history
    history = model.fit(xtrain, ytrain, 
        epochs           = epochs, 
        batch_size       = batch_size,
        validation_split = validation_split,
        verbose          = verbose)
    # save output in a dictionary
    output = {}
    output["model"]    = model
    output["scaler_x"] = scaler_x
    output["scaler_y"] = scaler_y
    output["history"]  = history
    output["Npars"]    = count_params(model.trainable_weights)
    return output


def plotLayersCrossVal(models_dict, threshold=0.9, Npars_lim=1e+6, \
                      hidden_activation='relu',   \
                      out_activation='linear_cut_mod', \
                      scaler_type='minmax', \
                      batch_size=256, epochs=250, \
                      metrics_idx=-1, labels=None):
    """
    Plots to check which NN-architecture produces the best results
    The metric used is R2
    Use metrics_idx=-1 to plot the mean of R2
    """
    dict_keys = models_dict.keys()
    i = 0
    max_neurons_l1 = 0
    max_neurons_l2 = 0
    max_score_l1   = 0
    max_score_l2   = 0
    max_score      = 0
    scores  = []
    Npars   = []
    hlayers = []
    layer1_size = []
    layer2_size = []
    tot_neurons = []
    for key in dict_keys:
        s = models_dict[key]
        if metrics_idx<0:
            score = s.metrics["R2mean"]
            mytitle = "mean of R2"
        else:
            score = s.metrics["R2"][metrics_idx]
            if labels is not None:
                mytitle = "R2 of "+labels[metrics_idx]+ \
                          " (feat n."+str(metrics_idx)+" )"
            else:
                mytitle = "R2 of feature n."+str(metrics_idx)
        mytitle += ", threshold: "+str(threshold)
        ha = s.hidden_activation
        oa = s.out_activation
        st = s.scaler_type
        bs = s.batch_size
        ep = s.epochs
        nl = s.Nlayers
        Np = s.Npars
        add2list = ha==hidden_activation and oa==out_activation and \
                   bs==batch_size and ep==epochs and nl<=2 and \
                   st==scaler_type and score>=threshold and Np<=Npars_lim
        if add2list:
            scores.append(score)
            Npars.append(s.Npars)
            hlayers.append(s.hlayers_sizes)
            neurons_l1 = s.hlayers_sizes[0]
            layer1_size.append(neurons_l1)
            tot_neurons_tmp = neurons_l1
            if s.Nlayers>1:
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
        print('no models found!')
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

    sc=axs[1].scatter(Npars, scores, c=tot_neurons, cmap='viridis')
    cbar = plt.colorbar(sc,ax=axs[1])
    cbar.set_label('total n. neurons')
    axs[1].title.set_text(mytitle)
    axs[1].set_xlabel('n. parameters')
    axs[1].set_ylabel('score')
    axs[1].set_ylim(threshold, min(np.max(scores)*1.005, 1)) 
    plt.subplots_adjust(wspace=0.4)
    plt.show()



