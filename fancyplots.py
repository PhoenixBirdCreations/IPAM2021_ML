import sys
import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils
import utils as ut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve

##################################################################
# Regression plots
#################################################################
def regrPredictionPlots(ytest, ypredicted, labels, scaler=None):
    """
    the usual injected vs predicted plots
    """
    if scaler is not None:
        ytest      = scaler.inverse_transform(ytest)
        ypredicted = scaler.inverse_transform(ypredicted)
    
    Nfeatures = len(ytest[0,:])
    if Nfeatures!=len(labels) or Nfeatures!=len(ypredicted[0,:]):
        print('Wrong input! Check shapes')
        sys.exit()

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
            ytest_1d      = ytest[:,feature]
            ypredicted_1d = ypredicted[:,feature]
            diff = np.abs(ytest_1d-ypredicted_1d)
            ax.scatter(ytest_1d, ypredicted_1d, s=15, c=diff, cmap="gist_rainbow")
            ax.plot(ytest_1d, ytest_1d, 'k')
            ymax = max(ytest_1d)
            xmin = min(ytest_1d)
            if xmin<0:
                xpos = xmin*0.7
            else:
                xpos = xmin*1.3

            if ymax<0:
                ypos = ymax*0.7
            else:
                ypos = ymax*1.3
            label = labels[feature]
            ax.set_ylabel('predicted - '+label, fontsize=25)
            ax.set_xlabel('injected - '+label, fontsize=25)
            feature+=1;
    plt.show()
    return

def plotHistory(history): 
    """
    history is the ouput of model.compile in TensorFlow
    """
    history_dict = history.history
    acc      = history_dict['R2metric']
    val_acc  = history_dict['val_R2metric']
    loss     = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs_plot=range(1,len(acc)+1)   
    plt.figure(figsize=(10,10))
    ax1=plt.subplot(221)
    ax1.plot(epochs_plot,acc,'b',label='Training R2')
    ax1.plot(epochs_plot,loss,'r',label='Training loss')
    ax1.set_title('loss and R2 of Training')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2=plt.subplot(222)
    ax2.plot(epochs_plot,val_acc,'b',label='Validation R2')
    ax2.plot(epochs_plot,val_loss,'r',label='Validation loss')
    ax2.set_title('loss and R2 of Validation')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('R2')
    ax2.legend()
    plt.show()
    return

def checkRegressionPlot(xtest, ytest, ypredicted, labels, scaler_y=None, scaler_x=None):
    """
    Plot recovered vs predicted
    """
    if scaler_y is not None:
        ytest      = scaler_y.inverse_transform(ytest)
        ypredicted = scaler_y.inverse_transform(ypredicted)
    
    if scaler_x is not None:
        xtest      = scaler_x.inverse_transform(xtest)
    
    Nfeatures = len(ytest[0,:])
    if Nfeatures!=len(labels) or Nfeatures!=len(ypredicted[0,:]):
        print('Wrong input! Check shapes')
        sys.exit()
    
    if Nfeatures<3:
        plot_cols    = Nfeatures
        fontsize_lab = 30
        fontsize_leg = 25
    else:
        plot_cols    = 3
        fontsize_lab = 20
        fontsize_leg = 20
    
    rows = round(Nfeatures/plot_cols)
    if rows>1:
        fig, axs  = plt.subplots(rows,plot_cols, figsize = (25,17))
    else: 
        fig, axs  = plt.subplots(rows,plot_cols, figsize = (22,9))

    feature = 0; 
    for i in range(0,rows):
        for j in range(0,plot_cols):
            if feature>=Nfeatures:
                break
            if rows>1:
                ax = axs[i,j]
            else: 
                ax = axs[j]
            ytest_plot = ytest[:,feature]
            xtest_plot = xtest[:,feature]
            ypred_plot = ypredicted[:,feature]
            
            ax.scatter(ytest_plot, xtest_plot, label='recovered', s=50)
            ax.scatter(ytest_plot, ypred_plot, label='predicted', marker='x', s=80)
            ax.set_xlabel('injected - '+labels[feature], fontsize=fontsize_lab)
            ax.set_ylabel(labels[feature], fontsize=fontsize_lab)
            ax.plot(ytest_plot, ytest_plot, 'k')
            ax.legend(fontsize=fontsize_leg)
            feature+=1
   # plt.show()
    return

def plotInjRecPred(injected, recovered, predicted, idx_m1=0, idx_m2=1, idx_Mc=None, hide_recovered=False):
    """
    Check consistency on predicted masses
    """
    asterisks = '*'*80

    m1_inj  = np.copy(injected[:,idx_m1])
    m2_inj  = np.copy(injected[:,idx_m2])
    m1_rec  = np.copy(recovered[:,idx_m1])
    m2_rec  = np.copy(recovered[:,idx_m2])
    m1_pred = np.copy(predicted[:,idx_m1])
    m2_pred = np.copy(predicted[:,idx_m2])

    # computed
    q_comp  = m2_pred/m1_pred 
    Mc_comp = (m2_pred*m1_pred)**(3/5)/((m1_pred+m2_pred)**(1/5))

    print(asterisks,'m1 vs m2: injected, recovered, predicted',asterisks,sep='\n')
    fig, axs = plt.subplots(1,3, figsize=(11, 3))
    axs[0].scatter(m1_inj, m2_inj, color=[1,0.5,0])
    axs[0].plot(m1_inj, m1_inj, 'r')
    axs[0].set_xlabel('m1_inj')
    axs[0].set_ylabel('m2_inj')
    axs[1].scatter(m1_rec, m2_rec, color=[0.2,0.4,1])
    axs[1].plot(m1_rec, m1_rec, 'r')
    axs[1].set_xlabel('m1_rec')
    axs[1].set_ylabel('m2_rec')
    axs[2].scatter(m1_pred, m2_pred, color=[0,0.8,0.2])
    axs[2].plot(m1_pred, m1_pred, 'r')
    axs[2].set_xlabel('m1_pred')
    axs[2].set_ylabel('m2_pred')
    plt.subplots_adjust(wspace=0.4)
    plt.show()

    plt.figure
    plt.figure(figsize=(4,5))
    if not hide_recovered:
        plt.scatter(m1_rec, m2_rec,   label='recovered', color=[0.2,0.4,1])
    plt.scatter(m1_inj, m2_inj,   label='injected', marker='x', color=[1,0.5,0])
    plt.scatter(m1_pred, m2_pred, label='predicted', color=[0,0.8,0.2])
    plt.xlabel('m1')
    plt.ylabel('m2')
    plt.legend()
    plt.show()

    print(asterisks,'m1 vs q: injected, recovered, predicted (indirectly)',asterisks,sep='\n')
    plt.figure(figsize=(4,5))
    if not hide_recovered:
        plt.scatter(m1_rec, m2_rec/m1_rec, label='recovered', color=[0.2,0.4,1])
    plt.scatter(m1_inj, m2_inj/m1_inj, label='injected', color=[1,0.5,0])
    plt.scatter(m1_pred, q_comp, label='predicted', color=[0,0.8,0.2])
    plt.xlabel('m1')
    plt.ylabel('q')
    plt.legend()
    plt.show()

    if idx_Mc is not None:
        Mc_inj  = np.copy(injected[:,idx_Mc])
        Mc_rec  = np.copy(recovered[:,idx_Mc])
        Mc_pred = np.copy(predicted[:,idx_Mc])

        print(asterisks,'m1 vs Mc: injected, recovered, predicted',asterisks,sep='\n')
        fig, axs = plt.subplots(1,3, figsize=(11, 3))
        axs[0].scatter(m1_inj, Mc_inj, color=[1,0.5,0])
        axs[0].set_xlabel('m1_inj')
        axs[0].set_ylabel('Mc_inj')
        axs[1].scatter(m1_rec, Mc_rec, color=[0.2,0.4,1])
        axs[1].set_xlabel('m1_rec')
        axs[1].set_ylabel('Mc_rec')
        axs[2].scatter(m1_pred, Mc_pred, color=[0,0.8,0.2])
        axs[2].set_xlabel('m1_pred')
        axs[2].set_ylabel('Mc_pred')
        plt.subplots_adjust(wspace=0.4)
        plt.show()

        plt.figure(figsize=(4,5))
        if not hide_recovered:
            plt.scatter(m1_rec, Mc_rec,   label='recovered', color=[0.2,0.4,1])
        plt.scatter(m1_inj, Mc_inj,   label='injected', marker='x', color=[1,0.5,0])
        plt.scatter(m1_pred, Mc_pred, label='predicted', color=[0,0.8,0.2])
        #plt.scatter(m1_pred, ut.chirpMass(m1_pred, m2_pred), label='computed', color=[0.7,0.7,0.7])
        plt.xlabel('m1')
        plt.ylabel('Mc')
        plt.legend()
        plt.show()

        #plt.figure
        #plt.scatter(m1_pred, ut.chirpMass(m1_pred, m2_pred)-Mc_pred, label='diff')
        #plt.legend()
        #plt.show()
    return

##################################################################
# Classification plots
##################################################################
def probLabelDensePlot(model, label_idx=0, mass_range=[1,3],  N=30000, idx_m1=0, idx_m2=1, \
                       dataset='GSTLAL_2m', verbose=False, cv=0, title=None):
    """
    Scatter plot in the (m1,m2) plane with colorbar 
    for the probability of a certain label.
    'dataset' can be: 'GSTLAL_2m' or 'NewRealistic'
    'cv' is used only for dataset='NewRealistic' 
    (cv is the flag for the f-function of the NewRealistic dataset,
    cv=0 -> f_conditional, cv=1 -> f_new)
    """
    if dataset=='GSTLAL_2m':
        m1 = np.linspace(mass_range[0],mass_range[1],N)
        m2 = np.linspace(mass_range[0],mass_range[1],N)
        np.random.shuffle(m1)
        np.random.shuffle(m2)
        for i in range(0, N):
            if m1[i]<m2[i]:
                tmp   = m2[i];
                m2[i] = m1[i];
                m1[i] = tmp;
        m1 = np.reshape(m1, (N,1))
        m2 = np.reshape(m2, (N,1))
        X  = np.concatenate((m1,m2), axis=1)
    
    elif dataset=='NewRealistic':
        X = ut.generateUniformMassRange(N, mass_range, cv=cv)
    
    m1 = np.reshape(X[:,idx_m1], (N,1))
    m2 = np.reshape(X[:,idx_m2], (N,1))
    proba_dense   = model.predict_proba(X)
    proba_dense1d = np.reshape(proba_dense[:,label_idx],(N,1))
    
    plt.figure
    sc=plt.scatter(m1, m2, c=proba_dense1d, vmin=0, vmax=1, s=40, cmap='viridis')
    plt.colorbar(sc)
    if title is not None:
        plt.title(title)
    plt.xlabel("m1", fontsize=20)
    plt.ylabel("m2", fontsize=20)
    plt.show()
    return

def plotROC(ytrue, prob_of_label):
    """
    Improve me! prob_of_label is the probability of having a certain
    label (i.e. is a 1D vector)
    """
    fpr, tpr, thresholds = roc_curve(ytrue, prob_of_label)
    plt.figure
    sc=plt.scatter(fpr[1:-1], tpr[1:-1], c=thresholds[1:-1], cmap='viridis')
    plt.colorbar(sc)
    plt.xlabel("false positive rate", fontsize=14)
    plt.ylabel("true positive rate",  fontsize=14)
    plt.show()
    return





