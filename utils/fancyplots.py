import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def regrPredictionPlots(ytest0, ypredicted0, labels, scaler=None):
    if (scaler is None):
        ytest      = ytest0
        ypredicted = ypredicted0
    else:
        ytest      = scaler.inverse_transform(ytest0)
        ypredicted = scaler.inverse_transform(ypredicted0)
    
    Nfeatures = len(ytest0[0,:])
    if Nfeatures!=len(labels) or Nfeatures!=len(ypredicted[0,:]):
        print('Wrong input! Check shapes')
        sys.exit()

    if Nfeatures<3:
        plot_cols = Nfeatures
    else:
        plot_cols = 3
    
    rows = round(Nfeatures/plot_cols)
    if rows>1:
        fig, axs  = plt.subplots(rows,plot_cols, figsize = (25,17))
    else: 
        fig, axs  = plt.subplots(rows,plot_cols, figsize = (22,9))
    feature     = 0
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
            diff          = np.abs(ytest_1d-ypredicted_1d)
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
    history_dict = history.history
    acc      = history_dict['accuracy']
    val_acc  = history_dict['val_accuracy']
    loss     = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs_plot=range(1,len(acc)+1)   
    plt.figure(figsize=(10,10))

    ax1=plt.subplot(221)
    ax1.plot(epochs_plot,acc,'b',label='Training acc')
    ax1.plot(epochs_plot,loss,'r',label='Training loss')
    ax1.set_title('loss and acc of Training')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2=plt.subplot(222)
    ax2.plot(epochs_plot,val_acc,'b',label='Validation acc')
    ax2.plot(epochs_plot,val_loss,'r',label='Validation loss')
    ax2.set_title('loss and acc of Validation')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Acc')
    ax2.legend()
    return

def checkRegressionPlot(xtest0, ytest0, ypredicted0, labels, scaler_y=None, scaler_x=None):
    if (scaler_y is None):
        ytest      = ytest0
        ypredicted = ypredicted0
    else:
        ytest      = scaler_y.inverse_transform(ytest0)
        ypredicted = scaler_y.inverse_transform(ypredicted0)
    
    if (scaler_x is None):
        xtest      = xtest0
    else:
        xtest      = scaler_x.inverse_transform(xtest0)
    
    Nfeatures = len(ytest0[0,:])
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
    return




