import csv
import numpy as np
import matplotlib.pyplot as plt

# function for I/O files
def extractData(filename, verbose=False):
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
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in data:
            spamwriter.writerow(row)
    if verbose:
        print(filename, 'saved')
            
# plot function
def predictionPlots(ytest, ypredicted, labels):
    Nfeatures = len(labels) # Nfeatures<=12 hard-coded 
    fig, axs  = plt.subplots(4,3, figsize = (25,17))
    param     = 0
    for i in range(0,4):
        for j in range(0,3):
            ytest_1d      = ytest[:,param]
            ypredicted_1d = ypredicted[:,param]
            diff = np.abs(ytest_1d-ypredicted_1d)
            axs[i,j].scatter(ytest_1d, ypredicted_1d, s=15, c=diff, cmap="gist_rainbow")
            axs[i,j].plot(ytest_1d, ytest_1d, 'k')
            #axs[i,j].set_title(labels[param])
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
            axs[i,j].set_title(labels[param])
            axs[i,j].set_ylabel('predicted')
            #axs[i,j].set_xlabel('injected')
            param+=1;
            if param>=Nfeatures:
                break
    plt.show()

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

def R2(y_true, y_pred):
    SS_res = np.sum((y_true - y_pred )**2)
    SS_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1-SS_res/SS_tot

 
