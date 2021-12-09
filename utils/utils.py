import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
def predictionPlots(ytest0, ypredicted0, labels, scaler=None):
    if (scaler is None):
        ytest      = ytest0
        ypredicted = ypredicted0
    else:
        ytest      = scaler.inverse_transform(ytest0)
        ypredicted = scaler.inverse_transform(ypredicted0)
    
    Nfeatures = len(labels) # Nfeatures<=12 hard-coded
    rows      = round(Nfeatures/3)
    fig, axs  = plt.subplots(rows,3, figsize = (25,17))
    param     = 0
    for i in range(0,rows):
        for j in range(0,3):
            if param>=Nfeatures:
                break
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

def removeSomeMassFromDataset(X0,Y0,labels,mass_cols):
    X = np.delete(X0,mass_cols,1);
    Y = np.delete(Y0,mass_cols,1);
    Nfeatures = len(X[0,:]);
    
    labels_copy = labels.copy();
    if type(mass_cols)==list:
        for i in range(0,len(mass_cols)):
            label2remove = labels[mass_cols[i]]
            labels_copy.remove(label2remove)
    else:
        labels_copy.remove(labels[mass_cols])

    return X,Y,labels_copy,Nfeatures


def regressionDatasetLoader(data_paths, labels, scaler_type="standard", remove_some_mass=False):
    # Load all the data for the specific version
    xtrain_notnormalized = extractData(data_paths['xtrain'], verbose=False)
    ytrain_notnormalized = extractData(data_paths['ytrain'], verbose=False)
    xtest_notnormalized  = extractData(data_paths['xtest'],  verbose=False)
    ytest_notnormalized  = extractData(data_paths['ytest'],  verbose=False)
    
    if remove_some_mass:
        # 1 and 9 are the indeces of 'm2' and 'q'
        xtrain_notnormalized, ytrain_notnormalized, _, _ = \
            removeSomeMassFromDataset(xtrain_notnormalized, ytrain_notnormalized, labels, [1,9])
        xtest_notnormalized,  ytest_notnormalized , labels, Nfeatures = \
            removeSomeMassFromDataset(xtest_notnormalized,  ytest_notnormalized,  labels, [1,9])
            
    # rescale
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
        print('scaler "',scaler_type,'" not recognized! Use standard, minmax or mixed.',sep='')
        sys.exit()
        
    xtrain   = scaler_x.transform(xtrain_notnormalized)
    ytrain   = scaler_y.transform(ytrain_notnormalized)
    xtest    = scaler_x.transform(xtest_notnormalized)
    ytest    = scaler_y.transform(ytest_notnormalized)

    out             = {}
    out['xtrain']   = xtrain
    out['ytrain']   = ytrain
    out['xtest']    = xtest
    out['ytest']    = ytest
    out['scaler_x'] = scaler_x
    out['scaler_y'] = scaler_y
    out['labels']   = labels

    return out

def evalutationMetricsDict(xtest,ytest,model,ypredicted=None): 
    Nfeatures = len(xtest[0,:])
    
    if (ypredicted is None):
        ypredicted = model.predict(xtest)
    
    R2_vec = np.zeros((Nfeatures,))
    for i in range(0,Nfeatures):
         R2_vec[i] = R2(ytest[:,i], ypredicted[:,i])
    
    metrics         = model.metrics
    metrics_results = model.evaluate(xtest, ytest, verbose=0)
    metrics_dict    = {};
    
    for i in range(0, len(metrics)):
        metrics_dict[metrics[i].name] = metrics_results[i]
    metrics_dict["R2"]     = R2_vec
    metrics_dict["R2mean"] = np.mean(R2_vec)
    
    return metrics_dict

def printMetrics(metrics_dict):
    print('\nFinal loss     : {:.5f}'.format(metrics_dict["loss"]))
    print('Final mse      : {:.5f}'.format(metrics_dict["mean_squared_error"]))
    print('Final accuracy : {:.5f}'.format(metrics_dict["accuracy"]), '\n')
    print('Final R2 mean  : {:.5f}'.format(metrics_dict["R2mean"]))
    i = 0
    R2_vec = metrics_dict["R2"]
    for R2 in metrics_dict["R2"]:
        print('R2[{:2d}]         : {:.5f}'.format(i,R2))
        i+=1
    return
