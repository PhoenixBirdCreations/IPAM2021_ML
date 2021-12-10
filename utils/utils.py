import csv
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import realistic
import sklearn
# function for I/O files
def extractData(filename, verbose=False):
    """ Reads data from csv file and returns it in array form.

    Parameters
    ----------
    filename : str
        File path of data file to read

    Returns
    -------
    data : arr
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

    Parameters
    ----------
    filename : str
        File path of data file to read

    data : arr
        Array of data to write in csv file
    """
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in data:
            spamwriter.writerow(row)
    if verbose:
        print(filename, 'saved')
            
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
    # rescale and return    
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

def generateUniformMassRange(N, mass_range, cv=0):
    X, _ = realistic.generateEvents(N, cv, verbose=False, mass_range=mass_range, distribution='uniform')
    return np.array(X)




