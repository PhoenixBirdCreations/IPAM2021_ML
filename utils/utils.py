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

def regressionDatasetLoader(data_paths, labels, scaler_type=None, remove_some_mass=False):
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
    if scaler_type is None:
        scaler_x = None
        scaler_y = None
    elif scaler_type=="standard":
        scaler_x = StandardScaler().fit(xtrain_notnormalized)
        scaler_y = StandardScaler().fit(ytrain_notnormalized)
    elif scaler_type=="minmax":
        scaler_x = MinMaxScaler(feature_range=(-1, 1)).fit(xtrain_notnormalized)
        scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(ytrain_notnormalized)
    elif scaler_type=="mixed":
        scaler_x = StandardScaler().fit(xtrain_notnormalized)
        scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(ytrain_notnormalized)
    else:
        print("scaler '",scaler_type,"' not recognized! Use None, 'standard', 'minmax' or 'mixed'.",sep='')
        sys.exit()
    # rescale and return 
    if scaler_type is None:
        xtrain = xtrain_notnormalized 
        ytrain = ytrain_notnormalized 
        xtest  = xtest_notnormalized 
        ytest  = ytest_notnormalized 
    else:
        xtrain = scaler_x.transform(xtrain_notnormalized)
        ytrain = scaler_y.transform(ytrain_notnormalized)
        xtest  = scaler_x.transform(xtest_notnormalized)
        ytest  = scaler_y.transform(ytest_notnormalized)
    out             = {}
    out['xtrain']   = xtrain
    out['ytrain']   = ytrain
    out['xtest']    = xtest
    out['ytest']    = ytest
    out['scaler_x'] = scaler_x
    out['scaler_y'] = scaler_y
    out['labels']   = labels
    return out

def R2(y_true, y_pred):
    SS_res = np.sum((y_true - y_pred )**2)
    SS_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1-SS_res/SS_tot

def evalutationMetricsDict(xtest,ytest,model,ypredicted=None):
    # xtest and ytest must be normalized! 
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

def chirpMass(m1, m2):
    return (m1*m2)**(3/5)/(m1+m2)**(1/5)

def symmetricMass(m1,m2):
    return m1*m2/(m1+m2)**2

def reducedMass(m1,m2):
    return m1*m2/(m1+m2)

def findSecondMassFromMc(Mc, m):
    """
    Find analytically one mass from Mc and the other mass.
    Mc and m can be vectors
    """
    if np.any(Mc<0) or np.any(m<0):
        print('negative masses in input!')
        sys.exit()
    Mc5 = Mc**5
    arg = 81*m**5-12*Mc5
    mysqrt = np.where(arg<0, 1j*np.sqrt(-arg), np.sqrt(arg))
    Mc5by3 = Mc5**(1/3)
    croot  = (9*m**(5/2)+mysqrt)**(1/3)
    num    = Mc5by3*(2*3**(1/3)*Mc5by3+2**(1/3)*croot**2)
    den    = (6**(2/3)*m**(3/2)*croot)
    out    = num/den
    if np.any(np.abs(out.imag)>1e-14):
        print('Warning: imaginary part bigger than 1e-14!')
    return out.real

def findm1m2FrompMc(p,Mc):
    p3 = p*p*p
    p5 = p3*p*p
    Mc5 = Mc**5
    Mc10 = Mc5*Mc5
    nu = Mc10/p5
    m1 = p3*(1+np.sqrt(1-4*nu))/Mc5/2
    m2 = p/m1
    return m1,m2

def findm1m2FrompMc_Mod(p,Mc):
    p3 = p*p*p
    p5 = p3*p*p
    Mc5 = Mc**5
    Mc10 = Mc5*Mc5
    nu   = Mc10/p5
    arg  = 1-4*nu
    root = np.where(arg>0, np.sqrt(arg), 0)
    m1 = p3*(1+root)/Mc5/2
    m2 = p3*(1-root)/Mc5/2
    return m1,m2

def findm1m2FromsMc(s,Mc):
    Mc5by3 = Mc**(5/3)
    s1by3  = s**(1/3)
    s2     = s*s
    arg    = -4*Mc5by3*s1by3+s2
    root   = np.sqrt(arg)
    m1     = 0.5*(s+root)
    m2     = 0.5*(s-root)
    return m1,m2

def findm1m2Fromps(p,s):
    rootp=p**(1/3);
    arg=s*s-4*rootp
    m1 = np.where(arg>0, (s+np.sqrt(arg))*0.5, s*0.5)
    m2=s-m1
    return m1,m2

def findm1m2FromMcTm(Mc,s):
    C=(s*Mc**5)**(1.0/3)
    arg=s*s-4*C;
    m2 = np.where(arg>0, (s-np.sqrt(arg))*0.5, s*0.5)
    m1=s-m2
    return m1,m2

def findm1m2FromMcq(Mc,q):
    m1=Mc*((1+q)/q**3)**1.0/5
    m2=q*m1
    return m1,m2

def findm1m2FromMcSymm(Mc,nu):
    arg=Mc**2/nu**(1.0/5)*(1/nu-4)
    m1= np.where(arg>0, (Mc/nu**(3.0/5)+np.sqrt(arg))*0.5, (Mc/nu**(3.0/5))*0.5)
    m2=(Mc**10/nu)**(1.0/5)/m1
    return m1,m2

def findm1m2FromMcmu(Mc,mu):
    A=np.sqrt(Mc**5/mu**3)
    arg=A*A-4*A*mu
    m1 = np.where(arg>0, (A+np.sqrt(arg))*0.5, A*0.5)
    m2=m1*mu/(m1-mu)
    return m1,m2

def findm1m2Fromsmu(s,mu):
    arg=s*s-4*mu*s;
    m2 = np.where(arg>0, (s-np.sqrt(arg))*0.5, s*0.5)
    m1=mu*s/m2
    return m1,m2

