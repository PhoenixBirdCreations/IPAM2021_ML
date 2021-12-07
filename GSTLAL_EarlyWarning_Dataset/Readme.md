The dataset for Regression is in "Dataset/"

----------Regression------------------------

To load the dataset for regression:

path = 'Dataset/'

xtrain = np.loadtxt(path + 'train_recover.csv', delimiter=",")
ytrain = np.loadtxt(path + 'train_inject.csv', delimiter=",")
xtest = np.loadtxt(path + 'test_recover.csv', delimiter=",")
ytest = np.loadtxt(path + 'test_inject.csv', delimiter=",")

To save the predicted dataset:

np.savetxt("NN_prediction.csv", y_predict, delimiter=",")

---------Classification------------
To load the dataset

path = './Dataset/'
predicted = np.loadtxt('prediction.csv', delimiter=",")
label = np.loadtxt(path + 'test_label.csv', delimiter=",")

def split(x,y,split_ratio):
    nevents = len(x)
    print ('There are ',nevents,' events totally')
    n_train = int(nevents*split_ratio)
    print ("There are ",n_train," for training")
    print ("There are ",nevents - n_train," for testing")
    
    
    xtrain =  x[:n_train]
    ytrain =  y[:n_train]

    xtest =  x[n_train:]
    ytest =  y[n_train:]
    
    return (xtrain,ytrain,xtest,ytest)




Details see the note:
https://docs.google.com/document/d/1ZtnfuPUh-OPXi80-so60vXOiLInxsuv251UeztTWD6w/edit?usp=sharing
