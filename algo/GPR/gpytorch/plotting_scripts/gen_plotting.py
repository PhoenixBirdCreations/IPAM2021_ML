import sys
import argparse
import re

import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
import gpytorch
from sklearn import preprocessing

sys.path.insert(0, '/ddn/home1/r2566/IPAM2021_ML/utils')
from utils import *

# Read data 
xtrain = extractData('../../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/train_recover.csv')
ytrain = extractData('../../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/train_inject.csv')
xtest = extractData('../../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/test_recover.csv')
ytest = extractData('../../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/test_inject.csv')

xtrain_scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain_scaled = xtrain_scaler.transform(xtrain)
ytrain_scaler = preprocessing.StandardScaler().fit(ytrain)
ytrain_scaled = ytrain_scaler.transform(ytrain)

xtest_scaler = preprocessing.StandardScaler().fit(xtest)
xtest_scaled = xtest_scaler.transform(xtest)
ytest_scaler = preprocessing.StandardScaler().fit(ytest)
ytest_scaled = ytest_scaler.transform(ytest)

train_x = torch.from_numpy(xtrain_scaled).float()
train_y = torch.from_numpy(ytrain_scaled).float()
test_x = torch.from_numpy(xtest_scaled).float()
test_y = torch.from_numpy(ytest_scaled).float()

train_x = train_x.unsqueeze(0).repeat(2, 1, 1)
train_y = train_y.transpose(-2, -1)

# plotting stuff

color_cycle = [
        (53/255.,  74/255.,  93/255.),   # black
        (59/255.,  153/255., 217/255.),  # blue
        (229/255., 126/255., 49/255.),   # orange
        (53/255.,  206/255., 116/255.),  # green
        (230/255., 78/255.,  67/255.),   # red
        (154/255., 91/255.,  179/255.),  # purple
        (240/255., 195/255., 48/255.),   # gold
        '#e377c2',                       # pink
        '#8c564b',                       # brown
        '#7f7f7f',                       # gray
        '#17becf',                       # teal
        '#bcbd22',                       # lime
    ]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='gen_plotting.py', description=__doc__)

    parser.add_argument('tensor_file',
                        help='Directory containing metadata and rhOverM_Asymptotic_GeometricUnits_CoM.h5')
    parser.add_argument('--log', default='INFO',
                        help='Log level (default: %(default)s)')

    args = parser.parse_args()

    ########################################

    predictions = torch.load(args.tensor_file)['pred']
    preds = torch.transpose(predictions.mean,0,1).numpy()
    predicted_data = ytest_scaler.inverse_transform(preds)
    writeResult('PRBF50_1.csv', predicted_data, verbose=False)

    file_prefix = re.split('/|t', args.tensor_file)[1][:-1]
    
    with torch.no_grad():
    # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    #Recovered
        plt.plot(xtest[:,0], xtest[:,1], 'b*', label='Testing Recovered')
          
    #Injected
        plt.plot(ytest[:,0], ytest[:,1], 'rx', label='Testing Injected')
    
    #Predicted
        plt.plot(predicted_data[:,0], predicted_data[:,1], 'go', label='Predicted')       

        plt.xlabel(r'$m_1$', fontsize=14)
        plt.ylabel(r'$m_2$', fontsize=14)
        plt.legend()
        outfile = '../figs/'+file_prefix+'.pdf'
        plt.savefig(outfile)
        print('Saved figure to {}'.format(outfile))


    with torch.no_grad():
    # Initialize plot
        f, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    #Recovered
        ax[0].plot(ytest[:,0], xtest[:,0], 'b*', label='Recovered')
        ax[1].plot(ytest[:,1], xtest[:,1], 'b*', label='Recovered')
          
    #Injected
    #plt.plot(test_y.transpose(-2, -1).numpy()[0], test_y.transpose(-2, -1).numpy()[1], 'rx', label='Injected')
    
    #Predicted
        ax[0].plot(ytest[:,0], predicted_data[:,0], 'go', label='Predicted')       
        ax[1].plot(ytest[:,1], predicted_data[:,1], 'go', label='Predicted') 
    
    #True values
        ax[0].plot(ytest[:,0], ytest[:,0], 'r-', label='True value')     
        ax[1].plot(ytest[:,1], ytest[:,1], 'r-', label='True value') 
    
        ax[0].set_xlabel(r'$m_1$ injected', fontsize=14)
        ax[0].set_ylabel(r'$m_1$ GSTLAL', fontsize=14)
        ax[0].legend(fontsize=12)
    
        ax[1].set_xlabel(r'$m_2$ injected', fontsize=14)
        ax[1].set_ylabel(r'$m_2$ GSTLAL', fontsize=14)
        ax[1].legend(fontsize=12)
    
        outfile = '../figs/'+file_prefix+'_m1m2.pdf'
        plt.savefig(outfile)
        print('Saved figure to {}'.format(outfile))

    rel_err = np.abs((ytest - predicted_data) /ytest)
    abs_diff = np.abs(ytest - predicted_data)

#from matplotlib.transforms import ScaledTranslation

    fig, axis = plt.subplots(1, figsize=(7,5.25))

    combined_epsilons = [rel_err[:,0], rel_err[:,1]]

    hist, bins, _ = axis.hist(combined_epsilons, 20)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    axis.clear()

    axis.set_xlabel(r'$\epsilon$', fontsize=14)
    axis.set_ylabel(r'Counts', fontsize=14)
    axis.tick_params(axis='both', which='major', labelsize=14, pad=10)

#point_0 = axis.scatter(x=0, y=0, s=50, facecolors='none', edgecolors=color_cycle[0], label='Medians')
#point_0.set_clip_on(False)

    styles = ['-','-']
    labels = ['$m_1$ errors','$m_2$ errors']
    results = [rel_err[0], rel_err[1]]

    color = ['blue', 'orange']
    for i in range(len(combined_epsilons)):
        axis.hist(combined_epsilons[i], bins=logbins, histtype=u'step', label=labels[i], color=color_cycle[i+1], ls=styles[i])

        point = axis.scatter(x=np.median(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color_cycle[i+1])
        print(np.median(combined_epsilons[i]))
        point.set_clip_on(False)

    axis.set_xscale('log')
#axis.set_xlim(3e-5,4e-1)

    axis.legend(ncol = 1, loc="upper left", fontsize=12)
    outfile = '../figs/'+file_prefix+'_Epsilon_Analysis.pdf'
    plt.savefig(outfile)
    print('Saved figure to {}'.format(outfile))
