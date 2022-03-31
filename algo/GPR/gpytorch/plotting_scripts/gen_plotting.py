import sys
import argparse
#import re

from sklearn.utils import shuffle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

sys.path.insert(0, '/Users/Lorena/ML_IPAM/IPAM2021_ML/utils')
from utils import *

# Read data 
xtrain = extractData('../../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/train_recover.csv')
ytrain = extractData('../../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/train_inject.csv')
xtest = extractData('../../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/test_recover.csv')
ytest = extractData('../../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/test_inject.csv')

xtrain = shuffle(xtrain, random_state=5)
ytrain = shuffle(ytrain, random_state=5)
xtest = shuffle(xtest, random_state=42)
ytest = shuffle(ytest, random_state=42)

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

    parser.add_argument('predicted_data_file',
                        help='Directory containing metadata and rhOverM_Asymptotic_GeometricUnits_CoM.h5')
    parser.add_argument('--log', default='INFO',
                        help='Log level (default: %(default)s)')

    args = parser.parse_args()

    ########################################

    predicted_data = extractData(args.predicted_data_file)
    #predicted_data = extractData('../data_files/just_RBF.csv')

    abs_rel_err = np.abs((ytest - predicted_data) /ytest)
    rel_err = (ytest - predicted_data) /ytest
    abs_pred_diff = np.abs(ytest - predicted_data)
    pred_diff = ytest - predicted_data
    rec_diff = ytest - xtest
    rec_rel_err = (ytest - xtest)/ytest


    # Figure 1
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    plt.plot(xtest[:,0], xtest[:,1], 'b*', label='Testing Recovered')     
    plt.plot(ytest[:,0], ytest[:,1], 'rx', label='Testing Injected')
    plt.plot(predicted_data[:,0], predicted_data[:,1], 'go', label='Predicted') 
    plt.xlabel(r'$m_1$', fontsize=14)
    plt.ylabel(r'$m_2$', fontsize=14)
    plt.legend()
    outfile = '../figs/m1m2_comparison.pdf'
    plt.savefig(outfile, bbox_inches="tight")

    # Figure 2
    f, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(ytest[:,0], xtest[:,0], 'b*', label='Recovered')
    ax[1].plot(ytest[:,1], xtest[:,1], 'b*', label='Recovered')
    ax[0].plot(ytest[:,0], predicted_data[:,0], 'go', label='Predicted')       
    ax[1].plot(ytest[:,1], predicted_data[:,1], 'go', label='Predicted') 
    ax[0].plot(ytest[:,0], ytest[:,0], 'r-', label='True value')     
    ax[1].plot(ytest[:,1], ytest[:,1], 'r-', label='True value') 
    
    ax[0].set_xlabel(r'$m_1$ injected', fontsize=14)
    ax[0].set_ylabel(r'$m_1$ GSTLAL', fontsize=14)
    ax[0].legend(fontsize=12)
    ax[1].set_xlabel(r'$m_2$ injected', fontsize=14)
    ax[1].set_ylabel(r'$m_2$ GSTLAL', fontsize=14)
    ax[1].legend(fontsize=12)
    
    outfile = '../figs/m1_m2_comparisons.pdf'
    plt.savefig(outfile, bbox_inches="tight")

    # Figure 3
    fig, axis = plt.subplots(1, 1, figsize=(7,6))
    combined_epsilons = [abs_rel_err[:,0], abs_rel_err[:,1]]
    hist, bins, _ = axis.hist(combined_epsilons, 20)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    axis.clear()
    axis.set_xlabel(r'$|y_{\mathrm{inj}}-y_{\mathrm{pred}}/y_{\mathrm{inj}}|$', fontsize=14)
    axis.set_ylabel(r'Counts', fontsize=14)
    axis.tick_params(axis='both', which='major', labelsize=14, pad=10)
    styles = ['-','-']
    labels = ['$m_1$ errors','$m_2$ errors']
    color = ['blue', 'orange']

    for i in range(len(combined_epsilons)):
        axis.hist(combined_epsilons[i], bins=logbins, histtype=u'step', label=labels[i], color=color_cycle[i+1], ls=styles[i])
        point = axis.scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color_cycle[i+1])
        #print('The median is: ',np.median(combined_epsilons[i]), ' and the mean is: ',np.mean(combined_epsilons[i]))
        point.set_clip_on(False)

    axis.set_xscale('log')
    axis.legend(ncol = 1, loc="upper left", fontsize=12)
    outfile = '../figs/m1m2_absolute_errors.pdf'
    plt.savefig(outfile, bbox_inches="tight")

    # Figure 4
    fig, axis = plt.subplots(1, 1, figsize=(7,6))
    combined_epsilons = [rel_err[:,0], rel_err[:,1]]
    hist, bins, _ = axis.hist(combined_epsilons, 20)
    axis.clear()
    axis.set_xlabel(r'$\left(y_{\mathrm{inj}}-y_{\mathrm{pred}}\right)/y_{\mathrm{inj}}$', fontsize=14)
    axis.set_ylabel(r'Counts', fontsize=14)
    axis.tick_params(axis='both', which='major', labelsize=14, pad=10)
    styles = ['-','-']
    labels = ['$m_1$ errors','$m_2$ errors']
    color = ['blue', 'orange']

    for i in range(len(combined_epsilons)):
        axis.hist(combined_epsilons[i], bins=bins, histtype=u'step', label=labels[i], color=color_cycle[i+1], ls=styles[i])
        point = axis.scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color_cycle[i+1])
        #print('The median is: ',np.median(combined_epsilons[i]), ' and the mean is: ',np.mean(combined_epsilons[i]))
        point.set_clip_on(False)

    axis.legend(ncol = 1, loc="upper left", fontsize=12)
    outfile = '../figs/m1m2_errors.pdf'
    plt.savefig(outfile, bbox_inches="tight")
    
    #Figure 5
    fig, axis = plt.subplots(1,2, figsize=(15,5.25))
    combined_epsilons = [rel_err[:,0], rec_rel_err[:,0]]
    hist, bins, _ = axis[0].hist(combined_epsilons, 20)
    axis[0].clear()
    axis[0].set_xlabel(r'$\left(y_{\mathrm{inj}}-y_{\mathrm{pred}}\right)/y_{\mathrm{inj}}$', fontsize=14)
    axis[0].set_ylabel(r'Counts', fontsize=14)
    axis[0].tick_params(axis='both', which='major', labelsize=14, pad=10)
    styles = ['-','-']
    labels = ['$m_1$ predicted','$m_1$ recovered']
    color = ['blue','gray']
    
    for i in range(len(combined_epsilons)):
        axis[0].hist(combined_epsilons[i], bins=bins, label=labels[i], ls=styles[i],facecolor=color[i], alpha=0.5)
        point = axis[0].scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color[i])
        points = axis[0].scatter(x=np.median(combined_epsilons[i]), y=0, s=60, facecolors='none', edgecolors=color[i])
        #print('The median is: ',np.mean(combined_epsilons[i]), ' and the mean is: ',np.mean(combined_epsilons[i]))
        point.set_clip_on(False)
        points.set_clip_on(False)

    axis[0].scatter(x=[None],y=0, s=120, facecolors='none', color=color_cycle[0], label='$m_1$ mean')
    axis[0].scatter(x=[None],y=0, s=60, facecolors='none',color=color_cycle[0], label='$m_1$ median')
    axis[0].legend(ncol = 1, loc="upper left", fontsize=12)
    
    ###################

    combined_epsilons = [rel_err[:,1], rec_rel_err[:,1]]
    hist, bins, _ = axis[1].hist(combined_epsilons, 20)
    axis[1].clear()
    axis[1].set_xlabel(r'$\left(y_{\mathrm{inj}}-y_{\mathrm{pred}}\right)/y_{\mathrm{inj}}$', fontsize=14)
    axis[1].set_ylabel(r'Counts', fontsize=14)
    axis[1].tick_params(axis='both', which='major', labelsize=14, pad=10)
    styles = ['-','-']
    labels = ['$m_2$ predicted','$m_2$ recovered']
    color = ['orange','gray']
    
    for i in range(len(combined_epsilons)):
        axis[1].hist(combined_epsilons[i], bins=bins, label=labels[i], ls=styles[i],facecolor=color[i], alpha=0.5)
        point = axis[1].scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color[i])
        points = axis[1].scatter(x=np.median(combined_epsilons[i]), y=0, s=60, facecolors='none', edgecolors=color[i])
        #print('The median is: ',np.median(combined_epsilons[i]), ' and the mean is: ',np.mean(combined_epsilons[i]))
        point.set_clip_on(False)
        points.set_clip_on(False)

    axis[1].scatter(x=[None],y=0, s=120, facecolors='none', color=color_cycle[0], label='$m_2$ mean')
    axis[1].scatter(x=[None],y=0, s=60, facecolors='none',color=color_cycle[0], label='$m_2$ median')
    axis[1].legend(ncol = 1, loc="upper right", fontsize=12)
    outfile = '../figs/m1_m2_error_analysis.pdf'
    plt.savefig(outfile, bbox_inches="tight")
