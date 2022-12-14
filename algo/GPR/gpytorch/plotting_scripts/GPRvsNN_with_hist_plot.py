"""Make a plot of the regressed values using GPR and NN"""

import sys
import argparse
from pathlib import Path

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0,'/Users/Lorena/ML_IPAM/IPAM2021_ML/algo/GPR/gpytorch')
from data_conditioning import extract_data, flip_mass_values

# Plot colors
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

plt.style.use('paper.mplstyle')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='GPRvsNN_with_hist_plot.py', description=__doc__)

    args = parser.parse_args()

    ########################################
    
    print('Reading data ...')
    _, ytrain = extract_data('../data_files/complete_ytrain.csv')
    _, xtrain = extract_data('../data_files/complete_xtrain.csv')
    _, ytest = extract_data('../data_files/complete_ytest.csv')
    _, xtest = extract_data('../data_files/complete_xtest.csv')
    _, GPR_predicted_data = extract_data('../data_files/complete_predictions.csv')
    _, NN_predicted_data = extract_data('../../../classy_NN/sklassy_prediction/complete_prediction_m1m2chi1chi2.csv')
    GPR_predicted_data = flip_mass_values(GPR_predicted_data)
    NN_predicted_data = flip_mass_values(NN_predicted_data)

    ytest_masses = ytest[:,:2]
    xtest_masses = xtest[:,:2]
    ytest_spins = ytest[:,2:]
    xtest_spins = xtest[:,2:]
    GPR_predicted_masses = GPR_predicted_data[:,:2]
    GPR_predicted_spins = GPR_predicted_data[:,2:]
    NN_predicted_masses = NN_predicted_data[:,:2]
    NN_predicted_spins = NN_predicted_data[:,2:]
    
    GPR_rel_err_M = (ytest_masses - GPR_predicted_masses)/ytest_masses
    GPR_diff_S = ytest_spins - GPR_predicted_spins

    NN_rel_err_M = (ytest_masses - NN_predicted_masses)/ytest_masses
    rec_rel_err_M = (ytest_masses - xtest_masses)/ytest_masses
    NN_diff_S = ytest_spins - NN_predicted_spins
    rec_diff_S = ytest_spins - xtest_spins

    print('Plotting ...')
    f, ax = plt.subplots(2, 4, figsize=(20, 10))

    ax[0,0].plot(ytest[:,0], ytest[:,0], '-', linewidth=1.7, color=color_cycle[0], zorder=-1, label='Injected')
    ax[0,0].plot(ytest[:,0], xtest[:,0], 'o', markersize=2, color=color_cycle[9],
            zorder=-3, alpha=0.3, label='Recovered')
    ax[0,0].plot(ytest[:,0], GPR_predicted_data[:,0], 'o', markersize=2,
            color=color_cycle[2], zorder=-2, label='GPR')   
    ax[0,0].set_xlabel(r'$m_1$ injected')
    ax[0,0].set_ylabel(r'$m_1$ recovered/regressed')

    ax[0,1].plot(ytest[:,1], ytest[:,1], '-', linewidth=1.7, color=color_cycle[0], zorder=-1)
    ax[0,1].plot(ytest[:,1], xtest[:,1], 'o', markersize=2, color=color_cycle[9],
            zorder=-3, alpha=0.3)
    ax[0,1].plot(ytest[:,1], GPR_predicted_data[:,1], 'o', markersize=2, color=color_cycle[2], zorder=-2)   
    ax[0,1].set_xlabel(r'$m_2$ injected')
    ax[0,1].set_ylabel(r'$m_2$ recovered/regressed')


    ax[0,2].plot(ytest[:,2], ytest[:,2], '-', linewidth=1.7, color=color_cycle[0], zorder=-1)
    ax[0,2].plot(ytest[:,2], xtest[:,2], 'o', markersize=2, color=color_cycle[9],
            zorder=-3, alpha=0.3)
    ax[0,2].plot(ytest[:,2], GPR_predicted_data[:,2], 'o', markersize=2, color=color_cycle[2], zorder=-2)
    ax[0,2].set_xlabel(r'$\chi_1$ injected ')
    ax[0,2].set_ylabel(r'$\chi_1$ recovered/regressed')

    ax[0,3].plot(ytest[:,3], ytest[:,3], '-', linewidth=1.7, color=color_cycle[0], zorder=-1)
    ax[0,3].plot(ytest[:,3], xtest[:,3], 'o', markersize=2, color=color_cycle[9],
            zorder=-3, alpha=0.3)
    ax[0,3].plot(ytest[:,3], GPR_predicted_data[:,3], 'o', markersize=2, color=color_cycle[2], zorder=-2)
    ax[0,3].set_xlabel(r'$\chi_2$ injected ')
    ax[0,3].set_ylabel(r'$\chi_2$ recovered/regressed')
#########
    ax[0,0].plot(ytest[:,0], NN_predicted_data[:,0], 'o', markersize=2,
            mfc='none', color=color_cycle[1], zorder=-2, label='NN')
    ax[0,1].plot(ytest[:,1], NN_predicted_data[:,1], 'o', markersize=2,
            mfc='none', color=color_cycle[1], zorder=-2)   
    ax[0,2].plot(ytest[:,2], NN_predicted_data[:,2], 'o', markersize=2,
            mfc='none', color=color_cycle[1], zorder=-2)
    ax[0,3].plot(ytest[:,3], NN_predicted_data[:,3], 'o', markersize=2,
            mfc='none', color=color_cycle[1], zorder=-2)
    ax[0,0].legend()
    

    combined_epsilons = [rec_rel_err_M[:,0], GPR_rel_err_M[:,0], NN_rel_err_M[:,0]]
    hist, bins, _ = ax[1,0].hist(combined_epsilons, 100)
    ax[1,0].clear()
    ax[1,0].set_xlabel(r'$\Delta m_1 / m_1$')
    ax[1,0].set_ylabel(r'Count')
    styles = ['-','-','--']
    labels = ['Recovered','GPR','NN']
    color = [color_cycle[9], color_cycle[2], color_cycle[1]]
    histtype=['stepfilled','step', 'step']
    alpha= [0.5, 1, 1]
    for i in range(len(combined_epsilons)):
        ax[1,0].hist(combined_epsilons[i], histtype=histtype[i],bins=bins, log=False, label=labels[i], ls=styles[i],edgecolor=color[i], facecolor=color[i], alpha=alpha[i], linewidth=2)
        point = ax[1,0].scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color[i])
        point.set_clip_on(False)

    ax[1,0].scatter(x=[None],y=0, s=120, facecolors='none', color=color_cycle[0], label='Mean')

    ax[1,0].legend(loc="upper left")

    ###################

    combined_epsilons = [rec_rel_err_M[:,1], GPR_rel_err_M[:,1], NN_rel_err_M[:,1]]

    hist, bins, _ = ax[1,1].hist(combined_epsilons, 200)
    ax[1,1].clear()
    ax[1,1].set_xlabel(r'$\Delta m_2 / m_2$')
    ax[1,1].set_ylabel(r'Count')
    styles = ['-','-','--']
    color = [color_cycle[9], color_cycle[2], color_cycle[1]]
    histtype=['stepfilled', 'step', 'step']
    alpha= [0.5, 1, 1]
    for i in range(len(combined_epsilons)):
        ax[1,1].hist(combined_epsilons[i], histtype=histtype[i], bins=bins, log=False, ls=styles[i],edgecolor=color[i],facecolor=color[i], alpha=alpha[i], linewidth=2)
        point = ax[1,1].scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color[i])
        point.set_clip_on(False)

    ax[1,1].scatter(x=[None],y=0, s=120, facecolors='none', color=color_cycle[0])

    ###################
    combined_epsilons = [rec_diff_S[:,0], GPR_diff_S[:,0], NN_diff_S[:,0]]

    hist, bins, _ = ax[1,2].hist(combined_epsilons, 100)
    ax[1,2].clear()
    ax[1,2].set_xlabel(r'$\Delta \chi_1$')
    ax[1,2].set_ylabel(r'Counts')
    styles = ['-','-','--']
    color = [color_cycle[9], color_cycle[2], color_cycle[1]]
    histtype=['stepfilled', 'step','step']
    alpha= [0.5, 1, 1]
    for i in range(len(combined_epsilons)):
        ax[1,2].hist(combined_epsilons[i], histtype=histtype[i], bins=bins, log=False, ls=styles[i],edgecolor=color[i], facecolor=color[i], alpha=alpha[i], linewidth=2)
        point = ax[1,2].scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color[i])
        point.set_clip_on(False)

    ax[1,2].scatter(x=[None],y=0, s=120, facecolors='none', color=color_cycle[0])

    ###################

    combined_epsilons = [rec_diff_S[:,1], GPR_diff_S[:,1], NN_diff_S[:,1]]

    hist, bins, _ = ax[1,3].hist(combined_epsilons, 100)
    ax[1,3].clear()
    ax[1,3].set_xlabel(r'$\Delta \chi_2$')
    ax[1,3].set_ylabel(r'Counts')
    styles = ['-','-','--']
    color = [color_cycle[9], color_cycle[2], color_cycle[1]]
    histtype=['stepfilled', 'step','step']
    alpha= [0.5, 1, 1]
    for i in range(len(combined_epsilons)):
        ax[1,3].hist(combined_epsilons[i], histtype=histtype[i], bins=bins, log=False, ls=styles[i],edgecolor=color[i], facecolor=color[i], alpha=alpha[i], linewidth=2)
        point = ax[1,3].scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color[i])
        point.set_clip_on(False)

    ax[1,3].scatter(x=[None],y=0, s=120, facecolors='none', color=color_cycle[0])    

    ax[1,0].set_xlim((-3,1.2))
    ax[1,1].set_xlim((-3,1.2))

    plt.tight_layout()
    outfile = '/Users/Lorena/ML_IPAM/IPAM2021_ML/papers/regression/figs/GPRvsNN_with_hist.png'
    plt.savefig(outfile, bbox_inches="tight")
    outfile = '/Users/Lorena/ML_IPAM/IPAM2021_ML/papers/regression/figs/GPRvsNN_with_hist.pdf'
    plt.savefig(outfile, bbox_inches="tight")
    print('Saved figure to {}'.format(outfile))
