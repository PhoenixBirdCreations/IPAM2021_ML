"""Make a histogram of the recovered vs. regressed errors"""

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

    parser = argparse.ArgumentParser(prog='m1_m2_chi_error_analysis_wboxes_plot.py', description=__doc__)

    args = parser.parse_args()

    ########################################
    
    print('Reading ...')
    _, ytrain = extract_data('../data_files/complete_ytrain.csv')
    _, xtrain = extract_data('../data_files/complete_xtrain.csv')
    _, ytest = extract_data('../data_files/complete_ytest.csv')
    _, xtest = extract_data('../data_files/complete_xtest.csv')
    _, predicted_data = extract_data('../data_files/complete_predictions.csv')
    predicted_data = flip_mass_values(predicted_data)

    ytest_masses = ytest[:,:2]
    xtest_masses = xtest[:,:2]
    ytest_spins = ytest[:,2:]
    xtest_spins = xtest[:,2:]
    predicted_masses = predicted_data[:,:2]
    predicted_spins = predicted_data[:,2:]

    rel_err_M = (ytest_masses - predicted_masses)/ytest_masses
    rec_rel_err_M = (ytest_masses - xtest_masses)/ytest_masses
    diff_S = ytest_spins - predicted_spins
    rec_diff_S = ytest_spins - xtest_spins

    print('Plotting ...')
    fig, axis = plt.subplots(2,2, figsize=(14,12))

    combined_epsilons = [rel_err_M[:,0], rec_rel_err_M[:,0]]
    hist, bins, _ = axis[0,0].hist(combined_epsilons, 100)
    axis[0,0].clear()
    axis[0,0].set_xlabel(r'$\Delta m_1 / m_1$')
    axis[0,0].set_ylabel(r'Count')
    styles = ['-','-']
    labels = ['$m_1$ predicted','$m_1$ recovered']
    #for_printing = ['m1 predicted','m1 recovered']
    color = [color_cycle[6], color_cycle[9]]
    for i in range(len(combined_epsilons)):
        axis[0,0].hist(combined_epsilons[i], bins=bins, label=labels[i], ls=styles[i],facecolor=color[i], alpha=0.5)
        point = axis[0,0].scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color[i])
        #print('The mean error of', for_printing[i] , 'is: ',np.mean(combined_epsilons[i]))
        point.set_clip_on(False)

    axis[0,0].scatter(x=[None],y=0, s=120, facecolors='none', color=color_cycle[0], label='$m_1$ mean')
    axis[0,0].legend(ncol = 1, loc="upper left", fontsize=14)

    ###################

    combined_epsilons = [rel_err_M[:,1], rec_rel_err_M[:,1]]

    hist, bins, _ = axis[0,1].hist(combined_epsilons, 200)
    axis[0,1].clear()
    axis[0,1].set_xlabel(r'$\Delta m_2 / m_2$')
    axis[0,1].set_ylabel(r'Count')
    styles = ['-','-']
    labels = ['$m_2$ predicted','$m_2$ recovered']
    #for_printing = ['m2 predicted','m2 recovered']
    color = [color_cycle[6], color_cycle[9]]
    for i in range(len(combined_epsilons)):
        axis[0,1].hist(combined_epsilons[i], bins=bins, label=labels[i], ls=styles[i],facecolor=color[i], alpha=0.5)
        point = axis[0,1].scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color[i])
        #print('The mean error of', for_printing[i] , 'is: ',np.mean(combined_epsilons[i]))
        point.set_clip_on(False)

    axis[0,1].scatter(x=[None],y=0, s=120, facecolors='none', color=color_cycle[0], label='$m_2$ mean')
    axis[0,1].legend(ncol = 1, loc="upper left", fontsize=14)

    ###################

    combined_epsilons = [diff_S[:,0], rec_diff_S[:,0]]

    hist, bins, _ = axis[1,0].hist(combined_epsilons, 100)
    axis[1,0].clear()
    axis[1,0].set_xlabel(r'$\Delta \chi_1$')
    axis[1,0].set_ylabel(r'Counts')
    styles = ['-','-']
    labels = ['$\chi_1$ predicted','$\chi_1$ recovered']
    #for_printing = ['spin1 predicted','spin1 recovered']
    color = [color_cycle[6], color_cycle[9]]
    for i in range(len(combined_epsilons)):
        axis[1,0].hist(combined_epsilons[i], bins=bins, label=labels[i], ls=styles[i],facecolor=color[i], alpha=0.5)
        point = axis[1,0].scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color[i])
        #print('The mean error of', for_printing[i] , 'is: ',np.mean(combined_epsilons[i]))
        point.set_clip_on(False)

    axis[1,0].scatter(x=[None],y=0, s=120, facecolors='none', color=color_cycle[0], label='$m_1$ mean')
    axis[1,0].legend(ncol = 1, loc="upper left", fontsize=14)

    ###################

    combined_epsilons = [diff_S[:,1], rec_diff_S[:,1]]

    hist, bins, _ = axis[1,1].hist(combined_epsilons, 100)
    axis[1,1].clear()
    axis[1,1].set_xlabel(r'$\Delta \chi_2$')
    axis[1,1].set_ylabel(r'Counts')
    styles = ['-','-']
    labels = ['$\chi_2$ predicted','$\chi_2$ recovered']
    #for_printing = ['spin2 predicted','spin2 recovered']
    color = [color_cycle[6], color_cycle[9]]
    for i in range(len(combined_epsilons)):
        axis[1,1].hist(combined_epsilons[i], bins=bins, label=labels[i], ls=styles[i],facecolor=color[i], alpha=0.5)
        point = axis[1,1].scatter(x=np.mean(combined_epsilons[i]), y=0, s=120, facecolors='none', edgecolors=color[i])
        #print('The mean error of', for_printing[i] , 'is: ',np.mean(combined_epsilons[i]))
        point.set_clip_on(False)

    axis[1,1].scatter(x=[None],y=0, s=120, facecolors='none', color=color_cycle[0], label='$m_2$ mean')    
    axis[1,1].legend(ncol = 1, loc="upper left", fontsize=14)

    axis[0,0].set_xlim((-3,1.2))
    axis[0,1].set_xlim((-3,1.2))

    plt.tight_layout()
    outfile = '/Users/Lorena/ML_IPAM/IPAM2021_ML/papers/regression/figs/m1_m2_chi_error_analysis_wboxes.png'
    plt.savefig(outfile, bbox_inches="tight")
    outfile = '/Users/Lorena/ML_IPAM/IPAM2021_ML/papers/regression/figs/m1_m2_chi_error_analysis_wboxes.pdf'
    plt.savefig(outfile, bbox_inches="tight")
    print('Saved figure to {}'.format(outfile))
