"""Make a plot of the recovered vs. regressed values"""

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

    parser = argparse.ArgumentParser(prog='m_chi_comparisons_plot.py', description=__doc__)

    args = parser.parse_args()

    ########################################
    
    print('Reading data ...')
    _, ytrain = extract_data('../data_files/complete_ytrain.csv')
    _, xtrain = extract_data('../data_files/complete_xtrain.csv')
    _, ytest = extract_data('../data_files/complete_ytest.csv')
    _, xtest = extract_data('../data_files/complete_xtest.csv')
    _, predicted_data = extract_data('../data_files/complete_predictions.csv')
    predicted_data = flip_mass_values(predicted_data)

    print('Plotting ...')
    f, ax = plt.subplots(2, 2, figsize=(14, 12))

    ax[0,0].plot(ytest[:,0], ytest[:,0], '-', linewidth=1.7, color=color_cycle[0], zorder=-1, label='Injected')
    ax[0,0].plot(ytest[:,0], xtest[:,0], 'o', markersize=4, color=color_cycle[9], zorder=-3, label='Recovered')
    ax[0,0].plot(ytest[:,0], predicted_data[:,0], 'o', markersize=4, color=color_cycle[6], zorder=-2, label='Regressed')   
    ax[0,0].set_xlabel(r'$m_1$ injected')
    ax[0,0].set_ylabel(r'$m_1$ recovered/regressed')
    ax[0,0].legend()

    ax[0,1].plot(ytest[:,1], ytest[:,1], '-', linewidth=1.7, color=color_cycle[0], zorder=-1)
    ax[0,1].plot(ytest[:,1], xtest[:,1], 'o', markersize=4, color=color_cycle[9], zorder=-3)
    ax[0,1].plot(ytest[:,1], predicted_data[:,1], 'o', markersize=4, color=color_cycle[6], zorder=-2)   
    ax[0,1].set_xlabel(r'$m_2$ injected')
    ax[0,1].set_ylabel(r'$m_2$ recovered/regressed')


    ax[1,0].plot(ytest[:,2], ytest[:,2], '-', linewidth=1.7, color=color_cycle[0], zorder=-1)
    ax[1,0].plot(ytest[:,2], xtest[:,2], 'o', markersize=4, color=color_cycle[9], zorder=-3)
    ax[1,0].plot(ytest[:,2], predicted_data[:,2], 'o', markersize=4, color=color_cycle[6], zorder=-2)
    ax[1,0].set_xlabel(r'$\chi_1$ injected ', fontsize=18)
    ax[1,0].set_ylabel(r'$\chi_1$ recovered/regressed', fontsize=18)

    ax[1,1].plot(ytest[:,3], ytest[:,3], '-', linewidth=1.7, color=color_cycle[0], zorder=-1)
    ax[1,1].plot(ytest[:,3], xtest[:,3], 'o', markersize=4, color=color_cycle[9], zorder=-3)
    ax[1,1].plot(ytest[:,3], predicted_data[:,3], 'o', markersize=4, color=color_cycle[6], zorder=-2)
    ax[1,1].set_xlabel(r'$\chi_2$ injected ')
    ax[1,1].set_ylabel(r'$\chi_2$ recovered/regressed')

    plt.tight_layout()
    outfile = '/Users/Lorena/ML_IPAM/IPAM2021_ML/papers/regression/figs/m_chi_comparisons.png'
    plt.savefig(outfile, bbox_inches="tight")
    outfile = '/Users/Lorena/ML_IPAM/IPAM2021_ML/papers/regression/figs/m_chi_comparisons.pdf'
    plt.savefig(outfile, bbox_inches="tight")
    print('Saved figure to {}'.format(outfile))
