"""Make a plot of the parameter space spanned by the training data"""

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

    parser = argparse.ArgumentParser(prog='inj_rec_pred_masses_plot.py', description=__doc__)

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
    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(ytest[:,0], ytest[:,1], 'o', markersize=2, color=color_cycle[10],
            zorder=-1, label='Injected')
    ax[0].plot(xtest[:,0], xtest[:,1], 'o', markersize=2, color=color_cycle[9],
            label='Recovered')
    ax[0].plot(predicted_data[:,0], predicted_data[:,1], 'o', markersize=2,
            color=color_cycle[6], label='Regressed')
    ax[0].set_xlabel(r'$m_1$')
    ax[0].set_ylabel(r'$m_2$')
    ax[0].legend()
    
    ax[1].plot(ytest[:,2], ytest[:,3], 'o', markersize=2, color=color_cycle[10], zorder=-1)
    ax[1].plot(xtest[:,2], xtest[:,3], 'o', markersize=2, color=color_cycle[9])
    ax[1].plot(predicted_data[:,2], predicted_data[:,3], 'o', markersize=2, color=color_cycle[6])
    ax[1].set_xlabel(r'$\chi_1$')
    ax[1].set_ylabel(r'$\chi_2$')

    plt.tight_layout()
    outfile = '/Users/Lorena/ML_IPAM/IPAM2021_ML/papers/regression/figs/inj_rec_pred_masses.png'
    plt.savefig(outfile, bbox_inches="tight")
    outfile = '/Users/Lorena/ML_IPAM/IPAM2021_ML/papers/regression/figs/inj_rec_pred_masses.pdf'
    plt.savefig(outfile, bbox_inches="tight")
    print('Saved figure to {}'.format(outfile))
