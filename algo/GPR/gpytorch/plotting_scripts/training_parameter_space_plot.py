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

    parser = argparse.ArgumentParser(prog='training_parameter_space_plot.py', description=__doc__)

    args = parser.parse_args()

    ########################################
    
    print('Reading data ...')
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

    bbh = np.where((ytest_masses[:,0]>=5) & (ytest_masses[:,1]>=5))
    bns = np.where((ytest_masses[:,0]<5) & (ytest_masses[:,1]<5))
    nsbh = np.where((ytest_masses[:,0]>=5) & (ytest_masses[:,1]<5))

    bbh_m = ytest_masses[bbh,:]
    bns_m = ytest_masses[bns,:]
    nsbh_m = ytest_masses[nsbh,:]
    bbh_s = ytest_spins[bbh,:]
    bns_s = ytest_spins[bns,:]
    nsbh_s = ytest_spins[nsbh,:]

    print('Plotting ...')
    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(bbh_m[0][:,0], bbh_m[0][:,1], 'o', markersize=2, color=color_cycle[1], zorder=-1, label='BBH')
    ax[0].plot(bns_m[0][:,0], bns_m[0][:,1], 'o', markersize=2, color=color_cycle[2], label='BNS')
    ax[0].plot(nsbh_m[0][:,0], nsbh_m[0][:,1], 'o', markersize=2, color=color_cycle[3], label='NSBH')
    ax[0].set_xlabel(r'$m_1$')
    ax[0].set_ylabel(r'$m_2$')
    ax[0].legend()

    ax[1].plot(bbh_s[0][:,0], bbh_s[0][:,1], 'o', markersize=2, color=color_cycle[1], zorder=-3)
    ax[1].plot(bns_s[0][:,0], bns_s[0][:,1], 'o', markersize=2, color=color_cycle[2], zorder=-1)
    ax[1].plot(nsbh_s[0][:,0], nsbh_s[0][:,1], 'o', markersize=2, color=color_cycle[3], zorder=-2)
    ax[1].set_xlabel(r'$\chi_1$')
    ax[1].set_ylabel(r'$\chi_2$')

    plt.tight_layout()
    outfile = '/Users/Lorena/ML_IPAM/IPAM2021_ML/papers/regression/figs/training_parameter_space.png'
    plt.savefig(outfile, bbox_inches="tight")
    outfile = '/Users/Lorena/ML_IPAM/IPAM2021_ML/papers/regression/figs/training_parameter_space.pdf'
    plt.savefig(outfile, bbox_inches="tight")
    print('Saved figure to {}'.format(outfile))
