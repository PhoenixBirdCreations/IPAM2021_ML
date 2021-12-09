import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

def plotting_f(testing_data, predicted_data, kernel_type, show_fig=True):
    """Plots the predicted data along with the true values show by a red line.
    Parameters
    ----------
    testing_data : arr
        The dataset saved for testing the GPR algorithm.
    
    predicted_data : arr
        The dataset predicted by the GPR algorithm.
    
    kernel_type : str
        Kernel used to fit the data.
    
    show_fig: bool, optional [Default: True]
        Display figure
    """
    
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(50, 30), sharex=False)
    #plt.title('mass 1 recovery')
    for i in [0,1,2]:
        for j in [0,1]:
            axs[i,j].tick_params(axis='x', direction='in',length=12, width=2, pad=10, labelsize=30)
            axs[i,j].tick_params(axis='y', which='both', direction='in',length=12, width=2, pad=10, labelsize=30)

    axs[0,0].set_xlabel('Injected m1', fontsize=35)
    axs[0,0].set_ylabel('Predicted m1', fontsize=35)
    axs[0,1].set_xlabel(r'Injected $\chi_1$', fontsize=35)
    axs[0,1].set_ylabel(r'Predicted $\chi_1$', fontsize=35)
    axs[1,0].set_xlabel(r'Injected $\chi_2$', fontsize=35)
    axs[1,0].set_ylabel(r'Predicted $\chi_2$', fontsize=35)
    axs[1,1].set_xlabel(r'Injected $\theta_{12}$', fontsize=35)
    axs[1,1].set_ylabel(r'Predicted $\theta_{12}$', fontsize=35)
    axs[2,0].set_xlabel(r'Injected $q$', fontsize=35)
    axs[2,0].set_ylabel(r'Predicted $q$', fontsize=35)
    axs[2,1].set_xlabel(r'Injected $\mathcal{M}_c$', fontsize=35)
    axs[2,1].set_ylabel(r'Predicted $\mathcal{M}_c$', fontsize=35)

    it = [[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]]
    for q, it in enumerate(it):
        for i in np.arange(0,len(predicted_data)):
            axs[it[0],it[1]].plot(testing_data[i][q], predicted_data[i][q], 'ob', label="Prediction")
        axs[it[0],it[1]].plot(testing_data[:,q], testing_data[:,q], 'r-', lw=4.0, label="True value")

    fig.tight_layout()
    if show_fig==True:
        plt.show()
    outfile = 'figures/GPR_'+kernel_type+'.pdf'
    fig.savefig(outfile)
    print('Saved figure to {}'.format(outfile))


def plotting_feats(testing_data, predicted_data, kernel_type, setv, show_fig=True):
    """Plots the predicted data along with the true values show by a red line.
    This function works when the x_lim is the same for all plots. If it depends
    on the subplot, then use the function plot_all().

    Parameters
    ----------
    testing_data : arr
        The dataset saved for testing the GPR algorithm.
    
    predicted_data : arr
        The dataset predicted by the GPR algorithm.
    
    kernel_type : str
        Kernel used to fit the data.
    
    show_fig: bool, optional [Default: True]
        Display figure
    """
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(50, 30), sharex=True) 
    features = ['$m_1$', '$m_2$', '$s_1^x$', '$s_1^y$', '$s_1^z$', '$s_2^x$', '$s_2^y$', 
            '$s_2^z$', '$\cos(\phi)$', '$q$', '$\mathcal{M}_c$']
    it = [0,1,2,3,4,5]
    k = 0
    for i in it:
        for j in [0,1]:
            if k==len(features):
                break
            f = features[k]
            fstri = 'Inj '+ str(f)
            fstrp = 'Pred '+ str(f)
            axs[i,j].tick_params(axis='x', direction='in',length=12, width=2, pad=10, labelsize=30)
            axs[i,j].tick_params(axis='y', which='both', direction='in',length=12, width=2, pad=10, labelsize=30)
            axs[i,j].set_xlabel(fstri, fontsize=35)
            axs[i,j].set_ylabel(fstrp, fontsize=35)
            k+=1

    ax_it = [[j,k] for j in np.arange(6) for k in [0,1]][:-1]
    for q, it in enumerate(ax_it):
        for i in np.arange(0,len(predicted_data)):
            axs[it[0],it[1]].plot(testing_data[i][q], predicted_data[i][q], 'ob', label="Prediction")
        axs[it[0],it[1]].plot(testing_data[:,q], testing_data[:,q], 'r-', lw=4.0, label="True value")
    fig.delaxes(axs[5,1])
    fig.tight_layout()
    if show_fig==True:
        plt.show()
    outfile = 'figures/GPR_'+kernel_type+setv+'.pdf'
    fig.savefig(outfile)
    print('Saved figure to {}'.format(outfile))
    return fig, axs

def plot_all(testing_data, predicted_data, kernel_func, data_set):
    """TODO"""

    ax1 = plt.subplot(621); ax1.set_xlim(left=-0.1, right=100.1)
    ax2 = plt.subplot(622, sharey=ax1, sharex=ax1)
    ax3 = plt.subplot(623); ax3.set_xlim(left=-1.1, right=1.1)
    ax4 = plt.subplot(624, sharey=ax3, sharex=ax3); ax5 = plt.subplot(625, sharey=ax3, sharex=ax3)
    ax6 = plt.subplot(626, sharey=ax3, sharex=ax3); ax7 = plt.subplot(627, sharey=ax3, sharex=ax3)
    ax8 = plt.subplot(628, sharey=ax3, sharex=ax3); ax9 = plt.subplot(629, sharey=ax3, sharex=ax3)
    ax10 = plt.subplot(6,2,10, sharey=ax1, sharex=ax1); ax11 = plt.subplot(6,2,11, sharey=ax1, sharex=ax1)
    mpl.rcParams['xtick.labelsize'] = 30
    mpl.rcParams['ytick.labelsize'] = 30
    mpl.rcParams['figure.figsize'] = [55, 40]
    plt.rcParams["figure.autolayout"] = True
    
    features = ['$m_1$', '$m_2$', '$s_1^x$', '$s_1^y$', '$s_1^z$', '$s_2^x$', '$s_2^y$', 
            '$s_2^z$', '$\cos(\phi)$', '$q$', '$\mathcal{M}_c$']
    it = [0,1,2,3,4,5]
    ejes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]
    
    k = 0
    for eje in ejes:
        if k==len(features):
            break
        f = features[k]
        fstri = 'Inj '+ str(f)
        fstrp = 'Pred '+ str(f)
        eje.set_xlabel(fstri, fontsize=35)
        eje.set_ylabel(fstrp, fontsize=35)
        k+=1

    for q, eje in enumerate(ejes):
        for i in np.arange(0,len(predicted_data)):
            eje.plot(testing_data[i][q], predicted_data[i][q], 'ob', label="Prediction")
        eje.plot(testing_data[:,q], testing_data[:,q], 'r-', lw=4.0, label="True value")
   
    fig = plt.gcf()
    fig.show()
    plt.draw()
    outfile = 'figures/GPR_'+kernel_func+'_'+data_set+'.pdf'
    fig.savefig(outfile)
    print('Saved figure to {}'.format(outfile))
