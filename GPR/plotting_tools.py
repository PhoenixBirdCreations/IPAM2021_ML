import numpy as np
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

