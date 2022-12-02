import argparse, sys, os
import numpy as np
import matplotlib.pyplot as plt 

repo_paths = ['/Users/Lorena/ML_IPAM/IPAM2021_ML/', '/Users/simonealbanesi/repos/IPAM2021_ML/']
for rp in repo_paths:
    if os.path.isdir(rp):
        repo_path = rp
        break
sys.path.insert(0, repo_path+'algo/classy_NN/')
from split_GstLAL_data import split_GstLAL_data  
from sklassyNN import extract_data

# this is hard-coded, but at this point I don't think we will change this number
NFEATURES = 4 

def escapeLatex(text):
    if text: import matplotlib
    if text and matplotlib.rcParams['text.usetex']:
        return text.replace('_', '{\\textunderscore}')
    else:
        return text

###################################################
# Plots 
###################################################
def plot_recovered_vs_predicted(data):
    dot_size = 1
    edge_color_factor = 1
    fig, axs = plt.subplots(2,2,figsize = (8,8))
    color_rec  = np.array([0.7,0.7,0.7]);
    color_pred = np.array([1,0.8,0]);
    for i in range(NFEATURES):
        ax = axs[int(i/2), i%2]
        ax.scatter(data.inj[:,i], data.rec[:,i],  label='recovered', s=dot_size, 
                   color=color_rec,  edgecolors=color_rec/edge_color_factor)
        ax.scatter(data.inj[:,i], data.pred[:,i], label='predicted', s=dot_size, 
                   color=color_pred, edgecolors=color_pred/edge_color_factor)
        ax.plot(data.inj[:,i],data.inj[:,i], color='k')
        ylabel = escapeLatex(data.var_names_tex[i]) + ' -  recovered/predicted'
        ax.set_ylabel(ylabel, fontsize=15)
        if i>1:
            ax.set_xlabel(r'injected', fontsize=20)
    #plt.legend()
    plt.subplots_adjust(wspace=0.4)
    plt.show() 
    if data.savepng:
        figname = data.plots_prefix+'m_chi_comparisons.png'
        fullname = data.plots_dir+'/'+figname
        plt.savefig(figname,dpi=200,bbox_inches='tight')
        if data.verbose:
            print(figname, 'saved in', data.plots_dir)

    return 


###################################################
# Main 
###################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='paper_plots', description='plots to use in the regression paper')
    parser.add_argument('--NN', dest='use_NN_data', action='store_true',
                        help="Use NN-data (path and filename hardcoded).")
    parser.add_argument('--GPR', dest='use_GPR_data', action='store_true',
                        help="Use GPR-data (path and filename hardcoded).")
    parser.add_argument('-p', '--plots', dest='plots2do', nargs='+',default=['rec_vs_pred'],
                        help='Identifiers of the plots to do, e.g. >> -p rec_vs_pred')
    parser.add_argument('--vars', type=str, dest='regr_vars', default='m1m2chi1chi2', 
                        help="Variables used in the regression. Can be 'm1m2chi1chi2' or 'm1Mchi1chi2'")
    parser.add_argument('--dataset_path', type=str, dest='dataset_path', default=repo_path+'datasets/GstLAL/', 
                        help="Path where there are the O2 injections (test_NS.csv)")
    parser.add_argument('-s', '--save',  dest='savepng', action='store_true', 
                        help="Save plots in PNG format")
    parser.add_argument('--plots_dir', type=str, dest='plots_dir', default=os.getcwd(),
                        help="Directory where to save plots (default is current dir)")
    parser.add_argument('-v', '--verbose',  dest='verbose', action='store_true', 
                        help="Print stuff")
    
    args = parser.parse_args()
    verbose = args.verbose

    # load injected and recovered  
    X = extract_data(args.dataset_path+'/test_NS.csv', skip_header=True, verbose=verbose)
    if args.regr_vars=='m1Mcchi1chi2':
        splitted_data = split_GstLAL_data(X, features='mass&spin')
        var_names     = ['m1', 'Mc', 'chi1', 'chi2']
        var_names_tex = ['$m_1$', '${\cal{M}}_c$', '$\chi_1$', '$\chi_2$']
        var_idx         = {}
        var_idx['m1']   = 0
        var_idx['m2']   = None
        var_idx['Mc']   = 1
        var_idx['chi1'] = 2
        var_idx['chi2'] = 3
         
    elif args.regr_vars=='m1m2chi1chi2':
        splitted_data = split_GstLAL_data(X, features='m1m2chi1chi2')
        var_names     = ['m1', 'm2', 'chi1', 'chi2']
        var_names_tex = ['$m_1$', '$m_2$', '$\chi_1$', '$\chi_2$']
        var_idx         = {}
        var_idx['m1']   = 0
        var_idx['m2']   = 1
        var_idx['Mc']   = None
        var_idx['chi1'] = 2
        var_idx['chi2'] = 3


    injected  = splitted_data['inj']
    recovered = splitted_data['rec']

    # load prediction 
    plots_prefix = args.regr_vars
    if args.use_NN_data:
        fname = repo_path+'algo/classy_NN/sklassy_prediction/prediction_'+args.regr_vars+'.csv' 
        plots_prefix += '_NN_'
    elif args.use_GPR_data:
        fname = repo_path+'algo/GPR/something.csv'
        plots_prefix += '_GPR_'
    else:
        raise RuntimeError('Invalid input. Use --NN or --GPR')
    predicted = extract_data(fname, verbose=verbose)

    def order_data(X, old_idx):
        x1 = X[:,old_idx[0]]
        x2 = X[:,old_idx[1]]
        x3 = X[:,old_idx[2]]
        x4 = X[:,old_idx[3]]
        return np.column_stack((x1,x2,x3,x4))

    if args.use_NN_data and args.regr_vars=='m1Mcchi1chi2':
        # in this case re-order the prediction and the input
        injected  = order_data(injected,  [0,3,1,2])
        recovered = order_data(recovered, [0,3,1,2])
        predicted = order_data(predicted, [0,3,1,2])


    dashes = '-'*50
    if verbose:
        print(dashes)
        print('Shape of injected  matrix:', np.shape(injected))
        print('Shape of recovered matrix:', np.shape(recovered))
        print('Shape of predicted matrix:', np.shape(predicted))
        print(dashes)

    data               = lambda:0
    data.inj           = injected
    data.rec           = recovered
    data.pred          = predicted
    data.var_names     = var_names
    data.var_names_tex = var_names_tex
    data.var_idx       = var_idx
    data.savepng       = args.savepng
    data.plots_dir     = args.plots_dir
    data.plots_prefix  = plots_prefix
    data.verbose       = verbose

    for plot_id in args.plots2do:
        if plot_id=='rec_vs_pred':
            plot_recovered_vs_predicted(data)
        else:
            print('Unknown plot: '+plot_id)

