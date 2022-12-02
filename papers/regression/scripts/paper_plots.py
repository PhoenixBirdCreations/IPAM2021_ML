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

###################################################
# Plots 
###################################################
def plot_recovered_vs_predicted(data):
    dot_size = 2
    fig, axs = plt.subplots(2,2,figsize = (8,8))
    axs[0,0].scatter(data.inj[:,0], data.rec[:,0],  label='recovered', s=dot_size)
    axs[0,0].scatter(data.inj[:,0], data.pred[:,0], label='predicted', s=dot_size)
    axs[0,1].scatter(data.inj[:,1], data.rec[:,1],  s=dot_size)
    axs[0,1].scatter(data.inj[:,1], data.pred[:,1], s=dot_size)
    axs[1,0].scatter(data.inj[:,2], data.rec[:,2],  s=dot_size)
    axs[1,0].scatter(data.inj[:,2], data.pred[:,2], s=dot_size)
    axs[1,1].scatter(data.inj[:,3], data.rec[:,3],  s=dot_size)
    axs[1,1].scatter(data.inj[:,3], data.pred[:,3], s=dot_size)
    plt.legend()
    plt.show() 
    return 


###################################################
# Main 
###################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='paper_plots', description='plots to use in the regression paper')
    parser.add_argument('--NN', dest='use_NN_data', action='store_true',
                        help="Use NN-data (path and filename hardcoded). Ignored if --filename is used")
    parser.add_argument('--GPR', dest='use_GPR_data', action='store_true',
                        help="Use GPR-data (path and filename hardcoded). Ignored if --filename is used")
    parser.add_argument('-f', '--filename', type=str, dest='input_fname', default=None,
                        help="name of the file with the data (csv format). If this is used, the options --NN and --GPR are ignored")
    parser.add_argument('-p', '--plot', type=str, dest='plot2do', default='recovered_vs_predicted',
                        help='Identifier of the plot to do (string). Options: ...')
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
        var_names_tex = ['$m_1$', '${\cal{M}_c$', '$\chi_1$', '$\chi_2$']
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
    if args.input_fname is not None:
        fname = args.input_fname
    elif args.use_NN_data:
        fname = repo_path+'algo/classy_NN/sklassy_prediction/prediction.csv' 
    elif args.use_GPR_data:
        fname = repo_path+'algo/GPR/something.csv'
    else:
        raise RuntimeError('Specify the prediction to use. Use --NN, --GPR or --filename <filename>')
    predicted = extract_data(fname, verbose=verbose)
    
    if verbose:
        print('Shape of injected  matrix:', np.shape(injected))
        print('Shape of recovered matrix:', np.shape(recovered))
        print('Shape of predicted matrix:', np.shape(predicted))

    data               = lambda:0
    data.inj           = injected
    data.rec           = recovered
    data.pred          = predicted
    data.var_names     = var_names
    data.var_names_tex = var_names_tex
    data.var_idx       = var_idx
    data.savepng       = args.savepng
    data.plots_dir     = args.plots_dir

    if args.plot2do=='recovered_vs_predicted':
        plot_recovered_vs_predicted(data)
    else:
        raise RuntimeError('Unknown plot.')

