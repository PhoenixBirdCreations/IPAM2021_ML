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
    """ Usual plot with injection on x-axis and 
    recovered/predicted on y. data is the struct
    produced in the main
    """
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
    if data.savepng:
        figname = data.plots_prefix+'m_chi_comparisons.png'
        fullname = data.plots_dir+'/'+figname
        plt.savefig(figname,dpi=200,bbox_inches='tight')
        if data.verbose:
            print(figname, 'saved in', data.plots_dir)
    plt.show() 
    return 

def plot_parspace(data):
    """ Plot injections
    """
    dot_size = 1
    color    = [0.3,0.3,1]
    fig, axs = plt.subplots(1,2,figsize=(9,5))
    axs[0].scatter(data.inj[:,0], data.inj[:,1], s=dot_size, color=color)
    axs[1].scatter(data.inj[:,2], data.inj[:,3], s=dot_size, color=color)
    xlab1 = escapeLatex(data.var_names_tex[0])
    ylab1 = escapeLatex(data.var_names_tex[1])
    xlab2 = escapeLatex(data.var_names_tex[2])
    ylab2 = escapeLatex(data.var_names_tex[3])
    axs[0].set_xlabel(xlab1, fontsize=15)
    axs[0].set_ylabel(ylab1, fontsize=15)
    axs[1].set_xlabel(xlab2, fontsize=15)
    axs[1].set_ylabel(ylab2, fontsize=15)
    plt.subplots_adjust(wspace=0.4)
    if data.savepng:
        figname = data.plots_prefix+'parspace.png'
        fullname = data.plots_dir+'/'+figname
        plt.savefig(figname,dpi=200,bbox_inches='tight')
        if data.verbose:
            print(figname, 'saved in', data.plots_dir)
    plt.show()
    return

def plot_histograms(data):
    fig, axs = plt.subplots(2,2,figsize=(8,8))
    color_rec  = np.array([0.7,0.7,0.7]);
    color_pred = np.array([1,0.8,0]);
    for i in range(NFEATURES):
        ax = axs[int(i/2), i%2]
        if data.var_names[i]=='chi1' or data.var_names[i]=='chi2':
            y_rec  = data.stats['diffs_rec'][:,i]
            y_pred = data.stats['diffs_pred'][:,i]
        else:
            y_rec  = data.stats['errors_rec'][:,i]
            y_pred = data.stats['errors_pred'][:,i]
        fmin  = -3
        fmax  =  3
        nbins = 30
        fstep = (fmax-fmin)/nbins
        ax.hist(y_rec, bins=np.arange(fmin, fmax, fstep), color=color_rec, histtype='bar', ec='black')
        ax.hist(y_pred, bins=np.arange(fmin, fmax, fstep),color=color_pred, histtype='bar', ec='black')
    if data.savepng:
        figname = data.plots_prefix+'histo.png'
        fullname = data.plots_dir+'/'+figname
        plt.savefig(figname,dpi=200,bbox_inches='tight')
        if data.verbose:
            print(figname, 'saved in', data.plots_dir)
    plt.show()
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

    inj = splitted_data['inj']
    rec = splitted_data['rec']

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
    pred = extract_data(fname, verbose=verbose)

    def order_data(X, old_idx):
        x1 = X[:,old_idx[0]]
        x2 = X[:,old_idx[1]]
        x3 = X[:,old_idx[2]]
        x4 = X[:,old_idx[3]]
        return np.column_stack((x1,x2,x3,x4))

    if args.use_NN_data and args.regr_vars=='m1Mcchi1chi2':
        # in this case re-order the prediction and the input
        inj  = order_data(inj,  [0,3,1,2])
        rec  = order_data(rec, [0,3,1,2])
        pred = order_data(pred, [0,3,1,2])


    dashes = '-'*50
    if verbose:
        print(dashes)
        print('Shape of injected  matrix:', np.shape(inj))
        print('Shape of recovered matrix:', np.shape(rec))
        print('Shape of predicted matrix:', np.shape(pred))
        print(dashes)

    data               = lambda:0
    data.inj           = inj
    data.rec           = rec
    data.pred          = pred
    data.var_names     = var_names
    data.var_names_tex = var_names_tex
    data.var_idx       = var_idx
    data.savepng       = args.savepng
    data.plots_dir     = args.plots_dir
    data.plots_prefix  = plots_prefix
    data.verbose       = verbose
    
    data.stats = {}
    for i in range(NFEATURES):
        data.stats['diffs_rec']   =  inj-rec
        data.stats['diffs_pred']  =  inj-pred
        with np.errstate(divide='ignore'):
            data.stats['errors_rec']  = (inj-rec )/inj 
            data.stats['errors_pred'] = (inj-pred)/inj 

    for plot_id in args.plots2do:
        if plot_id=='rec_vs_pred':
            plot_recovered_vs_predicted(data)
        elif plot_id=='parspace':
            plot_parspace(data)
        elif plot_id=='histo':
            plot_histograms(data)
        else:
            print('Unknown plot: '+plot_id)

