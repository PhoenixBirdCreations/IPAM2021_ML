import argparse, sys, os
import numpy as np
import matplotlib.pyplot as plt 

repo_paths = ['/Users/Lorena/ML_IPAM/IPAM2021_ML/', '/Users/simonealbanesi/repos/IPAM2021_ML/']
for rp in repo_paths:
    if os.path.isdir(rp):
        repo_path = rp
        break
sys.path.insert(0, repo_path+'algo/classy_NN/')
sys.path.insert(0, repo_path+'utils/')
from split_GstLAL_data import split_GstLAL_data  
from sklassyNN import extract_data
from utils import chirpMass, findSecondMassFromMc

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
    fig, axs = plt.subplots(2,2,figsize = (7,9))
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
        xlabel = escapeLatex(data.var_names_tex[i]) + ' -  injected'
        ax.set_xlabel(xlabel, fontsize=15)
    #plt.legend()
    plt.subplots_adjust(wspace=0.4)
    if data.savepng:
        figname = data.plots_prefix+'recvspred.png'
        fullname = data.plots_dir+'/'+figname
        plt.savefig(fullname,dpi=200,bbox_inches='tight')
        if data.verbose:
            print(figname, 'saved in', data.plots_dir)
    plt.show() 
    return 

def plot_parspace(data):
    """ Plot injections
    """
    dot_size = 1
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    
    if data.parspace_colorful:
        m1 = data.inj[:,0]
        if data.regr_vars=='m1m2chi1chi2':
            m2 = data.inj[:,1]
        elif data.regr_vars=='m1Mcchi1chi2':
            Mc = data.inj[:,1]
            m2 = findSecondMassFromMc(Mc,m1)
        else:
            raise RuntimeError('Invalid regression variables')
        chi1 = data.inj[:,2]
        chi2 = data.inj[:,3]
        mask   = {}
        colors = {}
        mask['bbh']  = np.where(( (m1>=5) & (m2>=5) ))
        mask['bhns'] = np.where(( (m1>=5) & (m2<5)  ))
        mask['bns']  = np.where(( (m1<5)  & (m2<5)  ))
        colors['bbh']  = [1,0,0]
        colors['bhns'] = [0,0,1]
        colors['bns']  = [0,1,0]
        m1_dict   = {}
        m2_dict   = {}
        chi1_dict = {}
        chi2_dict = {}
        y_dict    = {}
        keys = mask.keys()
        for k in keys:
            m1_dict[k]   = m1[mask[k]]
            m2_dict[k]   = m2[mask[k]]
            chi1_dict[k] = chi1[mask[k]]
            chi2_dict[k] = chi2[mask[k]]
            if data.regr_vars=='m1m2chi1chi2':
                y_dict[k]  = m2_dict[k]
            elif data.regr_vars=='m1Mcchi1chi2':
                y_dict[k]  = chirpMass(m1_dict[k],  m2_dict[k])
         
            axs[0].scatter(m1_dict[k],   y_dict[k],    s=dot_size, color=colors[k])
            axs[1].scatter(chi1_dict[k], chi2_dict[k], s=dot_size, color=colors[k])

    else:
        color = [0.3,0.3,1]
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
        plt.savefig(fullname,dpi=200,bbox_inches='tight')
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
            dy_rec  = data.stats['diffs_rec'][:,i]
            dy_pred = data.stats['diffs_pred'][:,i]
        else:
            dy_rec  = data.stats['errors_rec'][:,i]
            dy_pred = data.stats['errors_pred'][:,i]
        
        fmin  = data.histo_fmins[i]
        fmax  = data.histo_fmaxs[i]
        nbins = data.histo_nbins[i]
        fstep = (fmax-fmin)/nbins
        ax.hist(dy_rec, bins=np.arange(fmin, fmax, fstep), color=color_rec, histtype='bar')
        ax.hist(dy_pred, bins=np.arange(fmin, fmax, fstep),color=color_pred, histtype=u'step', linewidth=2.)
        ax.set_xlabel(escapeLatex(data.var_names_tex[i]), fontsize=15)
        if data.histo_logs[i]==1:
            ax.set_yscale('log')      
        if data.verbose:
            tmp = np.where(dy_rec <fmin)
            print('For {:4s} there are {:4d} recoveries  smaller than fmin={:7.3f}'.format(data.var_names[i], np.shape(tmp)[1], fmin))
            tmp = np.where(dy_pred<fmin)
            print('For {:4s} there are {:4d} predictions smaller than fmin={:7.3f}'.format(data.var_names[i], np.shape(tmp)[1], fmin))
            tmp = np.where(dy_rec >fmax)
            print('For {:4s} there are {:4d} recoveries  bigger  than fmax={:7.3f}'.format(data.var_names[i], np.shape(tmp)[1], fmax))
            tmp = np.where(dy_pred>fmax)
            print('For {:4s} there are {:4d} predictions bigger  than fmax={:7.3f}'.format(data.var_names[i], np.shape(tmp)[1], fmax))
            print(' ')
    if data.savepng:
        figname = data.plots_prefix+'histo.png'
        fullname = data.plots_dir+'/'+figname
        plt.savefig(fullname,dpi=200,bbox_inches='tight')
        if data.verbose:
            print(figname, 'saved in', data.plots_dir)
    plt.show()
    return

###################################################
# Tables 
###################################################
def num2tex(x,precision=3):
    #out = '${:.'+str(precision)+'e}'
    #out = out.format(x)
    #out = out.replace('e+00','')
    #if 'e-0' in out:
    #    out = out.replace('e-0', '\\times 10^{-')+'}'
    #if 'e+0' in out:
    #    out = out.replace('e+0', '\\times 10^{')+'}'
    out = '${:.'+str(precision)+'f}'
    out = out.format(x)
    return out+'$'

def print_errortab(data):
    header = '\n{:14s} {:15s} {:15s} {:15s}     {:15s} {:15s} {:15s}'

    dashes='-'*120
    print('\n', dashes, sep='', end='')
    print(header.format('name  ', '  mean_diff_rec', '  mean_err_rec', '        std_rec', 
                                  ' mean_diff_pred', ' mean_err_pred', '       std_pred'))
    print(dashes)
    
    err_rec    = data.stats['errors_rec']
    err_pred   = data.stats['errors_pred']
    diffs_rec  = data.stats['diffs_rec']
    diffs_pred = data.stats['diffs_pred']
    for i in range(NFEATURES):
        var_name = data.var_names[i]
        mean_diff_rec  = np.mean(np.abs(diffs_rec[:,i]))
        mean_diff_pred = np.mean(np.abs(diffs_pred[:,i]))
        if var_name=='chi1' or var_name=='chi2':
            mean_err_rec  = np.nan  # then substitute with '/' while printing
            mean_err_pred = np.nan  # then substitute with '/' while printing
            std_rec       = np.std(diffs_rec[:,i])
            std_pred      = np.std(diffs_pred[:,i])
        else:
            mean_err_rec  = np.mean(np.abs(err_rec[:,i]))
            mean_err_pred = np.mean(np.abs(err_pred[:,i]))
            std_rec       = np.std(err_rec[:,i])
            std_pred      = np.std(err_pred[:,i])
        
        if data.tab_format=='txt':
            line_format = '{:14s} {:15.3e} {:15.3e} {:15.3e}     {:15.3e} {:15.3e} {:15.3e}'
            myline = line_format.format(var_name, mean_diff_rec,  mean_err_rec,  std_rec, 
                                                  mean_diff_pred, mean_err_pred, std_pred) 
        elif data.tab_format=='tex':
            tex_name = data.var_names_tex[i]
            line_format = escapeLatex('{:14s} & {:s} & {:s} & {:s} & {:s} & {:s} & {:s} \\\\')
            myline = line_format.format(tex_name, num2tex(mean_diff_rec),  num2tex(mean_err_rec),  num2tex(std_rec), 
                                                  num2tex(mean_diff_pred), num2tex(mean_err_pred), num2tex(std_pred)) 
        else:
            raise RuntimeError("'{:s}' is not a valid tab-format".format(data.tab_format))
        print(myline.replace('$nan$', ' / '))
    
    # add missing variable (i.e. Mc or m2)
    if data.regr_vars=='m1m2chi1chi2':
        Mc_inj  = chirpMass(data.inj[:,0], data.inj[:,1]) # (m1, m2)
        Mc_rec  = chirpMass(data.rec[:,0], data.rec[:,1])
        Mc_pred = chirpMass(data.pred[:,0],data.pred[:,1])
        diffs_rec  = Mc_inj-Mc_rec
        diffs_pred = Mc_inj-Mc_pred
        err_rec    = (Mc_inj-Mc_rec)/Mc_inj
        err_pred   = (Mc_inj-Mc_pred)/Mc_inj
        var_name   = 'Mc'
        tex_name   = '${\cal{M}}_c$'
    elif data.regr_vars=='m1Mcchi1chi2':
        m2_inj  = findSecondMassFromMc(data.inj[:,1], data.inj[:,0]) # (Mc, m1)
        m2_rec  = findSecondMassFromMc(data.rec[:,1], data.rec[:,0]) 
        m2_pred = findSecondMassFromMc(data.pred[:,1],data.pred[:,0])
        diffs_rec  = m2_inj-m2_rec
        diffs_pred = m2_inj-m2_pred
        err_rec    = (m2_inj-m2_rec)/m2_inj
        err_pred   = (m2_inj-m2_pred)/m2_inj
        var_name   = 'm2'
        tex_name   = '$m_2$'
    
    mean_err_rec   = np.mean(np.abs(err_rec))
    mean_err_pred  = np.mean(np.abs(err_pred))
    mean_diff_rec  = np.mean(np.abs(diffs_rec))
    mean_diff_pred = np.mean(np.abs(diffs_pred))
    std_rec        = np.std(err_rec)
    std_pred       = np.std(err_pred)

    if data.tab_format=='txt':
        line_format = '{:14s} {:15.3e} {:15.3e} {:15.3e}     {:15.3e} {:15.3e} {:15.3e}'
        myline = line_format.format(var_name, mean_diff_rec,  mean_err_rec,  std_rec, 
                                              mean_diff_pred, mean_err_pred, std_pred) 
        print(dashes,myline,dashes,sep='\n')
    elif data.tab_format=='tex':
        line_format = escapeLatex('{:14s} & {:s} & {:s} & {:s} & {:s} & {:s} & {:s} \\\\')
        myline = line_format.format(tex_name, num2tex(mean_diff_rec),  num2tex(mean_err_rec),  num2tex(std_rec), 
                                              num2tex(mean_diff_pred), num2tex(mean_err_pred), num2tex(std_pred)) 
        print('\hline',myline,dashes,sep='\n')


    print(' \n+++ Warning +++: for spin-variables the std is computed on the difference ' +
          'distribution [e.g. abs(inj-rec)],\nwhile for mass-variables on the error distributions'+
          ' [e.g. abs(inj-rec)/inj]\n')
    
    return

###################################################
# Main 
###################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='paper_plots', description='plots to use in the regression paper')
    parser.add_argument('--NN', dest='use_NN_data', action='store_true',
                        help="use NN-data (path and filename hardcoded).")
    parser.add_argument('--GPR', dest='use_GPR_data', action='store_true',
                        help="use GPR-data (path and filename hardcoded).")
    parser.add_argument('-p', '--plots', dest='plots2do', nargs='+',default=[],
                        help='identifiers of the plots to do, e.g. >> -p rec_vs_pred')
    parser.add_argument('--vars', type=str, dest='regr_vars', default='m1m2chi1chi2', 
                        help="variables used in the regression. Can be 'm1m2chi1chi2' or 'm1Mchi1chi2'")
    parser.add_argument('--dataset_path', type=str, dest='dataset_path', default=repo_path+'datasets/GstLAL/', 
                        help="path where there are the O2 injections (test_NS.csv)")
    parser.add_argument('-s', '--save',  dest='savepng', action='store_true', 
                        help="save plots in PNG format")
    parser.add_argument('--plots_dir', type=str, dest='plots_dir', default=os.getcwd(),
                        help="directory where to save plots (default is current dir)")
    parser.add_argument('-v', '--verbose',  dest='verbose', action='store_true', 
                        help="Print stuff")
    
    parser.add_argument('--histo_fmin', dest='histo_fmins', default=[-3,-3,-2,-2], nargs=4, type=int, 
                        help="fmin used in histograms, one float for each feature, e.g. --histo_fmin -1 -2 -3 -4 ")
    parser.add_argument('--histo_fmax', dest='histo_fmaxs', default=[2,2,2,2], nargs=4, type=int, 
                        help="fmax used in histograms, one float for each feature, e.g. --histo_fmax 1 2 3 4 ")
    parser.add_argument('--histo_nbins', dest='histo_nbins', default=[50,50,50,50], nargs=4, type=int, 
                        help="nbins used in histograms, one float for each feature, e.g. --histo_nbins 50 30 40 10 ")
    parser.add_argument('--histo_logs', dest='histo_logs', default=[0,0,0,0], nargs=4, type=int, 
                        help="if i-element is 1, use logscale in i-subplot, e.g. --histo_logs 1 1 0 0 ")
    
    parser.add_argument('--errortab', dest='errortab', action='store_true', 
                        help="print a table with mean errors/differences")
    parser.add_argument('--tab_format', dest='tab_format', type=str, default='txt', 
                        help="format of printed tables, 'txt' (default) or 'tex'")
    
    parser.add_argument('--parspace_colorful', dest='parspace_colorful', action='store_true', 
                        help='Use different colors in parspace plot fr BBH, BNS, BHNS')

    args = parser.parse_args()
    verbose = args.verbose
    
    # load injected and recovered  
    X = extract_data(args.dataset_path+'/test_NS.csv', skip_header=True, verbose=verbose)
    if args.regr_vars=='m1Mcchi1chi2':
        splitted_data = split_GstLAL_data(X, features='mass&spin')
        var_names     = ['m1', 'Mc', 'chi1', 'chi2']
        var_names_tex = ['$m_1$', '${\cal{M}}_c$', '$\chi_1$', '$\chi_2$']
        #var_idx         = {}
        #var_idx['m1']   = 0
        #var_idx['m2']   = None
        #var_idx['Mc']   = 1
        #var_idx['chi1'] = 2
        #var_idx['chi2'] = 3
         
    elif args.regr_vars=='m1m2chi1chi2':
        splitted_data = split_GstLAL_data(X, features='m1m2chi1chi2')
        var_names     = ['m1', 'm2', 'chi1', 'chi2']
        var_names_tex = ['$m_1$', '$m_2$', '$\chi_1$', '$\chi_2$']
        #var_idx         = {}
        #var_idx['m1']   = 0
        #var_idx['m2']   = 1
        #var_idx['Mc']   = None
        #var_idx['chi1'] = 2
        #var_idx['chi2'] = 3

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
    data.regr_vars     = args.regr_vars
    data.inj           = inj
    data.rec           = rec
    data.pred          = pred
    data.var_names     = var_names
    data.var_names_tex = var_names_tex
    #data.var_idx       = var_idx
    data.savepng       = args.savepng
    data.plots_dir     = args.plots_dir
    data.plots_prefix  = plots_prefix
    data.verbose       = verbose
    data.histo_fmins   = args.histo_fmins
    data.histo_fmaxs   = args.histo_fmaxs
    data.histo_nbins   = args.histo_nbins
    data.histo_logs    = args.histo_logs
    data.errortab      = args.errortab
    data.tab_format    = args.tab_format
    data.parspace_colorful = args.parspace_colorful

    data.stats = {}
    for i in range(NFEATURES):
        data.stats['diffs_rec']   =  inj-rec
        data.stats['diffs_pred']  =  inj-pred
        with np.errstate(divide='ignore'):
            data.stats['errors_rec']  = (inj-rec )/inj 
            data.stats['errors_pred'] = (inj-pred)/inj 
    
    if args.errortab:
        print_errortab(data)

    for plot_id in args.plots2do:
        if plot_id=='rec_vs_pred':
            plot_recovered_vs_predicted(data)
        elif plot_id=='parspace':
            plot_parspace(data)
        elif plot_id=='histo':
            plot_histograms(data)
        else:
            print('Unknown plot: '+plot_id)

