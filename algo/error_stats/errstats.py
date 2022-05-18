"""
Accurate description:
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class ErrorStats:
    """ x and y are 1D arrays.
    y are the true values, while x are the guesses 
    """
    def __init__(self, x, y, n=3000, project=False, sigma=None):
        self.x       = np.array(x)
        self.y       = np.array(y)
        self.Nx      = len(x)
        self.n       = n
        self.sigma   = sigma
        self.project = project
        bins_dict      = self.__create_bins_fixed_size(n=n, project=project)
        bins_dict      = self.__remove_outliers(bins_dict, sigma=sigma)
        self.bins_dict = bins_dict
        return

    def __bins_stats(self, xbins, ybins):
        nbins = len(xbins)
        xmid  = np.empty((nbins,))
        xmin  = np.empty((nbins,))
        xmax  = np.empty((nbins,))
        mean  = np.empty((nbins,)) # stats on ybins
        var   = np.empty((nbins,))
        skew  = np.empty((nbins,))
        kurt  = np.empty((nbins,)) # it's not Cobain (also because he wasn't a numpy array)
        for i in range(nbins):
            xbin    = xbins[i]
            ybin    = ybins[i]
            xmin[i] = min(xbin)
            xmax[i] = max(xbin)
            xmid[i] = (xmax[i]+xmin[i])/2
            mean[i] = stats.tmean(ybin)
            var[i]  = stats.tvar(ybin)
            skew[i] = stats.skew(ybin)
            kurt[i] = stats.kurtosis(ybin)
        bins_dict          = {}
        bins_dict['xbins'] = xbins
        bins_dict['ybins'] = ybins
        bins_dict['mean']  = mean
        bins_dict['var']   = var
        bins_dict['std']   = np.sqrt(var)
        bins_dict['skew']  = skew
        bins_dict['kurt']  = kurt
        bins_dict['xmid']  = xmid
        bins_dict['xmin']  = xmin
        bins_dict['xmax']  = xmax
        return bins_dict

    def __create_bins_fixed_size(self, n=3000, project=False):
        """ Create bins with same number of points. However, if 
        two adiacent bins have the same x-middle value, then the
        bins are merged
        """
        self.project = project
        xmid_tol     = 1e-5
        x = self.x
        y = self.y
        # sort x-array so that we have increasing values
        # y is sorted accordingly
        xs, ys = zip(*sorted(zip(x,y))) 
        Nx          = self.Nx
        nbins_guess = int(np.ceil(Nx/n))
        xbins = []
        ybins = []
        xmid  = []
        for i in range(nbins_guess):
            new_xbin  = np.array(xs[ i*n : (i+1)*n ])
            new_ybin  = np.array(ys[ i*n : (i+1)*n ])
            new_xmid  = (new_xbin[-1]+new_xbin[0])/2
            if project:
                new_ybin = new_ybin + new_xmid - new_xbin
                new_xbin = new_xmid + new_xbin*0 # *0 is needed for the broadcast
            if i>0 and np.abs(new_xmid-xmid[-1])<xmid_tol:
                # if xmid is the same of the previous bin, then merge the two bins
                old_xbin  = xbins[-1]
                old_ybin  = ybins[-1]
                new_xbin  = np.concatenate((old_xbin, new_xbin), axis=0)
                new_ybin  = np.concatenate((old_ybin, new_ybin), axis=0)
                xbins[-1] = new_xbin
                ybins[-1] = new_ybin
            else:
                xbins.append(new_xbin)
                ybins.append(new_ybin)
                xmid.append(new_xmid)
        bins_dict = self.__bins_stats(xbins, ybins)
        return bins_dict

    def __remove_outliers(self, bins_dict, sigma=None):
        xbins = bins_dict['xbins']
        ybins = bins_dict['ybins']
        if sigma is not None:
            nbins = len(xbins)
            mean  = bins_dict['mean']
            std   = bins_dict['std']
            new_xbins = []
            new_ybins = []
            for i in range(nbins):
                xbin = xbins[i]
                ybin = ybins[i]
                below_sigma = np.argwhere(ybin <= mean[i]-sigma*std[i])
                ybin = np.delete(ybin, below_sigma)
                xbin = np.delete(xbin, below_sigma)
                above_sigma = np.argwhere(ybin >= mean[i]+sigma*std[i])
                ybin = np.delete(ybin, above_sigma)
                xbin = np.delete(xbin, above_sigma)
                new_xbins.append(xbin)
                new_ybins.append(ybin)
            bins_dict = self.__bins_stats(new_xbins, new_ybins)
        return bins_dict
    
    def plot_bins(self):
        x = self.x
        bins_dict = self.bins_dict
        xbins = bins_dict['xbins']
        ybins = bins_dict['ybins']
        mean  = bins_dict['mean']
        xmax  = bins_dict['xmax']
        xmin  = bins_dict['xmin']
        xmid  = bins_dict['xmid']
        std   = bins_dict['std']
        nbins = len(xbins)
        if np.all(x>0):
            ncols = 2
            plt.figure(figsize=(8, 4))
        else:
            ncols = 1
            plt.figure(figsize=(5, 3))
        ax1 = plt.subplot(1,ncols,1)
        for i in range(nbins):
            ax1.scatter(xbins[i], ybins[i], s=1)
        ax1.plot(xmid, mean, '-ko')
        ax1.plot(xmid, mean - std, c=[1,0,0])
        ax1.plot(xmid, mean + std, c=[1,0,0])
        ax1.set_xlabel('guess', fontsize=15)
        ax1.set_ylabel('true', fontsize=15)
        if np.all(x>0):
            ax2 = plt.subplot(1,ncols,2)
            for i in range(nbins):
                ax2.scatter(xbins[i], ybins[i], s=1)
            ax2.plot(xmid, mean, '-ko')
            ax2.plot(xmid, mean - std, c=[1,0,0])
            ax2.plot(xmid, mean + std, c=[1,0,0])
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.grid(visible=True)
            ax2.set_xlabel('guess', fontsize=15)
            ax2.set_ylabel('true', fontsize=15)
        plt.tight_layout()
        plt.show()
        return 

    def plot_skewness(self):
        xmid = self.bins_dict['xmid']
        skew = self.bins_dict['skew']
        plt.figure
        plt.plot(xmid, skew, 'r', label='skew')
        plt.xlabel(r'$\bar{x}$', fontsize=15)
        plt.ylabel(r'$\gamma_1$', fontsize=15)
        plt.legend()
        plt.show()
        return 
    
    def plot_xstep(self):
        if not self.project:
            xmid = self.bins_dict['xmid']
            xmin = self.bins_dict['xmin']
            xmax = self.bins_dict['xmax']
            plt.figure 
            plt.plot(xmid, xmax-xmin)
            plt.xlabel(r'$\bar{x}$' , fontsize=18)
            plt.ylabel(r'$\Delta x$', fontsize=18)
            plt.yscale('log')
            plt.grid(visible=True)
            plt.show()
        else:
            print('self.plot_xstep: if project is True, then xmax-xmin=0 by construction!')
        return

    def plot_stats(self, bins_hist = 30, show_info=True, plot_xbins=False):
        bins_dict = self.bins_dict
        xbins = bins_dict['xbins']
        ybins = bins_dict['ybins']
        mean  = bins_dict['mean']
        xmax  = bins_dict['xmax']
        xmin  = bins_dict['xmin']
        xmid  = bins_dict['xmid']
        std   = bins_dict['std']
        i = 0
        subplot_cols = 8
        while i<len(ybins):
            j = 0
            if plot_xbins:
                plt.figure(figsize=(3*subplot_cols, 6))
            elif show_info:
                plt.figure(figsize=(3*subplot_cols, 4.5))
            else:
                plt.figure(figsize=(3*subplot_cols, 3))
            for j in range(subplot_cols):
                if i<len(ybins):
                    fmin = min(ybins[i])-mean[i]
                    fmax = max(ybins[i])-mean[i]
                    fstep = (fmax-fmin)/bins_hist
                    if plot_xbins:
                        ax1 = plt.subplot(2,subplot_cols,j+1)
                    else:
                        ax1 = plt.subplot(1,subplot_cols,j+1)
                    ax1.hist(ybins[i]-mean[i], bins=bins_hist, range=(fmin,fmax), ec='black')
                    if show_info:
                        ax1.set_title(r"$N$: {:d}".format(len(xbins[i])) + "\n" +
                                      r"$x \in ({:.2f},{:.2f})$".format(xmin[i], xmax[i]) + "\n" +
                                      r"$\bar{{x}}$: {:.2f}".format(xmid[i])  + "\n" +
                                      r"$\hat{{y}}$: {:.2f}".format(mean[i]) + "\n" +
                                      r"$\sigma_{{y}}$: {:.3f}".format(std[i]), fontsize=21)
                    if j==0:
                        ax1.set_ylabel('y', fontsize=20)
                    if plot_xbins:
                        ax2 = plt.subplot(2,subplot_cols,subplot_cols+j+1)
                        ax2.hist(xbins[i], bins=bins_hist, ec='black', color=[1,0.5,0])
                        if j==0:
                            ax2.set_ylabel('x', fontsize=20)
                else:
                    if plot_xbins:
                        ax1 = plt.subplot(2,subplot_cols,j+1)
                        ax2 = plt.subplot(2,subplot_cols,subplot_cols+j+1)
                    else:
                        ax1 = plt.subplot(1,subplot_cols,j+1)
                i += 1
                j += 1
            plt.tight_layout()
            plt.show()
            print('-'*114)
        return
        
