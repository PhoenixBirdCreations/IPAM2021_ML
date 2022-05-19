"""
Accurate description:
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skewnorm

class ErrorStats:
    """ x and y are 1D arrays.
    y are the true values, while x are the guesses 
    """
    def __init__(self, x, y, n=3000, project=False, shift_extrema=False, sigma=None):
        self.x       = np.array(x)
        self.y       = np.array(y)
        self.Nx      = len(x)
        self.n       = n
        self.sigma   = sigma
        self.project = project
        bins_dict      = self.__create_bins_fixed_size(n=n, project=project, shift_extrema=shift_extrema)
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

    def __create_bins_fixed_size(self, n=3000, project=False, shift_extrema=False):
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
                if shift_extrema and i==0:
                    new_xmid = min(new_xbin)
                elif shift_extrema and i==nbins_guess-1:
                    new_xmid = max(new_xbin)
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
    
    def distr_moments(self, x):
        moments = {}
        moments["mean"] = stats.tmean(x)
        moments["var"]  = stats.tvar(x)
        moments["skew"] = stats.skew(x)
        return moments

    def moments_to_pars(self, moments):
        """ Recover location, scale and shape (alpha)
        from mean, variance and skewness
        """
        mean = moments["mean"]
        var  = moments["var"]
        skew = moments["skew"]

        skew_tol = 1e-5
        if skew>1:
            skew = 1-skew_tol
            #print('Warning: skew>1! Using skew =', skew)
        if skew<-1:
            skew = -1+skew_tol
            #print('Warning: skew<-1! Using skew =', skew)
        pi = np.pi
        if skew>=0:
            beta = ((2*skew)/(4-pi))**(1/3)
        else:
            beta = -((2*np.abs(skew))/(4-pi))**(1/3)
        delta  = np.sqrt(pi/2)*beta/np.sqrt(1+beta*beta)
        delta2 = delta*delta
        pars   = {}
        if 1-delta2<0:
            delta2 = 1-1e-10
        shape = delta/np.sqrt(1-delta2) 
        scale = np.sqrt(var/(1-2*delta2/pi))
        loc   = mean-scale*delta*np.sqrt(2/pi)
        pars  = {"loc":loc, "scale":scale, "shape":shape}
        return pars

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
            plt.figure(figsize=(10, 4))
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

    def plot_stats(self, bins_hist = 30, show_info=True, plot_xbins=False, plot_distr=True, show_gauss=False):
        bins_dict = self.bins_dict
        xbins = bins_dict['xbins']
        ybins = bins_dict['ybins']
        mean  = bins_dict['mean']
        xmax  = bins_dict['xmax']
        xmin  = bins_dict['xmin']
        xmid  = bins_dict['xmid']
        std   = bins_dict['std']
        var   = bins_dict['var']
        skew  = bins_dict['skew']
        if plot_distr:
            density_hist = True
        else: 
            density_hist = False
        i = 0
        subplot_cols = 6
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
                    #ydistr = ybins[i]-mean[i]
                    ydistr = ybins[i]
                    fmin = min(ydistr)
                    fmax = max(ydistr)
                    fstep = (fmax-fmin)/bins_hist
                    if plot_xbins:
                        ax1 = plt.subplot(2,subplot_cols,j+1)
                    else:
                        ax1 = plt.subplot(1,subplot_cols,j+1)
                    ax1.hist(ydistr, bins=bins_hist, range=(fmin,fmax), ec=[0,.2,1], density=density_hist, alpha=0.5, color=[0,.2,0.8])
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
                    if plot_distr:
                        moments = self.distr_moments(ydistr)
                        pars    = self.moments_to_pars(moments)
                        rv      = skewnorm(a=pars["shape"], loc=pars["loc"], scale=pars["scale"])
                        #x_rv = np.linspace(fmin, fmax, 1000)
                        x_rv  = np.linspace(min(fmin, rv.ppf(0.001)), max(fmax, rv.ppf(0.999))) 
                        if show_gauss:
                            gauss = skewnorm(a=0, loc=moments["mean"], scale=np.sqrt(moments["var"]))
                            x_rv  = np.linspace(min(fmin, gauss.ppf(0.001)), max(fmax, gauss.ppf(0.999))) 
                            ax1.plot(x_rv, gauss.pdf(x_rv), c=[0.9,0,0], lw=3, label='Gauss')
                        ax1.plot(x_rv, rv.pdf(x_rv), c=[0,0,1], lw=3, label='PDF') # plot recovered distribution
                        ax1.legend()
                        #ax1.set_xlim([fmin,fmax])
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
        