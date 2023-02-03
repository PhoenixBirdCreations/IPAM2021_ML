#!/usr/bin/env python
# S. Albanesi
# Part of the ML project at IPAM 2021

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate

class ErrorSurface:
    """ ErrorSurface class
    
    During initialization we create a density surface in the 
    (X,Y) plane, where X are predicted/recovery data
    and Y are the true values. 

    Given the surface and point x0, we can consider a slice, 
    compute the fx0(y) distribution (method distribution) and 
    then the confidence interval (method confidence_interval)
    
    Since fx0(y) is discrete, the confidence interval obtained
    does not always correspond exactly to the requested one 
    (i.e. if we ask for 0.90 confidence, maybe we get something
    around ~0.88). To overcome this problem, we add the
    possibility to spline fx0(y).

    The data can be plotted using the following methods 
    plot_surf, plot_interp, plot_inst. 

    See algo/classy_NN/GstLAL.ipynb for a working example of 
    this class.
    """

    def __init__(self, X, Y, Nx=50, Ny=50, 
                 exp_step=False, dx_expstep=1.05, dy_expstep=1.05, 
                 method='linear', Nx_igrid=200, Ny_igrid=200):
        """ Initialize the class to compute the error-surface and
        then the confidence intervals.
        
        The initialization works in this way: 
          1) Create a grid in the (X,Y) plane according to input. 
             If the exp_step is False, the grid is uniform with
             spacing (max(X)-min(X)/Nx and (max(Y)-min(Y))/Ny.
             If exp_step is True, then the grid is created 
             according to {dx,dy}_expstep and it is not uniform.
         
          2) We count the number of (X,Y) points in each cell,
             so that we have a discrete density distribution.

          3) The original discrete density distribution is then
             interpolated on a more refined grid. The new 
             interpolated grid is always uniform with spancing 
             (max(X)-min(X))/Nx_igrid and (max(Y)-min(Y))/Ny_igrid.


        Parameters
        ----------
        X : 1d-array
            Predicted or recovered data for a specific feature
        
        Y : 1d-array
            True values that correspond to X
        
        Nx : int
            Number of points used in the x-direction for the non-interpolated surface. 
            If exp_step=True, the value in unput is ignored and Nx is then obtained from
            the non-uniform grid
        
        Ny : int
            as Nx, but in the y-direction
        
        exp_step : bool
            Use exponential-step instead of uniform-step. Can be used only if X[i]>=0 and Y[i]>=0 for every i
        
        dx_expstep : float
            exp_step in the x-direction for the non-interpolated surface (must be >1)
        
        dy_expstep : float
            as dx_expstep, but in the y-direction
        
        method : str
            Method used for the surface-interpolation. Can be 'nearest', 'linear' or 'cubic'
        
        Nx_igrid : int
            number of points used in the x-direction for the intepolated grid
        
        Ny_igrid : int
            as Nx_igrid, but in the y-direction
        """
        self.X     = X
        self.Y     = Y
        self.X_min = min(X)
        self.X_max = max(X)
        self.Y_min = min(Y)
        self.Y_max = max(Y)
        
        # build non-interpolated grid
        if exp_step:
            if np.any(X<0):
                raise ValueError('exp_step=True can be used only if X[i]>=0 for every i')
            step_dict = self.__exp_step(dx_expstep,dy_expstep)
        else:
            step_dict = self.__uniform_step(Nx,Ny)
        
        self.step_dict = step_dict 
        self.Nx      = step_dict['Nx']
        self.Ny      = step_dict['Ny']
        self.xg0     = step_dict['xg0']
        self.yg0     = step_dict['yg0']
        self.S0      = step_dict['S0']
        self.x_mid   = step_dict['x_mid']
        self.y_mid   = step_dict['y_mid']
        self.x_edges = step_dict['x_edges']
        self.y_edges = step_dict['y_edges']
        
        # interpolation
        self.method  = method
        self.Nx_igrid = Nx_igrid
        self.Ny_igrid = Ny_igrid
        self.__interpolate_surface()
        
        return


    def __uniform_step(self, Nx, Ny):
        """ 
        Create the uniform grid and the corresponding 
        density surface (not interpolated). Used during
        initialization.
        """
        X     = self.X
        Y     = self.Y
        X_min = self.X_min
        X_max = self.X_max
        Y_min = self.Y_min
        Y_max = self.Y_max
        dx = (X_max-X_min)/Nx
        dy = (Y_max-Y_min)/Ny
        # compute middle points and edges
        x_mid   = np.linspace(X_min+dx/2, X_max-dx/2, Nx-1)
        y_mid   = np.linspace(Y_min+dy/2, Y_max-dy/2, Ny-1)
        x_edges = np.linspace(X_min, X_max, Nx)
        # compute uniform non-interpolated surface
        xg0, yg0 = self.__tmesh(x_mid, y_mid)
        S0 = np.empty((Nx-1, Ny-1))
        for i in range(0,Nx-1):
            for j in range(0,Ny-1):
                x1 = X_min + dx*i
                y1 = Y_min + dy*j
                x2 = x1 + dx
                y2 = y1 + dy
                mask_x = np.argwhere((X>=x1) & (X<x2))
                mask_y = np.argwhere((Y>=y1) & (Y<y2))
                mask   = np.intersect1d(mask_x, mask_y)
                S0[i,j] = len(mask)
        y_edges = np.linspace(Y_min, Y_max, Ny)
        # save stuff in the dictionary
        step_dict            = {}
        step_dict['xg0']     = xg0
        step_dict['yg0']     = yg0
        step_dict['S0']      = S0
        step_dict['x_mid']   = x_mid
        step_dict['y_mid']   = y_mid
        step_dict['x_edges'] = x_edges
        step_dict['y_edges'] = y_edges
        step_dict['Nx']      = Nx
        step_dict['Ny']      = Ny
        return step_dict

    def __exp_step(self, dx_expstep, dy_expstep):
        """
        Create the non-uniform grid and the corresponding 
        density surface (not interpolated). Used during
        initialization.
        """
        X       = self.X
        Y       = self.Y
        X_min   = self.X_min
        X_max   = self.X_max
        Y_min   = self.Y_min
        Y_max   = self.Y_max
        # compute x_edges and x_edges
        x_edges = [X_min]
        while x_edges[-1]<X_max:
            next_x = x_edges[-1]*dx_expstep
            x_edges.append(next_x)
        y_edges = [Y_min]
        while y_edges[-1]<Y_max:
            next_y = y_edges[-1]*dy_expstep
            y_edges.append(next_y)
        x_edges = np.array(x_edges)
        y_edges = np.array(y_edges)
        Nx = len(x_edges)
        Ny = len(y_edges)
        # compute middle points (to optimize...)
        x_mid = np.empty((Nx-1,))
        y_mid = np.empty((Ny-1,))
        for i in range(Nx-1):
            x_mid[i] = (x_edges[i]+x_edges[i+1])/2
        for i in range(Ny-1):
            y_mid[i] = (y_edges[i]+y_edges[i+1])/2
        # compute the non-interpolated density-surface
        xg0, yg0 = self.__tmesh(x_mid, y_mid)
        S0 = np.empty((Nx-1, Ny-1))
        for i in range(0,Nx-1):
            for j in range(0,Ny-1):
                x1 = x_edges[ i ]
                x2 = x_edges[i+1]
                y1 = y_edges[ j ]
                y2 = y_edges[j+1]
                mask_x = np.argwhere((X>=x1) & (X<x2))
                mask_y = np.argwhere((Y>=y1) & (Y<y2))
                mask   = np.intersect1d(mask_x, mask_y)
                S0[i,j] = len(mask)
        # save stuff in the dictionary
        step_dict            = {}
        step_dict['xg0']     = xg0
        step_dict['yg0']     = yg0
        step_dict['S0']      = S0
        step_dict['x_mid']   = x_mid
        step_dict['y_mid']   = y_mid
        step_dict['x_edges'] = x_edges
        step_dict['y_edges'] = y_edges
        step_dict['Nx']      = Nx
        step_dict['Ny']      = Ny
        return step_dict
    
    def __tmesh(self,x,y):
        xg, yg = np.meshgrid(x, y)
        xg = np.transpose(xg)
        yg = np.transpose(yg)
        return xg, yg
    
    def __interpolate_surface(self):
        """ 
        Interpolate the surface previously
        found to a new uniform grid.
        Used during initialization
        """
        Nx_igrid = self.Nx_igrid
        Ny_igrid = self.Ny_igrid
        method  = self.method
        X_min   = self.X_min
        X_max   = self.X_max
        Y_min   = self.Y_min
        Y_max   = self.Y_max
        Nx      = self.Nx
        Ny      = self.Ny
        S0      = self.S0
        xg0     = self.xg0
        yg0     = self.yg0
        # compute grid to use for interpolation
        x_interp = np.linspace(X_min, X_max, Nx_igrid)
        y_interp = np.linspace(Y_min, Y_max, Ny_igrid)
        xg_interp, yg_interp = self.__tmesh(x_interp, y_interp)
        points = np.empty(((Nx-1)*(Ny-1), 2))
        values = np.empty(((Nx-1)*(Ny-1),))
        k = 0
        for i in range(Nx-1):
            for j in range(Ny-1):
                points[k,0] = xg0[i,j]
                points[k,1] = yg0[i,j]
                values[k]   = S0[i,j]
                k += 1
        S_interp = interpolate.griddata(points, values, (xg_interp, yg_interp), method=self.method)
        S_interp = np.around(S_interp, decimals=0)
        S_interp = np.nan_to_num(S_interp, copy=True, nan=0)      
        self.x_interp   = x_interp
        self.y_interp   = y_interp
        self.xg_interp  = xg_interp
        self.yg_interp  = yg_interp
        self.S_interp   = S_interp
        return
    
    def distribution(self, x0, verbose=False):
        """ 
        Given a certain x0, compute the corresponding
        distribution fx0(y) using the interpolated
        surface. Used for the computation of the confidence
        interval.

        Parameters
        ----------
          x0      : float
          verbose : bool

        Return y-points of the fx0(y) distribution
        """
        x_interp = self.x_interp
        y_interp = self.y_interp
        S_interp = self.S_interp
        i = 0
        while x0>=x_interp[i]:
            i+=1
        y_N1 = S_interp[i-1,:]
        y_N2 = S_interp[ i ,:]
        dx   = x_interp[i] - x_interp[i-1]
        w1   = (x0-x_interp[i-1])/dx
        w2   = (x_interp[i]-x0)/dx
        y_N  = w1*y_N1 + w2*y_N2
        if verbose:
            print('x1, x2   : {:.1f}, {:.1f}'.format(x_interp[i-1], x_interp[i]))
            print('n1 events:', sum(y_N1))
            print('n2 events:', sum(y_N2))
            print('n. events:', sum(y_N))
        N = len(y_N)
        y_values = np.array([])
        for j in range(N):
            nj = round(y_N[j])
            if nj>0:
                ones = np.ones((nj,))
                y_values = np.concatenate((y_values, ones*y_interp[j]))
        return y_values
    
    def confidence_interval(self, x0, cfi=0.9, verbose=False, nbins=50, spline=False, spline_sample=1000, spline_plot=False):
        """
        Compute the confidence interval [xl,xr] for x0 
        according to the confidence requested (cfi).
        Note that since the distribution fx0(y) used 
        is discrete, so the final probabily to have
        an event in [xl,xr] could be different to the
        requested confidence (e.g. if cfi=0.90,
        the probability to have an event in [xl,xr]
        could be P(x\in[xl,xr]=0.88). 
        To overcome this problem, it is possible to use
        a spline on the distribution (spline switched
        off by default). This reduce the difference
        between cfi and P(x0\in[xl,xr])
        
        Parameters
        ---------
          x0 : float
            Point in which we compute the y-distribution

          cfi : float
            Nominal confidence

          verbose : bool
            Print some info

          nbins : int
            Number of bins to use on the y-points 
            given in output by self.distribution()
          
          spline : bool
            Use a spline algorithm on the histogram
            computed from the y-points of self.distribution()
          
          spline_sample : int
            Sampling for the spline algo

          spline_plot : bool
            Plot the original histogram and 
            the splined one. Useful to check the
            spline procedure (not always optimal)

         Return xl, xr and P(x\in[xl,xr])
        """
        y  = self.distribution(x0,verbose=False)
        hist, bin_edges = np.histogram(y,bins=nbins)
        density = hist/hist.sum()
        tail = (1-cfi)/2
        suml = 0
        sumr = 0
        xl   = None
        xr   = None

        if spline:
            x = (bin_edges[0:-1]+bin_edges[1:])/2
            y = density
            tck = interpolate.splrep(x, y)
            edges_fine = np.linspace(bin_edges[0], bin_edges[-1], spline_sample)
            x_fine     = (edges_fine[0:-1]+edges_fine[1:])/2
            new_hist    = interpolate.splev(x_fine, tck)
            new_bin_edges  = edges_fine
            mask = np.argwhere(new_hist<0)
            new_hist[mask] = 0
            if spline_plot:
                plt.figure
                plt.bar(x, density, width=bin_edges[2]-bin_edges[1], ec='black')
                plt.bar(x_fine, new_hist, width=new_bin_edges[2]-new_bin_edges[1], alpha=0.4)
                plt.show()
            density   = new_hist/new_hist.sum()
            bin_edges = new_bin_edges
            nbins     = len(bin_edges)-1

        for i in range(0,nbins):
            if xl is None:
                suml += density[i]
                if suml>=tail:
                    il = i+1
                    xl = bin_edges[il]
            if xr is None:
                j = nbins-1-i
                sumr += density[j]
                if xr is None and sumr>=tail:
                    ir = j-1
                    xr = bin_edges[ir]
            if xl is not None and xr is not None:
                break
        if xl is None or xr is None:
            print(suml)
            print(sumr)
            raise ValueError('confidence interval not found!')
        cfi_final = np.sum(density[il:ir+1])
        if verbose:
            ccfi = np.sum(density[0:il])+np.sum(density[ir+1:])
            print('-'*100)
            print('cfi requested                  : {:f}'.format(cfi))
            print('number of bins                 :', nbins)
            print('left-idf, right-idx            : {:d}, {:d}'.format(il, ir))
            print('left-tail prob, left-tail prob : {:.5f}, {:.5f}'.format(suml, sumr))
            print('final cfi (diff from initial)  : {:.5f} ({:f} %)'.format(cfi_final, 100*(cfi_final-cfi)/cfi))
            print('sum of final cfi and compl-cfi : {:.10f}'.format(cfi_final+ccfi))
            print('-'*100)
        return xl, xr, cfi_final
    
    #---------------------------------------------
    # Plots
    #---------------------------------------------
    def plot_surf(self, show_grid=True, log_bar=False, log_scale=False, bisectrix=True):
        """
        Plot the original density surface
        (not interpolated)
        """
        x_mid   = self.x_mid
        y_mid   = self.y_mid
        x_edges = self.x_edges
        y_edges = self.y_edges
        S0      = self.S0
        X_min   = self.X_min
        X_max   = self.X_max
        Y_min   = self.Y_min
        Y_max   = self.Y_max
        Nx      = self.Nx
        Ny      = self.Ny
        xg0     = self.xg0
        yg0     = self.yg0
        if log_bar:
            np.seterr(divide='ignore', invalid='ignore') 
            S0_plot = np.log10(S0)
            min_S = max(S0_plot.min(), 0)
            lev_decimals = 3
            #for i in range(0,Nx-1):
            #    for j in range(0, Ny-1):
            #        if S0_plot[i,j]<0:
            #                S0_plot[i,j]=0
        else:
            S0_plot = S0
            min_S = S0_plot.min()
            lev_decimals = 0
        levels = np.around(np.linspace(min_S, S0_plot.max(), 30), decimals=lev_decimals)
        fig,ax=plt.subplots(1,1, figsize=(10,6))
        cp = ax.contourf(xg0, yg0, S0_plot, cmap=plt.get_cmap('viridis'), levels=levels)
        cb = fig.colorbar(cp)
        if show_grid:
            for i in range(Nx):
                plt.axvline(x_edges[i], lw=0.5, color=[0,0,0])
            for j in range(Ny):
                plt.axhline(y_edges[j], lw=0.5, color=[0,0,0])    
        ax.set_ylim([Y_min,Y_max])
        ax.set_xlim([X_min,X_max])
        if log_scale:
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(r'$log_{10}(x)$', fontsize=25)
            ax.set_ylabel(r'$log_{10}(y)$', fontsize=25)
        else:
            ax.set_xlabel(r'$x$', fontsize=25)
            ax.set_ylabel(r'$y$', fontsize=25)
        if log_bar:
            cb.set_label(r'$log_{10}(N_s)$', fontsize=25)
        else:
            cb.set_label(r'$N_s$', fontsize=25)
        if bisectrix:
            ax.plot(x_mid,x_mid,c='r',lw=1)
        plt.show()
        return

    def plot_interp(self, x0_line=None, log_bar=False, log_scale=False):
        """
        Plot the interpolated surface  
        """
        Nx_igrid = self.Nx_igrid
        Ny_igrid = self.Ny_igrid
        xg_interp = self.xg_interp
        yg_interp = self.yg_interp
        X_min   = self.X_min
        X_max   = self.X_max
        Y_min   = self.Y_min
        Y_max   = self.Y_max
        S_interp = self.S_interp
        if log_bar:
            np.seterr(divide='ignore', invalid='ignore') 
            S_interp_plot = np.log10(S_interp)
            min_S = max(S_interp_plot.min(), 0)
            lev_decimals = 3
        else:
            S_interp_plot = S_interp
            min_S = S_interp_plot.min() 
            lev_decimals = 0
        levels = np.around(np.linspace(min_S, S_interp_plot.max(), 30), decimals=lev_decimals)
        #S_interp_plot = np.empty(np.shape(S_interp))
        #for i in range(Nx_igrid):
        #    for j in range(Ny_igrid):
        #        if np.isnan(S_interp[i,j]):
        #            S_interp[i,j] = 0;
        #        if S_interp[i,j]<1:
        #            S_interp_plot[i,j] = 0
        #        else:
        #            S_interp_plot[i,j] = np.log10(S_interp[i,j])
        fig,ax=plt.subplots(1,1, figsize=(10,6))
        cp = ax.contourf(xg_interp, yg_interp, S_interp_plot, cmap=plt.get_cmap('plasma'), levels=levels)
        cb = fig.colorbar(cp)
        ax.set_ylim([Y_min,Y_max])
        ax.set_xlim([X_min,X_max])
        ax.plot(self.x_interp, self.x_interp,c='r',lw=1)
        if x0_line is not None:
            ax.axvline(x0_line, color=[0,1,0])
        if log_scale:
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(r'$log_{10}(x)$', fontsize=25)
            ax.set_ylabel(r'$log_{10}(y)$', fontsize=25)
        else:
            ax.set_xlabel(r'$x$', fontsize=25)
            ax.set_ylabel(r'$y$', fontsize=25)
        if log_bar:
            cb.set_label(r'$log_{10}(N_s)$', fontsize=25)    
        else:
            cb.set_label(r'$N_s$', fontsize=25)    
        plt.show()
        return

    def plot_hist(self, x0, nbins=50, axvlines=None):
        """
        Plot the y-values of a fx0(y) distribution
        found with the interpolated surface
        """
        X_min   = self.X_min
        X_max   = self.X_max
        if x0<X_min or x0>X_max:
            print('x={:f} is outside the prediction range [{:6f},{:6f}]'.format(x0, X_min, X_max))
            return
        y_values = self.distribution(x0)
        if len(y_values)<1:
            print('Empty bin!')
            return
        plt.figure(figsize=(12,4))
        ax1=plt.subplot(1,2,1)
        ax1.hist(y_values, bins=nbins, ec='black')
        ax1.axvline(x0, c='r')
        ax1.set_xlabel(r'$y$', fontsize=20)
        ax1.set_ylabel(r'$N$', fontsize=20)
        ax2=plt.subplot(1,2,2)
        ax2.hist(y_values, bins=nbins, ec='black')
        ax2.axvline(x0, c='r')
        ax2.set_xlabel(r'$y$', fontsize=20)
        ax2.set_ylabel(r'$log_{10}(N)$', fontsize=20)
        ax2.set_yscale('log')
        plt.tight_layout()
        if axvlines is not None:
            for axvl in axvlines:
                ax1.axvline(axvl, color=[0,1,0])
                ax2.axvline(axvl, color=[0,1,0])
        plt.show()
        return 


