import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate

class ErrorSurface:
    def __init__(self, X, Y, Nx=50, Ny=50, 
                 exp_step=False, dx_expstep=1.05, dy_expstep=1.05, 
                 method='linear', Nx_grid=200, Ny_grid=200):
        """ Initialize the class to compute the error-surface

        Parameters
        ----------
        X : 1d-array
            Predicted or recovered data for a specific feature
        
        Y : 1d-array
            True values that correspon to X
        
        Nx : int
            Number of points used in the x-direction for the non-interpolated surface. 
            If exp_step=True, the value in unput is ignored and Nx is then obtained from
            the non-uniform grid
        
        Ny : int
            as Nx, but in the y-direction
        
        exp_step : bool
            Wse exponential-step instead of uniform. Can be used only if X[i]>=0 and Y[i]>=0 for every i
        
        dx_expstep : float
            exp_step in the x-direction for the non-interpolated surface (must be >1)
        
        dy_expstep : float
            as dx_expstep, but in the y-direction
        
        method : str
            Method used for the surface-interpolation. Can be 'nearest', 'linear' or 'cubic'
        
        Nx_grid : int
            number of points used in the x-direction for the intepolated grid
        
        Ny_grid : int
            as Nx_grid, but in the y-direction
        """

        self.X     = X
        self.Y     = Y
        self.X_min = min(X)
        self.X_max = max(X)
        self.Y_min = min(Y)
        self.Y_max = max(Y)
        # buld non-interpolated grid
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
        self.Nx_grid = Nx_grid
        self.Ny_grid = Ny_grid
        self.__interpolate_surface()
        return


    def __uniform_step(self, Nx, Ny):
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
        Nx_grid = self.Nx_grid
        Ny_grid = self.Ny_grid
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
        x_interp = np.linspace(X_min, X_max, Nx_grid)
        y_interp = np.linspace(Y_min, Y_max, Ny_grid)
        xg_interp, yg_interp = self.__tmesh(x_interp, y_interp)
        # put the surface S and the original
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

    def plot_surf(self, show_grid=True, log_bar=False, log_scale=False, bisectrix=True):
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
        Nx_grid = self.Nx_grid
        Ny_grid = self.Ny_grid
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
        #for i in range(Nx_grid):
        #    for j in range(Ny_grid):
        #        if np.isnan(S_interp[i,j]):
        #            S_interp[i,j] = 0;
        #        if S_interp[i,j]<1:
        #            S_interp_plot[i,j] = 0
        #        else:
        #            S_interp_plot[i,j] = np.log10(S_interp[i,j])
        fig,ax=plt.subplots(1,1, figsize=(10,6))
        cp = ax.contourf(xg_interp, yg_interp, S_interp_plot, cmap=plt.get_cmap('viridis'), levels=levels)
        cb = fig.colorbar(cp)
        ax.set_ylim([Y_min,Y_max])
        ax.set_xlim([X_min,X_max])
        ax.plot(self.x_interp, self.x_interp,c='r',lw=1)
        if x0_line is not None:
            ax.axvline(x0_line, color=[1,0,1])
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

    def plot_hist(self, x0, nbins=50):
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
        plt.subplot(1,2,1)
        plt.hist(y_values, bins=nbins, ec='black')
        plt.axvline(x0, c='r')
        plt.xlabel(r'$y$', fontsize=20)
        plt.ylabel(r'$N$', fontsize=20)
        plt.subplot(1,2,2)
        plt.hist(y_values, bins=nbins, ec='black')
        plt.axvline(x0, c='r')
        plt.xlabel(r'$y$', fontsize=20)
        plt.ylabel(r'$log_{10}(N)$', fontsize=20)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
        return 


