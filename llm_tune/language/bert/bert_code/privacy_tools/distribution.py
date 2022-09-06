# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 15:01:15 2021
"""
import numpy as np
import scipy
import scipy.stats

## This is the Edgeworth Class that can do the calculation

class Distribution:
    """
    The base class for computing the Edgeworth approximation of the sum of distribution
    """
    
    def __init__(self, dens_func, log_likelihood_ratio_func, max_order):
        """
        Initiate one distribution follows the dens_func, and take values via log_likelihood_ratio_func. 
        For example, if the desired distribution is f(X), then dens_func is the density of X, and the
        log_likelihood_ratio_func is f. The name log_likelihood_ratio_func is specifically for the purpose 
        of doing Edgeworth expansion, where the transformation f is exactly the log_likelihood_ratio.
        @param:
            dens_func: the density function of the distribution
            log_likelihood_ratio_func: the value function
            max_order: the order of Edgeworth expansion, support 1, 2, 3
        """
        assert max_order in [1, 2, 3], "the Edgworth Expansion only support for order in [1, 2, 3]!"
        self.dens_func = dens_func
        self.log_likelihood_ratio_func = log_likelihood_ratio_func
        self.max_order = max_order
        
    @property
    def moments(self):
        """
        Get the moments of the given distribution. Upto the necessary order for Edgeworth.
        """
        return [self._compute_moments(order) for order in range(1, self.max_order + 3)]
        
    def _compute_moments(self, order, left = -np.inf, right = np.inf):
        """
        Get moment of a specific ORDER of the given distribution.
        """
        integrand = lambda x: self.log_likelihood_ratio_func(x) ** order * self.dens_func(x)
        return scipy.integrate.quad(integrand, left, right, 
                                 epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0]
    
    
    @property
    def cumulants(self):
        """
        Get the cumulants of the given distribution. Upto the necessary order for Edgeworth.
        """
        moments = self.moments
        kappas = [0] * len(moments)
        kappas[0] = moments[0]
        kappas[1] = moments[1] - moments[0] ** 2
        kappas[2] = moments[2] - 3 * moments[1] * moments[0] + 2 * moments[0] ** 3
        if len(moments) > 3:
            kappas[3] = (
                moments[3]
                - 4 * moments[2] * moments[0]
                - 3 * moments[1] ** 2
                + 12 * moments[1] * moments[0] ** 2
                - 6 * moments[0] ** 4
            )
        if len(moments) > 4:
            kappas[4] = (
                moments[4]
                - 5 * moments[3] * moments[0]
                - 10 * moments[2] * moments[1]
                + 20 * moments[2] * moments[0] ** 2
                + 30 * moments[1] ** 2 * moments[0]
                - 60 * moments[1] * moments[0] ** 3
                + 24 * moments[0] ** 5
            )      
        return kappas
    
    @property
    def abs_moments(self):
        """
        Get the absolute moments upto the desired order. This implementation is not 

        Returns
        -------
        am : TYPE
            DESCRIPTION.

        """
        am = self.moments
        for i in range(0, self.max_order + 2, 2):
            integrand = lambda x: abs(self.log_likelihood_ratio_func(x)) ** (i + 1) * self.dens_func(x)
            am[i] = scipy.integrate.quad(integrand, -np.inf, np.inf, 
                                 epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0]
        return am