#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:27:35 2021

"""

import numpy as np
import scipy
import scipy.stats
from .distribution import Distribution
from .Edgeworth import *
from .utils import *
from functools import lru_cache

class LogLikelihoodPair:
    """
    A wrapper of the combination of two DistributionSequence that support taking two densities and their log_likelihood_ratio,
    and generate the edgeworth of the likelihood ratio under each distribution sequence using the repeated method.
    """
    def __init__(self, log_likelihood_ratio_func, dens_func_X, dens_func_Y, order = 2):
        """
        log_likelihood_ratio_func should be log(dens_func_Y / dens_func_X), 
        but should be implemented using log-of-exp technique to avoid underflow.
        """
        self.p = None ## only work if consider Gaussian
        self.mu = None
        self.dsX = IIDDistributionSequence(Distribution(dens_func_X, log_likelihood_ratio_func, 2), order = order)
        self.dsY = IIDDistributionSequence(Distribution(dens_func_Y, log_likelihood_ratio_func, 2), order = order)
    
    def approx_delta_from_eps_edgeworth(self, eps, numbers, uniform = True):
        """
        The corresponding epsilon and delta for the given cutoff X for testing sum(Xi) vs sum(Yi).
        
        Parameters
        ----------
        eps : the desired epsilon
        numbers : number of iterations
        at_esp : the value at which we evaluate ex

        Returns
        -------
        the 3-tuple of (estimated delta, lower bound of delta, upper bound of delta)

        """
        # logfx = self.dsX.approx_log_fn_edgeworth(x)
        # logfy = self.dsY.approx_log_fn_edgeworth(x)

        Fx = self.dsX.approx_Fn_edgeworth(eps, numbers)
        Fy = self.dsY.approx_Fn_edgeworth(eps, numbers)
        if uniform:
            ex = self.approx_edgeworth_errorX(numbers)
        else:
            ex = self.get_subgaussian_bound(eps, numbers)
        #print("")
        #print(f"For eps = {eps}, the ex is {ex}, sub_ex is {sub_ex}.")
        
            # temp = np.exp(- 0.1 * (at_eps - numbers * self.dsX.distribution.moments[0]) ** 2)
            # if ex > temp:
            #     ex = np.exp(- 0.1 * (eps - numbers * self.dsX.distribution.moments[0]) ** 2)
        ey = self.approx_edgeworth_errorY(numbers)
        ## The current problem is that the uniform bound is too large to be timed with e^eps for large eps.
        return (1 - Fy - np.exp(eps) * (1 - Fx), 
                1 - Fy - ey - np.exp(eps) * (1 - Fx + ex),
                1 - Fy + ey - np.exp(eps) * (1 - Fx - ex)
               )
    
    def get_subgaussian_bound(self, eps, numbers):
        a, ex_ = search_best_a(self.p, self.mu, self.dsX.distribution.moments[0])
        tau = max((a - np.log(1-self.p) + self.mu * (a_plus(a) - a)) / 2, self.mu)
        
        ## subgaussian rate
        e1 = np.exp(-(eps - numbers * ex_) ** 2 / 8 / numbers / tau ** 2)
        ## treat as CLT, since Edgeworth approximate better, so triangle with CLT is of course okay.
        e2 = 1 - scipy.stats.norm.cdf((eps - numbers * self.dsX.distribution.moments[0]) / numbers ** 0.5)
        #print(f"E[X~] is {ex_}, and tau is {tau}, a is {a}.   e1 = {e1}, e2 = {e2}")
        return max(e1, e2)
    
    def approx_eps_from_delta_edgeworth(self, delta, numbers, method = "estimate"): #default uniform
        ## a heuristic initialization
        start = (self.dsX.distribution.moments[0] + self.dsY.distribution.moments[0]) / 2 * numbers
        gest = lambda eps: self.approx_delta_from_eps_edgeworth(eps, numbers)[0] - delta
        epsest = scipy.optimize.fsolve(gest, x0 = start, xtol = 1e-12)[0]
        start = epsest
        if method != "estimate":
            ## find eps +:
            ##uniform upper bound
            gp = lambda eps: self.approx_delta_from_eps_edgeworth(eps, numbers, uniform = True)[2] - delta
            epsp_uniform = scipy.optimize.fsolve(gp, x0 = start, xtol = 1e-12)[0]
            ##exponential upper bound
            if method != "uniform":
                gp = lambda eps: self.approx_delta_from_eps_edgeworth(eps, numbers, uniform = False)[2] - delta
                epsp_exp = scipy.optimize.fsolve(gp, x0 = start, xtol = 1e-12)[0]
            ## find eps -:
            ## uniform
            gm = lambda eps: self.approx_delta_from_eps_edgeworth(eps, numbers, uniform = True)[1] - delta
            epsm_uniform = scipy.optimize.fsolve(gm, x0 = start, xtol = 1e-12)[0]
            ##exponential 
            if method != "uniform":
                gm = lambda eps: self.approx_delta_from_eps_edgeworth(eps, numbers, uniform = False)[1] - delta
                epsm_exp = scipy.optimize.fsolve(gm, x0 = start, xtol = 1e-12)[0]
            
            tol = 1e-8
            if epsest > epsp_uniform + tol or epsest < epsm_uniform - tol:
                print("The uniform estimate fails to converge!")
            if method != "uniform" and (epsest > epsp_exp+ tol or epsest < epsm_exp - tol):
                print("The exponential estimate fails to converge!")
        else:
            return epsest
        
        if method != "uniform":    
            return epsest, epsm_uniform, epsp_uniform, epsm_exp, epsp_exp 
        else:
            return epsest, epsm_uniform, epsp_uniform
    
    @lru_cache(None)
    def approx_edgeworth_errorX(self, numbers):
        """
        Parameters
        ----------
        numbers : number of iterations.
        Returns
        -------
        Errors of edgeworth expansion on X
        """
        
        return self.dsX.error_bound_edgeworth_1(0, numbers)
    
    @lru_cache(None)
    def approx_edgeworth_errorY(self, numbers):
        """
        Parameters
        ----------
        numbers : number of iterations.
        Returns
        -------
        Errors of edgeworth expansion on Y
        """
        return self.dsY.error_bound_edgeworth_1(0, numbers)
    
    
class GaussianSGD(LogLikelihoodPair):
    """
    Initiate a SGD accountant, whose noise_multiplier is sigma, and subsampling rate is p (Poisson sampling).
    """
    def __init__(self, sigma, p, order = 1):
        mu = 1 / sigma
        def log_likelihood_ratio_func(x):
            if x > 0:
                return mu * x + np.log((1 - p) * np.exp(-mu * x) + p * np.exp(- mu * mu / 2))
            return np.log(1 - p + p * np.exp(mu * x - mu * mu / 2))
        def dens_func_X(x):
            return scipy.stats.norm.pdf(x)
        def dens_func_Y(x):
            return (1 - p) * scipy.stats.norm.pdf(x) + p * scipy.stats.norm.pdf(x, loc = mu)
        
        super().__init__(log_likelihood_ratio_func, dens_func_X, dens_func_Y, order)
        self.p = p
        self.mu = mu

