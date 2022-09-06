# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 15:43:08 2021
"""

import numpy as np
import scipy
import scipy.stats
from .distribution import Distribution
from .utils import *

        
        
class DistributionSequence:
    """
    The container class for a sequence of Distributions, whose sum is to be approximated if cumulants are known.
    """
    def __init__(self, distribution_order, order = 2):
        self.order = order
        assert order <= 3, "Edgeworth Expansion supports only for order in [1, 2, 3]."
        assert distribution_order >= order, f"The provided distribution's order {distribution_order} cannot calculate Edgworth upto to order {order}."
        

    def _approx_Fn_edgeworth(self, x, cumulants):
        """
        Compute the approximated value of Fn(x) with the given order Edgeworth expansion.
        Input:
            x - The data point where you want to evaluate Fn.
        """
        m = cumulants[0]
        inv_sigma_n = 1.0 / np.sqrt(cumulants[1])
        kap_3 = cumulants[2]
        x = (x - m) * inv_sigma_n
        expansion = (-1.0 / 6.0 * inv_sigma_n ** 3 * kap_3 * (x ** 2 - 1.0)) 
        if self.order > 1:
            kap_4 = cumulants[3]
            expansion -= (
            + 1.0 / 24.0 * inv_sigma_n ** 4 * kap_4 * (x ** 3 - 3 * x)
            + 1.0 / 72.0 * inv_sigma_n ** 6 * kap_3 ** 2 * (x ** 5 - 10 * x ** 3 + 15 * x)
            )
        if self.order == 3:
            kap_5 = cumulants[4]
            expansion -= (
            + 1.0 / 120.0 * inv_sigma_n ** 5 * kap_5 * (x ** 4 - 6 * x ** 2 + 3)
            + 1.0 / 144.0 * inv_sigma_n ** 7 * kap_3 * kap_4 * (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15)
            + 1.0 / 1296.0 * inv_sigma_n ** 9 * kap_3 ** 3 * (x ** 8 - 28 * x ** 6 + 210 * x ** 4 - 420 * x ** 2 + 105)
            )
        return scipy.stats.norm.cdf(x) + expansion * scipy.stats.norm.pdf(x)
    
    def _approx_log_fn_edgeworth(self, x, cumulants):
        """
        Compute the approximated value of log(fn(x)) with the given order Edgeworth expansion. Log is for numerical stability.
        Input:
            x - The data point where you want to evaluate fn.
        """
        m = cumulants[0]
        inv_sigma_n = 1.0 / np.sqrt(cumulants[1])
        kap_3 = cumulants[2]
        x = (x - m) * inv_sigma_n
        expansion = (1.0 + 1.0 / 6.0 * inv_sigma_n ** 3 * kap_3 * (x ** 3 - 3 * x))
        if self.order > 1:
            kap_4 = cumulants[3]
            expansion += (
             + 1.0 / 24.0 * inv_sigma_n ** 4 * kap_4 * (x ** 4 - 6 * x ** 2 + 3)
             + 1.0 / 72.0 * inv_sigma_n ** 6 * kap_3 ** 2 * (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15)
            )
        if self.order == 3:
            kap_5 = cumulants[4]
            expansion += (
            + 1.0 / 120.0 * inv_sigma_n ** 5 * kap_5 * (x ** 5 - 10 * x ** 3 + 15 * x)
            + 1.0 / 144.0 * inv_sigma_n ** 7 * kap_3 * kap_4 * (x ** 7 - 21 * x ** 5 + 105 * x ** 3 - 105 * x)
            + 1.0 / 1296.0 * inv_sigma_n ** 9 * kap_3 ** 3 * (x ** 9 - 36 * x ** 7 + 378 * x ** 5 - 1260 * x ** 3 + 945 * x)
            )
        return np.log(expansion) - np.log(2 * np.pi) / 2 - x ** 2 / 2
    

    
class IIDDistributionSequence(DistributionSequence):
    """
    The special case of DistributionSequence that is the sum of iid distributions.
    """
    def __init__(self, distribution, order = 2):
        super().__init__(distribution.max_order, order)
        #TODO
        self.distribution = distribution
        self.order = order
    
    def approx_Fn_edgeworth(self, x, numbers):
        """
        Compute the approximated value of Fn(x) with the given order Edgeworth expansion.
        Input:
            x - The data point where you want to evaluate Fn.
            numbers - number of copies of the iid distribution to be summed.
        """
        cumulants = [c * numbers for c in self.distribution.cumulants]
        return self._approx_Fn_edgeworth(x, cumulants)
    
    def error_bound_edgeworth_1(self, x, numbers):
        """
        Implement the second order bound only. Invariant of x!

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        numbers : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        Bn_ = (self.distribution.moments[1]) ** 0.5
        Knp = [xp / Bn_ ** (p + 1) for p, xp in enumerate(self.distribution.abs_moments)]
        lambda_np = [xp / Bn_ ** (p + 1) for p, xp in enumerate(self.distribution.moments)]
        lambda3, lambda4 = lambda_np[2], lambda_np[3]
        K3, K4 = Knp[2], Knp[3]
        K3_ = (K3 + self.distribution.abs_moments[0] / Bn_)
        t0, T = 1 / np.pi, 2 * np.pi * numbers ** 0.5 / K3_ ## Change T to be scale with n
        error = Omega1(T, lambda3, numbers) + Omega2(t0, T, lambda3, K3, K3_, K4, numbers) + Omega3(t0, T, lambda3, K3, K3_, K4, numbers)
        return error
        
        
     

class NIIDDistributionSequence(DistributionSequence):
    """
    The class that contains a sequence of non-iid Distributions, whose sum is to be approximated.
    """
    def __init__(self, distributions, order = 2):
        super().__init__(min(distributions.max_order), order)
        self.distributions = distributions
        self.order = order
        
    def approx_Fn_edgeworth(self, x, numbers):
        """
        Compute the approximated value of Fn(x) with the given order Edgeworth expansion upto the given number.
        Input:
            x - The data point where you want to evaluate Fn.
            numbers - the total numbers of the fronts of distributions to be summed.
        """
        cumulants = [sum([self.distributions[i].cumulants[j] for i in range(numbers)]) for j in range(4)]
        ##TODO: momeorization of the prefix sum called in a lru_cache fashion.
        return self._approx_Fn_edgeworth(x, cumulants)
    
    
