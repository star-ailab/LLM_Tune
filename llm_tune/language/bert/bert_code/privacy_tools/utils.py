#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:33:18 2021

"""

import numpy as np
import scipy
import scipy.stats
from .distribution import Distribution
from functools import lru_cache



"""
Some generic choice of T: 
    in first order:
        if aimed for 1/sqrt(n), then choose T = 2pi sqrt(n) / K3,n_tilde, t0 = 1/pi
        if aimed for 1/n, then choose 
"""

# def Omega1(t0, T, f1, f2):
#      quad1 = lambda t: Phi_t_i(t) * np.exp(-(T * t) ** 2 / 2) * (abs(f1(T * t)) + f2(T * t))
#      quad2 = lambda t: np.exp(-(T * t) ** 2 / 2) / t * (abs(f1(T * t)) + f2(T * t))
#      s1 = scipy.integrate.quad(quad1, 0, t0, 
#                               epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0] 
#      s2 = scipy.integrate.quad(quad1, t0, np.inf,
#                               epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0] 
#      return s1 * 2 + s2 / np.pi


def Omega1(T, lambda3, n, compare_both = True):
    
    #if compare_both:
    #    ret = Omega1_1st_order_upper_bound(T, lambda3, n)
        #print(f"Omega 1: The current numbers is {n}, the error using bound is {ret}.")
    
    const = abs(lambda3) / n ** 0.5 / 6 
    f1 = lambda Tt: 1 + const * Tt ** 3 
    quad1 = lambda t: Phi_t_i(t) * np.exp(-(T * t) ** 2 / 2) * f1(T * t)
    quad2 = lambda t: np.exp(-(T * t) ** 2 / 2) / t * f1(T * t)
    t0 = 1 / np.pi
    s1 = scipy.integrate.quad(quad1, 0, t0, 
                            epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0] 
    s2 = scipy.integrate.quad(quad2, t0, np.inf,
                            epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0] 
    ret = s1 * 2 + s2 / np.pi

    #print(f"Omega 1: The current numbers is {n}, the error using intgration is {ret}.")
    return ret


def Omega1_1st_order_upper_bound(T, lambda3, n):
    
    return 1.2533 / T + 0.3334 * abs(lambda3) / T / n ** 0.5 + 14.1961 / T ** 4 + abs(lambda3) * np.exp(- T ** 2 / 39.47842) / 3 / np.pi / n ** 0.5
 
def Omega2_1st_order_upper_bound(T):
    """
    The upper bound of page 17. The choice of t1 si 1/pi.
    Here the choice of T need to be exactly 2pi sqrt(n)/K3,n_tilde

    Parameters
    ----------
    T : numerical
        The choice of hyperparameter T.

    Returns 
    -------
    The first oder upper bound of Omega2.

    """
    return 67.0415 / T ** 4 + 1.2187 / T ** 2
 
def Omega2(t0, T, lambda3, K3, K3_, K4, n):
    # quad = lambda t: Phi_t(t) * abs(f_Sn(T * t))
    # return 2 * scipy.integrate.quad(quad, t0, 1,
    #                          epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0] 
    ret = Omega2_1st_order_upper_bound(T)
    #print(f"Omega 2: The current numbers is {n}, the error is {ret}.")
    return ret

def Omega3(t0, T, lambda3, K3, K3_, K4, n, eps = 0.1):
    # quad = lambda t: Phi_t(t) * abs(f_Sn(T * t) - np.exp(-(T * t) ** 2) / 2 * (f1(T * t) - f2(T * t)))
    # return 2 * scipy.integrate.quad(quad, 0, t0,
    #                          epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0] 
    i31 = I31(T, lambda3, K4, n, eps) 
    i32 = I32(T, K3, K3_, K4, n, eps) 
    i33 = I33(T, lambda3, K4, n, eps)
    #print(f"Omega 3: The current numbers is {n}, the i31 is {i31}, i32 is {i32}, i33 is {i33}.")
    return i31 + i32 + i33
    

def Phi_t(t):
    """
    The bound of the function Phi(t): |Phi(t)| <= 1.0253 / 2pi|t| = 0.16318 / |t|
    """
    return 0.16318 / abs(t)

def Phi_t_i(t):
    """
    The bound of the function: |Phi(t) - i/2pi t| <= 0.5(1 - |t| + pi ^ 2 * t^2 / 18)
    """
    return 0.5 * (1 - abs(t) + 0.54831 * t ** 2)


def I31(T, lambda3, K4, n, eps):
    quad = lambda u: u * np.exp(- u ** 2 / 2) * R1n(u, eps, lambda3, K4, n)
    l, m = 0, (2 * eps) ** 0.5 * (n / K4) ** 0.25
    return (0.327 * K4 / n * (1 / 12 + 1 / 4 / (1 - 3 * eps) ** 2) + 0.036278 * e1(eps) * lambda3 ** 2 / n + 
           0.32636 * scipy.integrate.quad(quad, l, m, epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0])

def I32(T, K3, K3_, K4, n, eps):
    return K3 / 3 / n ** 0.5 * J2(3, (2 * eps) ** 0.5 * (n / K4) ** 0.25, T / np.pi, K3_, K4, T, n)

def I33(T, lambda3, K4, n, eps):
    return lambda3 / 3 / n ** 0.5 * J1(3, (2 * eps) ** 0.5 * (n / K4) ** 0.25, T / np.pi, T)



def R1n(t, eps, lambda3, K4, n):
    temp = 1 / (1 - 3 * eps) ** 2
    return ((U11n(t, K4 / n) + U12n(t, K4 / n)) / 2 * temp + 
           e1(eps) * (t ** 8 * K4 ** 2 / 2 / n ** 2 * (1 / 24 + P1n(eps) / 2 * temp) ** 2
                      + t ** 7 * abs(lambda3) * K4 / 6 / n ** 1.5 * (1/24 + P1n(eps) / 2 * temp)))

def P1n(eps):
    return (144 + 48 * eps + 4 * eps ** 2 + (96 * (2 * eps) ** 0.5 + 32 * eps + 16 * 2 ** 0.5 * eps ** 1.5)) / 576

def e1(eps):
    return np.exp(eps ** 2 * (1 / 6 + 2 * P1n(eps) / (1 - 3 * eps) ** 2))

def U11n(t, K4dn):
    return t ** 6 / 24 * (K4dn) ** 1.5 + t ** 8 / 576 * K4dn ** 2

def U12n(t, K4dn):
    return (t ** 5 / 6 * K4dn ** 1.25 + t ** 6 / 36 * K4dn ** 1.5 + t ** 7 / 72 * K4dn ** 1.75)

def J1(p, l, m, T):
    quad = lambda u: u ** (p - 1) * np.exp(- u ** 2 / 2)
    return 0.16318 * scipy.integrate.quad(quad, l, m,
                             epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0] 

def J2(p, l, m, K3_, K4, T, n):
    quad = lambda u: Phi_t(u / T) * u ** p * np.exp(- u ** 2 / 2 * 
            (1 - 0.198324 * abs(u) * K3_ / n ** 0.5 - (K4 / n) ** 0.5))
    return 1/T * scipy.integrate.quad(quad, l, m,
                             epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0] 


    
## The following is the subgaussian bound part

"""
The code is to calculate the difference between the Xi and Xi~, 
which is easy to obtain the sub-Gaussian bound.

X = log(1 - p + p * exp(Z * mu - mu ** 2 / 2)), where Z ~ N(0, 1)

We will choose a parameter a, s.t. 
- X~ = X, when Z < a               (1)
- X~ = Z * mu - mu ** 2 / 2 + D,   (2)
where D = Z * mu - mu ** 2 / 2 - X|_{Z = a}.

This will always gaurantee that X~ >= X for all value of Z.

Thus, we only need to choose a, s.t. we still have E[X~] <= 0.
This will allow us to bound 
P[sum Xi >= eps] <= P[sum Xi~ >= eps] 
= P[sum Xi~ - sum E[Xi~] >= eps - sum E[Xi~]]
<= exp((eps - sum E[Xi~]) ** 2 / 2 / tau ** 2).

Here, the parameter tau is the sub-gaussian paramter of the variable X~,
we have: tau ** 2 = max{mu ** 2, (a - log(1-p)) ** 2 / 4}
The first is the SG paramter for the truncated normal part (2), and 
the second is the SG parameter for the bounded distribution (1). Specifically, 
we know that the distribution Xi is larger than log(1-p).
"""

def D(p, mu, z):
    return np.log((1 - p) / np.exp(z * mu - mu ** 2 / 2) + p)


def X(p, mu, z):
    return np.log(1 - p + p * np.exp(z * mu - mu ** 2 / 2))

def EX(p, mu):
    quad = lambda z: X(p, mu, z) * scipy.stats.norm.pdf(z)
    return scipy.integrate.quad(quad, -20, 20, 
                            epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0] 

def error_difference(p, mu, a):
    """
    This function evaluates the difference between E[X~] - E[X], 
    which I would like to roughly equal to E[X]/2.
    """
    quad = lambda z: (D(p, mu, a) - D(p, mu, z)) * scipy.stats.norm.pdf(z)
    return scipy.integrate.quad(quad, a, 20,
                            epsabs = 1e-8, epsrel = 1e-8, limit = 50)[0] 

@lru_cache(None)
def search_best_a(p, mu, ex, tol = 1e-8):
    """
    Would like the difference of E[x~] and E[X] to roughly equal to E[X]/2.
    Return the best a, and the E[x~] - E[X] evaluate at a.
    """
    ex = - ex
    left = right = 0
    while error_difference(p, mu, right) > ex / 2:
        left, right = right, right * 2 - np.log(1 - p)
        
    while right - left > tol:
        mid = (right + left) / 2
        err = error_difference(p, mu, mid)
        if err > ex / 2:
            left = mid
        else:
            right = mid
    return mid, err - ex
    
    
def a_plus(a):
    return scipy.stats.norm.pdf(a) / (1 - scipy.stats.norm.cdf(a))
    