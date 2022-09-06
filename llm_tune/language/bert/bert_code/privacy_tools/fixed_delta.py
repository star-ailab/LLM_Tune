#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 20:22:55 2022

"""

from prv_accountant import Accountant
import numpy as np
from .eps_delta_edgeworth import * 
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_eps_poisson
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_mu_poisson
from prv_accountant import Accountant
import numpy as np
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition 
from autodp import mechanism_zoo, transformer_zoo



def rdp(step, sigma, delta, prob):
    mech = mechanism_zoo.GaussianMechanism(sigma=sigma)
    subsample = transformer_zoo.AmplificationBySampling(PoissonSampling=True) # by default this is using poisson sampling
      
    SubsampledGaussian_mech = subsample(mech,prob,improved_bound_flag=True)
    compose = transformer_zoo.Composition()
    mech = compose([SubsampledGaussian_mech],[step])
    rdp_total = mech.RenyiDP
    noisysgd = Mechanism()
    noisysgd.propagate_updates(rdp_total,type_of_update='RDP')
    return mech.get_approxDP(delta=delta)

#%%

regimes = ["1\sqrt{m}", "1\sqrt{m logm}", "C sqrt{logm \ m}", "C\m"] # The regimes to be tried

epoch = 1
#prob = 1e-2
num_examples = 100000
sigma = 0.8
number_lst = points = [50000, 100000, 150000, 200000, 300000, 400000, 500000, 700000, 850000, 1000000]
#total_epoches = 1 / prob
delta = 0.1
C = 0.08
regime = regimes[2]

def calc_prob(C, n, regime):
    prob = C / n ** 0.5
    if regime == regimes[1]:
        prob = prob / np.log(n) ** 0.5
    elif regime == regimes[2]:
        prob = prob * np.log(n) ** 0.5
    elif regime == regimes[3]:
        prob = prob / n ** 0.5
    return prob
## The key to retrive the pickled data
key = (delta, sigma, C, regime)
#%%

filename = "data/fixed_delta_vary_p/Fixed_delta_vary_p_plot.pickle"
if os.path.isfile(filename):
    with open(filename, "rb") as f:
        cache_list = pickle.load(f)
else:
    cache_list = {}

#%%

if key in cache_list:
    cache = cache_list[key]
    eps_gdp = cache["eps_gdp"]
    eps_ew_est = cache["eps_ew_est"]
    eps_ew_upp = cache["eps_ew_upp"]
    eps_ew_low = cache["eps_ew_low"]
    eps_low = cache["eps_low"]
    eps_upper = cache["eps_upper"]
    eps_rdp = cache["eps_rdp"]
    # edgeeps = cache["edgeeps"]
    # edgeeps3 = cache["edgeeps3"]

else:
    cache = {}
    
    ## GDP
    print("\n\nNow calculating GDP...")
    eps_gdp = []
    # for n in tqdm(number_lst):
    #     prob = calc_prob(C, n, regime)
    
    #     eps_gdp.append(compute_eps_poisson(prob * n, sigma, num_examples, num_examples * prob, delta))
    cache["eps_gdp"] = eps_gdp
   

#%%    ## FFT (PRV, Gopi)
    print("\n\nNow calculating FFT...")
    eps_low, eps_upper = [], []
    for n in tqdm(number_lst):
        prob = calc_prob(C, n, regime)
        accountant = Accountant(
         	noise_multiplier=sigma,
         	sampling_probability=prob,
         	delta=delta,
         	eps_error=0.1,
        max_compositions = n + 200000
        )
        results = accountant.compute_epsilon(num_compositions=n)
        eps_low.append(results[0])
        eps_upper.append(results[2])
    cache["eps_low"] = eps_low
    cache["eps_upper"] = eps_upper
   #%% 
    
    ## Edgeworth
    eps_ew_est, eps_ew_upp, eps_ew_low = [], [], []
    print("\n\nNow calculating Edgeworth...")
    for n in tqdm(number_lst):
        prob = calc_prob(C, n, regime)
        sgd = GaussianSGD(sigma = sigma, p = prob, order = 1)
        eest, elow, eupp = sgd.approx_eps_from_delta_edgeworth(delta, n)
        eps_ew_est.append(eest)
        eps_ew_upp.append(eupp)
        eps_ew_low.append(elow)
        
    cache["eps_ew_est"] = eps_ew_est
    cache["eps_ew_upp"] = eps_ew_upp
    cache["eps_ew_low"] = eps_ew_low
    
    
    ## RDP: (Mironov)
    print("\n\nNow calculating RDP...")
    eps_rdp = []
    for n in tqdm(number_lst):
        prob = calc_prob(C, n, regime)
        eps_rdp.append(rdp(n, sigma, delta, prob))
    
    cache["eps_rdp"] = eps_rdp
#%%
    ## save it to pickle
    cache_list[key] = cache
    with open(filename, "wb") as f:
        pickle.dump(cache_list, f)


#%%  
# Plot
# plt.plot(number_lst, eps_low, label = "FFT_LOW", linestyle = "dashed", color = "black")
# plt.plot(number_lst, eps_upper, label = "FFT_UPP", linestyle = "dashed", color = "black")
#plt.plot(number_lst, eps_gdp, label = "GDP")
figure(figsize=(4, 4))
# plt.plot(number_lst, eps_rdp, label = "RDP")
# plt.plot(number_lst, eps_ew_est, label = "EW_EST")
# plt.plot(number_lst, eps_ew_upp, label = "EW_UPP")
# plt.plot(number_lst, eps_ew_low, label = "EW_UPP")

# #ax.plot(points, eps_estimate, label = "EPS_EST")
# #ax.plot(points, edgeeps, label = "Edgeworth (2nd)")
# #ax.plot(points, edgeeps3, label = "Edgeworth (3rd)")
# #ax.plot(points, edgeeps_prv, label = "Edgeworth")

# plt.legend()
# plt.ylabel(r"$\epsilon$")
# plt.xlabel("m")
# #plt.title(r"$\delta$" + f" = {delta}")
# #plt.title("Eps as function of iterations.")

# plt.savefig(f"figs/fixed_delta_vary_p/Fixed_delta_vary_p_key={key}.pdf", format='pdf',  bbox_inches = 'tight')
# plt.show()

plt.plot(number_lst, eps_low, label = "FFT_LOW", linestyle = "dashed", color = "black")
plt.plot(number_lst, eps_upper, label = "FFT_UPP", linestyle = "dashed", color = "black")
#plt.plot(number_lst, eps_gdp, label = "GDP")
plt.plot(number_lst, eps_rdp, label = "RDP")
plt.plot(number_lst, eps_ew_est, label = "EW_EST")
plt.plot(number_lst, eps_ew_upp, label = "EW_UPP")
plt.plot(number_lst, eps_ew_low, label = "EW_LOW")

#ax.plot(points, eps_estimate, label = "EPS_EST")
#ax.plot(points, edgeeps, label = "Edgeworth (2nd)")
#ax.plot(points, edgeeps3, label = "Edgeworth (3rd)")
#ax.plot(points, edgeeps_prv, label = "Edgeworth")

plt.legend(fontsize=10)
plt.ylabel(r"$\epsilon$", fontsize=15, rotation=90)
plt.xlabel("m", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#plt.title(r"$\delta$" + f" = {delta}")
#plt.title("Eps as function of iterations.")

plt.savefig(f"figs/fixed_delta_vary_p/Fixed_delta_vary_p_key={key}_with_FFT.pdf", format='pdf', bbox_inches = 'tight')